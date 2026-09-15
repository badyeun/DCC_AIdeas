
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

from common import (
    SAMPLE_RATE,
    INDEX_TO_GENDER,
    load_annotation,
    load_caller_audio,
)

from model import WavLMHead


# =========================================================
# 3-chunk
# =========================================================

def make_three_chunks(waveform, max_samples):

    waveform = torch.as_tensor(
        waveform,
        dtype=torch.float32
    ).flatten()

    n = waveform.numel()

    # 8초 이하 → 한 번만 처리
    if n <= max_samples:

        if n < max_samples:

            waveform = torch.nn.functional.pad(
                waveform,
                (0, max_samples - n)
            )

        return [waveform]

    # 8초 초과 → 앞 / 중간 / 뒤
    max_start = n - max_samples

    starts = [
        0,
        max_start // 2,
        max_start,
    ]

    return [
        waveform[
            start:start + max_samples
        ]
        for start in starts
    ]


# =========================================================
# WavLM embedding
# =========================================================

def extract_embedding(
    waveform,
    feature_extractor,
    encoder,
    device,
    max_seconds,
):

    max_samples = int(
        SAMPLE_RATE * max_seconds
    )

    chunks = make_three_chunks(
        waveform,
        max_samples
    )

    chunk_embeddings = []

    for chunk in chunks:

        audio_np = (
            chunk
            .detach()
            .cpu()
            .numpy()
        )

        inputs = feature_extractor(
            audio_np,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )

        input_values = (
            inputs["input_values"]
            .to(device)
        )

        attention_mask = inputs.get(
            "attention_mask"
        )

        if attention_mask is not None:

            attention_mask = (
                attention_mask
                .to(device)
            )

        with torch.inference_mode():

            outputs = encoder(
                input_values=input_values,
                attention_mask=attention_mask,
            )

            hidden = outputs.last_hidden_state

            # Validation 3-chunk embedding을 만들 때와
            # 동일한 pooling
            if attention_mask is None:

                pooled = hidden.mean(
                    dim=1
                )

            else:

                feature_mask = (
                    encoder
                    ._get_feature_vector_attention_mask(
                        hidden.shape[1],
                        attention_mask,
                    )
                )

                mask = (
                    feature_mask
                    .unsqueeze(-1)
                    .to(hidden.dtype)
                )

                pooled = (
                    (hidden * mask).sum(dim=1)
                    /
                    mask.sum(dim=1).clamp(
                        min=1.0
                    )
                )

        chunk_embeddings.append(
            pooled
            .squeeze(0)
            .cpu()
            .numpy()
        )

    # 앞 / 중간 / 뒤 embedding 평균
    embedding = np.mean(
        np.stack(
            chunk_embeddings,
            axis=0
        ),
        axis=0,
    )

    return embedding.astype(
        np.float32
    )


# =========================================================
# scaler
# =========================================================

def apply_scaler(
    embedding,
    scaler_mean,
    scaler_scale,
):

    # sklearn StandardScaler.transform과 동일:
    # (x - mean) / scale
    scaled = (
        embedding - scaler_mean
    ) / scaler_scale

    return scaled.astype(
        np.float32
    )


# =========================================================
# JSON 찾기
# =========================================================

def build_json_map(label_dir):

    json_files = sorted(
        label_dir.rglob("*.json")
    )

    json_map = {}

    for path in json_files:

        # 같은 stem이 중복되면 위험하므로 검사
        if path.stem in json_map:

            raise RuntimeError(
                f"중복 JSON stem 발견: "
                f"{path.stem}"
            )

        json_map[path.stem] = path

    return json_map


# =========================================================
# main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio_dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--label_dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--ckpt_path",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=str,
    )

    args = parser.parse_args()


    # -----------------------------------------------------
    # Path
    # -----------------------------------------------------

    audio_dir = Path(
        args.audio_dir
    )

    label_dir = Path(
        args.label_dir
    )

    ckpt_path = Path(
        args.ckpt_path
    )

    output_path = Path(
        args.output
    )


    if not audio_dir.exists():

        raise FileNotFoundError(
            f"audio_dir 없음: "
            f"{audio_dir}"
        )

    if not label_dir.exists():

        raise FileNotFoundError(
            f"label_dir 없음: "
            f"{label_dir}"
        )

    if not ckpt_path.is_file():

        raise FileNotFoundError(
            f"checkpoint 파일 없음: "
            f"{ckpt_path}"
        )


    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )


    # -----------------------------------------------------
    # checkpoint
    # -----------------------------------------------------

    print(
        "Checkpoint 로드 중..."
    )

    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = [
        "head_state_dict",
        "input_dim",
        "hidden_dim",
        "dropout",
        "scaler_mean",
        "scaler_scale",
        "model_name",
        "max_seconds",
    ]

    for key in required_keys:

        if key not in checkpoint:

            raise KeyError(
                f"checkpoint에 "
                f"'{key}' 없음"
            )


    scaler_mean = np.asarray(
        checkpoint["scaler_mean"],
        dtype=np.float32,
    )

    scaler_scale = np.asarray(
        checkpoint["scaler_scale"],
        dtype=np.float32,
    )

    model_name = checkpoint[
        "model_name"
    ]

    max_seconds = float(
        checkpoint["max_seconds"]
    )


    print(
        "Model:",
        model_name
    )

    print(
        "Chunk strategy:",
        checkpoint.get(
            "inference_strategy",
            "front_middle_end_mean"
        )
    )


    # -----------------------------------------------------
    # WavLM
    # -----------------------------------------------------

    print(
        "\nWavLM encoder 로드 중..."
    )

    feature_extractor = (
        AutoFeatureExtractor
        .from_pretrained(
            model_name
        )
    )

    encoder = (
        AutoModel
        .from_pretrained(
            model_name
        )
        .to(device)
    )

    encoder.eval()

    for parameter in (
        encoder.parameters()
    ):
        parameter.requires_grad = False


    print(
        "WavLM encoder 로드 완료."
    )


    # -----------------------------------------------------
    # Head
    # -----------------------------------------------------

    head = WavLMHead(
        input_dim=int(
            checkpoint["input_dim"]
        ),
        hidden_dim=int(
            checkpoint["hidden_dim"]
        ),
        dropout=float(
            checkpoint["dropout"]
        ),
    ).to(device)

    head.load_state_dict(
        checkpoint[
            "head_state_dict"
        ]
    )

    head.eval()


    # -----------------------------------------------------
    # 파일 목록
    # -----------------------------------------------------

    audio_files = sorted(
        audio_dir.rglob("*.wav")
    )

    if len(audio_files) == 0:

        raise RuntimeError(
            f"WAV 파일이 없습니다: "
            f"{audio_dir}"
        )


    print(
        f"\nAudio files: "
        f"{len(audio_files):,}"
    )


    print(
        "JSON index 생성 중..."
    )

    json_map = build_json_map(
        label_dir
    )

    print(
        f"JSON files: "
        f"{len(json_map):,}"
    )


    # -----------------------------------------------------
    # inference
    # -----------------------------------------------------

    predictions = []

    for i, audio_path in enumerate(
        audio_files,
        start=1
    ):

        json_path = json_map.get(
            audio_path.stem
        )

        if json_path is None:

            raise FileNotFoundError(
                f"JSON 없음: "
                f"{audio_path.name}"
            )


        # ---------------------------------------------
        # annotation
        # ---------------------------------------------

        annotation = load_annotation(
            json_path
        )


        # ---------------------------------------------
        # 신고자 음성만 추출
        # ---------------------------------------------

        waveform = load_caller_audio(
            audio_path,
            annotation,
            timestamp_unit="auto",
        )


        # ---------------------------------------------
        # WavLM 3-chunk embedding
        # ---------------------------------------------

        embedding = extract_embedding(
            waveform,
            feature_extractor,
            encoder,
            device,
            max_seconds,
        )


        # ---------------------------------------------
        # scaler
        # ---------------------------------------------

        embedding_scaled = (
            apply_scaler(
                embedding,
                scaler_mean,
                scaler_scale,
            )
        )

        x = torch.from_numpy(
            embedding_scaled
        ).unsqueeze(0).to(
            device
        )


        # ---------------------------------------------
        # Head
        # ---------------------------------------------

        with torch.inference_mode():

            logits = head(x)

            pred_idx = int(
                logits.argmax(
                    dim=1
                ).item()
            )


        gender = INDEX_TO_GENDER[
            pred_idx
        ]


        predictions.append(
            [
                audio_path.name,
                gender,
            ]
        )


        if (
            i % 100 == 0
            or i == len(audio_files)
        ):

            print(
                f"[{i:,}/"
                f"{len(audio_files):,}]"
            )


    # -----------------------------------------------------
    # CSV
    # -----------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "audio file name",
                "gender",
            ]
        )

        writer.writerows(
            predictions
        )


    print(
        "\n============================"
    )

    print(
        "✅ Mission 1 inference 완료"
    )

    print(
        "Output:",
        output_path
    )

    print(
        "Rows:",
        len(predictions)
    )

    print(
        "============================"
    )


if __name__ == "__main__":
    main()

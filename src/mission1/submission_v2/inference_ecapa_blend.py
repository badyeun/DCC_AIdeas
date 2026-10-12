"""Mission 1 최종 추론: 예은 WavLM 3모델 앙상블 + ECAPA 성별 모델 (EXP-012).

    P(F) = (1 - b) * [0.4 * Attention+F0 + 0.4 * Stats+F0 + 0.2 * Stats(No F0)] + b * ECAPA-LR
    b = ecapa_gender/ecapa_lr.npz의 blend (내부 holdout으로 고른 0.75)

예은의 `inference_3model_final.py`는 고치지 않고 모델·특징 함수를 그대로 불러 쓴다.
파일 하나에 대한 예은 3모델 계산은 그 스크립트 `main()`의 반복문 본문과 같은 식이다.

한쪽이 실패하면 다른 쪽 확률만 쓰고, 둘 다 실패하면 예은 스크립트처럼 F(Training 다수 클래스)로 둔다.
`--save_probs`(선택)를 주면 파일별 확률을 npz로 저장한다 — 학습 때 값과 같은지 검증하는 용도.

사용:
    python inference.py --audio_dir AUDIO_DIR --label_dir LABEL_DIR --ckpt_path CKPT --output ./outputs/mission1.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import inference_3model_final as yeeun  # noqa: E402  (예은 원본, 수정하지 않음)
from common import INDEX_TO_GENDER, load_annotation, load_caller_audio, waveform_to_f0_feature  # noqa: E402
from ecapa_gender import EcapaGender  # noqa: E402

YEEUN_WEIGHTS = (0.4, 0.4, 0.2)


def yeeun_probabilities(audio_path, annotation, items, feature_extractor, encoder, device) -> np.ndarray:
    """예은 3모델의 F 확률 [Attention+F0, Stats+F0, Stats(No F0)]. 예은 main() 반복문과 같은 계산."""
    f0_item, stats_item, no_f0_item = items
    waveform = load_caller_audio(audio_path, annotation, timestamp_unit="auto")
    chunk_embeddings, stats_raw, chunk_mask = yeeun.extract_chunk_features_combined(
        waveform, feature_extractor, encoder, device, 8.0)
    x_raw = chunk_embeddings[None, :, :]
    mask_np = chunk_mask[None, :]
    mask_tensor = torch.from_numpy(mask_np).to(device)

    f0_raw = np.asarray(waveform_to_f0_feature(waveform), dtype=np.float32).reshape(1, 6)
    if not np.isfinite(f0_raw).all():
        raise ValueError("Invalid F0 features")

    f0_x_scaled = (x_raw - f0_item["scaler_mean"][None, None, :]) / f0_item["scaler_scale"][None, None, :]
    f0_x_scaled = f0_x_scaled.astype(np.float32)
    f0_x_scaled[~mask_np.astype(bool)] = 0.0
    f0_scaled = ((f0_raw - f0_item["f0_scaler_mean"][None, :]) / f0_item["f0_scaler_scale"][None, :]).astype(np.float32)
    with torch.inference_mode():
        f0_logits = f0_item["model"](torch.from_numpy(f0_x_scaled).to(device),
                                     torch.from_numpy(f0_scaled).to(device), mask_tensor.bool())
        f0_probs = torch.softmax(f0_logits, dim=1).squeeze(0).cpu().numpy()

    stats_scaled = (stats_raw[None, :, :] - stats_item["stats_scaler_mean"][None, None, :]) / stats_item["stats_scaler_scale"][None, None, :]
    stats_scaled = stats_scaled.astype(np.float32)
    stats_scaled[:, ~chunk_mask.astype(bool), :] = 0.0
    stats_f0_scaled = ((f0_raw - stats_item["f0_scaler_mean"][None, :]) / stats_item["f0_scaler_scale"][None, :]).astype(np.float32)
    if not np.isfinite(stats_scaled).all():
        raise ValueError("Invalid Stats features")
    if not np.isfinite(stats_f0_scaled).all():
        raise ValueError("Invalid Stats F0 features")
    with torch.inference_mode():
        stats_logits = stats_item["model"](torch.from_numpy(stats_scaled).to(device),
                                           torch.from_numpy(stats_f0_scaled).to(device), mask_tensor.bool())
        stats_probs = torch.softmax(stats_logits, dim=1).squeeze(0).cpu().numpy()

    no_f0_scaled = (stats_raw[None, :, :] - no_f0_item["stats_scaler_mean"][None, None, :]) / no_f0_item["stats_scaler_scale"][None, None, :]
    no_f0_scaled = no_f0_scaled.astype(np.float32)
    no_f0_scaled[:, ~chunk_mask.astype(bool), :] = 0.0
    with torch.inference_mode():
        no_f0_logits = no_f0_item["model"](torch.from_numpy(no_f0_scaled).to(device), mask_tensor.bool())
        no_f0_probs = torch.softmax(no_f0_logits, dim=1).squeeze(0).cpu().numpy()

    # 클래스 1 = F (common.GENDER_TO_INDEX)
    return np.array([f0_probs[1], stats_probs[1], no_f0_probs[1]], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True, type=str, nargs="+")
    parser.add_argument("--label_dir", required=True, type=str, nargs="+")
    parser.add_argument("--ckpt_path", required=False, type=str, default=None,
                        help="공식 명령 형식 호환용. 체크포인트는 이 폴더 기준 상대 경로로 읽는다.")
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--save_probs", type=str, default=None, help="(검증용) 파일별 확률 npz 경로")
    args = parser.parse_args()

    audio_dirs = [Path(path) for path in args.audio_dir]
    label_dirs = [Path(path) for path in args.label_dir]
    for folder in audio_dirs + label_dirs:
        if not folder.exists():
            raise FileNotFoundError(f"폴더 없음: {folder}")
    pretrained_dir = SCRIPT_DIR / "pretrained" / "wavlm-base-plus"
    if not pretrained_dir.is_dir():
        raise FileNotFoundError(f"로컬 WavLM 모델 폴더 없음: {pretrained_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    items = (
        yeeun.load_f0_model(SCRIPT_DIR / "attention_f0" / "attention_f0_seed_42.pt", device),
        yeeun.load_stats_f0_model(SCRIPT_DIR / "stats_f0" / "stats_f0_seed42.pt", device),
        yeeun.load_stats_no_f0_model(SCRIPT_DIR / "stats_no_f0" / "stats_attention_seed42.pt", device),
    )
    feature_extractor = yeeun.AutoFeatureExtractor.from_pretrained(pretrained_dir, local_files_only=True)
    encoder = yeeun.AutoModel.from_pretrained(pretrained_dir, local_files_only=True).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    ecapa = EcapaGender(SCRIPT_DIR / "ecapa_gender", str(device))
    blend = ecapa.blend
    print(f"예은 3모델 {YEEUN_WEIGHTS} + ECAPA, ECAPA 비율 {blend}")

    audio_files = sorted(path for folder in audio_dirs for path in folder.rglob("*.wav"))
    if not audio_files:
        raise RuntimeError(f"WAV 파일이 없습니다: {audio_dirs}")
    json_map = yeeun.build_json_map(label_dirs)
    print(f"Audio files: {len(audio_files):,} / JSON files: {len(json_map):,}")

    names, rows = [], []
    yeeun_saved = np.full((len(audio_files), 3), np.nan)
    ecapa_saved = np.full(len(audio_files), np.nan)
    final_saved = np.full(len(audio_files), np.nan)
    failures = 0
    for i, audio_path in enumerate(audio_files):
        p_yeeun = p_ecapa = None
        try:
            json_path = json_map.get(audio_path.stem)
            if json_path is None:
                raise FileNotFoundError(f"JSON 없음: {audio_path.name}")
            annotation = load_annotation(json_path)
        except Exception as error:  # noqa: BLE001 — 파일 하나의 실패가 전체를 멈추지 않게
            annotation = None
            print(f"⚠️ {audio_path.name}: {type(error).__name__}: {error}")
        if annotation is not None:
            try:
                yeeun_saved[i] = yeeun_probabilities(audio_path, annotation, items, feature_extractor, encoder, device)
                p_yeeun = float(np.dot(YEEUN_WEIGHTS, yeeun_saved[i]))
            except Exception as error:  # noqa: BLE001
                print(f"⚠️ {audio_path.name} WavLM 3모델 실패: {type(error).__name__}: {error}")
            try:
                p_ecapa = ecapa.probability(audio_path, annotation)
                ecapa_saved[i] = p_ecapa
            except Exception as error:  # noqa: BLE001
                print(f"⚠️ {audio_path.name} ECAPA 실패: {type(error).__name__}: {error}")
        if p_yeeun is not None and p_ecapa is not None:
            p = (1.0 - blend) * p_yeeun + blend * p_ecapa
        elif p_yeeun is not None or p_ecapa is not None:
            p = p_yeeun if p_yeeun is not None else p_ecapa
            failures += 1
        else:
            p = 1.0  # 예은 스크립트와 같은 fallback: F
            failures += 1
        final_saved[i] = p
        # 학습·검증 평가(blend_with_yeeun.py)와 같은 기준: P(F) >= 0.5면 F
        rows.append([audio_path.name, INDEX_TO_GENDER[int(p >= 0.5)]])
        names.append(audio_path.name)
        if (i + 1) % 100 == 0 or i + 1 == len(audio_files):
            print(f"[{i + 1:,}/{len(audio_files):,}]", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["audio file name", "gender"])
        writer.writerows(rows)
    if args.save_probs:
        Path(args.save_probs).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.save_probs, names=np.asarray(names), yeeun=yeeun_saved, ecapa=ecapa_saved,
                 final=final_saved, blend=blend)
    print(f"Output: {output_path} / Rows: {len(rows)} / 일부·전체 실패 {failures}")


if __name__ == "__main__":
    main()

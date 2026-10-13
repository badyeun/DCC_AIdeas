"""WavLM 3모델 (Attention+F0, Stats+F0, Stats No-F0)의 모델 정의·체크포인트 로딩·특징 추출.

신고자 음성을 최대 8초 구간(8초 이하면 1구간 + zero padding, 넘으면 앞·중간·끝 3구간)으로 나눠
고정된 WavLM-base-plus에 넣고, 구간별 평균 임베딩(768)과 통계 임베딩(평균·표준편차 1536)을 만든다.
세 모델은 구간 attention pooling 후 분류하며, 두 모델은 F0 통계 6개를 함께 쓴다.
(학습 때 추론 스크립트 inference_3model_final.py에서 최종 추론에 쓰는 부분만 그대로 옮겼다.)
"""

import numpy as np
import torch
import torch.nn as nn

from common import SAMPLE_RATE


def make_three_chunks(waveform, max_samples):

    waveform = np.asarray(
        waveform,
        dtype=np.float32
    ).reshape(-1)

    n = len(waveform)

    # 8초 이하 -> 1 chunk + zero padding
    if n <= max_samples:

        padded = np.zeros(
            max_samples,
            dtype=np.float32
        )

        padded[:n] = waveform

        return [padded]

    # 8초 초과 -> front / middle / end
    max_start = n - max_samples

    starts = np.linspace(
        0,
        max_start,
        num=3
    ).astype(int)

    return [
        waveform[
            start:start + max_samples
        ]
        for start in starts
    ]


def extract_chunk_features_combined(
    waveform,
    feature_extractor,
    encoder,
    device,
    max_seconds,
):
    """One WavLM forward pass per chunk for mean and stats."""

    max_samples = int(SAMPLE_RATE * max_seconds)
    chunks = make_three_chunks(waveform, max_samples)

    embeddings = np.zeros((3, 768), dtype=np.float32)
    stats_result = np.zeros((3, 1536), dtype=np.float32)
    chunk_mask = np.zeros(3, dtype=np.float32)

    for i, chunk in enumerate(chunks):
        inputs = feature_extractor(
            chunk,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )

        input_values = inputs["input_values"].to(device)
        attention_mask = inputs.get("attention_mask")

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.inference_mode():
            outputs = encoder(
                input_values=input_values,
                attention_mask=attention_mask,
            )

            hidden = outputs.last_hidden_state

            if attention_mask is None:
                pooled = hidden.mean(dim=1)
                frames = hidden[0]
            else:
                feature_mask = (
                    encoder._get_feature_vector_attention_mask(
                        hidden.shape[1],
                        attention_mask,
                    )
                )

                mask = (
                    feature_mask.unsqueeze(-1)
                    .to(hidden.dtype)
                )

                # Original Attention F0 mean pooling
                pooled = (
                    (hidden * mask).sum(dim=1)
                    / mask.sum(dim=1).clamp(min=1.0)
                )

                # Original Stats F0 frame selection
                frames = hidden[0][feature_mask[0].bool()]

            mean = frames.mean(dim=0)
            std = torch.sqrt(
                frames.var(dim=0, correction=0) + 1e-5
            )

            stats = torch.cat([mean, std], dim=0)

        embeddings[i] = (
            pooled.squeeze(0).cpu().numpy().astype(np.float32)
        )
        stats_result[i] = (
            stats.cpu().numpy().astype(np.float32)
        )
        chunk_mask[i] = 1.0

    return embeddings, stats_result, chunk_mask


class AttentionF0MLP(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(768, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(774, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x, f0, mask):
        scores = self.attention(x).squeeze(-1)

        scores = scores.masked_fill(
            ~mask,
            torch.finfo(scores.dtype).min,
        )

        weights = torch.softmax(scores, dim=1)

        pooled = (
            x * weights.unsqueeze(-1)
        ).sum(dim=1)

        combined = torch.cat(
            [pooled, f0],
            dim=1,
        )

        return self.classifier(combined)


def load_f0_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = AttentionF0MLP(
        dropout=float(checkpoint["dropout"])
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    return {
        "model": model,
        "scaler_mean": np.asarray(
            checkpoint["scaler_mean"],
            dtype=np.float32,
        ),
        "scaler_scale": np.asarray(
            checkpoint["scaler_scale"],
            dtype=np.float32,
        ),
        "f0_scaler_mean": np.asarray(
            checkpoint["f0_scaler_mean"],
            dtype=np.float32,
        ),
        "f0_scaler_scale": np.asarray(
            checkpoint["f0_scaler_scale"],
            dtype=np.float32,
        ),
    }


class StatsF0AttentionMLP(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(1536, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1542, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x, f0, mask):
        scores = self.attention(x).squeeze(-1)
        scores = scores.masked_fill(
            ~mask,
            torch.finfo(scores.dtype).min,
        )

        weights = torch.softmax(scores, dim=1)

        pooled = (
            x * weights.unsqueeze(-1)
        ).sum(dim=1)

        combined = torch.cat(
            [pooled, f0],
            dim=1,
        )

        return self.classifier(combined)


def load_stats_f0_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = StatsF0AttentionMLP(
        dropout=float(checkpoint.get("dropout", 0.2))
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.eval()

    return {
        "model": model,
        "stats_scaler_mean": np.asarray(
            checkpoint["stats_scaler_mean"],
            dtype=np.float32,
        ),
        "stats_scaler_scale": np.asarray(
            checkpoint["stats_scaler_scale"],
            dtype=np.float32,
        ),
        "f0_scaler_mean": np.asarray(
            checkpoint["f0_scaler_mean"],
            dtype=np.float32,
        ),
        "f0_scaler_scale": np.asarray(
            checkpoint["f0_scaler_scale"],
            dtype=np.float32,
        ),
    }


class StatsNoF0AttentionMLP(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(1536, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1536, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x, mask):
        scores = self.attention(x).squeeze(-1)
        scores = scores.masked_fill(
            ~mask,
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores, dim=1)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(pooled)


def load_stats_no_f0_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = StatsNoF0AttentionMLP().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return {
        "model": model,
        "stats_scaler_mean": np.asarray(
            checkpoint["scaler_mean"],
            dtype=np.float32,
        ),
        "stats_scaler_scale": np.asarray(
            checkpoint["scaler_scale"],
            dtype=np.float32,
        ),
    }


def build_json_map(label_dirs):

    json_files = sorted(
        json_path
        for label_dir in label_dirs
        for json_path in label_dir.rglob("*.json")
    )

    json_map = {}

    for path in json_files:

        if path.stem in json_map:

            raise RuntimeError(
                f"중복 JSON stem 발견: "
                f"{path.stem}"
            )

        json_map[path.stem] = path

    return json_map

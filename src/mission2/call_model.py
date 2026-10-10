"""Mission 2 통화 단위 트랜스포머 (EXP-010 계열).

멘토링 답변(2026-09-19): 같은 통화 안의 다른 발화와 비교해 판정하는 것은 허용된다.
지금까지는 그 비교를 손으로 만든 특징(통화 평균과의 차이, 유사도 가중 확률, 같은 화자
판별기)으로 했다. 여기서는 통화 하나의 모든 발화를 한꺼번에 넣고, 각 발화가 같은 통화의
다른 발화를 self-attention으로 직접 참고하도록 모델이 비교 방식 자체를 배우게 한다.

**위치 정보를 넣지 않는다.** 트랜스포머는 위치 인코딩이 없으면 입력 순서를 알 수 없다
(순열 등변). 발화를 어떤 차례로 넣어도 각 발화의 예측이 같으므로 "발화 순서 사용 금지"를
구조적으로 지킨다. startAt/endAt은 발화를 자르는 데만 쓰고 입력에 넣지 않는다.

발화 하나의 입력 (906차원 + 선택 1):
- ECAPA 임베딩(길이 1) 192 + 통화 평균을 뺀 값 192
- MFCC 통계 261(Train 평균·표준편차로 표준화) + 통화 평균을 뺀 값 261
- (선택) 1단계 LightGBM 확률의 logit 1 — 기존 파이프라인 위에 쌓을 때
"""

from __future__ import annotations

import numpy as np

ECAPA_DIM = 192
MFCC_DIM = 261


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norm, 1e-8)


def call_inputs(
    ecapa: np.ndarray, mfcc: np.ndarray, scale: dict[str, np.ndarray],
    stage1: np.ndarray | None = None, second: np.ndarray | None = None,
) -> np.ndarray:
    """통화 하나의 발화 입력 행렬. 통화 평균은 순서와 무관한 값이다.

    second: (선택) WavLM 화자 임베딩 512. 길이 1로 맞춘 값과 통화 평균을 뺀 값을 더한다.
    """
    unit = normalize_rows(np.asarray(ecapa, dtype=np.float32))
    standard = (np.asarray(mfcc, dtype=np.float32) - scale["mean"]) / scale["std"]
    parts = [unit, unit - unit.mean(axis=0, keepdims=True), standard, standard - standard.mean(axis=0, keepdims=True)]
    if second is not None:
        unit2 = normalize_rows(np.asarray(second, dtype=np.float32))
        parts += [unit2, unit2 - unit2.mean(axis=0, keepdims=True)]
    if stage1 is not None:
        clipped = np.clip(np.asarray(stage1, dtype=np.float64), 1e-4, 1 - 1e-4)
        parts.append(np.log(clipped / (1 - clipped)).astype(np.float32)[:, None])
    return np.hstack(parts).astype(np.float32)


def input_width(use_stage1: bool, second_dim: int = 0) -> int:
    return ECAPA_DIM * 2 + MFCC_DIM * 2 + second_dim * 2 + (1 if use_stage1 else 0)


def build_model(config: dict):
    """위치 인코딩 없는 트랜스포머 인코더. config는 체크포인트에 그대로 저장된다."""
    import torch
    from torch import nn

    class CallTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = int(config["d_model"])
            self.project = nn.Sequential(
                nn.Linear(int(config["in_dim"]), width), nn.LayerNorm(width), nn.GELU(),
                nn.Dropout(float(config["dropout"])),
            )
            layer = nn.TransformerEncoderLayer(
                width, int(config["heads"]), int(config["ff"]), float(config["dropout"]),
                activation="gelu", batch_first=True, norm_first=True,
            )
            # nested tensor 경로는 padding 위치에 따라 수치가 미세하게 달라질 수 있어 끈다.
            self.encoder = nn.TransformerEncoder(layer, int(config["layers"]), enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
            # 같은 화자 보조 과제용 투영. 학습 때만 쓰고 추론 판정에는 쓰지 않는다.
            pair_dim = int(config.get("pair_dim", 0))
            self.pair = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, pair_dim)) if pair_dim else None

        def forward(self, inputs: "torch.Tensor", padding: "torch.Tensor", return_hidden: bool = False):
            hidden = self.encoder(self.project(inputs), src_key_padding_mask=padding)
            logits = self.head(hidden).squeeze(-1)
            return (logits, hidden) if return_hidden else logits

    return CallTransformer()


def load_transformers(entries: list[dict], device: str = "cpu") -> list[tuple[object, dict]]:
    """체크포인트에 넣어 둔 트랜스포머들(가중치는 numpy 배열)을 불러온다."""
    import torch

    loaded = []
    for entry in entries:
        model = build_model(entry["config"])
        model.load_state_dict({key: torch.from_numpy(np.asarray(value)) for key, value in entry["state_dict"].items()})
        loaded.append((model.to(device).eval(), entry))
    return loaded


def transformer_probability(
    loaded: list[tuple[object, dict]], ecapa: np.ndarray, mfcc: np.ndarray,
    stage1: np.ndarray | None = None, device: str = "cpu", second: np.ndarray | None = None,
) -> np.ndarray:
    """통화 하나에 대해 트랜스포머 여러 개(시드 앙상블)의 확률 평균.

    학습 캐시는 ECAPA·MFCC·WavLM을 float16으로 저장해 썼다. 같은 값으로 맞춰야 Validation 점수가
    재현되므로 여기서도 float16으로 반올림한다.
    """
    ecapa = np.asarray(ecapa, dtype=np.float16).astype(np.float32)
    mfcc = np.asarray(mfcc, dtype=np.float16).astype(np.float32)
    if second is not None:
        second = np.asarray(second, dtype=np.float16).astype(np.float32)
    outputs = []
    for model, entry in loaded:
        inputs = call_inputs(ecapa, mfcc, entry["scale"], stage1 if entry.get("use_stage1") else None,
                             second if entry.get("use_second") else None)
        outputs.append(predict_calls(model, [inputs], device, batch_calls=1)[0])
    return np.mean(outputs, axis=0)


def predict_calls(model, calls: list[np.ndarray], device: str = "cpu", batch_calls: int = 32) -> list[np.ndarray]:
    """통화별 입력 행렬 목록 → 통화별 신고자(1) 확률 목록. 길이가 비슷한 통화끼리 묶는다."""
    import torch

    model.eval()
    outputs: list[np.ndarray | None] = [None] * len(calls)
    order = np.argsort([len(call) for call in calls], kind="stable")
    with torch.no_grad():
        for begin in range(0, len(order), batch_calls):
            chosen = order[begin:begin + batch_calls]
            longest = max(len(calls[index]) for index in chosen)
            width = calls[chosen[0]].shape[1]
            batch = np.zeros((len(chosen), longest, width), dtype=np.float32)
            padding = np.ones((len(chosen), longest), dtype=bool)
            for row, index in enumerate(chosen):
                batch[row, :len(calls[index])] = calls[index]
                padding[row, :len(calls[index])] = False
            logits = model(torch.from_numpy(batch).to(device), torch.from_numpy(padding).to(device))
            probabilities = torch.sigmoid(logits).cpu().numpy()
            for row, index in enumerate(chosen):
                outputs[index] = probabilities[row, :len(calls[index])].astype(np.float64)
    return outputs  # type: ignore[return-value]

"""Mission 1 통화 단위 성별 트랜스포머 (Mission 2 `call_model.py` 변형).

통화 하나의 모든 발화(신고자 + 대원)를 한 시퀀스로 넣는다. Mission 1은 추론 때도 `speaker`를
쓸 수 있으므로 각 발화에 역할(신고자=1/대원=0)을 입력으로 붙인다. 대원 발화는 같은 회선의
기준점(채널·녹음 상태)을 알려 주는 용도다. 판정은 **신고자 발화들만** attention pooling해서
통화 하나의 성별 logit을 낸다.

위치 인코딩이 없어 발화 순서와 무관하다(순서를 섞어도 출력이 같다). startAt/endAt은
발화를 자르는 데만 쓰였고, 입력에는 발화 길이(MFCC 특징의 마지막 열)만 들어간다.

발화 하나의 입력 (907차원):
- ECAPA(길이 1) 192 + 통화 평균을 뺀 값 192
- MFCC 통계 261(Train 평균·표준편차로 표준화) + 통화 평균을 뺀 값 261
- 역할 1 (신고자=1)
"""

from __future__ import annotations

import numpy as np

ECAPA_DIM = 192
MFCC_DIM = 261
INPUT_DIM = ECAPA_DIM * 2 + MFCC_DIM * 2 + 1


def call_inputs(ecapa: np.ndarray, mfcc: np.ndarray, role: np.ndarray, scale: dict[str, np.ndarray]) -> np.ndarray:
    ecapa = np.asarray(ecapa, dtype=np.float32)
    unit = ecapa / np.maximum(np.linalg.norm(ecapa, axis=1, keepdims=True), 1e-8)
    standard = (np.asarray(mfcc, dtype=np.float32) - scale["mean"]) / scale["std"]
    return np.hstack([
        unit, unit - unit.mean(axis=0, keepdims=True),
        standard, standard - standard.mean(axis=0, keepdims=True),
        np.asarray(role, dtype=np.float32)[:, None],
    ]).astype(np.float32)


def build_model(config: dict):
    import torch
    from torch import nn

    class GenderCallTransformer(nn.Module):
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
            self.encoder = nn.TransformerEncoder(layer, int(config["layers"]), enable_nested_tensor=False)
            self.norm = nn.LayerNorm(width)
            self.score = nn.Linear(width, 1)       # 신고자 발화별 pooling 가중치
            self.head = nn.Linear(width, 1)        # 통화 성별 logit (F=1)
            self.token_head = nn.Linear(width, 1)  # 보조: 신고자 발화 하나하나의 성별 logit

        def forward(self, inputs, padding, caller):
            """inputs (B,T,D), padding (B,T) True=빈칸, caller (B,T) True=신고자 발화."""
            hidden = self.norm(self.encoder(self.project(inputs), src_key_padding_mask=padding))
            score = self.score(hidden).squeeze(-1).masked_fill(~caller, float("-inf"))
            weight = torch.softmax(score, dim=1)
            pooled = (weight.unsqueeze(-1) * hidden).sum(dim=1)
            return self.head(pooled).squeeze(-1), self.token_head(hidden).squeeze(-1)

    return GenderCallTransformer()


def batch_arrays(calls: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    longest = max(len(call) for call in calls)
    width = calls[0].shape[1]
    x = np.zeros((len(calls), longest, width), dtype=np.float32)
    padding = np.ones((len(calls), longest), dtype=bool)
    for row, call in enumerate(calls):
        x[row, :len(call)] = call
        padding[row, :len(call)] = False
    caller = (x[:, :, -1] > 0.5) & ~padding
    return x, padding, caller


def predict_calls(model, calls: list[np.ndarray], device: str = "cpu", batch_calls: int = 32) -> np.ndarray:
    """통화별 입력 행렬 목록 → 통화별 여성(F) 확률. 길이가 비슷한 통화끼리 묶는다."""
    import torch

    model.eval()
    outputs = np.zeros(len(calls), dtype=np.float64)
    order = np.argsort([len(call) for call in calls], kind="stable")
    with torch.no_grad():
        for begin in range(0, len(order), batch_calls):
            chosen = order[begin:begin + batch_calls]
            x, padding, caller = batch_arrays([calls[index] for index in chosen])
            logits, _ = model(*(torch.from_numpy(a).to(device) for a in (x, padding, caller)))
            outputs[chosen] = torch.sigmoid(logits).cpu().numpy()
    return outputs

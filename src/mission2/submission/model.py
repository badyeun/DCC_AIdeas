"""Mission 2 모델 코드: 통화 단위 트랜스포머와 10개 앙상블.

통화 하나의 발화 전체를 한 시퀀스로 넣고, 각 발화가 신고자(1)일 확률을 낸다. self-attention으로
같은 통화의 다른 발화 목소리와 비교해, "통화 안에 두 사람이 있고 그중 어느 쪽인가"를 판단한다.

**위치 인코딩을 넣지 않는다.** 위치 인코딩이 없는 트랜스포머는 입력 순서를 알 수 없다(순열 등변).
발화를 어떤 차례로 넣어도 각 발화의 예측이 같으므로 발화 순서(홀짝 등)를 쓰지 않는다.
startAt/endAt은 발화를 자르는 데만 쓰고 입력에 넣지 않는다.

발화 하나의 입력 906차원:
- ECAPA 임베딩(길이 1로 정규화) 192 + 그 값에서 통화 평균을 뺀 값 192
- MFCC 통계(Train 평균·표준편차로 표준화) 261 + 그 값에서 통화 평균을 뺀 값 261
"""

from __future__ import annotations

import numpy as np

ECAPA_DIM = 192
MFCC_DIM = 261
INPUT_DIM = (ECAPA_DIM + MFCC_DIM) * 2


def call_inputs(ecapa: np.ndarray, mfcc: np.ndarray, scale: dict[str, np.ndarray]) -> np.ndarray:
    """통화 하나의 입력 행렬 (발화 수, 906). 통화 평균은 순서와 무관한 값이다.

    scale: Train 통화로 계산한 MFCC 평균·표준편차 (체크포인트에 저장).
    """
    unit = np.asarray(ecapa, dtype=np.float32)
    unit = unit / np.maximum(np.linalg.norm(unit, axis=-1, keepdims=True), 1e-8)
    standard = (np.asarray(mfcc, dtype=np.float32) - scale["mean"]) / scale["std"]
    parts = [unit, unit - unit.mean(axis=0, keepdims=True), standard, standard - standard.mean(axis=0, keepdims=True)]
    return np.hstack(parts).astype(np.float32)


def build_model(config: dict):
    """위치 인코딩 없는 트랜스포머 인코더. config는 체크포인트에 그대로 저장된다.

    config: in_dim, d_model, layers, heads, ff, dropout, (선택) pair_dim
    """
    import torch
    from torch import nn

    class CallTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = int(config["d_model"])
            # 발화 특징 906 → d_model
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
            # 발화마다 신고자(1) logit 하나
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
            # 같은 화자 보조 과제(학습 때만 사용)의 투영. 판정에는 쓰지 않지만 가중치를 그대로 불러오기 위해 만든다.
            pair_dim = int(config.get("pair_dim", 0))
            self.pair = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, pair_dim)) if pair_dim else None

        def forward(self, inputs: "torch.Tensor", padding: "torch.Tensor", return_hidden: bool = False):
            """inputs (통화 수, 발화 수, 906), padding (통화 수, 발화 수) True=빈 자리 → logit (통화 수, 발화 수)."""
            hidden = self.encoder(self.project(inputs), src_key_padding_mask=padding)
            logits = self.head(hidden).squeeze(-1)
            return (logits, hidden) if return_hidden else logits

    return CallTransformer()


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
            batch = np.zeros((len(chosen), longest, calls[chosen[0]].shape[1]), dtype=np.float32)
            padding = np.ones((len(chosen), longest), dtype=bool)
            for row, index in enumerate(chosen):
                batch[row, :len(calls[index])] = calls[index]
                padding[row, :len(calls[index])] = False
            logits = model(torch.from_numpy(batch).to(device), torch.from_numpy(padding).to(device))
            probabilities = torch.sigmoid(logits).cpu().numpy()
            for row, index in enumerate(chosen):
                outputs[index] = probabilities[row, :len(calls[index])].astype(np.float64)
    return outputs  # type: ignore[return-value]


class CallEnsemble:
    """체크포인트의 트랜스포머 10개를 불러와 확률을 단순 평균한다."""

    def __init__(self, entries: list[dict], device: str = "cpu") -> None:
        import torch

        self.device = device
        self.models = []
        for entry in entries:
            model = build_model(entry["config"])
            model.load_state_dict({key: torch.from_numpy(np.asarray(value)) for key, value in entry["state_dict"].items()})
            self.models.append((model.to(device).eval(), entry["scale"]))

    def probability(self, ecapa: np.ndarray, mfcc: np.ndarray) -> np.ndarray:
        """통화 하나의 발화별 신고자(1) 확률 (발화 수,).

        학습 캐시는 ECAPA·MFCC를 float16으로 저장해 썼다. 같은 값으로 맞춰야 Validation 점수가
        재현되므로 여기서도 float16으로 반올림한다.
        """
        ecapa = np.asarray(ecapa, dtype=np.float16).astype(np.float32)
        mfcc = np.asarray(mfcc, dtype=np.float16).astype(np.float32)
        outputs = [predict_calls(model, [call_inputs(ecapa, mfcc, scale)], self.device, batch_calls=1)[0]
                   for model, scale in self.models]
        return np.mean(outputs, axis=0)

"""Fine-tune a Korean encoder for the nine symptoms, on this machine's GPU.

Deliberately independent of ver4: weights start from the public pretrained
checkpoint, the split is our own Training-internal grouped split, and official
Validation is never read. That keeps every result here free of the anchoring
that ver5-ver8 ran into.

Decisions are scored at the mandated 0.5 threshold for every symptom.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)

sys.path.insert(0, str(Path(__file__).parent))
from common_local import TARGETS, TRAIN_DIR, load, macro_f1, split          # noqa: E402
from normalize import VARIANTS, normalize                                    # noqa: E402

OUT = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"


class Calls(Dataset):
    """통화 하나 = 전사 텍스트 1개 + 증상 9개 타깃.

    타깃은 0/1 하드 라벨일 수도, 교사 확률을 섞은 소프트 타깃일 수도 있습니다
    (`--soft_targets`). `aux` 는 보조 과제용이며 기본은 쓰지 않습니다.
    """

    def __init__(self, texts, Y=None, aux=None):
        self.texts, self.Y, self.aux = texts, Y, aux

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        y = self.Y[i] if self.Y is not None else 0
        return self.texts[i], y, (self.aux[i] if self.aux is not None else -100)


class Collate:
    """배치 안에서 가장 긴 전사에 맞춰 토큰화합니다(동적 패딩).

    512 로 고정 패딩하면 중앙값이 230 토큰인 이 데이터에서 절반 이상이
    낭비됩니다. 배치마다 길이를 맞추면 같은 시간에 더 많이 학습합니다.
    """

    def __init__(self, tokenizer, max_length):
        self.tok, self.max_length = tokenizer, max_length

    def __call__(self, batch):
        enc = self.tok([b[0] for b in batch], padding=True, truncation=True,
                       max_length=self.max_length, return_tensors="pt")
        enc["labels"] = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
        enc["aux"] = torch.tensor([b[2] for b in batch], dtype=torch.long)
        return enc


class Classifier(nn.Module):
    """Mean pooling, or one attention query per symptom (LAAT / PLM-ICD style).

    Mean pooling forces all nine symptoms to share a single sentence vector.
    Label attention lets each symptom weight the tokens itself, so "열상" can
    look at "피가 나요" while "오심" looks somewhere else entirely -- which is
    the pooling ver4 used and this pipeline did not.
    """

    def __init__(self, name, n_labels, dropout=0.1, head="mean", attn_dim=256,
                 label_corr=False, aux_dim=0, pool_layer=-1, freeze_layers=0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        self.drop = nn.Dropout(dropout)
        self.head_kind = head
        # Which transformer layer to read. The last one is tuned hardest to the
        # training labels; an earlier layer carries more general features and
        # is wrong about different calls, which is what the ensemble wants.
        self.pool_layer = pool_layer
        # Freezing the bottom layers leaves the general features alone and
        # adapts only the upper ones -- a different model from the same weights.
        if freeze_layers:
            for p_ in self.encoder.embeddings.parameters():
                p_.requires_grad = False
            for layer in self.encoder.encoder.layer[:freeze_layers]:
                for p_ in layer.parameters():
                    p_.requires_grad = False
        hidden = self.encoder.config.hidden_size
        # Nine independent sigmoids ignore that the labels travel together:
        # 오심 shows up beside 구토 and 어지러움 far more often than chance.
        # A zero-initialised residual mixer starts as the identity, so the run
        # only diverges from the plain head if the correlation actually pays.
        self.corr = None
        if label_corr:
            self.corr = nn.Linear(n_labels, n_labels)
            nn.init.zeros_(self.corr.weight)
            nn.init.zeros_(self.corr.bias)
        # Auxiliary softmax over whole label combinations. Only 188 of the 511
        # possible sets ever occur, and 어지러움+오심 is the ninth most common --
        # structure nine independent sigmoids cannot represent. The head is
        # dropped at inference, so it only shapes the encoder.
        self.aux = nn.Linear(hidden, aux_dim) if aux_dim else None
        if head == "meanmax":
            self.head = nn.Linear(hidden * 2, n_labels)
        elif head in ("mean", "cls", "max"):
            self.head = nn.Linear(hidden, n_labels)
        else:
            self.project = nn.Linear(hidden, attn_dim)
            self.queries = nn.Linear(attn_dim, n_labels, bias=False)
            self.out_weight = nn.Parameter(torch.empty(n_labels, hidden))
            self.out_bias = nn.Parameter(torch.zeros(n_labels))
            nn.init.xavier_uniform_(self.out_weight)

    def forward(self, input_ids, attention_mask, **kw):
        """(배치, 9) 로짓을 돌려줍니다.

        `pool_layer` 가 -1 이면 마지막 층을, 아니면 그 층의 출력을 씁니다.
        헤드 종류에 따라 토큰을 모으는 방식이 달라집니다 -- 추론 코드의
        같은 이름 함수와 **정확히 일치해야** 합니다. 학습기에만 있는 헤드를
        만들면 추론에서 state_dict 적재가 실패합니다.
        """
        if self.pool_layer == -1:
            hidden = self.encoder(input_ids=input_ids,
                                  attention_mask=attention_mask).last_hidden_state
        else:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                               output_hidden_states=True)
            hidden = out.hidden_states[self.pool_layer]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        if self.head_kind == "meanmax":
            mx = hidden.masked_fill(mask == 0, float("-inf")).max(1).values
            logits = self.head(self.drop(torch.cat([pooled, mx], dim=-1)))
            if self.corr is not None:
                logits = logits + self.corr(logits)
            return (logits, self.aux(self.drop(pooled))) if self.aux is not None else logits
        if self.head_kind in ("mean", "cls", "max"):
            # Three ways to collapse the tokens into one vector. They score
            # about the same on their own, but they are wrong about different
            # calls -- which is what an ensemble wants. "cls" reads the
            # position the encoder was pretrained to summarise with; "max"
            # takes the strongest activation per dimension, so a single
            # decisive phrase can carry the whole call.
            if self.head_kind == "cls":
                pooled = hidden[:, 0]
            elif self.head_kind == "max":
                pooled = hidden.masked_fill(mask == 0, float("-inf")).max(1).values
            logits = self.head(self.drop(pooled))
        else:
            # scores: (batch, tokens, labels); padding must never win attention.
            scores = self.queries(torch.tanh(self.project(hidden)))
            scores = scores.masked_fill(attention_mask.unsqueeze(-1) == 0, float("-inf"))
            weights = torch.softmax(scores.float(), dim=1).to(hidden.dtype)
            # per-label context: (batch, labels, hidden)
            context = torch.einsum("btl,bth->blh", weights, hidden)
            context = self.drop(context)
            logits = (context * self.out_weight).sum(-1) + self.out_bias
        if self.corr is not None:
            logits = logits + self.corr(logits)
        return (logits, self.aux(self.drop(pooled))) if self.aux is not None else logits


def pos_weight(Y, device, scheme="sqrt5"):
    """How hard to push each symptom's probability toward the positive side.

    With the decision threshold frozen at 0.5, this is the main remaining way
    to place a symptom's probabilities relative to the cut. ver4 could instead
    move the cut per symptom (오심 sat at 0.411); that option is gone, so the
    weighting has to carry it. "sqrt5" is ver4's inherited setting.
    """
    pos = Y.sum(axis=0).astype(np.float64)
    ratio = np.maximum((len(Y) - pos) / np.maximum(pos, 1.0), 1.0)
    weights = {
        "none": np.ones_like(ratio),
        "sqrt3": np.clip(np.sqrt(ratio), 1.0, 3.0),
        "sqrt5": np.clip(np.sqrt(ratio), 1.0, 5.0),
        "sqrt8": np.clip(np.sqrt(ratio), 1.0, 8.0),
        "sqrt": np.sqrt(ratio),
        # "none" (1.0) scored 0.619 and "sqrt" (1.8-3.0) scored 0.638, so the
        # useful direction is stronger weighting. These walk further along it.
        # Soft targets compress the outputs toward the middle, and with sqrt
        # already inflating them the result crosses 0.5 far too often (9,583
        # predictions against 8,305 true). These sit below sqrt so a distilled
        # run can be pulled back. The formula stays the same for all nine
        # symptoms -- this is global class weighting, not a per-symptom cut.
        "pow025": ratio ** 0.25,
        "pow035": ratio ** 0.35,
        "pow065": ratio ** 0.65,
        "pow080": ratio ** 0.80,
        "full": ratio,
    }[scheme]
    return torch.tensor(weights, dtype=torch.float32, device=device)


class SoftMacroF1(nn.Module):
    """Macro F1 computed on the probabilities themselves, as a loss.

    Every attempt so far has optimised BCE and hoped the 0.5 cut lands well.
    This optimises the metric directly: the counts are built from sigmoid
    outputs instead of hard decisions, so the whole thing is differentiable
    and a symptom's rarity is handled by the F1 formula rather than by a
    weight we have to guess.

    It is added to BCE rather than replacing it. Alone it gives no gradient
    to a symptom absent from the batch, and with sixteen calls per step the
    rarer symptoms often are.
    """

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        """확률 자체로 Macro F1 을 계산해 손실로 씁니다 (미분 가능).

        BCE 는 확률을 맞히고 0.5 판정이 잘 떨어지길 기대하지만, 이쪽은 지표를
        직접 최적화합니다. 측정 결과 BCE 보다 나쁘므로 제출에는 쓰지 않습니다.
        """
        p = torch.sigmoid(logits.float())
        y = targets.float()
        tp = (p * y).sum(0)
        fp = (p * (1 - y)).sum(0)
        fn = ((1 - p) * y).sum(0)
        f1 = 2 * tp / (2 * tp + fp + fn + self.eps)
        present = (y.sum(0) > 0).float()          # 배치에 없는 증상은 세지 않음
        return 1.0 - (f1 * present).sum() / present.sum().clamp(min=1.0)


def sample_weights(Y, scheme):
    """How often each call is drawn, so rare symptoms appear more.

    Distinct from ``pos_weight``, which leaves the data alone and rescales the
    loss. Oversampling changes what the model actually sees, so batch
    statistics and the implicit curriculum change with it -- an untested lever
    while loss reweighting has been swept five ways.

    A call's weight comes from its rarest symptom: a call carrying 두통 (9.9%)
    is drawn more often than one carrying only 복통 (23.2%).
    """
    freq = Y.mean(axis=0)
    per_call = np.array([(1.0 / freq[row == 1]).max() if row.any() else 1.0 for row in Y])
    if scheme == "sqrt":
        per_call = np.sqrt(per_call)
    elif scheme == "log":
        per_call = np.log1p(per_call)
    elif scheme != "full":
        raise ValueError(f"알 수 없는 oversample: {scheme}")
    return per_call / per_call.mean()


def symmetric_kl(logits_a, logits_b):
    """Bernoulli KL both ways, averaged -- the multi-label form of R-Drop."""
    a, b = logits_a.float(), logits_b.float()
    pa, pb = torch.sigmoid(a), torch.sigmoid(b)
    la, lb = nn.functional.logsigmoid(a), nn.functional.logsigmoid(b)
    na, nb = nn.functional.logsigmoid(-a), nn.functional.logsigmoid(-b)
    kl_ab = pa * (la - lb) + (1 - pa) * (na - nb)
    kl_ba = pb * (lb - la) + (1 - pb) * (nb - na)
    return (kl_ab + kl_ba).mean()


def aux_targets(Y, spec):
    """Turn each call's label set into one class index for the auxiliary head.

    "combo:K" keeps the K most frequent symptom combinations and marks the rest
    -100 (ignored), so the head learns the sets that actually recur rather than
    memorising singletons. "count" predicts how many symptoms the call carries;
    every call has at least one and none has more than five.
    """
    if spec == "none":
        return None, 0, 1.0
    if spec == "count":
        return Y.sum(axis=1).astype(np.int64) - 1, int(Y.sum(axis=1).max()), 1.0
    if spec.startswith("combo:"):
        k = int(spec.split(":", 1)[1])
        keys = [tuple(r) for r in Y.tolist()]
        order = sorted({kk: keys.count(kk) for kk in set(keys)}.items(),
                       key=lambda kv: -kv[1])[:k]
        index = {kk: i for i, (kk, _) in enumerate(order)}
        covered = sum(c for _, c in order) / len(keys)
        return np.array([index.get(kk, -100) for kk in keys], np.int64), k, covered
    raise ValueError(f"알 수 없는 aux: {spec}")


def label_weight(Y, device, spec="none"):
    """How much of the loss each symptom gets -- capacity, not placement.

    Unlike ``pos_weight``, this scales a symptom's positive and negative terms
    by the same factor, so it does not move that symptom's probabilities
    relative to the 0.5 cut. It only decides how much of the shared encoder the
    symptom gets to shape. Macro F1 counts all nine equally while BCE does not,
    so a rare symptom like 오심 is under-represented in the gradient by default.
    """
    if spec.startswith("custom:"):
        weights = np.array([float(x) for x in spec.split(":", 1)[1].split(",")])
    elif spec == "invfreq":
        weights = 1.0 / np.sqrt(np.maximum(Y.sum(axis=0), 1.0))
    elif spec == "none":
        return None
    else:
        raise ValueError(f"알 수 없는 label_weight: {spec}")
    weights = weights / weights.mean()          # mean 1, so the overall LR scale is unchanged
    return torch.tensor(weights, dtype=torch.float32, device=device)


class AsymmetricLoss(nn.Module):
    """ASL: down-weight easy negatives harder than easy positives.

    Replaces pos_weight rather than stacking with it -- both push the same
    direction, and ver6 found doubling up unstable. gamma_neg > gamma_pos is
    what makes it asymmetric.
    """

    def __init__(self, gamma_neg=2.0, gamma_pos=0.0, margin=0.0, eps=1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.margin, self.eps = gamma_neg, gamma_pos, margin, eps

    def forward(self, logits, targets):
        """ASL (Asymmetric Loss): 쉬운 음성의 비중을 깎습니다.

        다중 라벨에서 음성이 압도적으로 많을 때 쓰는 손실입니다. 측정 결과
        음수였으므로 제출에는 쓰지 않습니다(`--loss bce` 가 기본).
        """
        p = torch.sigmoid(logits.float())
        p_neg = (p - self.margin).clamp(min=0) if self.margin > 0 else p
        pos = targets * ((1 - p) ** self.gamma_pos) * torch.log(p.clamp(min=self.eps))
        neg = (1 - targets) * (p_neg ** self.gamma_neg) * torch.log((1 - p_neg).clamp(min=self.eps))
        return -(pos + neg).mean()


class FocalLoss(nn.Module):
    """Focal loss, keeping the per-symptom positive weighting that works."""

    def __init__(self, gamma=2.0, pos_weight=None, eps=1e-8):
        super().__init__()
        self.gamma, self.eps = gamma, eps
        self.register_buffer("pw", pos_weight if pos_weight is not None else torch.tensor(1.0))

    def forward(self, logits, targets):
        """Focal Loss: 이미 잘 맞히는 샘플의 기울기를 줄입니다.

        측정 결과 BCE 보다 나쁘므로 제출에는 쓰지 않습니다.
        """
        p = torch.sigmoid(logits.float())
        pos = self.pw * targets * ((1 - p) ** self.gamma) * torch.log(p.clamp(min=self.eps))
        neg = (1 - targets) * (p ** self.gamma) * torch.log((1 - p).clamp(min=self.eps))
        return -(pos + neg).mean()


class EMA:
    """Shadow weights that trail the live ones; often steadier at evaluation."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        """가중치의 지수이동평균을 한 스텝 갱신합니다."""
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def swapped(self, model):
        """평균 가중치를 모델에 넣고, 원래 가중치를 돌려줍니다(복원용).

        평가할 때만 평균 가중치를 쓰고 학습은 원래 가중치로 계속하기 위한
        것입니다. 호출자가 반환값으로 반드시 복원해야 합니다.
        """
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict({k: v.to(dtype=backup[k].dtype) for k, v in self.shadow.items()},
                              strict=False)
        return backup


def param_groups(model, lr, llrd=1.0):
    """Optimiser groups, optionally with a lower rate on the lower layers.

    Layer-wise decay (ver4 used 0.95) rests on the idea that early layers hold
    general language and need moving less than the task-specific top. llrd=1.0
    reproduces the flat behaviour this pipeline had before.
    """
    no_decay = ("bias", "LayerNorm")
    layers = getattr(getattr(model.encoder, "encoder", None), "layer", None)
    depth = len(layers) if layers is not None else 0

    def depth_of(name):
        """파라미터 이름에서 층 깊이를 읽습니다 (층별 학습률 감쇠용).

        아래쪽 층일수록 학습률을 낮춰 사전학습 지식을 덜 흔들려는 기법입니다.
        """
        if llrd >= 1.0 or not depth:
            return depth                       # every parameter gets the top rate
        if "embeddings" in name:
            return 0
        if ".layer." in name:
            return int(name.split(".layer.")[1].split(".")[0]) + 1
        return depth + 1                       # head and pooler sit above the stack

    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        key = (depth_of(name), any(k in name for k in no_decay))
        groups.setdefault(key, []).append(param)
    out = []
    for (level, is_no_decay), params in sorted(groups.items()):
        out.append({"params": params,
                    "lr": lr * (llrd ** max(0, depth + 1 - level)),
                    "weight_decay": 0.0 if is_no_decay else 0.01})
    return out


@torch.no_grad()
def predict(model, loader, device, amp):
    """(통화 수, 9) 확률 행렬을 돌려줍니다. 평가 전용(기울기 없음)."""
    model.eval()
    chunks = []
    for batch in loader:
        batch.pop("labels", None)
        batch.pop("aux", None)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=amp):
            logits = model(**batch)
        if isinstance(logits, tuple):        # auxiliary head is training-only
            logits = logits[0]
        chunks.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(chunks)


def score_both(Y_true, P):
    """Plain 0.5 decisions, and the same with at least one symptom forced."""
    pred = (P >= 0.5).astype(np.int8)
    plain, per = macro_f1(Y_true, pred, per_symptom=True)
    blank = ~pred.any(axis=1)
    forced = pred.copy()
    if blank.any():
        forced[blank, P[blank].argmax(axis=1)] = 1
    with_force, _ = macro_f1(Y_true, forced, per_symptom=True)
    return plain, with_force, per, int(blank.sum())


def run(args, log):
    """한 번의 학습 실행 전체: 데이터 적재 → 학습 → 평가 → 저장.

    같은 홀드아웃 분할(seed 2026)을 모든 실험이 공유하므로 실행끼리 짝비교가
    됩니다. `--full_train` 이면 29,200통화 전부로 학습하고 홀드아웃을 남기지
    않습니다 -- 그때 출력되는 Macro F1 은 학습에 쓴 통화로 잰 값이라
    일반화 성능이 아니며, 수렴 확인용입니다.

    실행이 끝나면 확률 · 학습 기록(JSON) · 선택적으로 체크포인트를 남깁니다.
    체크포인트에는 전처리 변형과 인코더 설정을 함께 적어, 추론이 네트워크
    없이 같은 구조를 세울 수 있게 합니다.
    """
    device = torch.device("cuda")
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    names, texts, Y = load(TRAIN_DIR, "train")
    # Split on the raw transcripts first: every preprocessing variant must be
    # judged on the same holdout calls, or the comparison measures the split.
    tr, te = split(texts, Y, seed=2026)
    if args.full_train:
        # Submission models train on every Training call. Nothing is left to
        # score against, so the epoch count and recipe must already be settled
        # on the holdout -- this run only re-fits them on more data.
        tr = np.arange(len(Y))
    if args.folds > 0:
        # Teacher pass for self-distillation: carve the train split into folds,
        # fit on all but one, predict the one left out. Stitching the folds
        # gives a probability for every training call that no model that saw
        # that call produced. The main holdout stays out of all of it.
        rng = np.random.default_rng(777)
        fold_of = np.empty(len(tr), int)
        fold_of[rng.permutation(len(tr))] = np.arange(len(tr)) % args.folds
        te = tr[fold_of == args.fold]
        tr = tr[fold_of != args.fold]
    if args.train_frac < 1.0:
        # Bagging (vary --bag_seed) and the data-size curve (vary --train_frac)
        # both come from here. The holdout never changes, so runs stay paired.
        picked = np.random.default_rng(args.bag_seed).permutation(len(tr))
        tr = tr[picked[: int(len(tr) * args.train_frac)]]
    if args.normalize != "raw":
        texts = [normalize(t, **VARIANTS[args.normalize]) for t in texts]
    log(f"모델 {args.model} | 전처리 {args.normalize} | 손실 {args.loss}"
        + (f" | EMA {args.ema}" if args.ema else "")
        + (f" | 데이터 {args.train_frac:.0%} bag{args.bag_seed}" if args.train_frac < 1.0 else "")
        + f" | 학습 {len(tr):,} / 자체검증 {len(te):,}")

    # Hard 0/1 targets tell the model that every 오심 label is certain. The
    # data says otherwise: even transcripts that plainly say 메스꺼워요 carry the
    # label only 63% of the time. Soft targets from a teacher let the model fit
    # that uncertainty instead of memorising which coin flip landed which way.
    targets = Y.astype(np.float32)
    if args.soft_targets:
        soft = np.load(args.soft_targets).astype(np.float32)
        if soft.shape != Y.shape:
            raise SystemExit(f"soft_targets 모양 불일치: {soft.shape} vs {Y.shape}")
        if args.soft_debias and args.pos_weight != "none":
            # pos_weight 와 소프트 타깃의 충돌을 상쇄한다.
            #
            # 하드 라벨에서 pos_weight w 는 두 항의 기울기 균형만 바꾸고
            # 최적해는 0 또는 1 그대로다. 소프트 타깃 y 에서는 최적해가
            # w*y / (w*y + 1 - y) 로 **이동**한다 -- w=5, y=0.23 이면 모델이
            # 0.60 을 예측하도록 학습되는데 판정은 0.5 다. 측정에서도 alpha 가
            # 커질수록 단조 악화했다(0.5 -> 0.62675, 1.0 -> 0.60369).
            #
            # 그래서 교사 확률 p 를 그대로 넣지 않고, 가중 손실의 최적해가 p 가
            # 되는 타깃 y* = p / (w - (w-1)p) 를 넣는다. 이러면 pos_weight 의
            # 이점(불균형 보정)과 교사 정보를 동시에 쓸 수 있다.
            w = pos_weight(Y[tr], torch.device("cpu"), args.pos_weight).numpy()
            den = w - (w - 1.0) * soft
            soft = np.clip(soft / np.maximum(den, 1e-6), 0.0, 1.0).astype(np.float32)
            log(f"  소프트 타깃 역보정 적용 (pos_weight {args.pos_weight}, "
                f"w {w.min():.2f}~{w.max():.2f})")
        targets = (1 - args.soft_alpha) * targets + args.soft_alpha * soft
        log(f"  소프트 타깃 {Path(args.soft_targets).name} · alpha {args.soft_alpha}")

    aux_vec, aux_dim, aux_cover = aux_targets(Y, args.aux)
    if aux_dim:
        log(f"  보조 과제 {args.aux} · 클래스 {aux_dim}개 · 학습 통화의 {aux_cover:.1%} 커버"
            f" · 가중치 {args.aux_weight}")

    sampler = None
    if args.oversample != "none":
        w = sample_weights(Y[tr], args.oversample)
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(w, dtype=torch.double), num_samples=len(tr), replacement=True)
        log(f"  오버샘플링 {args.oversample} · 통화 가중치 {w.min():.2f}~{w.max():.2f}")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    collate = Collate(tok, args.max_length)
    train_loader = DataLoader(Calls([texts[i] for i in tr], targets[tr],
                                    None if aux_vec is None else aux_vec[tr]),
                              batch_size=args.batch_size,
                              shuffle=sampler is None, sampler=sampler,
                              collate_fn=collate, pin_memory=True, drop_last=True)
    eval_loader = DataLoader(Calls([texts[i] for i in te]), batch_size=args.eval_batch_size,
                             shuffle=False, collate_fn=collate, pin_memory=True)

    model = Classifier(args.model, len(TARGETS), args.dropout, head=args.head,
                       label_corr=args.label_corr, aux_dim=aux_dim,
                       pool_layer=args.pool_layer,
                       freeze_layers=args.freeze_layers).to(device)
    if args.grad_checkpoint:
        model.encoder.gradient_checkpointing_enable()
    pw = pos_weight(Y[tr], device, args.pos_weight)
    lw = label_weight(Y[tr], device, args.label_weight)
    if lw is not None:
        log("  라벨 손실 가중치 " + " ".join(f"{s} {w:.2f}" for s, w in zip(TARGETS, lw.tolist())))
    loss_fn = {
        "bce": lambda: nn.BCEWithLogitsLoss(pos_weight=pw, weight=lw),
        "asl": lambda: AsymmetricLoss(gamma_neg=args.gamma_neg, gamma_pos=0.0),
        "focal": lambda: FocalLoss(gamma=args.gamma_neg, pos_weight=pw).to(device),
    }[args.loss]()
    aux_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    softf1_fn = SoftMacroF1() if args.softf1 else None
    ema = EMA(model, args.ema) if args.ema else None
    opt = torch.optim.AdamW(param_groups(model, args.lr, args.llrd), lr=args.lr)
    steps = max(1, len(train_loader) // args.accum * args.epochs)
    warm = int(steps * args.warmup)
    sched = (get_cosine_schedule_with_warmup(opt, warm, steps) if args.scheduler == "cosine"
             else get_linear_schedule_with_warmup(opt, warm, steps))

    history, best, best_probs = [], {"best_macro_f1": -1.0}, None
    snapshot, snapshot_at = None, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, total = time.time(), 0.0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(device, non_blocking=True)
            aux_true = batch.pop("aux").to(device, non_blocking=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=amp):
                out = model(**batch)
                if aux_dim:
                    out, aux_logits = out
                loss = loss_fn(out, labels)
                if softf1_fn is not None:
                    loss = loss + args.softf1 * softf1_fn(out, labels)
                if aux_dim:
                    loss = loss + args.aux_weight * aux_loss_fn(aux_logits.float(), aux_true)
                if args.rdrop:
                    # R-Drop: run the same call through two dropout masks and
                    # penalise the disagreement. Where the label is noise the
                    # two passes have nothing to agree on, so the penalty pulls
                    # the model off memorising that particular coin flip.
                    second = model(**batch)
                    if aux_dim:
                        second = second[0]
                    loss = 0.5 * (loss + loss_fn(second, labels))
                    loss = loss + args.rdrop * symmetric_kl(out, second)
            (loss / args.accum).backward()
            total += loss.item()
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
            if step % 200 == 0:
                done = (step + 1) / len(train_loader)
                eta = (time.time() - started) / max(done, 1e-9) * (1 - done) / 60
                print(f"  epoch {epoch} {done:5.1%}  loss {total/(step+1):.4f}  ETA {eta:4.1f}분",
                      flush=True)
        minutes = (time.time() - started) / 60

        if ema is not None:
            backup = ema.swapped(model)
            P = predict(model, eval_loader, device, amp)
            model.load_state_dict(backup, strict=False)
        else:
            P = predict(model, eval_loader, device, amp)
        plain, forced, per, blank = score_both(Y[te], P)
        # In full-train mode these calls were trained on, so the number is
        # in-sample and says nothing about generalisation -- it is printed only
        # to catch a run that collapsed.
        scope = "(학습에 포함됨·참고용)" if args.full_train else ""
        log(f"  epoch {epoch}: loss {total/len(train_loader):.4f} | Macro F1 {plain:.5f}{scope} | "
            f"최소1개 강제 {forced:.5f} | 빈 예측 {blank} | {minutes:.1f}분")
        history.append({"epoch": epoch, "loss": total / len(train_loader), "macro_f1": plain,
                        "macro_f1_forced": forced, "best_macro_f1": max(plain, forced),
                        "empty": blank, "minutes": minutes,
                        "per_symptom": dict(zip(TARGETS, per))})
        # Forcing a symptom onto empty rows is optional at submission time and
        # does not always pay, so rank each epoch by whichever decision wins.
        if max(plain, forced) > best["best_macro_f1"]:
            best, best_probs = history[-1], P
        # Shipping the epoch that a longer schedule peaks at.
        #
        # The weights saved below are whichever are live when the loop ends, so
        # a 3-epoch run ships epoch 3. klue/roberta-large peaks at epoch 2 and
        # declines after, and shortening the run to 2 epochs does NOT give the
        # same model -- the learning rate then decays to zero over a shorter
        # horizon, which measured -0.00198 on the holdout for mean s1. So the
        # schedule keeps its full length and the weights are snapshotted at the
        # epoch that was validated. That epoch is fixed in advance from a
        # holdout run and never read off the numbers this run prints, so it
        # adds no degree of freedom -- which matters because --full_train has
        # no honest holdout to choose on.
        if args.save_at_epoch and epoch == args.save_at_epoch:
            snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            snapshot_at = epoch
            log(f"  epoch {epoch} 가중치를 보관했습니다 (스케줄은 {args.epochs} epoch 기준 유지)")
            if args.stop_at_save:
                log("  --stop_at_save: 남은 epoch 을 돌리지 않고 마칩니다")
                break

    head_tag = {"mean": "", "label_attention": "_attn"}.get(args.head, f"_{args.head}")
    pw_tag = "" if args.pos_weight == "sqrt5" else f"_pw{args.pos_weight}"
    extra = (("" if args.loss == "bce" else f"_{args.loss}{args.gamma_neg:g}")
             + (f"_ema{args.ema:g}" if args.ema else "")
             + (f"_f{args.train_frac:g}b{args.bag_seed}" if args.train_frac < 1.0 else "")
             # The teacher file has to be in the tag. Without it a run against
             # soft_targets_constant.npy and one against soft_targets_oof.npy
             # produced the same filename, and the later run silently
             # overwrote the earlier one's saved probabilities.
             + ("_db" if args.soft_debias and args.soft_targets else "")
             + (f"_{Path(args.soft_targets).stem.replace('soft_targets_', '')}"
                f"{args.soft_alpha:g}" if args.soft_targets else "")
             + ("" if args.label_weight == "none" else f"_lw{args.label_weight.replace(':', '')}")
             + ("_corr" if args.label_corr else "")
             + ("" if args.aux == "none" else f"_aux{args.aux.replace(':', '')}w{args.aux_weight:g}")
             + (f"_rdrop{args.rdrop:g}" if args.rdrop else "")
             + ("" if args.oversample == "none" else f"_os{args.oversample}")
             + ("" if args.pool_layer == -1 else f"_L{args.pool_layer}")
             + ("" if args.freeze_layers == 0 else f"_fz{args.freeze_layers}")
             + (f"_sf1{args.softf1:g}" if args.softf1 else "")
             + ("" if args.scheduler == "linear" else f"_{args.scheduler}")
             + ("" if args.warmup == 0.1 else f"_wu{args.warmup:g}")
             + ("" if args.dropout == 0.1 else f"_do{args.dropout:g}")
             + ("" if args.max_length == 512 else f"_ml{args.max_length}")
             ) + (f"_llrd{args.llrd:g}" if args.llrd < 1.0 else "")
    tag = (f"{Path(args.model).name}{head_tag}{pw_tag}{extra}"
           f"_{args.normalize}_s{args.seed}{args.tag_suffix}")
    if args.folds > 0:
        # Fold predictions are not comparable to the shared holdout, so they go
        # to their own names and never into the ensemble pool.
        np.save(OUT / f"oof_{tag}_k{args.folds}f{args.fold}.npy", best_probs)
        np.save(OUT / f"oofidx_{tag}_k{args.folds}f{args.fold}.npy", te)
    elif args.full_train:
        # No honest holdout exists here, so nothing goes into the pool that the
        # ensemble analysis reads from.
        np.save(OUT / f"insample_{tag}.npy", best_probs)
    else:
        np.save(OUT / f"probs_{tag}.npy", best_probs)
        np.save(OUT / "holdout_Y.npy", Y[te])
    if args.save_model:
        ckpt_dir = OUT / "submission_models"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        shipped = snapshot or {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save({"state_dict": shipped,
                    "model_name": args.arch_name or args.model, "head": args.head,
                    "max_length": args.max_length, "dropout": args.dropout,
                    "target_symptoms": list(TARGETS), "threshold": 0.5,
                    "normalize": args.normalize,
                    # The encoder's own config travels with the weights, so
                    # inference can build the architecture without fetching
                    # 440MB of pretrained weights it immediately overwrites.
                    "encoder_config": model.encoder.config.to_dict(),
                    "trained_on": int(len(tr)), "epochs": args.epochs,
                    "lr": args.lr, "seed": args.seed,
                    "saved_at_epoch": snapshot_at or args.epochs},
                   ckpt_dir / f"{tag}.ckpt")
        tok.save_pretrained(ckpt_dir / "tokenizer")
        log(f"  가중치 저장: submission_models/{tag}.ckpt")
    (OUT / f"history_{tag}.json").write_text(
        json.dumps({"args": vars(args), "history": history, "best": best},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  최고 epoch {best['epoch']}: {best['best_macro_f1']:.5f} "
        f"(그대로 {best['macro_f1']:.5f} / 강제 {best['macro_f1_forced']:.5f} 중 높은 쪽)")
    return best


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="klue/roberta-base")
    p.add_argument("--normalize", default="raw", choices=list(VARIANTS))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--head", default="mean",
                   choices=("mean", "cls", "max", "meanmax", "label_attention"))
    p.add_argument("--pos_weight", default="sqrt5",
                   choices=("none", "pow025", "pow035", "sqrt3", "sqrt5", "sqrt8", "sqrt",
                            "pow065", "pow080", "full"))
    p.add_argument("--loss", default="bce", choices=("bce", "asl", "focal"))
    p.add_argument("--gamma_neg", type=float, default=2.0, help="ASL/focal focusing strength.")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay; 0 disables.")
    p.add_argument("--train_frac", type=float, default=1.0, help="Fraction of the train split to use.")
    p.add_argument("--bag_seed", type=int, default=0, help="Which subsample, when train_frac < 1.")
    p.add_argument("--soft_targets", default="",
                   help="(29200, 9) teacher probabilities aligned with load() order.")
    p.add_argument("--soft_alpha", type=float, default=0.3,
                   help="Mixing weight: target = (1-a)*hard + a*soft.")
    p.add_argument("--label_weight", default="none",
                   help="none | invfreq | custom:w1,...,w9 -- per-symptom loss scale.")
    p.add_argument("--pool_layer", type=int, default=-1,
                   help="Transformer layer to pool from; -1 is the last.")
    p.add_argument("--freeze_layers", type=int, default=0,
                   help="Freeze embeddings and the bottom N encoder layers.")
    p.add_argument("--scheduler", default="linear", choices=("linear", "cosine"))
    p.add_argument("--warmup", type=float, default=0.1,
                   help="Fraction of steps spent warming up.")
    p.add_argument("--softf1", type=float, default=0.0,
                   help="Weight of a differentiable macro-F1 term added to BCE; 0 disables.")
    p.add_argument("--oversample", default="none", choices=("none", "log", "sqrt", "full"),
                   help="Draw calls carrying rare symptoms more often.")
    p.add_argument("--rdrop", type=float, default=0.0,
                   help="R-Drop strength; 0 disables. Costs a second forward pass.")
    p.add_argument("--aux", default="none",
                   help="none | count | combo:K -- training-only auxiliary head.")
    p.add_argument("--aux_weight", type=float, default=0.3)
    p.add_argument("--label_corr", action="store_true",
                   help="Zero-initialised residual mixer over the nine logits.")
    p.add_argument("--fold", type=int, default=-1,
                   help="With --folds k, train on all but this fold of the train split.")
    p.add_argument("--folds", type=int, default=0, help="Number of OOF folds; 0 disables.")
    p.add_argument("--llrd", type=float, default=1.0,
                   help="Layer-wise LR decay; 1.0 keeps every layer at --lr.")
    p.add_argument("--soft_debias", action="store_true",
                   help="With --soft_targets and a pos_weight, invert the weight's "
                        "effect on the target so the loss optimum equals the "
                        "teacher probability.")
    p.add_argument("--save_at_epoch", type=int, default=0,
                   help="Ship the weights from the end of this epoch rather than the last. "
                        "Fix it from a holdout run; never pick it from this run's output.")
    p.add_argument("--stop_at_save", action="store_true",
                   help="With --save_at_epoch, skip the remaining epochs.")
    p.add_argument("--full_train", action="store_true",
                   help="Train on all 29,200 calls; no holdout is left to score on.")
    p.add_argument("--arch_name", default="",
                   help="Public checkpoint name to record in the saved file. A domain-adapted run trains from a local directory, and that path will not exist on the grading machine -- the weights live in the state_dict, so only the architecture name needs to be portable.")
    p.add_argument("--save_model", action="store_true", help="Write a submission checkpoint.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad_checkpoint", action="store_true")
    p.add_argument("--tag_suffix", default="", help="Keeps runs that differ only by epochs from overwriting each other.")
    p.add_argument("--log", default=str(OUT / "train_log.md"))
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    logfile = Path(args.log)

    def log(line):
        """화면과 로그 파일에 동시에 적습니다 (제출 노트북이 이 로그를 씁니다)."""
        print(line, flush=True)
        with logfile.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log(f"\n### {time.strftime('%H:%M:%S')} {args.model} / {args.normalize} / seed {args.seed}")

    # 8GB is tight for the large encoder. Rather than lose a queued run to an
    # out-of-memory crash, halve the per-step batch and keep the effective batch
    # the same by doubling accumulation, then try again.
    while True:
        try:
            run(args, log)
            break
        except torch.cuda.OutOfMemoryError:
            gc.collect()          # drop the traceback's hold on the dead model
            torch.cuda.empty_cache()
            if args.batch_size <= 1:
                log("  VRAM 부족: batch 1에서도 실패해 이 설정은 건너뜁니다.")
                raise
            args.batch_size //= 2
            args.accum *= 2
            args.eval_batch_size = max(1, args.eval_batch_size // 2)
            log(f"  VRAM 부족 → batch {args.batch_size} / accum {args.accum} 로 재시도")

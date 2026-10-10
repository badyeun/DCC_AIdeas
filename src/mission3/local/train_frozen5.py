# FROZEN SNAPSHOT #5, 2026-10-02 20:40. Tag now records which soft-target
# file a run used; snapshot #4 did not, and a control run overwrote the
# distillation run it was meant to be compared against.
#
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
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
from common_local import TARGETS, TRAIN_DIR, load, macro_f1, split          # noqa: E402
from normalize import VARIANTS, normalize                                    # noqa: E402

OUT = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"


class Calls(Dataset):
    def __init__(self, texts, Y=None, aux=None):
        self.texts, self.Y, self.aux = texts, Y, aux

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        y = self.Y[i] if self.Y is not None else 0
        return self.texts[i], y, (self.aux[i] if self.aux is not None else -100)


class Collate:
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
                 label_corr=False, aux_dim=0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        self.drop = nn.Dropout(dropout)
        self.head_kind = head
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
        if head == "mean":
            self.head = nn.Linear(hidden, n_labels)
        else:
            self.project = nn.Linear(hidden, attn_dim)
            self.queries = nn.Linear(attn_dim, n_labels, bias=False)
            self.out_weight = nn.Parameter(torch.empty(n_labels, hidden))
            self.out_bias = nn.Parameter(torch.zeros(n_labels))
            nn.init.xavier_uniform_(self.out_weight)

    def forward(self, input_ids, attention_mask, **kw):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        if self.head_kind == "mean":
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
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def swapped(self, model):
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
                       label_corr=args.label_corr, aux_dim=aux_dim).to(device)
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
    sched = get_linear_schedule_with_warmup(opt, int(steps * 0.1), steps)

    history, best, best_probs = [], {"best_macro_f1": -1.0}, None
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

    head_tag = "" if args.head == "mean" else "_attn"
    pw_tag = "" if args.pos_weight == "sqrt5" else f"_pw{args.pos_weight}"
    extra = (("" if args.loss == "bce" else f"_{args.loss}{args.gamma_neg:g}")
             + (f"_ema{args.ema:g}" if args.ema else "")
             + (f"_f{args.train_frac:g}b{args.bag_seed}" if args.train_frac < 1.0 else "")
             # The teacher file has to be in the tag. Without it a run against
             # soft_targets_constant.npy and one against soft_targets_oof.npy
             # produced the same filename, and the later run silently
             # overwrote the earlier one's saved probabilities.
             + (f"_{Path(args.soft_targets).stem.replace('soft_targets_', '')}"
                f"{args.soft_alpha:g}" if args.soft_targets else "")
             + ("" if args.label_weight == "none" else f"_lw{args.label_weight.replace(':', '')}")
             + ("_corr" if args.label_corr else "")
             + ("" if args.aux == "none" else f"_aux{args.aux.replace(':', '')}w{args.aux_weight:g}")
             + (f"_rdrop{args.rdrop:g}" if args.rdrop else "")
             + ("" if args.oversample == "none" else f"_os{args.oversample}")
             + (f"_sf1{args.softf1:g}" if args.softf1 else "")
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
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "model_name": args.model, "head": args.head,
                    "max_length": args.max_length, "dropout": args.dropout,
                    "target_symptoms": list(TARGETS), "threshold": 0.5,
                    "trained_on": int(len(tr)), "epochs": args.epochs,
                    "lr": args.lr, "seed": args.seed},
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
    p.add_argument("--head", default="mean", choices=("mean", "label_attention"))
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
    p.add_argument("--full_train", action="store_true",
                   help="Train on all 29,200 calls; no holdout is left to score on.")
    p.add_argument("--save_model", action="store_true", help="Write a submission checkpoint.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad_checkpoint", action="store_true")
    p.add_argument("--tag_suffix", default="", help="Keeps runs that differ only by epochs from overwriting each other.")
    p.add_argument("--log", default=str(OUT / "train_log.md"))
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    logfile = Path(args.log)

    def log(line):
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

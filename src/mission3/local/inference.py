"""Mission 3 submission inference.

Required interface (출제문제 11쪽):

    python inference.py --audio_dir {wav folder} --label_dir {json folder} \
        --ckpt_path {checkpoint file} --output ./outputs/mission3.csv

One command produces the CSV. Mission 3 reads no audio, but --audio_dir is
accepted because the organisers' harness passes it.

What this does:
  * reads only utterances[].text from each annotation JSON
  * averages the probabilities of every checkpoint in the ensemble
  * calls a symptom present when its probability is >= 0.5 -- the same
    threshold for all nine, as the organisers require
  * writes `label file name,symptom` with the symptom list as a string

Pass a directory to --ckpt_path to ensemble every .ckpt inside it, or a single
.ckpt file for one model. Both work; the submitted ensemble is a directory.

If a `tfidf.pkl` sits beside the checkpoints it is blended in at 10%, which
is worth +0.0011 Macro F1 and +0.0026 AP on the holdout. See tfidf_model.py
for the split-half validation behind that weight.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

TARGETS = ("고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란")
THRESHOLD = 0.5
UTTERANCE_KEYS = ("utterances", "utterences")


class Classifier(nn.Module):
    """Must match the architecture the checkpoints were trained with."""

    def __init__(self, name, n_labels, dropout=0.1, head="mean", attn_dim=256,
                 encoder_config=None):
        super().__init__()
        # from_pretrained downloads the published weights and load_state_dict
        # then replaces every one of them. Building from the stored config
        # skips that: no network, and eleven checkpoints load in seconds.
        if encoder_config:
            cfg = dict(encoder_config)
            self.encoder = AutoModel.from_config(
                AutoConfig.for_model(cfg.pop("model_type"), **cfg))
        else:
            self.encoder = AutoModel.from_pretrained(name)
        self.drop = nn.Dropout(dropout)
        self.head_kind = head
        hidden = self.encoder.config.hidden_size
        if head in ("mean", "cls", "max"):
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
        if self.head_kind in ("mean", "cls", "max"):
            # Three ways to collapse tokens into one vector. They must match
            # train_local.py exactly -- a head the trainer knows and this file
            # does not fails at load_state_dict, which is how "cls" was caught.
            if self.head_kind == "cls":
                pooled = hidden[:, 0]
            elif self.head_kind == "max":
                pooled = hidden.masked_fill(mask == 0, float("-inf")).max(1).values
            else:
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            return self.head(self.drop(pooled))
        scores = self.queries(torch.tanh(self.project(hidden)))
        scores = scores.masked_fill(attention_mask.unsqueeze(-1) == 0, float("-inf"))
        weights = torch.softmax(scores.float(), dim=1).to(hidden.dtype)
        context = self.drop(torch.einsum("btl,bth->blh", weights, hidden))
        return (context * self.out_weight).sum(-1) + self.out_bias


def dialogue_text(annotation):
    """Only the transcript. speaker, startAt, endAt and the rest stay unread."""
    utterances = []
    for key in UTTERANCE_KEYS:
        value = annotation.get(key)
        if isinstance(value, list):
            utterances = value
            break
    said = [u["text"].strip() for u in utterances
            if isinstance(u, dict) and isinstance(u.get("text"), str) and u["text"].strip()]
    return "\n".join(said)


def read_inputs(label_dir: Path):
    """Every annotation JSON under label_dir, and nothing else.

    rglob("*.json") happily returns any JSON it finds, so a stray config or a
    checkpoint folder nested under label_dir would become extra rows in the
    submission CSV. An annotation always carries an utterance list, even when
    that list is empty, so that key is what identifies one.
    """
    names, texts, skipped = [], [], []
    for path in sorted(label_dir.rglob("*.json")):
        try:
            ann = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            skipped.append(path.name)
            continue
        if not isinstance(ann, dict) or not any(k in ann for k in UTTERANCE_KEYS):
            skipped.append(path.name)
            continue
        names.append(path.name)
        texts.append(dialogue_text(ann))
    if skipped:
        print(f"  주석 JSON이 아니라 건너뜀 {len(skipped)}개: "
              f"{', '.join(skipped[:4])}{' …' if len(skipped) > 4 else ''}", flush=True)
    if not names:
        raise SystemExit(f"주석 JSON을 찾지 못했습니다: {label_dir}")
    return names, texts


@torch.no_grad()
def probabilities(ckpt_path: Path, texts, device, batch_size):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Members of the submitted ensemble do not all read the same text: the
    # normalised-text models were trained on unified wording with fillers
    # stripped, and feeding them raw transcripts would be a train/test
    # mismatch. Each checkpoint records which variant it expects.
    variant = blob.get("normalize", "raw")
    if variant != "raw":
        sys.path.insert(0, str(Path(__file__).parent))
        from normalize import VARIANTS, normalize
        texts = [normalize(t, **VARIANTS[variant]) for t in texts]
    # One tokenizer folder is shared by the whole ensemble, which is fine only
    # while every member uses the same vocabulary. A heterogeneous ensemble can
    # break that quietly -- wrong token ids produce plausible-looking garbage
    # rather than an error -- so the vocabulary size is checked against the
    # checkpoint's own embedding matrix before the shared copy is trusted.
    vocab_rows = blob["state_dict"]["encoder.embeddings.word_embeddings.weight"].shape[0]
    tokenizer_dir = ckpt_path.parent / "tokenizer"
    tok = None
    if tokenizer_dir.exists():
        shared = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
        if len(shared) <= vocab_rows:
            tok = shared
        else:
            print(f"    공유 토크나이저 불일치 ({len(shared)} vs {vocab_rows}) — "
                  f"{blob['model_name']}에서 직접 불러옵니다", flush=True)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(blob["model_name"], use_fast=True)
    model = Classifier(blob["model_name"], len(blob["target_symptoms"]),
                       blob.get("dropout", 0.1), blob.get("head", "mean"),
                       encoder_config=blob.get("encoder_config"))
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    amp = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None

    out = []
    for start in range(0, len(texts), batch_size):
        batch = tok(texts[start:start + batch_size], padding=True, truncation=True,
                    max_length=blob.get("max_length", 512), return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        if amp is not None:
            with torch.autocast("cuda", dtype=amp):
                logits = model(**batch)
        else:
            logits = model(**batch)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", default="", help="미사용 (대회 인터페이스 호환).")
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--ckpt_path", required=True,
                        help=".ckpt 파일 또는 앙상블 체크포인트가 든 폴더.")
    parser.add_argument("--output", default="./outputs/mission3.csv")
    parser.add_argument("--tfidf_path", default="",
                        help="기본값은 체크포인트 폴더의 tfidf.pkl.")
    parser.add_argument("--tfidf_weight", type=float, default=0.10,
                        help="0이면 TF-IDF를 섞지 않습니다.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    names, texts = read_inputs(Path(args.label_dir))
    print(f"통화 {len(names):,}개 · 장치 {device}", flush=True)

    ckpt = Path(args.ckpt_path)
    checkpoints = sorted(ckpt.glob("*.ckpt")) if ckpt.is_dir() else [ckpt]
    if not checkpoints:
        raise SystemExit(f"체크포인트를 찾지 못했습니다: {ckpt}")

    total = None
    for i, path in enumerate(checkpoints, 1):
        print(f"  [{i}/{len(checkpoints)}] {path.name}", flush=True)
        p = probabilities(path, texts, device, args.batch_size)
        total = p if total is None else total + p
    probs = total / len(checkpoints)

    tfidf = Path(args.tfidf_path) if args.tfidf_path else \
        (ckpt if ckpt.is_dir() else ckpt.parent) / "tfidf.pkl"
    if tfidf.exists() and args.tfidf_weight > 0:
        sys.path.insert(0, str(Path(__file__).parent))
        import tfidf_model
        print(f"  TF-IDF {tfidf.name} 혼합 {args.tfidf_weight:g}", flush=True)
        probs = ((1 - args.tfidf_weight) * probs
                 + args.tfidf_weight * tfidf_model.predict(tfidf_model.load(tfidf), texts))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label file name", "symptom"])
        for name, row in zip(names, probs):
            found = [s for s, value in zip(TARGETS, row) if value >= THRESHOLD]
            writer.writerow([name, str(found)])
    print(f"저장 완료: {output}  (모델 {len(checkpoints)}개 평균, 임계값 {THRESHOLD})")


if __name__ == "__main__":
    main()

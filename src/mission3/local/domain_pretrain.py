"""Continue pretraining klue/roberta-base on the 119 call transcripts.

The encoder was pretrained on written Korean -- news, wiki, books. The input
here is transcribed emergency phone speech: disfluent, elliptical, full of
dispatcher formulae and dialect. Masked-language-model training on the
transcripts themselves closes that gap before any label is read, and it is
the one large lever nothing in this campaign has pulled.

Rules: this reads only utterances[].text from the provided Seoul Training set
and starts from a public open-source checkpoint. No labels, no external data.

Holdout transcripts are EXCLUDED. Including them would let the encoder
memorise the surface of the very calls we score on, inflating the holdout
relative to official Validation, whose text we have never seen. The
submission run can use everything, so this measurement is the conservative
one.

Run:  python src/mission3/local/domain_pretrain.py --epochs 3
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling,
                          get_linear_schedule_with_warmup)

sys.path.insert(0, str(Path(__file__).parent))
from common_local import TRAIN_DIR, load, split                            # noqa: E402

OUT = Path(__file__).resolve().parents[3] / "results" / "mission3" / "local"


class Chunks(Dataset):
    """Transcripts tokenised and cut into fixed-length blocks."""

    def __init__(self, texts, tok, length):
        self.blocks = []
        for t in texts:
            ids = tok(t, add_special_tokens=False)["input_ids"]
            body = length - 2
            for i in range(0, max(len(ids), 1), body):
                piece = ids[i:i + body]
                if len(piece) < 16:
                    continue
                self.blocks.append([tok.cls_token_id] + piece + [tok.sep_token_id])

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, i):
        return {"input_ids": self.blocks[i]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="klue/roberta-base")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--include_holdout", action="store_true",
                   help="제출 모델용. 평가용 실행에서는 쓰지 마십시오.")
    p.add_argument("--out", default=str(OUT / "domain_adapted"))
    args = p.parse_args()

    device = torch.device("cuda")
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.manual_seed(2026)

    names, texts, Y = load(TRAIN_DIR, "train")
    tr, te = split(texts, Y, seed=2026)
    used = np.arange(len(texts)) if args.include_holdout else tr
    print(f"MLM 학습 전사 {len(used):,}개"
          + ("" if args.include_holdout else f" (홀드아웃 {len(te):,}개 제외)"), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    data = Chunks([texts[i] for i in used], tok, args.max_length)
    collate = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=args.mask_prob)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, pin_memory=True, drop_last=True)
    print(f"블록 {len(data):,}개 · 스텝/epoch {len(loader)//args.accum:,}", flush=True)

    model = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = max(1, len(loader) // args.accum * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(steps * 0.06), steps)

    for epoch in range(1, args.epochs + 1):
        model.train()
        started, total, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=amp):
                loss = model(**batch).loss
            (loss / args.accum).backward()
            total += loss.item(); n += 1
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if step % 400 == 0:
                done = (step + 1) / len(loader)
                eta = (time.time() - started) / max(done, 1e-9) * (1 - done) / 60
                print(f"  epoch {epoch} {done:5.1%}  loss {total/n:.4f}  ETA {eta:4.1f}분",
                      flush=True)
        mean = total / max(n, 1)
        print(f"  epoch {epoch} 완료: loss {mean:.4f} · perplexity {math.exp(min(mean, 20)):.2f}"
              f" · {(time.time()-started)/60:.1f}분", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()

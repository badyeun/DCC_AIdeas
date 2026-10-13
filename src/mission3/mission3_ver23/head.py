"""Strict offline loading of the pinned source and adapted portable models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import torch

from . import data
from .neural import DialogueClassifier


MODEL_ID = "klue/roberta-base"
REVISION = "02f94ba5e3fcb7e2a58a390b8639b0fac974a8da"


def _safe(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Checkpoint asset paths must be nonempty relative paths")
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root) or result == root:
        raise ValueError("Checkpoint asset path escapes its directory")
    return result


def load_model(checkpoint, device):
    """Strictly restore local ver19/ver23 assets without fetching any model."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    checkpoint = Path(checkpoint)
    manifest = json.loads(checkpoint.read_text(encoding="utf-8-sig"))
    if (manifest.get("format") != "mission3_categorical_v1"
            or manifest.get("kind") != "neural"
            or manifest.get("version") not in ("ver19", "ver23")
            or tuple(manifest.get("labels", ())) != data.TARGET_SYMPTOMS
            or manifest.get("label_attention") is not False
            or manifest.get("contrastive") is not False
            or manifest.get("utterance_evidence") is not True
            or manifest.get("model_id") != MODEL_ID
            or manifest.get("resolved_revision") != REVISION):
        raise ValueError("Unsupported checkpoint, architecture, revision, or label ordering")
    architecture = {key: manifest[key] for key in
                    ("utterance_evidence", "evidence_hidden", "evidence_dropout")}
    assets = _safe(checkpoint.parent, manifest["assets"])
    weights_path = _safe(checkpoint.parent, manifest["weights"])
    config = AutoConfig.from_pretrained(assets / "encoder", local_files_only=True,
                                       trust_remote_code=False)
    if (config.model_type != "roberta" or config.num_hidden_layers != 12
            or config.hidden_size != 768):
        raise ValueError("The checkpoint must contain the pinned base encoder architecture")
    encoder = AutoModel.from_config(config, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(assets / "tokenizer", local_files_only=True,
                                              trust_remote_code=False)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Complete overflow windows require a fast tokenizer")
    model = DialogueClassifier(encoder, label_attention=False, contrastive=False,
                               **architecture)
    chunk_size = manifest["encoder_window_batch"]
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("encoder_window_batch must be a positive integer")
    model.encoder_window_batch = chunk_size
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Checkpoint weights must be a tensor-only state dictionary")
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.float().to(torch.device(device)).eval()
    return model, tokenizer, manifest


def _state_hashes(model):
    return {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in model.state_dict().items()}

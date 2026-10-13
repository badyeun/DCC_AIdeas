"""Training-only model selection and categorical Mission 3 inference."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from . import data
from .bootstrap import prepare_data


def dataset_digest(texts, targets):
    digest = hashlib.sha256()
    for text, target in zip(texts, targets):
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(target, dtype=np.int8).tobytes())
    return digest.hexdigest()


def feature_extractor(settings):
    return FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                                min_df=2, max_features=settings.get("char_features", 80000),
                                sublinear_tf=True, dtype=np.float32)),
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                max_features=settings.get("word_features", 20000),
                                sublinear_tf=True, dtype=np.float32)),
    ])


def sparse_estimator(name, seed):
    # These are standard categorical estimators; no score cutoff is searched.
    if name == "svm_c1":
        base = LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=seed)
    elif name == "svm_c4":
        base = LinearSVC(C=4.0, dual="auto", max_iter=5000, random_state=seed)
    elif name == "complement_nb":
        base = ComplementNB(alpha=1.0)
    else:
        raise ValueError(f"Unknown sparse candidate: {name}")
    return OneVsRestClassifier(base, n_jobs=1)


def train_sparse(paths, texts, targets, fit, holdout, local_dir, output_dir, settings):
    vectorizer = feature_extractor(settings)
    train_x = vectorizer.fit_transform([texts[i] for i in fit])
    holdout_x = vectorizer.transform([texts[i] for i in holdout])
    candidates = []
    best = None
    for name in settings.get("sparse_candidates", ["svm_c1", "svm_c4", "complement_nb"]):
        started = time.monotonic()
        estimator = sparse_estimator(name, settings["seed"])
        estimator.fit(train_x, targets[fit])
        predictions = estimator.predict(holdout_x).astype(np.int8)
        report = data.score_predictions(targets[holdout], predictions)
        candidates.append({"name": name, "seconds": time.monotonic() - started, **report})
        data.write_json(output_dir / "candidate_metrics.json", candidates)
        print(f"Training holdout {name}: Macro F1={report['macro_f1']:.6f}", flush=True)
        if best is None or report["macro_f1"] > best[0]:
            best = (report["macro_f1"], name, predictions, report)
    if best is None:
        raise ValueError("At least one sparse candidate is required")
    np.savez_compressed(output_dir / "holdout_predictions.npz", indices=holdout,
                        targets=targets[holdout], predictions=best[2])
    data.export_errors([paths[i] for i in holdout],
                       [texts[i] for i in holdout], targets[holdout], best[2],
                       output_dir / "training_errors")
    del train_x, holdout_x, estimator, vectorizer
    gc.collect()
    print(f"Selected using Training only: {best[1]}; refitting all Training", flush=True)
    vectorizer = feature_extractor(settings)
    full_x = vectorizer.fit_transform(texts)
    estimator = sparse_estimator(best[1], settings["seed"])
    estimator.fit(full_x, targets)
    joblib.dump({"vectorizer": vectorizer, "estimator": estimator}, local_dir / "sparse.joblib", compress=3)
    shutil.copy2(local_dir / "sparse.joblib", output_dir / "sparse.joblib")
    manifest = {"kind": "sparse", "artifact": "sparse.joblib", "selected_name": best[1]}
    return manifest, best[3]


def _save_weights(model, path):
    import torch
    temporary = path.with_name(path.name + ".tmp")
    # CPU tensors make the saved checkpoint device-independent.
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, temporary)
    temporary.replace(path)


def _load_weights(model, path):
    import torch
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)


def _loader(texts, targets, tokenizer, settings, shuffle):
    from torch.utils.data import DataLoader
    from .neural import EncodeDataset
    dataset = EncodeDataset(texts, targets, tokenizer,
                            max_length=settings["max_length"], overlap=settings["overlap"])
    return DataLoader(dataset, batch_size=settings["batch_size"], shuffle=shuffle,
                      num_workers=0, collate_fn=dataset.collate)


def _to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def _predict(model, loader, device):
    import torch
    from .neural import predictions
    result = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            logits, _ = model(_to_device(batch, device))
            result.append(predictions(logits).cpu().numpy())
    if not result:
        raise ValueError("Cannot predict an empty dataset")
    return np.concatenate(result).astype(np.int8)


def _fit_epochs(model, loader, settings, device, output_dir, prefix,
                holdout_loader=None, holdout_targets=None, fixed_epochs=None):
    import torch
    from .neural import classification_loss, binary_supervised_contrastive_loss
    encoder_parameters = list(model.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": settings["learning_rate"]},
        {"params": head_parameters, "lr": settings["head_learning_rate"]},
    ], weight_decay=0.01)
    epochs = fixed_epochs or settings["max_epochs"]
    schedule_epochs = settings["max_epochs"]
    if type(epochs) is not int or not 1 <= epochs <= schedule_epochs:
        raise ValueError("Executed epochs must fit within the selection schedule horizon")
    accumulation = settings["gradient_accumulation"]
    updates_per_epoch = (len(loader) + accumulation - 1) // accumulation
    # Preserve the selected epoch's relative LR trajectory during fresh full refit.
    total_updates = updates_per_epoch * schedule_epochs
    warmup = max(1, int(total_updates * 0.1))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_updates - step) / max(1, total_updates - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    use_cuda = device.type == "cuda"
    ce_views = settings.get("ce_views", 2 if settings["contrastive_weight"] else 1)
    if type(ce_views) is not int or ce_views not in (1, 2) or (settings["contrastive_weight"] and ce_views != 2):
        raise ValueError("ce_views must be 1 or 2, and contrastive training requires 2")
    dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and dtype == torch.float16)
    best_f1, best_epoch, stale = -1.0, 0, 0
    history = []
    weight_path = output_dir / f"{prefix}_weights.pt"
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_ce, total_contrastive = 0.0, 0.0, 0.0
        nausea_batch_counts = {}
        for step, batch in enumerate(loader):
            batch = _to_device(batch, device)
            positive_count = int(batch["targets"][:, 6].sum())
            nausea_batch_counts[str(positive_count)] = nausea_batch_counts.get(str(positive_count), 0) + 1
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_cuda):
                logits, features1 = model(batch)
                ce = classification_loss(logits, batch["targets"],
                                         nausea_positive_weight=settings.get("nausea_positive_weight", 1.0))
                contrastive = logits.new_zeros(())
                if ce_views == 2:
                    logits2, features2 = model(batch)
                    ce2 = classification_loss(logits2, batch["targets"],
                                              nausea_positive_weight=settings.get("nausea_positive_weight", 1.0))
                    ce = (ce + ce2) / 2
                if settings["contrastive_weight"]:
                    contrastive = binary_supervised_contrastive_loss(features1, features2, batch["targets"])
                loss = ce + settings["contrastive_weight"] * contrastive
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; no checkpoint was accepted")
            group_start = (step // accumulation) * accumulation
            group_size = min(accumulation, len(loader) - group_start)
            scaler.scale(loss / group_size).backward()
            total_loss += float(loss.detach())
            total_ce += float(ce.detach())
            total_contrastive += float(contrastive.detach())
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (step + 1) % settings.get("log_every", 250) == 0:
                print(f"{prefix} epoch {epoch}: {step + 1}/{len(loader)} batches", flush=True)
        row = {"epoch": epoch, "loss": total_loss / len(loader),
               "scheduler_horizon_epochs": schedule_epochs, "ce_views": ce_views,
               "classification_loss": total_ce / len(loader),
               "contrastive_loss": total_contrastive / len(loader),
               "nausea_positive_count_per_batch": nausea_batch_counts,
               "seconds": time.monotonic() - started}
        if holdout_loader is not None:
            predicted = _predict(model, holdout_loader, device)
            metrics = data.score_predictions(holdout_targets, predicted)
            row["holdout_macro_f1"] = metrics["macro_f1"]
            row["holdout_metrics"] = metrics
            if metrics["macro_f1"] > best_f1:
                best_f1, best_epoch, stale = metrics["macro_f1"], epoch, 0
                _save_weights(model, weight_path)
                data.write_json(output_dir / "selected_holdout_metrics.json", metrics)
            else:
                stale += 1
            print(f"Training holdout epoch {epoch}: Macro F1={metrics['macro_f1']:.6f}", flush=True)
        else:
            _save_weights(model, weight_path)
            best_epoch = epoch
            print(f"All-Training refit epoch {epoch}/{epochs}: loss={row['loss']:.5f}", flush=True)
        history.append(row)
        data.write_json(output_dir / f"{prefix}_history.json", history)
        if holdout_loader is not None and stale >= settings["patience"]:
            break
    return best_epoch, weight_path


def train_neural(paths, texts, targets, fit, holdout, local_dir, output_dir, settings):
    import torch
    from .neural import DialogueClassifier, prepare_encoder, seed_everything
    if not torch.cuda.is_available() and not settings.get("allow_cpu", False):
        raise RuntimeError("This version needs a Colab GPU runtime; ver10 runs on CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(settings["seed"])
    asset_dir = output_dir / "assets"
    encoder, tokenizer, revision = prepare_encoder(settings["model_id"], settings["revision"], asset_dir)
    settings = {**settings, "resolved_revision": revision}
    data.write_json(output_dir / "settings.json", settings)
    model = DialogueClassifier(encoder, label_attention=settings["label_attention"],
                               contrastive=bool(settings["contrastive_weight"]))
    model.encoder_window_batch = settings["encoder_window_batch"]
    model.to(device)
    fit_loader = _loader([texts[i] for i in fit], targets[fit], tokenizer, settings, True)
    holdout_loader = _loader([texts[i] for i in holdout], targets[holdout], tokenizer, settings, False)
    best_epoch, selected_path = _fit_epochs(model, fit_loader, settings, device, output_dir,
                                           "selection", holdout_loader, targets[holdout])
    _load_weights(model, selected_path)
    predicted = _predict(model, holdout_loader, device)
    metrics = data.score_predictions(targets[holdout], predicted)
    np.savez_compressed(output_dir / "holdout_predictions.npz", indices=holdout,
                        targets=targets[holdout], predictions=predicted)
    data.export_errors([paths[i] for i in holdout], [texts[i] for i in holdout],
                       targets[holdout], predicted, output_dir / "training_errors")
    if settings.get("refit_full_training", True):
        del model, encoder, fit_loader, holdout_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        seed_everything(settings["seed"])
        # Fresh public encoder; selected weights do not seed the refit.
        encoder, tokenizer, _ = prepare_encoder(settings["model_id"], revision, asset_dir)
        model = DialogueClassifier(encoder, label_attention=settings["label_attention"],
                                   contrastive=bool(settings["contrastive_weight"]))
        model.encoder_window_batch = settings["encoder_window_batch"]
        model.to(device)
        full_loader = _loader(texts, targets, tokenizer, settings, True)
        _, final_weights = _fit_epochs(model, full_loader, settings, device, output_dir,
                                       "final", fixed_epochs=best_epoch)
        fitted_samples = len(targets)
    else:
        final_weights = selected_path
        fitted_samples = len(fit)
    # A final artifact exists only once its declared refit has completed.
    manifest = {"kind": "neural", "weights": final_weights.name, "assets": "assets",
                "model_id": settings["model_id"], "resolved_revision": revision,
                "selected_epochs": best_epoch, "fitted_samples": fitted_samples,
                "label_attention": settings["label_attention"],
                "contrastive": bool(settings["contrastive_weight"]),
                "max_length": settings["max_length"], "overlap": settings["overlap"],
                "batch_size": settings["batch_size"], "encoder_window_batch": settings["encoder_window_batch"]}
    del model, encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return manifest, metrics


def refit_selected(texts, targets, settings, selected_epochs, output_dir):
    """Refit the chosen candidate once from its public encoder, using Training only."""
    import torch
    from .neural import DialogueClassifier, prepare_encoder, seed_everything
    if type(selected_epochs) is not int or selected_epochs < 1:
        raise ValueError("selected_epochs must be a positive integer")
    if not torch.cuda.is_available() and not settings.get("allow_cpu", False):
        raise RuntimeError("ver14 requires a GPU runtime for pretrained training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(settings["seed"])
    encoder, tokenizer, revision = prepare_encoder(settings["model_id"], settings["revision"], output_dir / "assets")
    model = DialogueClassifier(encoder, label_attention=settings["label_attention"],
                               contrastive=bool(settings["contrastive_weight"]))
    model.encoder_window_batch = settings["encoder_window_batch"]
    model.to(device)
    loader = _loader(texts, targets, tokenizer, settings, True)
    _, weights = _fit_epochs(model, loader, settings, device, output_dir, "final", fixed_epochs=selected_epochs)
    _load_weights(model, weights)
    probe_texts = texts[:min(12, len(texts))]
    probe_predictions = _predict(model, _loader(probe_texts, None, tokenizer, settings, False), device)
    manifest = {"kind": "neural", "weights": weights.name, "assets": "assets",
                "model_id": settings["model_id"], "resolved_revision": revision,
                "selected_epochs": selected_epochs, "fitted_samples": len(targets),
                "label_attention": settings["label_attention"], "contrastive": bool(settings["contrastive_weight"]),
                "nausea_positive_weight": settings["nausea_positive_weight"],
                "max_length": settings["max_length"], "overlap": settings["overlap"],
                "batch_size": settings["batch_size"], "encoder_window_batch": settings["encoder_window_batch"]}
    del model, encoder, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return manifest, probe_predictions


def _relative_artifact(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Model artifact path escapes checkpoint directory")
    return path


def predict_checkpoint(checkpoint, texts):
    checkpoint = Path(checkpoint)
    manifest = json.loads(checkpoint.read_text(encoding="utf-8-sig"))
    if manifest.get("format") != "mission3_categorical_v1" or tuple(manifest["labels"]) != data.TARGET_SYMPTOMS:
        raise ValueError("Unsupported checkpoint or symptom ordering")
    root = checkpoint.parent
    if manifest["kind"] == "sparse":
        bundle = joblib.load(_relative_artifact(root, manifest["artifact"]))
        return bundle["estimator"].predict(bundle["vectorizer"].transform(texts)).astype(np.int8)
    if manifest["kind"] != "neural":
        raise ValueError("Unknown checkpoint model kind")
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from .neural import DialogueClassifier
    assets = _relative_artifact(root, manifest["assets"])
    config = AutoConfig.from_pretrained(assets / "encoder", local_files_only=True, trust_remote_code=False)
    encoder = AutoModel.from_config(config, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(assets / "tokenizer", local_files_only=True, trust_remote_code=False)
    model = DialogueClassifier(encoder, label_attention=manifest["label_attention"],
                               contrastive=manifest["contrastive"])
    model.encoder_window_batch = manifest["encoder_window_batch"]
    _load_weights(model, _relative_artifact(root, manifest["weights"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = _loader(texts, None, tokenizer, manifest, False)
    return _predict(model, loader, device)


def inference(checkpoint, label_dir, output):
    paths, texts, _ = data.load_dataset(label_dir, include_targets=False)
    predicted = predict_checkpoint(checkpoint, texts)
    data.write_predictions(paths, predicted, output)
    return paths, predicted


def run(settings, runtime_sources=None):
    started = time.monotonic()
    settings = dict(settings)
    train_dir, validation_dir = prepare_data(settings["input_zip"], settings["work_dir"],
                                             settings.get("train_dir", ""), settings.get("validation_dir", ""))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output_dir = Path(settings["output_dir"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    local_dir = Path(settings["work_dir"]) / run_id
    local_dir.mkdir(parents=True, exist_ok=False)
    data.write_json(output_dir / "settings.json", settings)
    if runtime_sources:
        submission = output_dir / "submission"
        for name, source in runtime_sources.items():
            if name.startswith("test_") or name == "build_notebooks.py":
                continue
            target = submission / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")
    paths, texts, targets = data.load_dataset(train_dir, include_targets=True)
    fit, holdout = data.make_split(texts, targets, settings["holdout_size"], settings["seed"])
    data.write_json(output_dir / "training_diagnostics.json", data.training_diagnostics(texts, targets))
    np.savez_compressed(output_dir / "training_split.npz", fit=fit, holdout=holdout)
    data.write_json(output_dir / "run_identity.json", {
        "version": settings["version"], "training_digest": dataset_digest(texts, targets),
        "training_samples": len(texts), "fit_samples": len(fit), "holdout_samples": len(holdout),
        "model_selection": "Training internal holdout only", "decision_rule": "categorical predict/argmax",
    })
    print(f"Training {len(texts):,}; fit {len(fit):,}; holdout {len(holdout):,}", flush=True)
    print("Training label counts:", dict(zip(data.TARGET_SYMPTOMS, targets.sum(axis=0).tolist())), flush=True)
    try:
        trainer = train_sparse if settings["backend"] == "sparse" else train_neural
        manifest, metrics = trainer(paths, texts, targets, fit, holdout, local_dir, output_dir, settings)
        manifest.update(format="mission3_categorical_v1", labels=list(data.TARGET_SYMPTOMS),
                        version=settings["version"], decision_rule="categorical predict/argmax")
        checkpoint = output_dir / f"mission3_{settings['version']}_model.ckpt"
        data.write_json(output_dir / "training_metrics.json", metrics)
        data.write_json(checkpoint, manifest)
        # Reopen the serialized model on the holdout text before official inference.
        probe = [texts[i] for i in holdout[:min(12, len(holdout))]]
        roundtrip = predict_checkpoint(checkpoint, probe)
        if roundtrip.shape != (len(probe), len(data.TARGET_SYMPTOMS)):
            raise ValueError("Checkpoint roundtrip shape mismatch")
        if settings.get("run_validation", True):
            validation_paths, predicted = inference(checkpoint, validation_dir, output_dir / "mission3.csv")
            # Official targets are accessed only after predictions are saved.
            scored_paths, _, validation_targets = data.load_dataset(validation_dir, include_targets=True)
            if [p.name for p in scored_paths] != [p.name for p in validation_paths]:
                raise ValueError("Official evaluation file order mismatch")
            report = data.score_predictions(validation_targets, predicted)
            data.write_json(output_dir / "validation_metrics.json", report)
            print(f"Official Validation (final diagnostic): Macro F1={report['macro_f1']:.6f}", flush=True)
            for name, value in report["per_symptom"].items():
                print(f"  {name}: F1={value['f1']:.4f}, P={value['precision']:.4f}, R={value['recall']:.4f}", flush=True)
        data.write_json(output_dir / "runtime.json", {"status": "complete", "seconds": time.monotonic() - started})
    except Exception as error:
        data.write_json(output_dir / "runtime.json", {"status": "failed", "seconds": time.monotonic() - started,
                                                      "error_type": type(error).__name__})
        raise
    print("Saved results:", output_dir, flush=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True)  # Official interface; Mission 3 uses text only.
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    inference(args.ckpt_path, args.label_dir, args.output)


if __name__ == "__main__":
    main()

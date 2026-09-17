"""Training-only model selection, ensemble calibration, and portable inference."""
from __future__ import annotations

import ast
import csv
import gc
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from transformers import AutoTokenizer

if __package__:
    from .calibration import apply_calibrator, calibrate_crossfit
    from .data import TARGET_SYMPTOMS, load_dataset, make_splits, score_predictions, write_json, write_predictions
    from .neural import predict_checkpoint, tokenize_dialogues, train_model
else:
    from calibration import apply_calibrator, calibrate_crossfit
    from data import TARGET_SYMPTOMS, load_dataset, make_splits, score_predictions, write_json, write_predictions
    from neural import predict_checkpoint, tokenize_dialogues, train_model


DEFAULT_CONFIG = {
    "model_name": "klue/roberta-large",
    "seed": 42,
    "n_folds": 5,
    "calibration_size": 0.10,
    "calibration_folds": 5,
    "max_length": 512,
    "stride": 64,
    "max_epochs": 6,
    "patience": 2,
    "batch_size": 2,
    "inference_batch_size": 8,
    "window_batch_size": 2,
    "gradient_accumulation_steps": 16,
    "learning_rate": 1.5e-5,
    "head_learning_rate": 1e-4,
    "layerwise_lr_decay": 0.95,
    "weight_decay": 0.01,
    "warmup_ratio": 0.10,
    "dropout": 0.20,
    "pos_weight_power": 0.5,
    "gradient_checkpointing": True,
    "device": "cuda",
    "require_cuda": True,
    "char_features": 120_000,
    "word_features": 40_000,
    "min_df": 2,
    "logistic_c": 4.0,
}
RUNTIME_FILES = ("data.py", "calibration.py", "neural.py", "pipeline.py", "train.py", "inference.py", "evaluate.py")


def fit_sparse(texts, targets, config):
    """문자 표현과 단어 순서를 보완하는, Training에만 fit한 TF-IDF 모델."""
    features = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                                max_features=config["char_features"], min_df=config["min_df"],
                                sublinear_tf=True, dtype=np.float32)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b",
                                max_features=config["word_features"], min_df=config["min_df"],
                                sublinear_tf=True, dtype=np.float32)),
    ])
    matrix = features.fit_transform(texts)
    classifiers = []
    for column, name in enumerate(TARGET_SYMPTOMS):
        truth = targets[:, column]
        if len(np.unique(truth)) == 1:
            classifier = float(truth[0])
        else:
            classifier = LogisticRegression(C=config["logistic_c"], class_weight="balanced",
                                            solver="liblinear", max_iter=1500, random_state=config["seed"])
            classifier.fit(matrix, truth)
        classifiers.append(classifier)
        print(f"TF-IDF fit: {name}", flush=True)
    return {"features": features, "classifiers": classifiers}


def predict_sparse(model, texts):
    if not texts:
        return np.empty((0, len(TARGET_SYMPTOMS)), dtype=np.float32)
    # 큰 Test 폴더도 전체 sparse matrix를 RAM에 동시에 올리지 않는다.
    batches = []
    for start in range(0, len(texts), 1024):
        matrix = model["features"].transform(texts[start:start + 1024])
        columns = [np.full(matrix.shape[0], clf) if isinstance(clf, float)
                   else clf.predict_proba(matrix)[:, 1] for clf in model["classifiers"]]
        batches.append(np.column_stack(columns))
    return np.concatenate(batches).astype(np.float32)


def _release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _torch_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _fingerprint(paths, texts, targets, config):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for path, text, target in zip(paths, texts, targets):
        digest.update(json.dumps([path.name, text, target.tolist()], ensure_ascii=False).encode("utf-8"))
    # 변경된 코드로 기존 fold가 조용히 재사용되는 것을 방지한다.
    for name in RUNTIME_FILES:
        path = Path(__file__).parent / name
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def train_pipeline(train_label_dir, output_dir, config=None):
    """Validation 인자를 받지 않는다. 보정 후에도 모델을 재학습하지 않는다."""
    supplied = config or {}
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    config = {**DEFAULT_CONFIG, **supplied}
    train_label_dir, output_dir = Path(train_label_dir).resolve(), Path(output_dir).resolve()
    if train_label_dir.name != "TL_서울_구급" or "Validation" in train_label_dir.parts:
        raise ValueError("학습 입력은 서울 Training의 TL_서울_구급 폴더여야 합니다.")
    if config["require_cuda"] and (config["device"] != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("전체 학습에는 GPU가 필요합니다. Colab GPU 런타임에서 실행하세요.")
    if config["device"] == "cuda":
        torch.set_float32_matmul_precision("high")
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    paths, texts, targets = load_dataset(train_label_dir, include_targets=True)
    dev, calibration, folds, groups = make_splits(
        texts, targets, config["calibration_size"], config["n_folds"], config["seed"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = _fingerprint(paths, texts, targets, config)
    run_path = output_dir / "run.json"
    if run_path.exists() and json.loads(run_path.read_text(encoding="utf-8"))["signature"] != signature:
        raise ValueError("출력 폴더에 다른 데이터/설정/코드의 실험이 있습니다. 새 output_dir를 지정하세요.")
    write_json(run_path, {"signature": signature, "config": config, "training_samples": len(paths)})
    write_json(output_dir / "splits.json", {
        "calibration": [paths[i].name for i in calibration],
        "folds": [{"fit": [paths[i].name for i in fit], "stop": [paths[i].name for i in stop]}
                  for fit, stop in folds],
        "seed": config["seed"], "grouping": "SHA256 of whitespace-normalized transcript only",
    })
    print(f"Training={len(paths):,}, development={len(dev):,}, calibration={len(calibration):,}", flush=True)
    tokenizer_dir = output_dir / "tokenizer"
    if (tokenizer_dir / "tokenizer_config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
        tokenizer.save_pretrained(tokenizer_dir)
    documents = tokenize_dialogues(texts, tokenizer, config["max_length"], config["stride"])
    calibration_documents = [documents[i] for i in calibration]
    counts = np.asarray([len(document) for document in documents])
    print(f"대화 window 수: 평균={counts.mean():.2f}, 최대={counts.max()}, 긴 대화={int((counts > 1).sum()):,}", flush=True)
    neural_probs = np.zeros((len(calibration), len(TARGET_SYMPTOMS)), dtype=np.float64)
    fold_names, histories = [], []
    for number, (fit, stop) in enumerate(folds, 1):
        name = f"fold_{number}.ckpt"
        checkpoint_path, done_path = output_dir / name, output_dir / f"fold_{number}_done.json"
        print(f"\nFold {number}/{len(folds)}: fit={len(fit):,}, stop={len(stop):,}", flush=True)
        if done_path.exists() and checkpoint_path.exists():
            result = json.loads(done_path.read_text(encoding="utf-8"))
            print("완료된 fold 재사용 (동일 run signature 확인됨)", flush=True)
        else:
            result = train_model([documents[i] for i in fit], targets[fit],
                                 [documents[i] for i in stop], targets[stop], tokenizer,
                                 config, checkpoint_path, seed=config["seed"] + number * 1009)
            write_json(done_path, result)
        _release_memory()
        neural_probs += predict_checkpoint(checkpoint_path, calibration_documents, tokenizer, config)
        _release_memory()
        fold_names.append(name)
        histories.append(result)
    neural_probs /= len(folds)
    sparse_path = output_dir / "sparse.joblib"
    if sparse_path.exists():
        sparse = joblib.load(sparse_path)
    else:
        sparse = fit_sparse([texts[i] for i in dev], targets[dev], config)
        temporary = sparse_path.with_name("sparse.joblib.tmp")
        joblib.dump(sparse, temporary, compress=3)
        temporary.replace(sparse_path)
    sparse_probs = predict_sparse(sparse, [texts[i] for i in calibration])
    del sparse
    calibrator, report = calibrate_crossfit(targets[calibration], neural_probs, sparse_probs,
                                          groups[calibration], seed=config["seed"] + 2,
                                          n_splits=config["calibration_folds"])
    # OOF 확률을 final refit에 전용하지 않고, 실제 배포 ensemble의 확률을 보정한다.
    np.savez_compressed(output_dir / "calibration_predictions.npz", targets=targets[calibration],
                        neural=neural_probs, sparse=sparse_probs,
                        filenames=np.asarray([paths[i].name for i in calibration]))
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "numpy", "scikit-learn", "iterative-stratification")}
    manifest = {"format": "mission3_ver4", "version": 1, "target_symptoms": list(TARGET_SYMPTOMS),
                "config": config, "folds": fold_names, "tokenizer": "tokenizer", "sparse": "sparse.joblib",
                "calibrator": calibrator, "signature": signature, "versions": versions}
    checkpoint_path = output_dir / "mission3_ver4_model.ckpt"
    _torch_save(checkpoint_path, manifest)
    report.update({"training_samples": len(paths), "development_samples": len(dev),
                   "model": f"{config['model_name']} {len(folds)}-fold ensemble + char/word TF-IDF LR",
                   "calibrator": calibrator, "fold_best_epochs": [x["best_epoch"] for x in histories],
                   "versions": versions, "validation_used_for_training_or_calibration": False})
    write_json(output_dir / "mission3_ver4_metrics.json", report)
    write_json(output_dir / "mission3_ver4_history.json", histories)
    # 노트북 하나만 Colab에 올려도 제출 코드와 실행 당시 설정을 남긴다.
    submission = output_dir / "submission"
    submission.mkdir(exist_ok=True)
    for name in (*RUNTIME_FILES, "requirements.txt", "README.md"):
        path = Path(__file__).parent / name
        if path.exists():
            shutil.copy2(path, submission / path.name)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("Checkpoint:", checkpoint_path, flush=True)
    return checkpoint_path, report


def load_manifest(checkpoint_path):
    manifest = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if manifest.get("format") != "mission3_ver4" or manifest.get("version") != 1:
        raise ValueError("지원하지 않는 checkpoint입니다. ver4의 manifest .ckpt를 지정하세요.")
    if manifest["target_symptoms"] != list(TARGET_SYMPTOMS) or not manifest["folds"]:
        raise ValueError("Checkpoint class order/folds mismatch")
    root = Path(checkpoint_path).resolve().parent
    for name in [*manifest["folds"], manifest["sparse"], manifest["tokenizer"]]:
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.exists():
            raise FileNotFoundError(f"Checkpoint 구성 파일을 함께 보관해야 합니다: {name}")
    return manifest


def predict_pipeline(checkpoint_path, label_dir, output, device=None, batch_size=None):
    """저장된 encoder config/tokenizer만 사용하므로 학습 후 오프라인 추론 가능."""
    checkpoint_path = Path(checkpoint_path).resolve()
    manifest = load_manifest(checkpoint_path)
    root = checkpoint_path.parent
    config = dict(manifest["config"])
    config["device"] = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if batch_size is not None:
        config["inference_batch_size"] = batch_size
    paths, texts, _ = load_dataset(Path(label_dir), include_targets=False)
    tokenizer = AutoTokenizer.from_pretrained(root / manifest["tokenizer"], use_fast=True, local_files_only=True)
    documents = tokenize_dialogues(texts, tokenizer, config["max_length"], config["stride"])
    neural_probs = np.zeros((len(paths), len(TARGET_SYMPTOMS)), dtype=np.float64)
    for name in manifest["folds"]:
        print("Predict:", name, flush=True)
        neural_probs += predict_checkpoint(root / name, documents, tokenizer, config)
        _release_memory()
    neural_probs /= len(manifest["folds"])
    sparse = joblib.load(root / manifest["sparse"])
    sparse_probs = predict_sparse(sparse, texts)
    probabilities = apply_calibrator(neural_probs, sparse_probs, manifest["calibrator"])
    predictions = write_predictions(paths, probabilities, manifest["calibrator"]["thresholds"], Path(output))
    print(f"{len(paths):,} predictions saved: {output}", flush=True)
    return paths, probabilities, predictions


def evaluate_csv(label_dir, prediction_path):
    """이미 저장된 예측을 파일명으로 정답에 연결한다. 모델/임계값 변경 없음."""
    with Path(prediction_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["label file name", "symptom"]:
            raise ValueError("CSV header must be: label file name,symptom")
        rows = list(reader)
    by_name = {}
    for row in rows:
        if row["label file name"] in by_name:
            raise ValueError("Duplicate prediction filename")
        labels = ast.literal_eval(row["symptom"])
        if not isinstance(labels, list) or any(x not in TARGET_SYMPTOMS for x in labels) or len(set(labels)) != len(labels):
            raise ValueError("CSV symptom must be a list of unique target strings")
        by_name[row["label file name"]] = [int(x in labels) for x in TARGET_SYMPTOMS]
    paths, _, targets = load_dataset(Path(label_dir), include_targets=True)
    if set(by_name) != {path.name for path in paths}:
        raise ValueError("CSV와 JSON 파일명 집합이 일치하지 않습니다.")
    return score_predictions(targets, np.asarray([by_name[path.name] for path in paths], dtype=np.int8))

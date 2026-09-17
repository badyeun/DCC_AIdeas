"""Exercise the shipped standalone notebook with a tiny shared ZIP, without Colab/GPU/network."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
import zipfile

import pytest

from mission3_ver6 import bootstrap


SOURCE_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = SOURCE_DIR.parent / "mission3_ver6.ipynb"
LABELS = ["고열", "구토", "두통", "복통", "어지러움", "열상", "오심", "전신쇠약", "호흡곤란"]


@pytest.fixture
def notebook():
    saved = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = {cell["id"]: "".join(cell["source"]) for cell in saved["cells"]}
    return SimpleNamespace(saved=saved, cells=cells)


def make_input_zip(directory):
    names = ["mission3_ver4_model.ckpt", "splits.json", "calibration_predictions.npz", "sparse.joblib",
             *[f"fold_{i}.ckpt" for i in range(1, 6)], "tokenizer/tokenizer.json",
             "tokenizer/tokenizer_config.json", "tokenizer/vocab.txt", "tokenizer/special_tokens_map.json"]
    assets = {bootstrap.MODEL_PREFIX + name: ("dummy " + name).encode() for name in names}
    for prefix, name, text in [(bootstrap.TRAIN_PREFIX, "train.json", "학습 대화"),
                                (bootstrap.VALID_PREFIX, "valid.json", "검증 대화")]:
        assets[prefix + name] = json.dumps({"utterances": [{"text": text}], "symptom": ["두통"]},
                                           ensure_ascii=False).encode()
    manifest = {"format": "mission3_ver5_inputs_v1", "training_count": 1, "validation_count": 1,
                "files": [{"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                          for name, data in assets.items()]}
    directory.mkdir(parents=True)
    archive_path = directory / "mission3_ver5_inputs.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for name, data in assets.items():
            archive.writestr(name, data)
        archive.writestr("bundle_manifest.json", json.dumps(manifest, ensure_ascii=False))
    return archive_path, assets


@pytest.fixture
def prepared_colab(notebook, tmp_path, monkeypatch):
    content = tmp_path / "content"
    my_drive = content / "drive/MyDrive"
    my_drive.mkdir(parents=True)
    archive_path, assets = make_input_zip(tmp_path / "shared_with_me")
    shortcut = my_drive / archive_path.name
    shortcut.symlink_to(archive_path)
    mount = Mock()
    google, colab = ModuleType("google"), ModuleType("google.colab")
    google.__path__ = []
    colab.drive = SimpleNamespace(mount=mount)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.chdir(tmp_path)
    namespace = {"sys": sys, "subprocess": subprocess}
    original_archive = archive_path.read_bytes()
    original_mtime = archive_path.stat().st_mtime_ns

    def run_cell(identifier):
        code = notebook.cells[identifier]
        # Relocate only notebook filesystem configuration, never its embedded runtime sources.
        if identifier in ("ver6-settings", "ver6-paths"):
            code = code.replace("/content", content.as_posix())
        exec(compile(code, NOTEBOOK_PATH.name + ":" + identifier, "exec"), namespace)

    run_cell("ver6-settings")
    run_cell("ver6-paths")
    runtime = tmp_path / "embedded_runtime"
    with patch.object(tempfile, "mkdtemp", return_value=str(runtime)):
        run_cell("ver6-runtime")
    return SimpleNamespace(namespace=namespace, run_cell=run_cell, content=content, my_drive=my_drive,
                           archive=archive_path, original_archive=original_archive, original_mtime=original_mtime,
                           assets=assets, shortcut=shortcut, mount=mount, runtime=runtime)


def test_notebook_is_clean_and_embeds_every_current_runtime_source(notebook):
    assert notebook.saved["metadata"]["accelerator"] == "GPU"
    for cell in notebook.saved["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse("".join(cell["source"]))
    runtime_assignment = next(node for node in ast.parse(notebook.cells["ver6-runtime"]).body
                              if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                              and target.id == "RUNTIME_SOURCES" for target in node.targets))
    sources = ast.literal_eval(runtime_assignment.value)
    declaration = next(node for node in ast.parse((SOURCE_DIR / "pipeline.py").read_text()).body
                       if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                       and target.id == "RUNTIME_FILES" for target in node.targets))
    assert set(ast.literal_eval(declaration.value)) <= set(sources)
    expected_legacy = {"ver4_runtime/" + path.name for path in (SOURCE_DIR / "ver4_runtime").iterdir()
                       if path.is_file()}
    assert expected_legacy <= set(sources)
    for name, source in sources.items():
        assert source.encode("utf-8") == (SOURCE_DIR / name).read_bytes(), name


def test_shared_shortcut_restores_inputs_and_separates_drive_results_from_local_features(prepared_colab):
    env = prepared_colab
    values = env.namespace
    env.mount.assert_called_once_with(str(env.content / "drive"), force_remount=False)
    assert env.shortcut.is_symlink()
    assert values["bundle_path"] == env.archive.resolve()
    bundle = env.content / "mission3_ver5_inputs/DCC_AIdeas"
    assert values["BASE_DIR"] == bundle / "results/mission3/ver4"
    assert values["TRAIN_LABEL_DIR"] == bundle / "DATA/대학부 데이터/Training/2.라벨링데이터/TL_서울_구급"
    assert values["VALID_LABEL_DIR"] == bundle / "DATA/대학부 데이터/Validation/2.라벨링데이터/VL_서울_구급"
    assert values["OUTPUT_DIR"] == env.my_drive / "DCC_AIdeas/results/mission3/ver6"
    assert values["INFERENCE_CACHE_DIR"] == env.my_drive / "DCC_AIdeas/results/mission3/prediction_cache"
    assert values["FROZEN_FEATURE_CACHE_DIR"] == env.content / "mission3_ver6_features"
    assert not values["FROZEN_FEATURE_CACHE_DIR"].is_relative_to(env.my_drive)
    assert values["TRAIN_CONFIG"]["feature_cache_dir"] == str(env.content / "mission3_ver6_features")
    assert values["TRAIN_CONFIG"]["bundle_base"] is False
    for name, expected in env.assets.items():
        assert (env.content / "mission3_ver5_inputs" / name).read_bytes() == expected
    assert env.archive.read_bytes() == env.original_archive
    assert env.archive.stat().st_mtime_ns == env.original_mtime
    configuration = json.loads((env.runtime / "notebook_config.json").read_text())
    assert configuration == values["TRAIN_CONFIG"]
    for name, source in values["RUNTIME_SOURCES"].items():
        assert (env.runtime / name).read_bytes() == source.encode("utf-8")


def test_repeated_setup_reads_only_zip_manifest_and_preserves_extracted_files(prepared_colab, capsys):
    env = prepared_colab
    root = env.content / "mission3_ver5_inputs"
    times = {name: (root / name).stat().st_mtime_ns for name in env.assets}
    original_open = zipfile.ZipFile.open

    def manifest_only(archive, name, *args, **kwargs):
        name_value = name.filename if isinstance(name, zipfile.ZipInfo) else name
        assert name_value == "bundle_manifest.json", "An unchanged input member was extracted again"
        return original_open(archive, name, *args, **kwargs)

    with patch.object(zipfile.ZipFile, "open", new=manifest_only):
        env.run_cell("ver6-paths")
    assert {name: (root / name).stat().st_mtime_ns for name in env.assets} == times
    assert "재사용" in capsys.readouterr().out
    assert env.archive.read_bytes() == env.original_archive
    assert env.archive.stat().st_mtime_ns == env.original_mtime


def test_training_report_and_final_evaluation_cells_use_correct_cli_and_save_results(prepared_colab, capsys):
    env = prepared_colab
    values = env.namespace
    output = values["OUTPUT_DIR"]
    report = {
        "candidate_training": {"selected_profile": "D", "best_epoch": 0,
                               "selection_scope": "Training stop threshold cross-fit",
                               "head_trials": [{"profile": "A", "best_epoch": 0,
                                                "best_stop_macro_f1": 0.61, "best_stop_ap": 0.66},
                                               {"profile": "D", "best_epoch": 2,
                                                "best_stop_macro_f1": 0.68, "best_stop_ap": 0.70}]},
        "baseline_crossfit_macro_f1": 0.65, "candidate_crossfit_macro_f1": 0.66,
        "candidate_raw_crossfit_macro_f1": 0.66, "candidate_logit_crossfit_macro_f1": 0.655,
        "selected_name": "candidate_mixture", "selected_blend_space": "raw",
        "selection_gain": 0.01, "minimum_gain": 0.002, "candidate_improved_folds": 4,
        "minimum_improved_folds": 3, "candidate_fold_gains": [0.01, 0.01, -0.001, 0.01, 0.02],
        "fallback_reason": None,
        "calibrator": {"neural_weights": [0.5] * 9, "candidate_weights": [0.25] * 9, "thresholds": [0.5] * 9},
    }
    metrics = {"samples": 1, "macro_f1": 0.71,
               "per_symptom": {label: {"precision": 0.7, "recall": 0.72, "f1": 0.71, "support": 1}
                               for label in LABELS}}
    calls = []

    def fake_run(arguments):
        args = list(map(str, arguments))
        calls.append(args)
        if Path(args[0]).name == "train.py":
            output.mkdir(parents=True)
            (output / "mission3_ver6_model.ckpt").write_bytes(b"mock completed model")
            (output / "mission3_ver6_metrics.json").write_text(json.dumps(report), encoding="utf-8")
        elif Path(args[0]).name == "inference.py":
            destination = Path(args[args.index("--output") + 1])
            destination.parent.mkdir(parents=True)
            destination.write_text("filename," + ",".join(LABELS) + "\n", encoding="utf-8")
            destination.with_name("validation_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        else:
            pytest.fail(f"Unexpected notebook subprocess: {args}")

    values["run_python"] = fake_run
    env.run_cell("ver6-train")
    env.run_cell("ver6-report")
    env.run_cell("ver6-evaluate")
    training, inference = calls
    assert training[training.index("--train_label_dir") + 1] == str(values["TRAIN_LABEL_DIR"])
    assert str(values["VALID_LABEL_DIR"]) not in training
    assert training[training.index("--output_dir") + 1] == str(output)
    assert "--auto_batch" in training
    assert inference[inference.index("--label_dir") + 1] == str(values["VALID_LABEL_DIR"])
    assert inference[inference.index("--ckpt_path") + 1] == str(output / "mission3_ver6_model.ckpt")
    assert inference[inference.index("--base_dir") + 1] == str(values["BASE_DIR"])
    assert inference[inference.index("--cache_dir") + 1] == str(values["INFERENCE_CACHE_DIR"])
    assert inference[inference.index("--batch_size") + 1] == "8"
    assert "--evaluate" in inference
    assert values["PREDICTION_PATH"].is_relative_to(output / "predictions")
    assert values["PREDICTION_PATH"].is_file()
    timing_paths = list(output.glob("notebook_runtime_*.json"))
    assert len(timing_paths) == 1
    timing = json.loads(timing_paths[0].read_text())
    assert timing["output_dir"] == str(output)
    assert timing["prediction_csv"] == str(values["PREDICTION_PATH"])
    assert all(timing[key] >= 0 for key in ("preparation_minutes", "training_and_calibration_minutes",
                                           "inference_and_evaluation_minutes"))
    printed = capsys.readouterr().out
    assert "encoder 미세조정 최적 epoch: 0" in printed
    assert "D | 2 | 0.68 | 0.7" in printed
    assert "Macro F1: 0.710000000" in printed
    assert "Precision | Recall | F1" in printed
    assert "ver4 이전 기록 대비" in printed
    assert "ver5 이전 기록 대비" in printed


def test_validation_can_be_skipped_without_starting_inference(prepared_colab):
    env = prepared_colab
    values = env.namespace
    values["OUTPUT_DIR"].mkdir(parents=True)
    values["RUN_VALIDATION"] = False
    values["CHECKPOINT_PATH"] = values["OUTPUT_DIR"] / "mission3_ver6_model.ckpt"
    values["TRAINING_MINUTES"] = 0.0
    values["run_python"] = Mock(side_effect=AssertionError("Disabled evaluation must not start inference"))
    env.run_cell("ver6-evaluate")
    values["run_python"].assert_not_called()
    assert values["PREDICTION_PATH"] is None
    assert not (values["OUTPUT_DIR"] / "predictions").exists()

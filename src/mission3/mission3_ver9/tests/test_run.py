"""Supervisor behavior without installation, model downloads, datasets or GPU."""
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import pytest

from mission3_ver9 import run as supervisor
from mission3_ver9.budget import BudgetExceeded


@pytest.fixture
def controlled(tmp_path, monkeypatch):
    runtime = tmp_path / "embedded_runtime"
    runtime.mkdir()
    monkeypatch.setattr(supervisor, "HERE", runtime)
    status_root = tmp_path / "local_status"
    status_root.mkdir()
    original_mkdtemp = supervisor.tempfile.mkdtemp
    monkeypatch.setattr(supervisor.tempfile, "mkdtemp", lambda prefix: original_mkdtemp(prefix=prefix, dir=status_root))
    settings = {
        "output_dir": str(tmp_path / "drive/results/ver9"),
        "base_dir": str(tmp_path / "input/ver4"),
        "train_label_dir": str(tmp_path / "input/Training"),
        "valid_label_dir": str(tmp_path / "input/Validation"),
        "pretrained_dir": str(tmp_path / "public_model"),
        "train_config": {"max_models": 1, "final_reserve_seconds": 1500},
    }
    state = SimpleNamespace(settings=settings, calls=[], instances=[], failures={},
                            remaining=8000.0, status_root=status_root, publish_marker=True)

    class FakeBudget:
        def __init__(self, *, start_epoch=None):
            state.instances.append(self)
            self.start_epoch = start_epoch if start_epoch is not None else time.time()
            self.deadline_epoch = self.start_epoch + 10800

        def remaining_seconds(self):
            return state.remaining

        def run(self, command, **kwargs):
            command = [str(part) for part in command]
            if command[1] == "-c":
                stage = "status_copy"
            elif command[2] == "-m":
                stage = "dependencies"
            elif command[2] == "-c":
                stage = "input_restore" if "restore_input_bundle" in command[3] else "verify_model"
            elif Path(command[2]).name == "pretrained.py":
                stage = "public_pretrained"
            elif Path(command[2]).name == "train.py":
                stage = "training"
            elif Path(command[2]).name == "inference.py":
                stage = "candidate_diagnostic" if "--candidate_output" in command else "final_validation"
            else:
                raise AssertionError(f"Unexpected supervisor command: {command}")
            state.calls.append(SimpleNamespace(stage=stage, command=command, kwargs=kwargs, budget=self))
            if stage == "training" and state.publish_marker:
                marker = Path(command[command.index("--attempt_marker") + 1])
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(json.dumps({"signature": "verified-this-attempt", "output_dir": settings["output_dir"]}),
                                  encoding="utf-8")
            if stage in state.failures:
                raise state.failures[stage]
            if stage == "final_validation":
                destination = Path(command[command.index("--output") + 1])
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text("label file name,symptom\nsample.json,[]\n", encoding="utf-8")
            if stage == "status_copy":
                # Execute only the tiny real status writer against local temporary
                # paths; this catches broken inline Python or publication logic.
                return subprocess.run(command, check=True, capture_output=True, text=True)
            return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(supervisor, "TimeBudget", FakeBudget)
    state.run = lambda **kwargs: supervisor.run(settings, **kwargs)
    state.report = lambda: json.loads((Path(settings["output_dir"]) / "runtime.json").read_text())
    return state


def stages(state):
    return [call.stage for call in state.calls]


def test_every_stage_shares_original_budget_and_validation_never_enters_training(controlled):
    state = controlled
    original_start = time.time() - 27
    output = state.run(start_epoch=original_start)
    assert output == Path(state.settings["output_dir"])
    assert len(state.instances) == 1
    assert all(call.budget is state.instances[0] for call in state.calls)
    assert stages(state) == ["dependencies", "public_pretrained", "training", "final_validation", "candidate_diagnostic", "status_copy"]
    training = next(call for call in state.calls if call.stage == "training")
    assert training.command[training.command.index("--start_epoch") + 1] == str(original_start)
    assert float(training.command[training.command.index("--deadline_epoch") + 1]) == original_start + 10800
    assert state.settings["train_label_dir"] in training.command
    assert state.settings["valid_label_dir"] not in training.command
    assert not any("valid" in argument.lower() for argument in training.command if argument.startswith("--"))
    assert training.kwargs["reserve_seconds"] == 1500
    for call in state.calls[:-1]:
        assert call.kwargs["stream_output"] is True
        online = call.stage in ("dependencies", "public_pretrained")
        assert call.kwargs["env"]["HF_HUB_OFFLINE"] == ("0" if online else "1")
        assert call.kwargs["env"]["TRANSFORMERS_OFFLINE"] == ("0" if online else "1")
    report = state.report()
    assert report["start_epoch"] == original_start
    assert report["deadline_epoch"] == original_start + 10800
    assert report["final_validation_completed_this_run"] is True
    assert report["final_csv_exists"] is True
    assert report["failures"] == []


@pytest.mark.parametrize("failure", [BudgetExceeded("training timed out"), subprocess.CalledProcessError(2, ["train.py"])])
def test_training_failure_still_attempts_final_inference_with_published_fallback(controlled, failure):
    state = controlled
    state.failures["training"] = failure
    state.run()
    assert stages(state).index("training") < stages(state).index("final_validation")
    report = state.report()
    assert report["final_validation_completed_this_run"] is True
    assert report["final_csv_exists"] is True
    assert [item["phase"] for item in report["failures"]] == ["training"]


@pytest.mark.parametrize("failure", [None, BudgetExceeded("before portable model"),
                                     subprocess.CalledProcessError(2, ["train.py"])])
def test_unverified_attempt_never_evaluates_stale_model_or_csv(controlled, failure):
    state = controlled
    state.publish_marker = False
    output = Path(state.settings["output_dir"])
    output.mkdir(parents=True)
    (output / "mission3_ver9_model.ckpt").write_text("old unrelated experiment", encoding="utf-8")
    (output / "inference.csv").write_text("old complete CSV", encoding="utf-8")
    if failure is not None:
        state.failures["training"] = failure
    with pytest.raises(RuntimeError, match="did not verify a portable model"):
        state.run()
    assert "final_validation" not in stages(state)
    assert "candidate_diagnostic" not in stages(state)
    assert "verify_model" not in stages(state)
    assert state.report()["final_validation_completed_this_run"] is False
    assert (output / "inference.csv").read_text() == "old complete CSV"


def test_final_csv_is_published_before_optional_diagnostic(controlled):
    state = controlled
    state.failures["candidate_diagnostic"] = BudgetExceeded("diagnostic budget ended")
    state.run()
    assert stages(state).index("final_validation") < stages(state).index("candidate_diagnostic")
    output = Path(state.settings["output_dir"]) / "inference.csv"
    assert output.read_text().startswith("label file name,symptom\n")
    report = state.report()
    assert report["final_validation_completed_this_run"] is True
    assert [item["phase"] for item in report["failures"]] == ["candidate_diagnostic"]


@pytest.mark.parametrize("failure", [BudgetExceeded("final timed out"), subprocess.CalledProcessError(3, ["inference.py"])])
def test_final_validation_failure_skips_diagnostic_and_distinguishes_stale_csv(controlled, failure):
    state = controlled
    output = Path(state.settings["output_dir"])
    output.mkdir(parents=True)
    (output / "inference.csv").write_text("old complete CSV", encoding="utf-8")
    state.failures["final_validation"] = failure
    with pytest.raises(type(failure)):
        state.run()
    assert "candidate_diagnostic" not in stages(state)
    report = state.report()
    assert report["final_csv_exists"] is True
    assert report["final_validation_completed_this_run"] is False
    assert (output / "inference.csv").read_text() == "old complete CSV"
    assert [item["phase"] for item in report["failures"]] == ["workflow"]


@pytest.mark.parametrize("remaining", [0.0, 360.0])
def test_optional_diagnostic_is_skipped_when_reserve_is_low(controlled, remaining):
    controlled.remaining = remaining
    controlled.run()
    assert "final_validation" in stages(controlled)
    assert "candidate_diagnostic" not in stages(controlled)


def test_disabled_validation_verifies_model_without_reading_validation(controlled):
    state = controlled
    state.settings.update(run_validation=False)
    state.settings.pop("valid_label_dir")
    state.run()
    assert "verify_model" in stages(state)
    assert "final_validation" not in stages(state)
    assert "candidate_diagnostic" not in stages(state)
    assert state.report()["final_validation_completed_this_run"] is False
    assert state.report()["final_csv_exists"] is False


def test_input_restore_and_install_share_budget_without_importing_ml(controlled, tmp_path):
    state = controlled
    state.settings.update(input_zip=str(tmp_path / "drive/shared_inputs.zip"),
                          extract_dir=str(tmp_path / "restored"))
    state.run()
    assert stages(state)[:3] == ["dependencies", "input_restore", "public_pretrained"]
    restore = next(call for call in state.calls if call.stage == "input_restore")
    assert restore.budget is state.instances[0]
    assert restore.kwargs["reserve_seconds"] == 3900
    train = next(call for call in state.calls if call.stage == "training")
    assert "TL_서울_구급" in train.command[train.command.index("--train_label_dir") + 1]
    assert not any("VL_서울_구급" in argument for argument in train.command)


def test_dependency_failure_stops_later_stages_but_writes_status(controlled):
    state = controlled
    state.failures["dependencies"] = subprocess.CalledProcessError(1, ["pip"])
    with pytest.raises(subprocess.CalledProcessError):
        state.run()
    assert stages(state) == ["dependencies", "status_copy"]
    assert state.report()["final_validation_completed_this_run"] is False


def test_status_copy_timeout_preserves_local_report_and_successful_output(controlled):
    state = controlled
    state.failures["status_copy"] = BudgetExceeded("Drive status write timed out")
    output = state.run()
    assert (output / "inference.csv").is_file()
    assert not (output / "runtime.json").exists()
    reports = list(state.status_root.glob("*/runtime.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["final_validation_completed_this_run"] is True
    assert state.calls[-1].kwargs["timeout"] == 10


@pytest.mark.parametrize("reserve", [1499, 3601, float("nan"), float("inf")])
def test_invalid_final_reservation_rejected_before_any_subprocess(controlled, reserve):
    controlled.settings["train_config"]["final_reserve_seconds"] = reserve
    with pytest.raises(ValueError):
        controlled.run()
    assert controlled.calls == []

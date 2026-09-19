"""Standard-library supervisor for a single <= 3 hour automatic Colab run."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
if __package__:
    from .budget import TimeBudget, BudgetExceeded
else:
    from budget import TimeBudget, BudgetExceeded

HERE = Path(__file__).resolve().parent
LAST_REPORT = {}


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(settings, start_epoch=None):
    global LAST_REPORT
    budget = TimeBudget(start_epoch=start_epoch)
    stages, failures = {}, []
    # Do not resolve/stat Drive paths in the unsupervised notebook process.
    output = Path(os.path.abspath(os.path.expanduser(settings["output_dir"])))
    final_validation_completed = False
    config = dict(settings.get("train_config", {}))
    reserve = float(config.get("final_reserve_seconds", 1500))
    if not 1500 <= reserve <= 3600:
        raise ValueError("final_reserve_seconds must be between 1500 and 3600")
    common_env = dict(os.environ, HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
                      OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")

    def call(stage, arguments, *, online=False, reserve_seconds=0, timeout=None):
        before = time.time()
        print(f"\n[{stage}] remaining {budget.remaining_seconds() / 60:.1f} minutes", flush=True)
        offline = "0" if online else "1"
        try:
            return budget.run([sys.executable, "-u", *map(str, arguments)], cwd=HERE,
                reserve_seconds=reserve_seconds, timeout=timeout, stream_output=True,
                env=dict(common_env, HF_HUB_OFFLINE=offline, TRANSFORMERS_OFFLINE=offline))
        finally:
            stages[stage] = time.time() - before

    try:
        if settings.get("install_dependencies", True):
            call("dependencies", ["-m", "pip", "install", "--disable-pip-version-check", "-q", "-r", HERE / "requirements.txt"],
                 online=True, reserve_seconds=reserve + 2400, timeout=900)
        if settings.get("input_zip"):
            destination = Path(os.path.abspath(os.path.expanduser(settings.get("extract_dir", "/content/mission3_ver5_inputs"))))
            # Restore in a supervised process; large Drive I/O cannot block the notebook indefinitely.
            script = "import sys; from bootstrap import restore_input_bundle; restore_input_bundle(sys.argv[1], sys.argv[2])"
            call("input_restore", ["-c", script, settings["input_zip"], destination], reserve_seconds=reserve + 2400)
            bundle = destination / "DCC_AIdeas"
            base = bundle / "results/mission3/ver4"
            data_root = bundle / "DATA/대학부 데이터"
            training = data_root / "Training/2.라벨링데이터/TL_서울_구급"
            validation = data_root / "Validation/2.라벨링데이터/VL_서울_구급"
        else:
            base = Path(os.path.abspath(settings["base_dir"]))
            training = Path(os.path.abspath(settings["train_label_dir"]))
            validation = Path(os.path.abspath(settings["valid_label_dir"])) if settings.get("valid_label_dir") else None
        public_model = Path(os.path.abspath(os.path.expanduser(settings.get("pretrained_dir", "/content/mission3_ver9_pretrained/deberta_korean"))))
        call("public_pretrained", [HERE / "pretrained.py", "--destination", public_model],
             online=True, reserve_seconds=reserve + 2400, timeout=1800)
        config_path = HERE / "run_train_config.json"
        _write(config_path, config)
        attempt_marker = Path(tempfile.mkdtemp(prefix="mission3_ver9_attempt_")) / "portable_model_ready.json"
        try:
            call("training", [HERE / "train.py", "--train_label_dir", training, "--base_dir", base,
                 "--pretrained_dir", public_model, "--output_dir", output, "--config_json", config_path,
                 "--start_epoch", budget.start_epoch, "--deadline_epoch", budget.deadline_epoch, "--auto_batch",
                 "--attempt_marker", attempt_marker],
                 reserve_seconds=reserve)
        except (BudgetExceeded, subprocess.CalledProcessError) as error:
            failures.append({"phase": "training", "error": str(error)})
            print("Candidate training interrupted. Checking the already published fallback.", flush=True)
        if not attempt_marker.is_file():
            raise RuntimeError("This attempt did not verify a portable model. Inspect training errors; an older run will not be evaluated as the new experiment.")
        checkpoint = output / "mission3_ver9_model.ckpt"
        if settings.get("run_validation", True):
            if validation is None:
                raise FileNotFoundError("Final Validation JSON directory is missing")
            cache = Path(settings.get("cache_dir", output.parent / "prediction_cache"))
            arguments = [HERE / "inference.py", "--audio_dir", "", "--label_dir", validation,
                         "--ckpt_path", checkpoint, "--output", output / "inference.csv", "--cache_dir", cache,
                         "--batch_size", settings.get("inference_batch_size", 4), "--evaluate"]
            # Publish the requested final CSV first. Diagnostics cannot endanger it.
            call("final_validation", arguments, reserve_seconds=30)
            final_validation_completed = True
            if settings.get("run_candidate_diagnostic", True) and budget.remaining_seconds() > 360:
                try:
                    call("candidate_diagnostic", [*arguments, "--candidate_output", output / "candidate_inference.csv"], reserve_seconds=20)
                except (BudgetExceeded, subprocess.CalledProcessError) as error:
                    failures.append({"phase": "candidate_diagnostic", "error": str(error)})
        else:
            call("verify_model", ["-c", "import sys; from pipeline import load_manifest; load_manifest(sys.argv[1])", checkpoint], reserve_seconds=20)
        return output
    except (BudgetExceeded, subprocess.CalledProcessError) as error:
        failures.append({"phase": "workflow", "error": str(error)})
        print("Automatic run stopped within the shared budget; completed artifacts are preserved.", flush=True)
        raise
    finally:
        # Even a small Drive write may hang: keep a local report, copy under the same watchdog.
        report = {"start_epoch": budget.start_epoch, "deadline_epoch": budget.deadline_epoch,
            "elapsed_seconds": time.time() - budget.start_epoch, "budget_seconds": 10800,
            "stages_seconds": stages, "failures": failures,
            "final_validation_completed_this_run": final_validation_completed}
        local_status = Path(tempfile.mkdtemp(prefix="mission3_ver9_status_")) / "runtime.json"
        _write(local_status, report)
        LAST_REPORT = {"runtime.json": report}
        status_script = '''import json,sys,time
from pathlib import Path
p=Path(sys.argv[2]); p.parent.mkdir(parents=True,exist_ok=True)
v=json.loads(Path(sys.argv[1]).read_text())
v['final_csv_exists']=(p.parent/'inference.csv').is_file()
v['elapsed_seconds']=time.time()-v['start_epoch']
t=p.with_name(p.name+'.tmp')
t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\\n',encoding='utf-8'); t.replace(p)
summary={'runtime.json':v}
for name in ('training_metrics.json','validation_metrics.json','candidate_validation_metrics.json'):
    f=p.parent/name
    if f.is_file(): summary[name]=json.loads(f.read_text(encoding='utf-8'))
print(json.dumps(summary,ensure_ascii=False))
'''
        try:
            saved = budget.run([sys.executable, "-c", status_script, str(local_status), str(output / "runtime.json")],
                               timeout=10, capture_output=True, text=True)
            LAST_REPORT = json.loads(saved.stdout)
        except (BudgetExceeded, subprocess.CalledProcessError, ValueError):
            print("Drive status save did not finish before the deadline. Local report:", local_status, flush=True)
        print("Runtime report:", json.dumps(report, ensure_ascii=False), flush=True)
        print("Output:", output, flush=True)
        print(f"Total elapsed: {(time.time() - budget.start_epoch) / 60:.1f} minutes", flush=True)
        if failures:
            print("Interrupted phases:", json.dumps(failures, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--start_epoch", type=float)
    args = parser.parse_args()
    run(json.loads(Path(args.settings).read_text(encoding="utf-8")), args.start_epoch)


if __name__ == "__main__":
    main()

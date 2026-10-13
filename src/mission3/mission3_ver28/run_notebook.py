"""Fresh-process notebook entry point with privacy-safe error diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", required=True)
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args(argv)
    result_path = Path(args.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    before = set()
    try:
        from . import data, runtime
        from .build_notebook import source_files
        root = Path(__file__).resolve().parent
        sources = {name: (root / name).read_text(encoding="utf-8") for name in source_files()}
        settings = json.loads(Path(args.settings).read_text(encoding="utf-8-sig"))
        before = set(Path(settings["output_dir"]).glob("*/run_state.json"))
        result = runtime.run(settings, runtime_sources=sources)
        data.write_json(result_path, result)
        return 0
    except Exception as error:
        frames = [{"file": Path(frame.f_code.co_filename).name, "line": line, "function": frame.f_code.co_name}
                  for frame, line in traceback.walk_tb(error.__traceback__)]
        # No raw exception values, transcripts, or patient metadata are logged.
        report = {"status": "failed", "submission_ready": False,
                  "error_type": type(error).__name__, "frames": frames[-15:]}
        try:
            settings = json.loads(Path(args.settings).read_text(encoding="utf-8-sig"))
            runs = set(Path(settings["output_dir"]).glob("*/run_state.json")) - before
            if len(runs) == 1:
                report["run_dir"] = str(next(iter(runs)).parent)
        except (OSError, ValueError, KeyError):
            pass
        result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

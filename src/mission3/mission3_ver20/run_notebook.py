"""Launch the bounded ver20 experiment in a fresh interpreter after installation."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys

SOURCE_FILES = (
    "__init__.py", "data.py", "bootstrap.py", "neural.py", "budget.py", "runtime.py",
    "inference.py", "reference.py", "training_reference.json", "run_notebook.py", "requirements.txt", "README.md",
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", required=True)
    parser.add_argument("--result_file", required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    name = "mission3_ver20_execution"
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py",
                                                  submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    runtime = importlib.import_module(name + ".runtime")
    settings = json.loads(Path(args.settings).read_text(encoding="utf-8-sig"))
    sources = {name: (root / name).read_text(encoding="utf-8") for name in SOURCE_FILES}
    result = runtime.run(settings, runtime_sources=sources)
    Path(args.result_file).write_text(json.dumps(str(result)), encoding="utf-8")


if __name__ == "__main__":
    main()

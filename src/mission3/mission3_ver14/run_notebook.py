"""Launch the ver14 experiment suite in a fresh interpreter after installation."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", required=True)
    parser.add_argument("--result_file", required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    name = "mission3_ver14_execution"
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py",
                                                  submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    suite = importlib.import_module(name + ".runtime_suite")
    settings = json.loads(Path(args.settings).read_text(encoding="utf-8-sig"))
    sources = {path.name: path.read_text(encoding="utf-8") for path in sorted(root.iterdir())
               if path.is_file() and (path.suffix in (".py", ".md") or path.name == "requirements.txt")}
    result = suite.run(settings, runtime_sources=sources)
    Path(args.result_file).write_text(json.dumps(str(result)), encoding="utf-8")


if __name__ == "__main__":
    main()

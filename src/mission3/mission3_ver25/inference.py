"""One-command inference entry point for a portable ver25 result folder."""
import importlib.util
from pathlib import Path
import sys

if __package__:
    from .runtime import inference_main as main
else:
    root = Path(__file__).resolve().parent
    name = "mission3_ver25_submission"
    spec = importlib.util.spec_from_file_location(name, root / "__init__.py",
                                                  submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    from importlib import import_module
    main = import_module(name + ".runtime").inference_main

if __name__ == "__main__":
    main()

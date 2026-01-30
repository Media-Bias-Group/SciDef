"""Bootstrap helpers for scripts.

Ensures the repo root is on sys.path and loads the local config.py
without colliding with the PyPI `config` package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_config_module() -> ModuleType:
    """Load the repo-local config.py as a dedicated module."""
    existing = sys.modules.get("scidef_config")
    if existing is not None:
        return existing

    config_path = REPO_ROOT / "config.py"
    spec = importlib.util.spec_from_file_location(
        "scidef_config",
        config_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load config from {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["scidef_config"] = module
    return module

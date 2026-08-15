"""Import shim for tests; the executable keeps its conventional dashed filename."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_path = Path(__file__).with_name("train-environment-specialists.py")
_spec = importlib.util.spec_from_file_location("train_environment_specialists_impl", _path)
if _spec is None or _spec.loader is None:  # pragma: no cover - import machinery failure
    raise ImportError(f"cannot load {_path}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

hard_negative_augmented = _module.hard_negative_augmented

__all__ = ["hard_negative_augmented"]

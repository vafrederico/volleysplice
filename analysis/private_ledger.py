"""Resolve private research inputs without committing workstation or media names.

The ledger is deliberately external to Git. Missing configuration fails closed;
an alias must never silently become a relative output path on the system disk.
"""
import json
import os
from pathlib import Path


def private_value(index: str) -> str:
    platform = "WINDOWS" if os.name == "nt" else "POSIX"
    location = os.environ.get(f"VOLLEYCUT_PRIVATE_LEDGER_{platform}") or os.environ.get("VOLLEYCUT_PRIVATE_LEDGER")
    if not location:
        raise RuntimeError("Set VOLLEYCUT_PRIVATE_LEDGER to the private research ledger")
    ledger = json.loads(Path(location).read_text(encoding="utf-8"))
    value = ledger.get("privateValues", {}).get(index)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing private ledger index: {index}")
    return value

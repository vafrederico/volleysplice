#!/usr/bin/env python3
"""Export one canonical base-feature cache for the TypeScript parity test."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_cache", type=Path)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()

    with np.load(arguments.feature_cache, allow_pickle=False) as cache:
        times = np.ascontiguousarray(cache["times"], dtype="<f8")
        values = np.ascontiguousarray(cache["values"], dtype="<f4")
        names = [str(value) for value in cache["names"]]
    if values.shape != (len(times), len(names)):
        raise SystemExit("feature cache arrays are not aligned")

    analysis = json.loads(arguments.analysis.read_text(encoding="utf-8"))
    rallies = [
        {
            "id": rally["id"],
            "start": rally["start"],
            "end": rally["end"],
            "confidence": rally["confidence"],
        }
        for rally in analysis["rallies"]
    ]
    payload = times.tobytes(order="C") + values.tobytes(order="C")
    arguments.destination.mkdir(parents=True, exist_ok=True)
    binary_path = arguments.destination / "on-device-y9-base-features.bin.gz"
    binary_path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    metadata = {
        "schemaVersion": 1,
        "sourceFeatureCacheSha256": sha256(arguments.feature_cache),
        "sourceAnalysisSha256": sha256(arguments.analysis),
        "duration": analysis["source"]["duration"],
        "rows": int(values.shape[0]),
        "columns": int(values.shape[1]),
        "timesBytes": int(times.nbytes),
        "names": names,
        "rallies": rallies,
    }
    (arguments.destination / "on-device-y9-golden.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

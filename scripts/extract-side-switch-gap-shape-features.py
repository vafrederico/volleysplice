#!/usr/bin/env python3
"""Append preregistered G1 gap-shape values using existing production traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_feature_development import GAP_SHAPE_FEATURE_NAMES, gap_shape_features
from analysis.side_switch_production_replay import (
    load_frozen_heads,
    numeric_array,
    prepared_from_feedback,
    replay_trace,
)
from analysis.side_switch_production_state import PRODUCTION_MODEL_SOURCES


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
INTAKE = Path("/mnt/freenas/volleycut/intake-2026-08-13")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_MANIFEST = ROOT / "reports/full-nas-video-corpus-v1.json"
DEFAULT_FEATURES = REPORTS / "side-switch-full-union-v5-state-features-v1.json"
DEFAULT_OUTPUT = REPORTS / "side-switch-gap-shape-features-v1.json"
V2_BUNDLE = INTAKE / "experiments/environment-specialists-v2/models/all-labels/bundle.json"
OLD_MODEL_ROOT = (
    Path("/mnt/freenas/volleycut")
    / "labeling-v1-2026-08-09-no-beach-2026-08-12/models"
)
OLD_HEAD_NAMES = {
    "rally": "full-audiovisual-audio-normalized-v3",
    "serve": "serve-specialist-audio-normalized-v5",
    "dead": "dead-state-transition-audio-normalized-v5-no-legacy-final",
}
EXPECTED_SHA256 = {
    "manifest": "c177f2936dc1ea7f2953cf94a6b9720cd6d4bf4e652a4c7409499374c9fe1e24",
    "features": "9763cb3e5cd9baada64f4bf54f06140dcff5068bb8cd74a1d485c677d1e6c551",
}
PROBABILITY_PARITY_TOLERANCE = 5e-6


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _initial_dead_scores(feedback: Mapping[str, Any], rows: int) -> np.ndarray:
    return numeric_array(
        feedback["initialInference"]["probabilities"]["deadState"],
        "float32",
        (rows,),
    )


def extract(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "manifest": args.manifest.expanduser().resolve(),
        "features": args.features.expanduser().resolve(),
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite G1 feature artifact: {output}")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if args.enforce_source_hash and hashes != EXPECTED_SHA256:
        raise ValueError(f"G1 feature sources changed: {hashes}")
    script_path = Path(__file__).resolve()
    module_path = (
        script_path.parent.parent / "analysis/side_switch_feature_development.py"
    ).resolve()

    manifest = _load(paths["manifest"])
    feature_payload = _load(paths["features"])
    recording_ids = tuple(str(value) for value in feature_payload["scope"]["recordingIds"])
    records = {
        str(record["recordingId"]): record
        for record in manifest["records"]
        if str(record.get("recordingId")) in recording_ids
    }
    if set(records) != set(recording_ids):
        raise ValueError("G1 manifest does not cover the feature scope")
    source_rows = list(feature_payload["rows"])
    rows_by_recording = {
        recording_id: [
            row for row in source_rows if str(row["recordingId"]) == recording_id
        ]
        for recording_id in recording_ids
    }

    old_heads, v2_heads = load_frozen_heads(V2_BUNDLE, OLD_MODEL_ROOT, OLD_HEAD_NAMES)
    rows: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    for position, recording_id in enumerate(recording_ids, 1):
        record = records[recording_id]
        feedback_path = Path(str(record["labelPath"])).resolve()
        feedback = _load(feedback_path)
        source_audit = feature_payload["extractionAudit"][recording_id]
        if _sha256(feedback_path) != source_audit["feedbackWholeFileSha256"]:
            raise ValueError(f"G1 feedback file changed for {recording_id}")
        if _json_sha256(feedback["features"]) != source_audit["feedbackFeaturesSha256"]:
            raise ValueError(f"G1 feedback features changed for {recording_id}")

        prepared = prepared_from_feedback(record, feedback, v2_heads.rally)
        trace = replay_trace(prepared, old_heads, v2_heads)
        expected_dead = _initial_dead_scores(feedback, len(trace.times))
        dead_parity = float(
            np.max(
                np.abs(
                    expected_dead
                    - trace.dead_state_scores[PRODUCTION_MODEL_SOURCES[0]]
                )
            )
        )
        if dead_parity > PROBABILITY_PARITY_TOLERANCE:
            raise ValueError(f"G1 production replay drifted for {recording_id}")

        print(
            f"[{position}/{len(recording_ids)}] G1 trace features: {recording_id}",
            file=sys.stderr,
            flush=True,
        )
        for source in rows_by_recording[recording_id]:
            additions = gap_shape_features(
                trace.times,
                trace.dead_state_scores[PRODUCTION_MODEL_SOURCES[0]],
                trace.dead_state_scores[PRODUCTION_MODEL_SOURCES[1]],
                float(source["gapStart"]),
                float(source["gapEnd"]),
                trace.duration,
            )
            updated = dict(source)
            features = dict(source["features"])
            features.update(
                {name: round(float(value), 8) for name, value in additions.items()}
            )
            updated["features"] = features
            rows.append(updated)
        audits[recording_id] = {
            "candidates": len(rows_by_recording[recording_id]),
            "productionSamples": len(trace.times),
            "allLabelsV2DeadProbabilityMaximumAbsoluteDifference": dead_parity,
            "probabilityParityTolerance": PROBABILITY_PARITY_TOLERANCE,
            "feedbackPath": str(feedback_path),
            "feedbackSha256": _sha256(feedback_path),
        }

    if len(rows) != len(source_rows):
        raise ValueError("G1 extraction lost candidate rows")
    source_ids = [str(row["eventId"]) for row in source_rows]
    if [str(row["eventId"]) for row in rows] != source_ids:
        raise ValueError("G1 extraction changed candidate order")
    if any(
        any(row["features"][name] != source["features"][name] for name in source["features"])
        for row, source in zip(rows, source_rows, strict=True)
    ):
        raise ValueError("G1 extraction changed an existing feature")

    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-gap-shape-features-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "status": "opened-development-only",
        "scope": feature_payload["scope"],
        "profile": {
            **feature_payload["profile"],
            "name": "FULL-UNION-V5-STATE42-GAP-SHAPE8",
            "featureNames": [
                *feature_payload["profile"]["featureNames"],
                *GAP_SHAPE_FEATURE_NAMES,
            ],
            "gapShapeFeatureNames": list(GAP_SHAPE_FEATURE_NAMES),
            "gapShapeSource": "existing two production dead-state traces; no video decode",
        },
        "rows": rows,
        "extractionAudit": audits,
        "productionModels": feature_payload["productionModels"],
        "sources": {
            **{
                name: {"path": str(paths[name]), "sha256": hashes[name]}
                for name in paths
            },
            "extractor": {"path": str(script_path), "sha256": _sha256(script_path)},
            "featureModule": {
                "path": str(module_path),
                "sha256": _sha256(module_path),
            },
        },
        "limitations": [
            "All recordings are opened development scope.",
            "The fixed 0.80 consensus threshold is a feature reduction, not a candidate threshold sweep.",
            "Trace replay adds no new video frames but still depends on both frozen production bundles.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    payload = extract(_parser().parse_args())
    print(
        json.dumps(
            {
                "output": str(DEFAULT_OUTPUT),
                "rows": len(payload["rows"]),
                "featureNames": payload["profile"]["gapShapeFeatureNames"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

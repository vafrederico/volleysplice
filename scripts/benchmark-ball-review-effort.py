#!/usr/bin/env python3
"""Benchmark blind ball-presence review across GPT-5.6 models and efforts.

The model-facing pack contains only opaque image copies and frame IDs. Existing
Sol annotations are loaded only after every requested configuration has returned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
from typing import Any, Mapping


PILOT_ROOT = Path("/mnt/freenas/volleycut/ball-presence-v1/round-01")
DEFAULT_OUTPUT = Path(
    "/mnt/freenas/volleycut/ball-presence-v1/reports/ball-review-effort-screen12-v1"
)
CODEX_BIN = Path(os.environ.get("VOLLEYCUT_CODEX_BIN", shutil.which("codex") or "codex"))
FRAME_WIDTH = 960
FRAME_HEIGHT = 540

SAMPLE = (
    ("beach-source-02", "01-serve_window", "f000006484"),
    ("beach-source-01", "01-serve_window", "f000019842"),
    ("beach-source-01", "02-mid_live_a", "f000017202"),
    ("grass-source-04", "04-end_transition", "f000001626"),
    ("grass-source-01", "02-mid_live_a", "f000006682"),
    ("grass-source-09", "01-serve_window", "f000004988"),
    ("grass-source-09", "01-serve_window", "f000005032"),
    ("grass-source-10", "01-serve_window", "f000024500"),
    ("indoor-source-01", "03-mid_live_b", "f000016200"),
    ("indoor-source-01", "04-end_transition", "f000009226"),
    ("indoor-source-07", "03-mid_live_b", "f000025088"),
    ("indoor-source-07", "04-end_transition", "f000017696"),
)

# Interleaved order avoids putting every high-effort run at the end.
CONFIGS = (
    ("sol-medium", "gpt-5.6-sol", "medium"),
    ("luna-xhigh", "gpt-5.6-luna", "xhigh"),
    ("sol-max", "gpt-5.6-sol", "max"),
    ("terra-xhigh", "gpt-5.6-terra", "xhigh"),
    ("sol-low", "gpt-5.6-sol", "low"),
    ("luna-max", "gpt-5.6-luna", "max"),
    ("sol-xhigh", "gpt-5.6-sol", "xhigh"),
    ("terra-max", "gpt-5.6-terra", "max"),
    ("sol-high", "gpt-5.6-sol", "high"),
)

STATES = ("localizable", "fully_occluded", "out_of_frame", "indeterminate")
ROLES = ("primary-court", "other-court", "unknown")
VISIBILITIES = ("clear", "motion-blurred", "partially-occluded")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _source_image(recording: str, window: str, frame_id: str) -> Path:
    return PILOT_ROOT / "images" / recording / window / f"{frame_id}.png"


def _schema(frame_ids: list[str]) -> dict[str, Any]:
    number = {"type": "number", "minimum": 0.0, "maximum": 1.0}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["schemaVersion", "frames"],
        "properties": {
            "schemaVersion": {"type": "integer", "const": 1},
            "frames": {
                "type": "array",
                "minItems": len(frame_ids),
                "maxItems": len(frame_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "frameId",
                        "status",
                        "primaryBallState",
                        "objects",
                        "notes",
                    ],
                    "properties": {
                        "frameId": {"type": "string", "enum": frame_ids},
                        "status": {"type": "string", "const": "reviewed"},
                        "primaryBallState": {"type": "string", "enum": list(STATES)},
                        "objects": {
                            "type": "array",
                            "maxItems": 8,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "id",
                                    "category",
                                    "role",
                                    "bbox",
                                    "visibility",
                                    "truncated",
                                ],
                                "properties": {
                                    "id": {"type": "string", "minLength": 1},
                                    "category": {"type": "string", "const": "volleyball"},
                                    "role": {"type": "string", "enum": list(ROLES)},
                                    "bbox": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["x", "y", "width", "height"],
                                        "properties": {
                                            "x": number,
                                            "y": number,
                                            "width": {
                                                "type": "number",
                                                "exclusiveMinimum": 0.0,
                                                "maximum": 1.0,
                                            },
                                            "height": {
                                                "type": "number",
                                                "exclusiveMinimum": 0.0,
                                                "maximum": 1.0,
                                            },
                                        },
                                    },
                                    "visibility": {
                                        "type": "string",
                                        "enum": list(VISIBILITIES),
                                    },
                                    "truncated": {"type": "boolean"},
                                },
                            },
                        },
                        "notes": {"type": "string"},
                    },
                },
            },
        },
    }


def _prompt(frame_ids: list[str]) -> str:
    mapping = "\n".join(
        f"{index + 1:02d}. attachment {index + 1} -> {frame_id}"
        for index, frame_id in enumerate(frame_ids)
    )
    return f"""Perform a blind, independent ball-presence review of the {len(frame_ids)} attached 960x540 volleyball frames.

Do not use shell commands, tools, filesystem searches, or outside artifacts. Use only the attached pixels. The attachments are independent frames; where pixels alone cannot reliably establish whether a non-visible primary ball is occluded or out of frame, choose indeterminate.

Target every actual volleyball, including held, retrieved, or dead balls. Overlays and logos are not balls. A ball associated with the primary filmed court/camera is primary-court; a ball clearly associated with another court is other-court; otherwise use unknown. primaryBallState=localizable requires exactly one primary-court object. Other states require zero primary-court objects, though visible other-court or unknown balls should still be returned. Use tight normalized boxes. Set visibility from pixels and truncated when the box meets an image boundary. Object IDs only need to be unique within a frame.

Return exactly the required JSON schema, one row per frame, in this exact attachment mapping:
{mapping}
"""


def prepare_pack(output: Path) -> tuple[list[str], list[Path]]:
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    frame_ids: list[str] = []
    opaque_paths: list[Path] = []
    manifest_rows: list[dict[str, Any]] = []
    for index, (recording, window, frame_id) in enumerate(SAMPLE, start=1):
        source = _source_image(recording, window, frame_id)
        if not source.is_file():
            raise RuntimeError(f"missing benchmark image: {source}")
        opaque = inputs / f"image-{index:02d}.png"
        if opaque.exists():
            if _sha256(opaque) != _sha256(source):
                raise RuntimeError(f"opaque input changed: {opaque}")
        else:
            shutil.copyfile(source, opaque)
        frame_ids.append(frame_id)
        opaque_paths.append(opaque)
        manifest_rows.append(
            {
                "attachmentIndex": index,
                "frameId": frame_id,
                "image": opaque.name,
                "sha256": _sha256(opaque),
            }
        )
    schema = _schema(frame_ids)
    _atomic_json(output / "result.schema.json", schema)
    _atomic_json(
        output / "blind-pack.json",
        {
            "schemaVersion": 1,
            "policy": "volleyball-ball-presence-v1",
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT,
            "frames": manifest_rows,
        },
    )
    return frame_ids, opaque_paths


def _primary(annotation: Mapping[str, Any]) -> Mapping[str, Any] | None:
    rows = [row for row in annotation["objects"] if row["role"] == "primary-court"]
    return rows[0] if len(rows) == 1 else None


def _validate_result(value: Mapping[str, Any], frame_ids: list[str]) -> dict[str, Any]:
    if set(value) != {"schemaVersion", "frames"} or value["schemaVersion"] != 1:
        raise RuntimeError("result root does not match benchmark schema")
    rows = value["frames"]
    if not isinstance(rows, list) or len(rows) != len(frame_ids):
        raise RuntimeError("result does not contain exactly one row per benchmark frame")
    by_id: dict[str, Any] = {}
    for row in rows:
        frame_id = row["frameId"]
        if frame_id in by_id or frame_id not in frame_ids:
            raise RuntimeError(f"duplicate or unknown frame ID: {frame_id}")
        if row["status"] != "reviewed" or row["primaryBallState"] not in STATES:
            raise RuntimeError(f"invalid state for {frame_id}")
        primary_count = sum(item["role"] == "primary-court" for item in row["objects"])
        expected = 1 if row["primaryBallState"] == "localizable" else 0
        if primary_count != expected:
            raise RuntimeError(f"primary object/state mismatch for {frame_id}")
        for item in row["objects"]:
            box = item["bbox"]
            if box["x"] + box["width"] > 1.0 + 1e-9 or box["y"] + box["height"] > 1.0 + 1e-9:
                raise RuntimeError(f"out-of-bounds box for {frame_id}")
        by_id[frame_id] = row
    if set(by_id) != set(frame_ids):
        raise RuntimeError("result frame coverage differs from the blind pack")
    return by_id


def _usage(events_path: Path) -> Mapping[str, Any] | None:
    usage = None
    for line in events_path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.completed":
            usage = event.get("usage")
    return usage


def _codex_version() -> str:
    completed = subprocess.run(
        [str(CODEX_BIN), "--version"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def _execution_provenance(
    output: Path,
    frame_ids: list[str],
    opaque_paths: list[Path],
    model: str,
    effort: str,
) -> dict[str, Any]:
    return {
        "codexBinary": str(CODEX_BIN),
        "codexVersion": _codex_version(),
        "model": model,
        "effort": effort,
        "serviceTier": "default",
        "promptSha256": _text_sha256(_prompt(frame_ids)),
        "schemaSha256": _sha256(output / "result.schema.json"),
        "blindPackSha256": _sha256(output / "blind-pack.json"),
        "orderedInputSha256": [_sha256(path) for path in opaque_paths],
    }


def _validate_cached_run(
    result_path: Path,
    receipt_path: Path,
    frame_ids: list[str],
    expected_provenance: Mapping[str, Any],
) -> None:
    _validate_result(json.loads(result_path.read_text()), frame_ids)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("valid") is not True or receipt.get("exitCode") != 0:
        raise RuntimeError(f"cached run is not valid: {receipt_path}")
    if receipt.get("resultSha256") != _sha256(result_path):
        raise RuntimeError(f"cached result hash mismatch: {result_path}")
    if receipt.get("executionProvenance") != expected_provenance:
        raise RuntimeError(
            f"cached run provenance differs from this invocation: {receipt_path}"
        )


def _validate_receipt_identity(
    receipt: Mapping[str, Any],
    receipt_path: Path,
    config_id: str,
    model: str,
    effort: str,
) -> None:
    expected = {
        "configId": config_id,
        "model": model,
        "effort": effort,
        "serviceTier": "default",
    }
    actual = {key: receipt.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"benchmark receipt identity mismatch: {receipt_path}")


def run_config(
    output: Path,
    frame_ids: list[str],
    opaque_paths: list[Path],
    config_id: str,
    model: str,
    effort: str,
) -> None:
    run_dir = output / "runs" / config_id
    result_path = run_dir / "result.json"
    receipt_path = run_dir / "receipt.json"
    execution_provenance = _execution_provenance(
        output, frame_ids, opaque_paths, model, effort
    )
    if result_path.is_file() and receipt_path.is_file():
        _validate_cached_run(
            result_path,
            receipt_path,
            frame_ids,
            execution_provenance,
        )
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"
    stderr_path = run_dir / "stderr.log"
    command = [
        str(CODEX_BIN),
        "-a",
        "never",
        "-m",
        model,
        "-C",
        str(run_dir),
        "-s",
        "read-only",
        "-c",
        f"model_reasoning_effort={effort}",
        "-c",
        "service_tier=default",
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--strict-config",
        "--skip-git-repo-check",
        "--output-schema",
        str(output / "result.schema.json"),
        "--json",
        "-o",
        str(result_path),
    ]
    # Use the equals form because Codex's ``--image <FILE>...`` option is
    # variadic; a final ``-i path`` can otherwise consume the prompt itself.
    for path in opaque_paths:
        command.append(f"--image={path}")
    command.append(_prompt(frame_ids))
    started_epoch = time.time()
    started = time.monotonic()
    with events_path.open("w") as events, stderr_path.open("w") as stderr:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=events,
            stderr=stderr,
            check=False,
        )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        _atomic_json(
            receipt_path,
            {
                "schemaVersion": 1,
                "configId": config_id,
                "model": model,
                "effort": effort,
                "startedEpochSeconds": started_epoch,
                "elapsedSeconds": elapsed,
                "exitCode": completed.returncode,
                "valid": False,
                "executionProvenance": execution_provenance,
            },
        )
        raise RuntimeError(f"{config_id} exited {completed.returncode}; see {stderr_path}")
    by_id = _validate_result(json.loads(result_path.read_text()), frame_ids)
    _atomic_json(
        receipt_path,
        {
            "schemaVersion": 1,
            "configId": config_id,
            "model": model,
            "effort": effort,
            "serviceTier": "default",
            "startedEpochSeconds": started_epoch,
            "elapsedSeconds": elapsed,
            "exitCode": completed.returncode,
            "valid": len(by_id) == len(frame_ids),
            "usage": _usage(events_path),
            "resultSha256": _sha256(result_path),
            "executionProvenance": execution_provenance,
        },
    )


def load_reference(output: Path, frame_ids: list[str]) -> dict[str, Any]:
    sealed_path = output / "sealed-reference.json"
    if sealed_path.is_file():
        document = json.loads(sealed_path.read_text())
        if (
            document.get("schemaVersion") != 1
            or document.get("source")
            != "prior blind Sol xhigh pseudo-reference"
            or set(document.get("frames", {})) != set(frame_ids)
        ):
            raise RuntimeError(f"invalid sealed pseudo-reference: {sealed_path}")
        return document["frames"]

    rows: dict[str, Any] = {}
    source_hashes: dict[str, str] = {}
    for recording, _window, frame_id in SAMPLE:
        path = PILOT_ROOT / "sol-labels" / f"{recording}.ball-presence.json"
        document = json.loads(path.read_text())
        annotation = document["annotations"]["frames"][frame_id]
        if annotation["status"] != "reviewed":
            raise RuntimeError(f"benchmark reference is not reviewed: {recording}/{frame_id}")
        rows[frame_id] = annotation
        source_hashes[recording] = _sha256(path)
    if set(rows) != set(frame_ids):
        raise RuntimeError("reference coverage differs from benchmark pack")
    _atomic_json(
        sealed_path,
        {
            "schemaVersion": 1,
            "source": "prior blind Sol xhigh pseudo-reference",
            "sourceTaskSha256": source_hashes,
            "frames": rows,
        },
    )
    return rows


def _f1(true: list[bool], predicted: list[bool]) -> float:
    tp = sum(left and right for left, right in zip(true, predicted))
    fp = sum(not left and right for left, right in zip(true, predicted))
    fn = sum(left and not right for left, right in zip(true, predicted))
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 1.0


def _macro_state_f1(reference: list[str], predicted: list[str]) -> float:
    return statistics.mean(
        _f1([row == state for row in reference], [row == state for row in predicted])
        for state in STATES
    )


def _active_macro_state_f1(reference: list[str], predicted: list[str]) -> float:
    active_states = [
        state for state in STATES if state in reference or state in predicted
    ]
    return statistics.mean(
        _f1([row == state for row in reference], [row == state for row in predicted])
        for state in active_states
    )


def _iou(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    left = max(first["x"], second["x"])
    top = max(first["y"], second["y"])
    right = min(first["x"] + first["width"], second["x"] + second["width"])
    bottom = min(first["y"] + first["height"], second["y"] + second["height"])
    overlap = max(0.0, right - left) * max(0.0, bottom - top)
    union = first["width"] * first["height"] + second["width"] * second["height"] - overlap
    return min(1.0, max(0.0, overlap / union)) if union else 0.0


def _center_error(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    first_x = (first["x"] + first["width"] / 2) * FRAME_WIDTH
    first_y = (first["y"] + first["height"] / 2) * FRAME_HEIGHT
    second_x = (second["x"] + second["width"] / 2) * FRAME_WIDTH
    second_y = (second["y"] + second["height"] / 2) * FRAME_HEIGHT
    return math.hypot(first_x - second_x, first_y - second_y)


def _score_pair(
    reference: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    exclude_reference_indeterminate: bool,
) -> dict[str, Any]:
    frame_ids = list(reference)
    reference_states = [reference[row]["primaryBallState"] for row in frame_ids]
    predicted_states = [prediction[row]["primaryBallState"] for row in frame_ids]
    eligible = (
        [
            row
            for row in frame_ids
            if reference[row]["primaryBallState"] != "indeterminate"
        ]
        if exclude_reference_indeterminate
        else frame_ids
    )
    reference_presence = [reference[row]["primaryBallState"] == "localizable" for row in eligible]
    predicted_presence = [prediction[row]["primaryBallState"] == "localizable" for row in eligible]
    both_localizable = [
        row
        for row in eligible
        if reference[row]["primaryBallState"] == "localizable"
        and prediction[row]["primaryBallState"] == "localizable"
    ]
    ious = [
        _iou(_primary(reference[row])["bbox"], _primary(prediction[row])["bbox"])
        for row in both_localizable
    ]
    center_errors = [
        _center_error(_primary(reference[row])["bbox"], _primary(prediction[row])["bbox"])
        for row in both_localizable
    ]
    box_f1: dict[str, float] = {}
    for threshold_name, threshold in (("boxF1Iou25", 0.25), ("boxF1Iou50", 0.50)):
        tp = sum(value >= threshold for value in ious)
        ref_count = sum(reference_presence)
        pred_count = sum(predicted_presence)
        denominator = 2 * tp + (pred_count - tp) + (ref_count - tp)
        box_f1[threshold_name] = (2 * tp / denominator) if denominator else 1.0
    visibility_rows = [
        row
        for row, value in zip(both_localizable, ious)
        if value >= 0.25
    ]
    visibility_accuracy = (
        sum(
            _primary(reference[row])["visibility"] == _primary(prediction[row])["visibility"]
            for row in visibility_rows
        )
        / len(visibility_rows)
        if visibility_rows
        else None
    )
    return {
        "frameCount": len(frame_ids),
        "stateAccuracy": sum(
            a == b for a, b in zip(reference_states, predicted_states)
        )
        / len(frame_ids),
        "stateMacroF1": (
            _macro_state_f1(reference_states, predicted_states)
            if exclude_reference_indeterminate
            else _active_macro_state_f1(reference_states, predicted_states)
        ),
        "primaryPresenceF1": _f1(reference_presence, predicted_presence),
        "boxF1Iou25": box_f1["boxF1Iou25"],
        "boxF1Iou50": box_f1["boxF1Iou50"],
        "matchedPrimaryCount": len(ious),
        "matchedPrimaryMeanIou": statistics.mean(ious) if ious else None,
        "matchedPrimaryMedianCenterErrorPixels": (
            statistics.median(center_errors) if center_errors else None
        ),
        "matchedPrimaryVisibilityAccuracyIou25": visibility_accuracy,
        "objectCountAccuracy": sum(
            len(reference[row]["objects"]) == len(prediction[row]["objects"])
            for row in frame_ids
        )
        / len(frame_ids),
    }


def score_against_reference(
    reference: Mapping[str, Any], prediction: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare with a fixed pseudo-reference under its uncertainty policy."""

    return _score_pair(
        reference,
        prediction,
        exclude_reference_indeterminate=True,
    )


def score_symmetric_similarity(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    """Measure run-to-run similarity symmetrically over all sampled frames."""

    return _score_pair(
        first,
        second,
        exclude_reference_indeterminate=False,
    )


def score(output: Path, frame_ids: list[str]) -> None:
    reference = load_reference(output, frame_ids)
    results: dict[str, dict[str, Any]] = {}
    annotations: dict[str, dict[str, Any]] = {}
    for config_id, model, effort in CONFIGS:
        run_dir = output / "runs" / config_id
        result_path = run_dir / "result.json"
        receipt_path = run_dir / "receipt.json"
        annotation = _validate_result(json.loads(result_path.read_text()), frame_ids)
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("valid") is not True or receipt.get("exitCode") != 0:
            raise RuntimeError(f"benchmark run is not valid: {receipt_path}")
        _validate_receipt_identity(
            receipt,
            receipt_path,
            config_id,
            model,
            effort,
        )
        if receipt.get("resultSha256") != _sha256(result_path):
            raise RuntimeError(f"benchmark result hash mismatch: {result_path}")
        annotations[config_id] = annotation
        results[config_id] = {
            "model": model,
            "effort": effort,
            "elapsedSeconds": receipt["elapsedSeconds"],
            "usage": receipt.get("usage"),
            "similarityToPriorSolXhigh": score_against_reference(
                reference, annotation
            ),
        }
    pairwise: dict[str, Any] = {}
    for first_id, _, _ in CONFIGS:
        pairwise[first_id] = {}
        for second_id, _, _ in CONFIGS:
            pairwise[first_id][second_id] = score_symmetric_similarity(
                annotations[first_id], annotations[second_id]
            )
    for first_id, _, _ in CONFIGS:
        for second_id, _, _ in CONFIGS:
            if pairwise[first_id][second_id] != pairwise[second_id][first_id]:
                raise RuntimeError(
                    f"pairwise similarity is not symmetric: {first_id}/{second_id}"
                )
    baseline = results["sol-xhigh"]["elapsedSeconds"]
    for row in results.values():
        row["speedupVsFreshSolXhigh"] = baseline / row["elapsedSeconds"]
    report = {
        "schemaVersion": 1,
        "status": "complete",
        "sample": {
            "frameCount": len(frame_ids),
            "recordingCount": len({row[0] for row in SAMPLE}),
            "environments": ["beach", "grass", "indoor"],
            "reference": "prior blind Sol-xhigh pseudo-reference, not human truth",
            "sealedReferenceSha256": _sha256(output / "sealed-reference.json"),
        },
        "configurations": results,
        "pairwise": pairwise,
        "metricSemantics": {
            "similarityToPriorSolXhigh": (
                "Directional comparison to a fixed pseudo-reference; "
                "reference-indeterminate frames are excluded from presence "
                "and box metrics."
            ),
            "pairwise": (
                "Symmetric comparison over all frames; stateMacroF1 averages "
                "only ontology states present in either run."
            ),
        },
        "artifactIntegrity": {
            "blindPackSha256": _sha256(output / "blind-pack.json"),
            "schemaSha256": _sha256(output / "result.schema.json"),
            "sealedReferenceSha256": _sha256(
                output / "sealed-reference.json"
            ),
            "runs": {
                config_id: {
                    "resultSha256": _sha256(
                        output / "runs" / config_id / "result.json"
                    ),
                    "receiptSha256": _sha256(
                        output / "runs" / config_id / "receipt.json"
                    ),
                    "eventsSha256": _sha256(
                        output / "runs" / config_id / "events.jsonl"
                    ),
                    "stderrSha256": _sha256(
                        output / "runs" / config_id / "stderr.log"
                    ),
                }
                for config_id, _, _ in CONFIGS
            },
        },
        "limitations": [
            "One timed run per configuration; scheduler and stochastic variance are unmeasured.",
            "Frames are independent targets without their original adjacent-frame context.",
            "The reference is prior Sol-xhigh output, not human ground truth.",
            "Cross-family comparisons conflate model family and reasoning effort.",
            "Concurrent full-corpus Sol review was active during these sequential runs.",
            (
                "These completed legacy receipts predate executionProvenance; "
                "their identities and current artifact hashes were audited and "
                "pinned, but the original command cannot be reconstructed from "
                "the receipt alone."
            ),
        ],
    }
    _atomic_json(output / "report.json", report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument(
        "--score-existing",
        action="store_true",
        help="Re-score an already completed run set without executing models.",
    )
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame_ids, opaque_paths = prepare_pack(output)
    if arguments.score_existing:
        if arguments.only:
            parser.error("--score-existing cannot be combined with --only")
        missing = [
            config_id
            for config_id, _, _ in CONFIGS
            if not (output / "runs" / config_id / "result.json").is_file()
        ]
        if missing:
            raise RuntimeError(f"missing completed benchmark results: {missing}")
        score(output, frame_ids)
        print(f"REPORT {output / 'report.json'}", flush=True)
        return 0
    selected = set(arguments.only)
    for config_id, model, effort in CONFIGS:
        if selected and config_id not in selected:
            continue
        print(f"START {config_id} {model} {effort}", flush=True)
        run_config(output, frame_ids, opaque_paths, config_id, model, effort)
        receipt = json.loads((output / "runs" / config_id / "receipt.json").read_text())
        print(f"DONE {config_id} {receipt['elapsedSeconds']:.3f}s", flush=True)
    if not selected and all(
        (output / "runs" / config_id / "result.json").is_file()
        for config_id, _, _ in CONFIGS
    ):
        score(output, frame_ids)
        print(f"REPORT {output / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

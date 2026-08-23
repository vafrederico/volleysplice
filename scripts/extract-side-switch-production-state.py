#!/usr/bin/env python3
"""Build production-state and serve-grounding inputs for side-switch research.

``prepare`` evaluates the two frozen production bundles over the same F104
feature matrix already used on-device.  It writes a label-independent state
artifact and a manifest whose appearance sampling windows are anchored near the
production serve detections.  ``augment`` joins those state inputs onto either a
V5 or V6 appearance artifact.  Suppression scores are retained only under a
quarantined diagnostic key and can never enter the declared model signature.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.dead_ball_experiment import _prediction_inputs
from analysis.dead_state import DeadStateDecoderConfig
from analysis.dead_state_experiment import (
    DeadStateRefinementConfig,
    _predictions_for,
)
from analysis.features import FeatureSequence, VideoMetadata, contextualize
from analysis.model import LogisticModel, load_model
from analysis.pipeline import PreparedRecording
from analysis.schema import Interval, Recording
from analysis.serve import (
    ServeCompositionConfig,
    ServeDecoderConfig,
    decode_serve_probabilities,
)
from analysis.side_switch_production_state import (
    GROUNDING_WINDOW_AFTER_SECONDS,
    GROUNDING_WINDOW_BEFORE_SECONDS,
    PRODUCTION_MODEL_SOURCES,
    PRODUCTION_STATE_FEATURE_NAMES,
    SERVE_ANCHOR_FEATURE_NAMES,
    STATE_GATE_FEATURE_NAMES,
    ProductionTrace,
    ScoredTime,
    TimeRange,
    gap_state_features,
    merge_production_components,
    rally_evidence,
    rally_evidence_payload,
)
from analysis.side_switch_v3 import FROZEN_RECORDING_SPLIT, RECORDING_ROLE
from analysis.side_switch_v5 import VISUAL_FEATURE_NAMES as V5_FEATURE_NAMES
from analysis.side_switch_v6 import VISUAL_FEATURE_NAMES as V6_FEATURE_NAMES


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
INTAKE = Path("/mnt/freenas/volleycut/intake-2026-08-13")
DEFAULT_MANIFEST = ROOT / "reports/full-nas-video-corpus-v1.json"
DEFAULT_V5_FEATURES = (
    ROOT / "reports/side-switch/side-switch-v5-player-orientation-features.json"
)
DEFAULT_V6_FEATURES = (
    ROOT / "reports/side-switch/side-switch-v6-detected-adaptive-features.json"
)
DEFAULT_STATE_OUTPUT = (
    ROOT / "reports/side-switch/side-switch-production-state-v1.json"
)
DEFAULT_GROUNDED_MANIFEST_OUTPUT = (
    ROOT / "reports/side-switch/side-switch-production-grounded-corpus-v1.json"
)
FEATURE_CACHE = (
    INTAKE
    / "experiments/environment-specialists-v2/features/"
    "audiovisual-audio-normalized-v3"
)
FEEDBACK_ROOT = Path("/mnt/freenas/volleycut/model-feedback")
V2_BUNDLE = (
    INTAKE / "experiments/environment-specialists-v2/models/all-labels/bundle.json"
)
OLD_MODEL_ROOT = (
    Path("/mnt/freenas/volleycut")
    / "labeling-v1-2026-08-09-no-beach-2026-08-12/models"
)
OLD_HEAD_NAMES = {
    "rally": "full-audiovisual-audio-normalized-v3",
    "serve": "serve-specialist-audio-normalized-v5",
    "dead": "dead-state-transition-audio-normalized-v5-no-legacy-final",
}
SUPPRESSION_MODEL = (
    INTAKE
    / "experiments/feedback-suppression-v3-2026-08-16/models/"
    "suppression-overlap-exclusion-retrained"
)

EXPECTED_SHA256 = {
    "manifest": "c177f2936dc1ea7f2953cf94a6b9720cd6d4bf4e652a4c7409499374c9fe1e24",
    "v5Features": "4406fe47de0326c1257b8d9cf353116a9495f82d2e43c92e9721b1d6764ba5db",
    "v6Features": "364d9f050b4cc581059e6ad944737a9bfa2c43bdf845fae6bd57394247abeb4a",
}


@dataclass(frozen=True)
class Heads:
    rally: LogisticModel
    serve: LogisticModel
    dead: LogisticModel


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _numeric_array(
    payload: Mapping[str, Any], dtype: str, shape: tuple[int, ...]
) -> np.ndarray:
    if payload.get("encoding") != "base64" or payload.get("byteOrder") != "little-endian":
        raise ValueError("feedback numeric payload has unsupported encoding")
    if payload.get("dataType") != dtype or tuple(payload.get("shape", ())) != shape:
        raise ValueError(f"feedback numeric payload does not match {dtype} {shape}")
    raw = base64.b64decode(str(payload["data"]), validate=True)
    numpy_dtype = np.dtype("<f4" if dtype == "float32" else "<f8")
    result = np.frombuffer(raw, dtype=numpy_dtype).copy()
    if result.size != math.prod(shape) or not np.isfinite(result).all():
        raise ValueError("feedback numeric payload is malformed")
    return result.reshape(shape)


def _recording(record: Mapping[str, Any], duration: float) -> Recording:
    roi = record.get("roi")
    roi_tuple = (
        (
            float(roi["x"]),
            float(roi["y"]),
            float(roi["width"]),
            float(roi["height"]),
        )
        if isinstance(roi, Mapping)
        else None
    )
    return Recording(
        id=str(record["recordingId"]),
        video=Path(str(record["videoPath"])),
        split="challenge",
        source_group=str(record.get("sourceGroup", "unknown")),
        environment=str(record.get("environment", "unknown")),
        game={},
        rallies=(),
        ignored_intervals=(),
        roi=roi_tuple,
        capture={},
        consent={"analyze": True, "train": False},
        content_sha256=None,
        raw={"duration": duration},
    )


def _prepared_from_sequence(
    record: Mapping[str, Any], sequence: FeatureSequence, template: LogisticModel
) -> PreparedRecording:
    contextual_values, contextual_names = contextualize(
        sequence, template.feature_config
    )
    return PreparedRecording(
        recording=_recording(record, sequence.metadata.duration),
        sequence=sequence,
        contextual_values=contextual_values,
        contextual_names=contextual_names,
        labels=np.zeros(len(sequence.times), dtype=np.float32),
        sample_mask=np.ones(len(sequence.times), dtype=bool),
    )


def _historical_prepared(
    record: Mapping[str, Any], template: LogisticModel
) -> PreparedRecording:
    recording_id = str(record["recordingId"])
    matches = sorted(FEATURE_CACHE.glob(f"{recording_id}-*.npz"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one cached F104 sequence for {recording_id}, found {matches}"
        )
    with np.load(matches[0], allow_pickle=False) as artifact:
        times = artifact["times"].astype(np.float64, copy=True)
        values = artifact["values"].astype(np.float32, copy=True)
        names = tuple(str(value) for value in artifact["names"].tolist())
        metadata = json.loads(str(artifact["metadata_json"].item()))
    sequence = FeatureSequence(
        times=times,
        values=values,
        names=names,
        metadata=VideoMetadata(
            duration=float(metadata["duration"]),
            width=int(metadata["width"]),
            height=int(metadata["height"]),
            fps=float(metadata["fps"]),
            frame_count=int(metadata["frame_count"]),
            has_audio=bool(metadata.get("has_audio")),
        ),
    )
    return _prepared_from_sequence(record, sequence, template)


def _feedback_index() -> dict[str, tuple[Path, Mapping[str, Any]]]:
    result: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for path in sorted(FEEDBACK_ROOT.glob("*/bundle.json")):
        payload = _load(path)
        filename = str(payload.get("source", {}).get("file", {}).get("name", ""))
        if not filename:
            continue
        recording_id = f"raw-no-backup-{Path(filename).stem}"
        result[recording_id] = (path, payload)
    return result


def _feedback_prepared(
    record: Mapping[str, Any], payload: Mapping[str, Any], template: LogisticModel
) -> PreparedRecording:
    features = payload["features"]
    rows, columns = int(features["rows"]), int(features["columns"])
    times = _numeric_array(features["timestamps"], "float64", (rows,)).astype(
        np.float64
    )
    values = _numeric_array(
        features["values"], "float32", (rows, columns)
    ).astype(np.float32)
    names = tuple(str(item) for item in features["names"])
    media = payload["source"]["media"]
    # Native analysis preserves source-time timestamps but decodes and clips within
    # the selected game window.  Using the full media duration would extend a final
    # active interval by up to half a sample beyond the frozen browser result.
    duration = float(payload["source"]["gameWindow"]["end"])
    sequence = FeatureSequence(
        times=times,
        values=values,
        names=names,
        metadata=VideoMetadata(
            duration=duration,
            width=int(media["width"]),
            height=int(media["height"]),
            fps=float(features["analysisFps"]),
            frame_count=max(1, round(duration * float(features["analysisFps"]))),
            has_audio=bool(media.get("hasAudio")),
        ),
    )
    return _prepared_from_sequence(record, sequence, template)


def _load_heads() -> tuple[Heads, Heads]:
    old = Heads(
        *(load_model(OLD_MODEL_ROOT / OLD_HEAD_NAMES[key]) for key in ("rally", "serve", "dead"))
    )
    bundle = _load(V2_BUNDLE)["heads"]
    v2 = Heads(
        load_model(bundle["rally"]["path"]),
        load_model(bundle["serve"]["path"]),
        load_model(bundle["deadState"]["path"]),
    )
    signatures = {
        head.feature_names
        for group in (old, v2)
        for head in (group.rally, group.serve, group.dead)
    }
    configs = {
        json.dumps(head.feature_config.to_dict(), sort_keys=True)
        for group in (old, v2)
        for head in (group.rally, group.serve, group.dead)
    }
    if len(signatures) != 1 or len(configs) != 1:
        raise ValueError("production bundle feature contracts diverged")
    return old, v2


def _predict_ranges(item: PreparedRecording, heads: Heads) -> tuple[TimeRange, ...]:
    serve_decoder = ServeDecoderConfig.from_dict(
        heads.serve.training_summary["serveDecoder"]
    )
    composition = ServeCompositionConfig.from_dict(
        heads.serve.training_summary["composition"]
    )
    dead_decoder = DeadStateDecoderConfig.from_dict(
        heads.dead.training_summary["selectedDeadStateDecoder"]
    )
    refinement = DeadStateRefinementConfig.from_dict(
        heads.dead.training_summary["selectedRefinement"]
    )
    inputs = _prediction_inputs(
        item,
        heads.rally,
        heads.serve,
        heads.dead,
        serve_decoder,
        composition,
    )
    prediction = _predictions_for([inputs], dead_decoder, refinement)[0]
    return tuple(TimeRange(float(row.start), float(row.end)) for row in prediction.candidate)


def _head_scores(
    item: PreparedRecording, heads: Heads
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[ScoredTime, ...]]:
    rally = heads.rally.predict(item.contextual_values)
    serve = heads.serve.predict(item.contextual_values)
    dead = heads.dead.predict(item.contextual_values)
    serve_decoder = ServeDecoderConfig.from_dict(
        heads.serve.training_summary["serveDecoder"]
    )
    detections = decode_serve_probabilities(
        item.sequence.times,
        serve,
        serve_decoder,
        duration=item.sequence.metadata.duration,
    )
    return (
        rally,
        serve,
        dead,
        tuple(ScoredTime(row.time, row.confidence) for row in detections),
    )


def _raw_parity(
    payload: Mapping[str, Any], trace: ProductionTrace
) -> dict[str, Any]:
    initial = payload["initialInference"]
    rows = len(trace.times)
    differences: dict[str, float] = {}
    for output_name, payload_name in (
        ("rally", "rally"),
        ("serve", "serve"),
        ("deadState", "deadState"),
    ):
        expected = _numeric_array(
            initial["probabilities"][payload_name], "float32", (rows,)
        )
        source_values = {
            "rally": trace.rally_scores,
            "serve": trace.serve_scores,
            "deadState": trace.dead_state_scores,
        }[output_name]["all-labels-v2"]
        differences[output_name] = float(
            np.max(np.abs(expected - np.asarray(source_values)))
        )
    if max(differences.values()) > 5e-6:
        raise ValueError(f"raw primary probability replay drifted: {differences}")

    components = merge_production_components(trace.ranges)
    expected_ranges = initial["ranges"]
    if len(components) != len(expected_ranges):
        raise ValueError("raw production range replay count drifted")
    maximum_boundary_error = 0.0
    for component, expected in zip(components, expected_ranges, strict=True):
        maximum_boundary_error = max(
            maximum_boundary_error,
            abs(component.start - float(expected["start"])),
            abs(component.end - float(expected["end"])),
        )
        expected_support = 2 if expected.get("agreement") == "both-models" else 1
        if component.support_count != expected_support:
            raise ValueError("raw production agreement replay drifted")
    if maximum_boundary_error > 1e-6:
        raise ValueError(
            f"raw production range replay boundary drifted by {maximum_boundary_error}"
        )
    return {
        "maximumPrimaryProbabilityAbsoluteError": max(differences.values()),
        "perHeadPrimaryProbabilityAbsoluteError": differences,
        "rangeCount": len(components),
        "maximumRangeBoundaryAbsoluteErrorSeconds": maximum_boundary_error,
    }


def _rounded(values: Mapping[str, float]) -> dict[str, float]:
    return {name: round(float(value), 8) for name, value in values.items()}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "manifest": args.manifest.expanduser().resolve(),
        "v5Features": args.v5_features.expanduser().resolve(),
        "v6Features": args.v6_features.expanduser().resolve(),
    }
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if args.enforce_source_hash and hashes != EXPECTED_SHA256:
        raise ValueError(f"production-state source identity changed: {hashes}")
    state_output = args.state_output.expanduser().resolve()
    grounded_output = args.grounded_manifest_output.expanduser().resolve()
    if state_output.exists() or grounded_output.exists():
        raise FileExistsError("refusing to overwrite production-state artifacts")

    manifest = _load(paths["manifest"])
    v5_features = _load(paths["v5Features"])
    v6_features = _load(paths["v6Features"])
    frozen_split = {
        role: list(recording_ids)
        for role, recording_ids in FROZEN_RECORDING_SPLIT.items()
    }
    if (
        v5_features.get("frozenRecordingSplit") != frozen_split
        or v6_features.get("frozenRecordingSplit") != frozen_split
    ):
        raise ValueError("V5/V6 features do not share the frozen split")
    v5_ids = [str(row["eventId"]) for row in v5_features["rows"]]
    v6_ids = [str(row["eventId"]) for row in v6_features["rows"]]
    if v5_ids != v6_ids:
        raise ValueError("V5/V6 row identity or order diverged")

    records = {
        str(record["recordingId"]): record
        for record in manifest["records"]
        if str(record.get("recordingId", "")) in RECORDING_ROLE
    }
    if set(records) != set(RECORDING_ROLE):
        raise ValueError("full-corpus manifest does not cover the frozen split")
    rows_by_recording: dict[str, list[Mapping[str, Any]]] = {
        recording_id: [] for recording_id in RECORDING_ROLE
    }
    for row in v6_features["rows"]:
        rows_by_recording[str(row["recordingId"])].append(row)
    for rows in rows_by_recording.values():
        rows.sort(key=lambda row: int(row["gapOrder"]))

    old_heads, v2_heads = _load_heads()
    suppression = load_model(SUPPRESSION_MODEL)
    if suppression.feature_names != v2_heads.rally.feature_names:
        raise ValueError("suppression and production feature signatures diverged")
    feedback = _feedback_index()
    suppression_training_ids = set(
        str(value)
        for value in suppression.training_summary.get("trainingRecordingIds", [])
    )

    grounded_manifest = copy.deepcopy(dict(manifest))
    grounded_manifest["schemaVersion"] = 1
    grounded_manifest["kind"] = "volleycut-side-switch-production-grounded-corpus-v1"
    grounded_manifest["createdAt"] = datetime.now(UTC).isoformat()
    grounded_manifest["groundingProfile"] = {
        "name": "PRODUCTION-SERVE-GROUNDING1",
        "productionModels": {
            "allLabelsV2": "model-1ca43e38eefc",
            "previousProduction": "model-9c92b8e9333f",
            "ensemble": "overlap-union-disagreement-v1",
        },
        "windowBeforeServeSeconds": GROUNDING_WINDOW_BEFORE_SECONDS,
        "windowAfterServeSeconds": GROUNDING_WINDOW_AFTER_SECONDS,
        "fallbackOrder": [
            "two-head serve consensus",
            "highest-confidence production serve",
            "matched production component start",
            "source rally start (training-only missing-component fallback)",
        ],
        "labelIndependent": True,
    }
    grounded_manifest["sources"] = {
        "manifest": {"path": str(paths["manifest"]), "sha256": hashes["manifest"]}
    }
    grounded_records = {
        str(record["recordingId"]): record
        for record in grounded_manifest["records"]
        if str(record.get("recordingId", "")) in RECORDING_ROLE
    }

    state_rows: list[dict[str, Any]] = []
    extraction_audit: dict[str, Any] = {}
    suppression_overlap_ids: list[str] = []
    for role in ("train", "validation", "evaluation"):
        for recording_id in FROZEN_RECORDING_SPLIT[role]:
            record = records[recording_id]
            feedback_payload: Mapping[str, Any] | None = None
            feedback_project_id: str | None = None
            if recording_id in feedback:
                _, feedback_payload = feedback[recording_id]
                prepared = _feedback_prepared(record, feedback_payload, v2_heads.rally)
                feedback_project_id = str(feedback_payload["source"]["projectId"])
            else:
                prepared = _historical_prepared(record, v2_heads.rally)
            if prepared.contextual_names != v2_heads.rally.feature_names:
                raise ValueError(f"F104 signature drifted for {recording_id}")

            score_groups: dict[
                str,
                tuple[
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    tuple[ScoredTime, ...],
                ],
            ] = {}
            range_groups: dict[str, tuple[TimeRange, ...]] = {}
            for source_name, heads in zip(
                PRODUCTION_MODEL_SOURCES, (v2_heads, old_heads), strict=True
            ):
                score_groups[source_name] = _head_scores(prepared, heads)
                range_groups[source_name] = _predict_ranges(prepared, heads)
            suppression_scores = suppression.predict(prepared.contextual_values)
            trace = ProductionTrace(
                times=prepared.sequence.times,
                duration=prepared.sequence.metadata.duration,
                rally_scores={
                    source: values[0] for source, values in score_groups.items()
                },
                serve_scores={
                    source: values[1] for source, values in score_groups.items()
                },
                dead_state_scores={
                    source: values[2] for source, values in score_groups.items()
                },
                ranges=range_groups,
                serves={
                    source: values[3] for source, values in score_groups.items()
                },
                suppression_scores=suppression_scores,
            )
            trace.validate()
            parity = _raw_parity(feedback_payload, trace) if feedback_payload else None
            components = merge_production_components(trace.ranges)

            source_rallies = [
                TimeRange(float(row["start"]), float(row["end"]))
                for row in record["rallies"]
            ]
            evidences = [
                rally_evidence(rally, trace, components) for rally in source_rallies
            ]
            grounded_rallies: list[dict[str, Any]] = []
            for raw_rally, evidence in zip(
                record["rallies"], evidences, strict=True
            ):
                grounded_rally = copy.deepcopy(dict(raw_rally))
                grounded_rally["start"] = evidence.comparison_start
                grounded_rally["end"] = evidence.comparison_end
                grounded_rally["productionGrounding"] = rally_evidence_payload(evidence)
                grounded_rallies.append(grounded_rally)
            grounded_records[recording_id]["rallies"] = grounded_rallies

            training_overlap = (
                recording_id in suppression_training_ids
                or feedback_project_id in suppression_training_ids
            )
            if training_overlap:
                suppression_overlap_ids.append(recording_id)
            for row in rows_by_recording[recording_id]:
                gap_order = int(row["gapOrder"])
                if gap_order < 1 or gap_order >= len(evidences):
                    raise ValueError(f"invalid gap order for {recording_id}: {gap_order}")
                before = evidences[gap_order - 1]
                after = evidences[gap_order]
                state_features, suppression_diagnostic = gap_state_features(
                    before,
                    after,
                    float(row["gapStart"]),
                    float(row["gapEnd"]),
                    trace,
                )
                state_rows.append(
                    {
                        "eventId": str(row["eventId"]),
                        "sourceEventIds": list(row.get("sourceEventIds", [])),
                        "recordingId": recording_id,
                        "role": role,
                        "gapOrder": gap_order,
                        "features": _rounded(state_features),
                        "grounding": {
                            "before": rally_evidence_payload(before),
                            "after": rally_evidence_payload(after),
                        },
                        "suppressionDiagnostic": {
                            **_rounded(suppression_diagnostic),
                            "eligibleForModelInput": False,
                            "trainingRecordingOverlap": training_overlap,
                        },
                    }
                )

            extraction_audit[recording_id] = {
                "role": role,
                "samples": len(trace.times),
                "durationSeconds": trace.duration,
                "sourceRallies": len(source_rallies),
                "allLabelsV2Ranges": len(trace.ranges["all-labels-v2"]),
                "previousProductionRanges": len(
                    trace.ranges["previous-production"]
                ),
                "ensembleComponents": len(components),
                "bothModelSourceRallies": sum(
                    value.support_count == 2 for value in evidences
                ),
                "oneModelSourceRallies": sum(
                    value.support_count == 1 for value in evidences
                ),
                "unmatchedSourceRallies": sum(
                    value.support_count == 0 for value in evidences
                ),
                "serveConsensusRallies": sum(
                    value.anchor_source == "serve-consensus" for value in evidences
                ),
                "sourceStartFallbackRallies": sum(
                    value.anchor_source == "source-start-fallback"
                    for value in evidences
                ),
                "suppressionTrainingRecordingOverlap": training_overlap,
                "rawFeedbackReplay": parity,
            }
            print(
                f"Prepared production state {role} {recording_id}: "
                f"{len(source_rallies)} rallies, {len(components)} components",
                file=sys.stderr,
                flush=True,
            )

    state_by_id = {str(row["eventId"]): row for row in state_rows}
    if len(state_by_id) != len(state_rows) or set(state_by_id) != set(v6_ids):
        raise ValueError("production-state row identity diverged from V5/V6")
    state_rows = [state_by_id[event_id] for event_id in v6_ids]
    created_at = datetime.now(UTC).isoformat()
    model_sources = {
        "allLabelsV2": {
            "modelId": "model-1ca43e38eefc",
            "rallyArtifactSha256": v2_heads.rally.artifact_sha256,
            "serveArtifactSha256": v2_heads.serve.artifact_sha256,
            "deadStateArtifactSha256": v2_heads.dead.artifact_sha256,
        },
        "previousProduction": {
            "modelId": "model-9c92b8e9333f",
            "rallyArtifactSha256": old_heads.rally.artifact_sha256,
            "serveArtifactSha256": old_heads.serve.artifact_sha256,
            "deadStateArtifactSha256": old_heads.dead.artifact_sha256,
        },
        "ensembleAlgorithmVersion": "overlap-union-disagreement-v1",
    }
    state_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-production-state-features-v1",
        "createdAt": created_at,
        "profile": {
            "name": "PRODUCTION-STATE20",
            "featureNames": list(PRODUCTION_STATE_FEATURE_NAMES),
            "stateGateFeatureNames": list(STATE_GATE_FEATURE_NAMES),
            "serveAnchorFeatureNames": list(SERVE_ANCHOR_FEATURE_NAMES),
            "analysisFps": 4.0,
            "labelIndependentExtraction": True,
            "onDeviceReuse": (
                "reuses the two production bundles' rally, serve, dead-state, and "
                "decoded-range outputs over the existing F104 matrix"
            ),
        },
        "frozenRecordingSplit": frozen_split,
        "rows": state_rows,
        "extractionAudit": extraction_audit,
        "suppressionQuarantine": {
            "modelId": "suppression-overlap-exclusion-retrained",
            "artifactSha256": suppression.artifact_sha256,
            "eligibleForModelInput": False,
            "reason": (
                "the suppression positive target explicitly includes side switches "
                "and its fitting set overlaps the frozen side-switch train, validation, "
                "and retrospective recordings"
            ),
            "overlappingRecordingIds": sorted(set(suppression_overlap_ids)),
            "overlapCounts": {
                role: sum(
                    recording_id in suppression_overlap_ids
                    for recording_id in FROZEN_RECORDING_SPLIT[role]
                )
                for role in ("train", "validation", "evaluation")
            },
            "diagnosticFields": [
                "suppressionGapMeanScore",
                "suppressionGapPeakScore",
            ],
        },
        "productionModels": model_sources,
        "sources": {
            name: {"path": str(paths[name]), "sha256": hashes[name]}
            for name in paths
        },
    }
    grounded_manifest["createdAt"] = created_at
    grounded_manifest["productionModels"] = model_sources
    _write(state_output, state_payload)
    _write(grounded_output, grounded_manifest)
    return {
        "stateOutput": str(state_output),
        "stateSha256": _sha256(state_output),
        "groundedManifestOutput": str(grounded_output),
        "groundedManifestSha256": _sha256(grounded_output),
        "rows": len(state_rows),
        "suppressionOverlapCounts": state_payload["suppressionQuarantine"][
            "overlapCounts"
        ],
    }


def augment(args: argparse.Namespace) -> dict[str, Any]:
    base_path = args.base_features.expanduser().resolve()
    state_path = args.state.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite augmented features: {output}")
    base = _load(base_path)
    state = _load(state_path)
    if state.get("kind") != "volleycut-side-switch-production-state-features-v1":
        raise ValueError("state input has the wrong kind")
    if state.get("suppressionQuarantine", {}).get("eligibleForModelInput") is not False:
        raise ValueError("suppression quarantine is not explicit")
    if base.get("frozenRecordingSplit") != state.get("frozenRecordingSplit"):
        raise ValueError("base and production-state splits diverged")
    base_names = V5_FEATURE_NAMES if args.family == "v5" else V6_FEATURE_NAMES
    state_rows = {str(row["eventId"]): row for row in state["rows"]}
    rows: list[dict[str, Any]] = []
    for raw_row in base["rows"]:
        row = copy.deepcopy(dict(raw_row))
        event_id = str(row["eventId"])
        state_row = state_rows.get(event_id)
        if state_row is None:
            raise ValueError(f"missing production-state row {event_id}")
        if list(row.get("sourceEventIds", [])) != list(
            state_row.get("sourceEventIds", [])
        ):
            raise ValueError(f"source event identity drifted for {event_id}")
        raw_features = row.get("features")
        features = dict(raw_features) if isinstance(raw_features, Mapping) else {}
        if tuple(features) != base_names:
            raise ValueError(f"base feature signature drifted for {event_id}")
        production_features = state_row["features"]
        if tuple(production_features) != PRODUCTION_STATE_FEATURE_NAMES:
            raise ValueError("production-state feature signature drifted")
        features.update(production_features)
        row["features"] = features
        row["productionStateContext"] = {
            "grounding": state_row["grounding"],
            "suppressionDiagnostic": state_row["suppressionDiagnostic"],
        }
        rows.append(row)
    if [str(row["eventId"]) for row in rows] != list(state_rows):
        raise ValueError("augmented row order diverged")

    created_at = datetime.now(UTC).isoformat()
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-production-state-augmented-features-v1",
        "createdAt": created_at,
        "family": args.family,
        "appearanceMode": args.appearance_mode,
        "profile": {
            "name": f"{args.family.upper()}-{args.appearance_mode.upper()}-PRODUCTION-STATE",
            "baseFeatureNames": list(base_names),
            "stateGateFeatureNames": list(STATE_GATE_FEATURE_NAMES),
            "serveAnchorFeatureNames": list(SERVE_ANCHOR_FEATURE_NAMES),
            "featureNames": [*base_names, *PRODUCTION_STATE_FEATURE_NAMES],
            "suppressionDiagnosticEligibleForModelInput": False,
            "appearanceSampling": (
                "frozen V5/V6 whole-rally sampling"
                if args.appearance_mode == "original"
                else (
                    f"{GROUNDING_WINDOW_BEFORE_SECONDS:g}s before through "
                    f"{GROUNDING_WINDOW_AFTER_SECONDS:g}s after production serve anchor"
                )
            ),
        },
        "frozenRecordingSplit": state["frozenRecordingSplit"],
        "collapseAudit": base.get("collapseAudit"),
        "extractionAudit": base.get("extractionAudit"),
        "personDetector": base.get("personDetector"),
        "rows": rows,
        "productionModels": state["productionModels"],
        "suppressionQuarantine": state["suppressionQuarantine"],
        "sources": {
            "baseFeatures": {
                "path": str(base_path),
                "sha256": _sha256(base_path),
                "kind": base.get("kind"),
                "createdAt": base.get("createdAt"),
            },
            "productionState": {
                "path": str(state_path),
                "sha256": _sha256(state_path),
                "createdAt": state.get("createdAt"),
            },
        },
    }
    _write(output, payload)
    return {
        "output": str(output),
        "sha256": _sha256(output),
        "rows": len(rows),
        "featureCount": len(payload["profile"]["featureNames"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    prepare_parser.add_argument("--v5-features", type=Path, default=DEFAULT_V5_FEATURES)
    prepare_parser.add_argument("--v6-features", type=Path, default=DEFAULT_V6_FEATURES)
    prepare_parser.add_argument("--state-output", type=Path, default=DEFAULT_STATE_OUTPUT)
    prepare_parser.add_argument(
        "--grounded-manifest-output",
        type=Path,
        default=DEFAULT_GROUNDED_MANIFEST_OUTPUT,
    )
    prepare_parser.add_argument(
        "--enforce-source-hash",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    augment_parser = commands.add_parser("augment")
    augment_parser.add_argument("--family", choices=("v5", "v6"), required=True)
    augment_parser.add_argument(
        "--appearance-mode", choices=("original", "serve-grounded"), required=True
    )
    augment_parser.add_argument("--base-features", type=Path, required=True)
    augment_parser.add_argument("--state", type=Path, default=DEFAULT_STATE_OUTPUT)
    augment_parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = prepare(args) if args.command == "prepare" else augment(args)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

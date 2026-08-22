#!/usr/bin/env python3
"""Audit frozen side-switch decoders against exhaustive full-video markers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_full_video import (
    event_metric_counts,
    monotonic_interval_match,
)
from analysis.side_switch_v2 import (
    DecoderSettings,
    V2Event,
    V2Model,
    decode_sequence,
    matrix_for,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
MODELS = ROOT / "models"
DEFAULT_MARKERS = REPORTS / "full-video-side-switch-markers-full-nas-v1.json"
DEFAULT_OUTPUT_PREFIX = REPORTS / "side-switch-full-video-marker-evaluation-2026-08-21"
PRIMARY_PADDING_SECONDS = 4.0
PADDING_SENSITIVITY_SECONDS = (0.0, PRIMARY_PADDING_SECONDS)


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _display_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes}:{remainder:06.3f}"


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}%"


def _event_gap(row: Mapping[str, Any]) -> int:
    if row.get("gapOrder") is not None:
        return int(row["gapOrder"])
    if row.get("rallyOrder") is not None:
        return int(row["rallyOrder"])
    return int(str(row["eventId"]).rsplit(":", 1)[-1])


def _geometry(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "eventId": str(row["eventId"]),
        "recordingId": str(row["recordingId"]),
        "gapOrder": _event_gap(row),
        "gapStart": float(row["gapStart"]),
        "gapEnd": float(row["gapEnd"]),
        "transitionTime": float(row["transitionTime"]),
    }


class ProposalRegistry:
    def __init__(self, recording_ids: Sequence[str]) -> None:
        self.recording_ids = tuple(recording_ids)
        self.recording_set = set(recording_ids)
        self.metadata: dict[str, dict[str, Any]] = {}
        self.proposals: dict[str, dict[str, list[dict[str, Any]]]] = {}

    def register(
        self,
        decoder_id: str,
        label: str,
        rows: Sequence[Mapping[str, Any]],
        selected: Callable[[Mapping[str, Any]], bool],
        *,
        contract: str,
        sources: Sequence[Path],
        comparison_scope: str = "common-11-recording-retrospective",
    ) -> None:
        if decoder_id in self.proposals:
            raise ValueError(f"duplicate decoder ID: {decoder_id}")
        by_recording = {recording_id: [] for recording_id in self.recording_ids}
        for row in rows:
            recording_id = str(row["recordingId"])
            if recording_id not in self.recording_set or not selected(row):
                continue
            proposal = _geometry(row)
            by_recording[recording_id].append(proposal)
        for values in by_recording.values():
            values.sort(
                key=lambda item: (
                    float(item["transitionTime"]),
                    int(item["gapOrder"]),
                    str(item["eventId"]),
                )
            )
        self.proposals[decoder_id] = by_recording
        self.metadata[decoder_id] = {
            "label": label,
            "contract": contract,
            "comparisonScope": comparison_scope,
            "sources": [str(path) for path in sources],
        }


def _build_registry(
    root: Path,
) -> tuple[
    ProposalRegistry,
    list[dict[str, Any]],
    list[Path],
    Mapping[str, Any],
]:
    reports = root / "reports/side-switch"
    models = root / "models"
    source_paths: set[Path] = set()

    peak_path = reports / "side-switch-v5-peak-cleanup-v1-evaluation.json"
    peak = _load(peak_path)
    source_paths.add(peak_path)
    recording_ids = [str(value) for value in peak["counts"]["recordingIds"]]
    registry = ProposalRegistry(recording_ids)

    appearance_path = reports / "appearance-diagnostic-full-nas-v1.json"
    appearance = _load(appearance_path)
    source_paths.add(appearance_path)
    appearance_by_event = {
        str(row["eventId"]): row
        for row in appearance["events"]
        if str(row["recordingId"]) in registry.recording_set
    }

    def register_v1(decoder_id: str, label: str, artifact_stem: str) -> None:
        evaluation_path = reports / f"{artifact_stem}-evaluation.json"
        evaluation = _load(evaluation_path)
        source_paths.add(evaluation_path)
        rows = []
        for prediction in evaluation["predictions"]:
            if str(prediction["recordingId"]) not in registry.recording_set:
                continue
            event_id = str(prediction["eventId"])
            source_row = appearance_by_event.get(event_id)
            if source_row is None:
                raise ValueError(f"V1 prediction has no appearance row: {event_id}")
            rows.append(
                {
                    **_geometry(source_row),
                    "selected": prediction["prediction"] == "switch",
                }
            )
        registry.register(
            decoder_id,
            label,
            rows,
            lambda row: bool(row["selected"]),
            contract="Legacy per-gap static classifier; no temporal decoder.",
            sources=[evaluation_path, appearance_path],
        )

    register_v1(
        "v1-static-original",
        "V1 static · original training",
        "side-switch-specialist-v1",
    )
    register_v1(
        "v1-static-no-blurry-beach",
        "V1 static · blurry beach excluded",
        "side-switch-specialist-v1-no-blurry-beach-2026-08-20",
    )

    v2_features_path = reports / "side-switch-v2-features.json"
    v2_model_path = models / "side-switch-specialist-v2/model.json"
    v2_evaluation_path = reports / "side-switch-specialist-v2-evaluation.json"
    v2_features = _load(v2_features_path)
    v2_model_payload = _load(v2_model_path)
    v2_evaluation = _load(v2_evaluation_path)
    source_paths.update({v2_features_path, v2_model_path, v2_evaluation_path})
    v2_model = V2Model.from_dict(v2_model_payload["classifier"])
    v2_decoder = DecoderSettings.from_dict(v2_model_payload["decoder"])
    v2_rows = [
        row
        for row in v2_features["events"]
        if str(row["recordingId"]) in registry.recording_set
    ]
    v2_events = [
        V2Event(
            event_id=str(row["eventId"]),
            recording_id=str(row["recordingId"]),
            role=str(row["role"]),
            rally_order=int(row["rallyOrder"]),
            decision="unopened",
            label=0,
            row=row,
        )
        for row in v2_rows
    ]
    v2_probabilities = v2_model.predict_proba(
        matrix_for(v2_events, v2_model.feature_set)
    )
    v2_selected: dict[str, bool] = {}
    for recording_id in recording_ids:
        indices = [
            index
            for index, event in enumerate(v2_events)
            if event.recording_id == recording_id
        ]
        ordered = sorted(indices, key=lambda index: v2_events[index].rally_order)
        decoded, _, _ = decode_sequence(
            [v2_events[index] for index in ordered],
            v2_probabilities[np.asarray(ordered)],
            v2_model.threshold,
            v2_decoder,
            calculate_scores=False,
        )
        for index, decision in zip(ordered, decoded, strict=True):
            v2_selected[v2_events[index].event_id] = bool(decision)
    for row in v2_evaluation["predictions"]:
        expected = row["decoderPrediction"] == "switch"
        actual = v2_selected[str(row["eventId"])]
        if actual != expected:
            raise ValueError(f"V2 replay mismatch: {row['eventId']}")
    registry.register(
        "v2-temporal-noop",
        "V2 temporal · selected no-op",
        v2_rows,
        lambda row: v2_selected[str(row["eventId"])],
        contract=(
            "Frozen V2 classifier and selected zero-penalty temporal decoder. "
            "All 11 immutable feature sequences are replayed; only four were "
            "frozen evaluation scope."
        ),
        sources=[v2_model_path, v2_features_path, v2_evaluation_path],
        comparison_scope="mixed-development-and-4-recording-frozen-evaluation",
    )

    modern_features_path = reports / "side-switch-v5-player-orientation-features.json"
    modern_features = _load(modern_features_path)
    source_paths.add(modern_features_path)
    modern_rows = [
        _geometry(row)
        | {
            "decision": str(row["decision"]),
            "label": int(row["label"]),
        }
        for row in modern_features["rows"]
        if str(row["recordingId"]) in registry.recording_set
    ]
    modern_by_event = {str(row["eventId"]): row for row in modern_rows}
    if len(modern_rows) != 352 or len(modern_by_event) != 352:
        raise ValueError("modern raw-phone candidate universe is not the frozen 352 rows")

    def register_selected(
        decoder_id: str,
        label: str,
        filename: str,
        contract: str,
    ) -> None:
        evaluation_path = reports / filename
        evaluation = _load(evaluation_path)
        source_paths.add(evaluation_path)
        rows = []
        for prediction in evaluation["predictions"]:
            if str(prediction["recordingId"]) not in registry.recording_set:
                continue
            event_id = str(prediction["eventId"])
            source_row = modern_by_event.get(event_id)
            if source_row is None:
                raise ValueError(f"prediction has no modern feature row: {event_id}")
            rows.append(
                {
                    **source_row,
                    "selected": bool(prediction["selectedPrediction"]),
                }
            )
        registry.register(
            decoder_id,
            label,
            rows,
            lambda row: bool(row["selected"]),
            contract=contract,
            sources=[evaluation_path, modern_features_path],
        )

    register_selected(
        "v3-original-cadence",
        "V3 · original cadence",
        "side-switch-specialist-v3-evaluation.json",
        "Frozen selected V3 cadence decoder.",
    )
    register_selected(
        "v3-reanchored-capped6",
        "V3 · re-anchored, cap 6",
        "side-switch-specialist-v3-reanchored-capped6-evaluation.json",
        "Frozen re-anchored V3 cadence decoder with a six-proposal cap.",
    )
    register_selected(
        "v4-multiframe-normalized",
        "V4 · multiframe normalized",
        "side-switch-specialist-v4-multiframe-normalized-evaluation.json",
        "Frozen selected V4 cadence decoder.",
    )
    register_selected(
        "v5-player-orientation",
        "V5 · player orientation",
        "side-switch-specialist-v5-player-orientation-evaluation.json",
        "Frozen selected V5 cadence decoder.",
    )
    register_selected(
        "v6-detected-adaptive",
        "V6 · detected adaptive",
        "side-switch-specialist-v6-detected-adaptive-evaluation.json",
        "Frozen selected V6 cadence decoder.",
    )
    register_selected(
        "v5-production-state-cadence",
        "V5 + production state · cadence",
        "side-switch-v5-production-state-v1-evaluation.json",
        "Validation-selected original:state-gate view with cadence decoding.",
    )
    register_selected(
        "v6-production-state-cadence",
        "V6 + production state · cadence",
        "side-switch-v6-production-state-v1-evaluation.json",
        "Validation-selected serve-grounded:combined V6 cadence view.",
    )
    register_selected(
        "v5-no-cadence",
        "V5 + state · no cadence",
        "side-switch-v5-no-cadence-v1-evaluation.json",
        "Frozen independent all-gap V5-state decoder.",
    )

    mechanism_labels = {
        "independent-control": "Cleanup · independent control",
        "independent-production-context": (
            "Cleanup · independent + production context"
        ),
        "independent-soft-count": "Cleanup · independent + soft count",
        "independent-soft-count-production-context": (
            "Cleanup · independent + soft count + context"
        ),
        "local-peak": "Cleanup · local peak",
        "local-peak-production-context": (
            "Cleanup · local peak + context (selected)"
        ),
        "local-peak-soft-count": "Cleanup · local peak + soft count",
        "local-peak-soft-count-production-context": (
            "Cleanup · local peak + soft count + context"
        ),
    }
    for mechanism, label in mechanism_labels.items():
        rows = []
        for prediction in peak["predictions"]:
            event_id = str(prediction["eventId"])
            source_row = modern_by_event[event_id]
            rows.append(
                {
                    **source_row,
                    "selected": bool(prediction["mechanismPredictions"][mechanism]),
                }
            )
        registry.register(
            f"cleanup/{mechanism}",
            label,
            rows,
            lambda row: bool(row["selected"]),
            contract="Validation-locked cadence-free cleanup mechanism.",
            sources=[peak_path, modern_features_path],
        )

    expected_totals = {
        "v1-static-original": 113,
        "v1-static-no-blurry-beach": 106,
        "v2-temporal-noop": 138,
        "v3-original-cadence": 55,
        "v3-reanchored-capped6": 44,
        "v4-multiframe-normalized": 37,
        "v5-player-orientation": 44,
        "v6-detected-adaptive": 48,
        "v5-production-state-cadence": 38,
        "v6-production-state-cadence": 32,
        "v5-no-cadence": 91,
        "cleanup/independent-control": 91,
        "cleanup/independent-production-context": 72,
        "cleanup/independent-soft-count": 44,
        "cleanup/independent-soft-count-production-context": 85,
        "cleanup/local-peak": 73,
        "cleanup/local-peak-production-context": 60,
        "cleanup/local-peak-soft-count": 62,
        "cleanup/local-peak-soft-count-production-context": 56,
    }
    if set(expected_totals) != set(registry.proposals):
        raise ValueError("frozen decoder registry changed")
    for decoder_id, expected in expected_totals.items():
        actual = sum(
            len(values) for values in registry.proposals[decoder_id].values()
        )
        if actual != expected:
            raise ValueError(
                f"{decoder_id}: expected {expected} proposals, found {actual}"
            )

    return registry, modern_rows, sorted(source_paths), peak


def _load_markers(
    path: Path, recording_ids: Sequence[str]
) -> tuple[dict[str, list[dict[str, Any]]], Mapping[str, Any]]:
    payload = _load(path)
    markers = payload.get("markers")
    reviewed = payload.get("reviewedRecordingIds")
    if not isinstance(markers, list) or not isinstance(reviewed, list):
        raise ValueError("marker artifact lacks markers/reviewedRecordingIds")
    missing_reviews = sorted(set(recording_ids) - {str(value) for value in reviewed})
    if missing_reviews:
        raise ValueError(
            f"raw-phone recordings are not marked fully reviewed: {missing_reviews}"
        )
    by_recording = {recording_id: [] for recording_id in recording_ids}
    seen_ids: set[str] = set()
    for raw_marker in markers:
        if not isinstance(raw_marker, Mapping):
            raise ValueError("marker entry is not an object")
        recording_id = str(raw_marker["recordingId"])
        if recording_id not in by_recording:
            continue
        marker_id = str(raw_marker["id"])
        if marker_id in seen_ids:
            raise ValueError(f"duplicate marker ID: {marker_id}")
        seen_ids.add(marker_id)
        marker = {
            "id": marker_id,
            "recordingId": recording_id,
            "time": float(raw_marker["time"]),
            "createdAt": raw_marker.get("createdAt"),
        }
        by_recording[recording_id].append(marker)
    for recording_id, values in by_recording.items():
        values.sort(key=lambda marker: (float(marker["time"]), str(marker["id"])))
        if not values:
            raise ValueError(
                f"fully reviewed raw-phone recording has no markers: {recording_id}"
            )
    return by_recording, payload


def _evaluate_one_decoder(
    proposals_by_recording: Mapping[str, Sequence[Mapping[str, Any]]],
    markers_by_recording: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_covered_marker_ids: Mapping[str, set[str]],
    padding_seconds: float,
) -> dict[str, Any]:
    by_recording: dict[str, Any] = {}
    precision_values: list[float] = []
    recall_values: list[float] = []
    proposal_counts: list[int] = []
    total_tp = total_fp = total_fn = 0
    total_upstream_misses = total_decoder_misses = 0

    for recording_id in markers_by_recording:
        proposals = list(proposals_by_recording[recording_id])
        markers = list(markers_by_recording[recording_id])
        result = monotonic_interval_match(proposals, markers, padding_seconds)
        metrics = event_metric_counts(result)
        proposal_counts.append(len(proposals))
        if metrics["precision"] is not None:
            precision_values.append(float(metrics["precision"]))
        if metrics["recall"] is not None:
            recall_values.append(float(metrics["recall"]))
        total_tp += int(metrics["truePositives"])
        total_fp += int(metrics["falsePositives"])
        total_fn += int(metrics["falseNegatives"])

        matched = []
        for pair in result.pairs:
            proposal = proposals[pair.proposal_index]
            marker = markers[pair.marker_index]
            matched.append(
                {
                    "markerId": marker["id"],
                    "humanTime": marker["time"],
                    "proposalEventId": proposal["eventId"],
                    "proposalGapOrder": proposal["gapOrder"],
                    "proposalTime": proposal["transitionTime"],
                    "proposalWindow": [proposal["gapStart"], proposal["gapEnd"]],
                    "anchorDistanceSeconds": pair.anchor_distance_seconds,
                }
            )
        missed = [markers[index] for index in result.unmatched_marker_indices]
        false_positives = [
            proposals[index] for index in result.unmatched_proposal_indices
        ]
        upstream_missed = [
            marker
            for marker in missed
            if str(marker["id"])
            not in candidate_covered_marker_ids[recording_id]
        ]
        decoder_missed = [
            marker
            for marker in missed
            if str(marker["id"]) in candidate_covered_marker_ids[recording_id]
        ]
        total_upstream_misses += len(upstream_missed)
        total_decoder_misses += len(decoder_missed)
        by_recording[recording_id] = {
            "humanEvents": len(markers),
            "proposals": len(proposals),
            **metrics,
            "upstreamCandidateMisses": len(upstream_missed),
            "decoderMissesInsideCandidateUniverse": len(decoder_missed),
            "matched": matched,
            "missedHumanTimes": [marker["time"] for marker in missed],
            "upstreamMissedHumanTimes": [
                marker["time"] for marker in upstream_missed
            ],
            "decoderMissedHumanTimes": [
                marker["time"] for marker in decoder_missed
            ],
            "falsePositiveProposals": false_positives,
        }

    pooled_precision = (
        total_tp / (total_tp + total_fp) if total_tp + total_fp else None
    )
    pooled_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else None
    pooled_f1 = (
        2.0 * pooled_precision * pooled_recall
        / (pooled_precision + pooled_recall)
        if pooled_precision is not None
        and pooled_recall is not None
        and pooled_precision + pooled_recall
        else None
    )
    return {
        "paddingSeconds": padding_seconds,
        "recordings": len(markers_by_recording),
        "humanEvents": sum(len(values) for values in markers_by_recording.values()),
        "proposals": sum(proposal_counts),
        "truePositives": total_tp,
        "falsePositives": total_fp,
        "falseNegatives": total_fn,
        "upstreamCandidateMisses": total_upstream_misses,
        "decoderMissesInsideCandidateUniverse": total_decoder_misses,
        "pooledPrecision": pooled_precision,
        "pooledRecall": pooled_recall,
        "pooledF1": pooled_f1,
        "macroPerVideoPrecision": statistics.mean(precision_values),
        "macroPerVideoPrecisionRecordingCount": len(precision_values),
        "macroPerVideoRecall": statistics.mean(recall_values),
        "macroPerVideoRecallRecordingCount": len(recall_values),
        "proposalDistribution": {
            "meanPerVideo": statistics.mean(proposal_counts),
            "medianPerVideo": statistics.median(proposal_counts),
            "minimumPerVideo": min(proposal_counts),
            "maximumPerVideo": max(proposal_counts),
        },
        "byRecording": by_recording,
    }


def _candidate_coverage(
    candidates: Sequence[Mapping[str, Any]],
    markers_by_recording: Mapping[str, Sequence[Mapping[str, Any]]],
    padding_seconds: float,
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    by_recording: dict[str, Any] = {}
    covered_ids: dict[str, set[str]] = {}
    total_covered = 0
    for recording_id, markers in markers_by_recording.items():
        proposals = [
            candidate
            for candidate in candidates
            if str(candidate["recordingId"]) == recording_id
        ]
        result = monotonic_interval_match(proposals, list(markers), padding_seconds)
        current_ids = {
            str(markers[pair.marker_index]["id"]) for pair in result.pairs
        }
        covered_ids[recording_id] = current_ids
        uncovered = [
            marker for marker in markers if str(marker["id"]) not in current_ids
        ]
        total_covered += len(current_ids)
        by_recording[recording_id] = {
            "humanEvents": len(markers),
            "candidateGaps": len(proposals),
            "coveredHumanEvents": len(current_ids),
            "uncoveredHumanEvents": len(uncovered),
            "uncoveredHumanTimes": [marker["time"] for marker in uncovered],
        }
    total_markers = sum(len(values) for values in markers_by_recording.values())
    return (
        {
            "paddingSeconds": padding_seconds,
            "candidateGaps": len(candidates),
            "humanEvents": total_markers,
            "coveredHumanEvents": total_covered,
            "uncoveredHumanEvents": total_markers - total_covered,
            "coverageRecallCeiling": total_covered / total_markers,
            "byRecording": by_recording,
        },
        covered_ids,
    )


def _ranking(
    decoders: Mapping[str, Mapping[str, Any]], padding_key: str, metric: str
) -> list[dict[str, Any]]:
    secondary = (
        "macroPerVideoPrecision"
        if metric == "macroPerVideoRecall"
        else "macroPerVideoRecall"
    )
    ordered = sorted(
        decoders,
        key=lambda decoder_id: (
            -float(decoders[decoder_id]["evaluations"][padding_key][metric]),
            -float(decoders[decoder_id]["evaluations"][padding_key][secondary]),
            decoder_id,
        ),
    )
    return [
        {
            "rank": rank,
            "decoderId": decoder_id,
            metric: decoders[decoder_id]["evaluations"][padding_key][metric],
            secondary: decoders[decoder_id]["evaluations"][padding_key][secondary],
        }
        for rank, decoder_id in enumerate(ordered, start=1)
    ]


def build_evaluation(root: Path, marker_path: Path) -> dict[str, Any]:
    registry, candidates, source_paths, _ = _build_registry(root)
    markers_by_recording, marker_payload = _load_markers(
        marker_path, registry.recording_ids
    )
    source_paths.append(marker_path)

    coverage: dict[str, Any] = {}
    covered_ids: dict[str, dict[str, set[str]]] = {}
    for padding in PADDING_SENSITIVITY_SECONDS:
        key = f"{padding:g}"
        coverage[key], covered_ids[key] = _candidate_coverage(
            candidates, markers_by_recording, padding
        )

    prior_switches = [row for row in candidates if row["decision"] == "switch"]
    prior_by_recording = {
        recording_id: [
            row for row in prior_switches if row["recordingId"] == recording_id
        ]
        for recording_id in registry.recording_ids
    }
    prior_evaluations = {
        key: _evaluate_one_decoder(
            prior_by_recording,
            markers_by_recording,
            covered_ids[key],
            padding,
        )
        for padding in PADDING_SENSITIVITY_SECONDS
        for key in [f"{padding:g}"]
    }

    decoders: dict[str, Any] = {}
    for decoder_id in registry.proposals:
        evaluations = {
            key: _evaluate_one_decoder(
                registry.proposals[decoder_id],
                markers_by_recording,
                covered_ids[key],
                padding,
            )
            for padding in PADDING_SENSITIVITY_SECONDS
            for key in [f"{padding:g}"]
        }
        decoders[decoder_id] = {
            **registry.metadata[decoder_id],
            "totalProposals": sum(
                len(values) for values in registry.proposals[decoder_id].values()
            ),
            "proposalInventory": registry.proposals[decoder_id],
            "evaluations": evaluations,
        }

    primary_key = f"{PRIMARY_PADDING_SECONDS:g}"
    sources = [
        {"path": str(path), "sha256": _sha256(path)}
        for path in sorted(set(source_paths))
    ]
    repository_root = Path(__file__).resolve().parents[1]
    implementation_paths = [
        Path(__file__).resolve(),
        repository_root / "analysis/side_switch_full_video.py",
    ]
    return {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-full-video-marker-evaluation-v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "implementation": {
            "files": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in implementation_paths
            ]
        },
        "scope": {
            "recordingIds": list(registry.recording_ids),
            "recordings": len(registry.recording_ids),
            "humanEvents": sum(
                len(values) for values in markers_by_recording.values()
            ),
            "humanEventsByRecording": {
                recording_id: [marker["time"] for marker in values]
                for recording_id, values in markers_by_recording.items()
            },
            "allRecordingsMarkedFullyReviewed": True,
            "markerArtifactSavedAt": marker_payload.get("savedAt"),
        },
        "humanTruthPolicy": {
            "source": "manual full-video point markers only",
            "canonicalForRawPhoneScope": True,
            "priorCandidateSwitchDecisionsAreSeedEvidenceOnly": True,
            "reason": (
                "The reviewer placed a complete physical-switch inventory across "
                "each raw-phone recording. Re-unioning the earlier candidate-conditioned "
                "switch decisions would duplicate confirmed events and restore the "
                "candidate-coverage bias this pass was created to remove."
            ),
        },
        "matching": {
            "primaryPaddingSeconds": PRIMARY_PADDING_SECONDS,
            "sensitivityPaddingSeconds": list(PADDING_SENSITIVITY_SECONDS),
            "eligibility": (
                "human marker time inside [proposal gapStart - padding, "
                "proposal gapEnd + padding]"
            ),
            "assignment": (
                "monotonic one-to-one; maximize matches, then minimize total "
                "distance from proposal transitionTime anchors"
            ),
            "paddingClarification": (
                "This is event-matching boundary allowance in seconds, not a cadence "
                "candidate margin measured in rally opportunities."
            ),
        },
        "candidateUniverseCoverage": coverage,
        "priorHeuristicSwitchDecisionAudit": {
            "proposals": len(prior_switches),
            "evaluations": prior_evaluations,
        },
        "decoders": decoders,
        "rankings": {
            "paddingSeconds": PRIMARY_PADDING_SECONDS,
            "byMacroPerVideoRecall": _ranking(
                decoders, primary_key, "macroPerVideoRecall"
            ),
            "byMacroPerVideoPrecision": _ranking(
                decoders, primary_key, "macroPerVideoPrecision"
            ),
        },
        "sources": sources,
        "knownLimitations": [
            (
                "All 11 recordings and their earlier candidate decisions were opened "
                "during prior model work. This is a post-hoc exhaustive audit, not a "
                "new threshold or decoder selection set."
            ),
            (
                "V2 is replayed across all 11 immutable feature sequences, but only four "
                "were its frozen evaluation split; its rank is not a held-out comparison."
            ),
            (
                "A point marker identifies one instant inside a switch, not the full "
                "physical switch interval. Strict and four-second-padded containment are "
                "reported together to show boundary sensitivity."
            ),
        ],
    }


def _render_markdown(payload: Mapping[str, Any]) -> str:
    primary_key = f"{float(payload['matching']['primaryPaddingSeconds']):g}"
    coverage = payload["candidateUniverseCoverage"][primary_key]
    prior = payload["priorHeuristicSwitchDecisionAudit"]["evaluations"][primary_key]
    decoders = payload["decoders"]
    lines = [
        "# Side-switch full-video marker audit — 2026-08-21",
        "",
        (
            f"The canonical truth is {payload['scope']['humanEvents']} manually placed "
            f"switch points across {payload['scope']['recordings']} fully reviewed "
            "raw-phone recordings. Earlier candidate-conditioned switch decisions are "
            "audited separately and are not unioned into the truth set."
        ),
        "",
        "## Matching contract",
        "",
        (
            "Primary matching expands each proposal's full inter-rally gap by 4 seconds "
            "on both sides. Assignment is monotonic and one-to-one, maximizing event "
            "matches before minimizing distance to the proposal anchor. Strict unexpanded "
            "gap containment is retained in the JSON sensitivity result."
        ),
        "",
        (
            f"Only **{coverage['coveredHumanEvents']}/{coverage['humanEvents']} "
            f"({_percent(coverage['coverageRecallCeiling'])})** confirmed switches are "
            "covered by any of the 352 modern candidate gaps at this allowance. The "
            f"remaining **{coverage['uncoveredHumanEvents']}** are upstream candidate "
            "misses and cannot be recovered by re-ranking the existing gaps."
        ),
        "",
        (
            f"The 35 earlier heuristic switch decisions match {prior['truePositives']} "
            f"manual events, leave {prior['falsePositives']} extra/duplicate proposals, "
            f"and miss {prior['falseNegatives']} manual events under the same contract."
        ),
        "",
        "## Sorted by average per-video recall",
        "",
        "| Rank | Decoder | Avg R | Avg P | TP/FP/FN | Pooled P/R/F1 | Proposals/video |",
        "| ---: | --- | ---: | ---: | --- | --- | ---: |",
    ]

    def ranking_row(
        entry: Mapping[str, Any], *, precision_first: bool = False
    ) -> str:
        decoder_id = str(entry["decoderId"])
        decoder = decoders[decoder_id]
        metric = decoder["evaluations"][primary_key]
        v2_note = "*" if decoder_id == "v2-temporal-noop" else ""
        first_metric = (
            metric["macroPerVideoPrecision"]
            if precision_first
            else metric["macroPerVideoRecall"]
        )
        second_metric = (
            metric["macroPerVideoRecall"]
            if precision_first
            else metric["macroPerVideoPrecision"]
        )
        return (
            f"| {entry['rank']} | {decoder['label']}{v2_note} | "
            f"{_percent(first_metric)} | "
            f"{_percent(second_metric)} | "
            f"{metric['truePositives']}/{metric['falsePositives']}/{metric['falseNegatives']} | "
            f"{_percent(metric['pooledPrecision'])} / "
            f"{_percent(metric['pooledRecall'])} / "
            f"{_percent(metric['pooledF1'])} | "
            f"{metric['proposalDistribution']['meanPerVideo']:.1f} |"
        )

    lines.extend(
        ranking_row(entry)
        for entry in payload["rankings"]["byMacroPerVideoRecall"]
    )
    lines.extend(
        [
            "",
            "## Sorted by average per-video precision",
            "",
            "| Rank | Decoder | Avg P | Avg R | TP/FP/FN | Pooled P/R/F1 | Proposals/video |",
            "| ---: | --- | ---: | ---: | --- | --- | ---: |",
        ]
    )
    lines.extend(
        ranking_row(entry, precision_first=True)
        for entry in payload["rankings"]["byMacroPerVideoPrecision"]
    )
    lines.extend(
        [
            "",
            "*V2 replays all 11 frozen feature sequences, but only four recordings were "
            "its held-out evaluation scope. Its rank is shown for output completeness, "
            "not as a fair held-out comparison.",
            "",
            "## Upstream candidate coverage by video",
            "",
            "| Video | Human | Covered | Upstream misses | Missed times |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for recording_id, row in coverage["byRecording"].items():
        missed = ", ".join(_display_time(value) for value in row["uncoveredHumanTimes"])
        lines.append(
            f"| {recording_id} | {row['humanEvents']} | "
            f"{row['coveredHumanEvents']} | {row['uncoveredHumanEvents']} | "
            f"{missed or '—'} |"
        )

    lines.extend(["", "## Per-video decoder breakdown", ""])
    for recording_id in payload["scope"]["recordingIds"]:
        lines.extend(
            [
                f"### {recording_id}",
                "",
                "| Decoder | Proposals | TP/FP/FN | P/R | Missed human times | False-positive anchors |",
                "| --- | ---: | --- | --- | --- | --- |",
            ]
        )
        for decoder_id, decoder in decoders.items():
            row = decoder["evaluations"][primary_key]["byRecording"][recording_id]
            missed = ", ".join(
                _display_time(value) for value in row["missedHumanTimes"]
            )
            false_positive_times = ", ".join(
                _display_time(float(value["transitionTime"]))
                for value in row["falsePositiveProposals"]
            )
            lines.append(
                f"| {decoder['label']} | {row['proposals']} | "
                f"{row['truePositives']}/{row['falsePositives']}/{row['falseNegatives']} | "
                f"{_percent(row['precision'])} / {_percent(row['recall'])} | "
                f"{missed or '—'} | {false_positive_times or '—'} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_csv(payload: Mapping[str, Any]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "padding_seconds",
            "video",
            "decoder",
            "decoder_label",
            "human_events",
            "proposals",
            "true_positives",
            "false_positives",
            "false_negatives",
            "precision",
            "recall",
            "upstream_candidate_misses",
            "decoder_misses_inside_candidate_universe",
            "matched_human_times",
            "missed_human_times",
            "upstream_missed_human_times",
            "decoder_missed_human_times",
            "false_positive_proposal_times",
        ]
    )
    for decoder_id, decoder in payload["decoders"].items():
        for padding_key, evaluation in decoder["evaluations"].items():
            for recording_id, row in evaluation["byRecording"].items():
                writer.writerow(
                    [
                        padding_key,
                        recording_id,
                        decoder_id,
                        decoder["label"],
                        row["humanEvents"],
                        row["proposals"],
                        row["truePositives"],
                        row["falsePositives"],
                        row["falseNegatives"],
                        row["precision"],
                        row["recall"],
                        row["upstreamCandidateMisses"],
                        row["decoderMissesInsideCandidateUniverse"],
                        ";".join(str(value["humanTime"]) for value in row["matched"]),
                        ";".join(str(value) for value in row["missedHumanTimes"]),
                        ";".join(
                            str(value) for value in row["upstreamMissedHumanTimes"]
                        ),
                        ";".join(
                            str(value) for value in row["decoderMissedHumanTimes"]
                        ),
                        ";".join(
                            str(value["transitionTime"])
                            for value in row["falsePositiveProposals"]
                        ),
                    ]
                )
    return output.getvalue()


def write_outputs(prefix: Path, payload: Mapping[str, Any]) -> list[Path]:
    paths = [
        Path(f"{prefix}.json"),
        Path(f"{prefix}.md"),
        Path(f"{prefix}.csv"),
    ]
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite full-video evaluation output: "
            + ", ".join(str(path) for path in existing)
        )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        paths[0], json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    atomic_write_text(paths[1], _render_markdown(payload))
    atomic_write_text(paths[2], _render_csv(payload))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--markers", type=Path, default=DEFAULT_MARKERS)
    parser.add_argument(
        "--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_evaluation(
        args.root.expanduser().resolve(), args.markers.expanduser().resolve()
    )
    paths = write_outputs(args.output_prefix.expanduser().resolve(), payload)
    primary_key = f"{PRIMARY_PADDING_SECONDS:g}"
    print(
        "candidate-coverage="
        f"{payload['candidateUniverseCoverage'][primary_key]['coveredHumanEvents']}/"
        f"{payload['scope']['humanEvents']}"
    )
    for entry in payload["rankings"]["byMacroPerVideoRecall"]:
        decoder_id = entry["decoderId"]
        metric = payload["decoders"][decoder_id]["evaluations"][primary_key]
        print(
            f"{entry['rank']:02d}\t{decoder_id}\t"
            f"macroR={metric['macroPerVideoRecall']:.6f}\t"
            f"macroP={metric['macroPerVideoPrecision']:.6f}\t"
            f"TP/FP/FN={metric['truePositives']}/"
            f"{metric['falsePositives']}/{metric['falseNegatives']}"
        )
    for path in paths:
        print(f"{path}\tsha256={_sha256(path)}")


if __name__ == "__main__":
    main()

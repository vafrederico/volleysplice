"""Fail-closed lineage preflight and pure targets for transition pilots.

No video or feature preparation occurs until a candidate-specific gate, a
development-only immutable snapshot, and its rebuilt manifest agree exactly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .annotations import load_label_document
from .artifacts import atomic_write_text
from .schema import DatasetManifest, ManifestError, Recording, load_manifest
from .transition_label_gate import (
    DEVELOPMENT_SPLITS,
    TRANSITION_GATE_KIND,
    require_transition_candidate_ready,
)


TRANSITION_PREFLIGHT_KIND = "volleycut-transition-pilot-development-preflight"
SNAPSHOT_KIND = "volleycut-completed-label-snapshot"
REGISTERED_EXPERIMENTS = (
    "reaction-supervised-serve-edge",
    "stand-down-supervised-terminal-edge",
    "verified-immediate-result-branch",
    "hard-negative-dead-state",
)


@dataclass(frozen=True)
class ReactionCue:
    rally_index: int
    serve_time: float
    reaction_time: float
    reaction_delay: float
    start_confidence: float


@dataclass(frozen=True)
class TerminalCue:
    rally_index: int
    end_time: float
    stand_down_time: float
    stand_down_offset: float
    terminal_cue: str
    observability: str
    end_confidence: float


@dataclass(frozen=True)
class ImmediateResultTarget:
    rally_index: int
    value: bool


@dataclass(frozen=True)
class HardNegativeTarget:
    start: float
    end: float
    category: str


@dataclass(frozen=True)
class TransitionPilotRecording:
    recording_id: str
    reaction_cues: tuple[ReactionCue, ...]
    terminal_cues: tuple[TerminalCue, ...]
    immediate_results: tuple[ImmediateResultTarget, ...]
    hard_negatives: tuple[HardNegativeTarget, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: str | Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read {label} {resolved}: {error}") from error
    if not isinstance(value, dict):
        raise ManifestError(f"{label} root must be an object")
    return resolved, value


def _baseline_development_rows(path: str | Path) -> tuple[Path, dict[str, Mapping[str, Any]]]:
    resolved, payload = _load_json(path, label="baseline manifest")
    rows = payload.get("recordings")
    if not isinstance(rows, list):
        raise ManifestError("baseline manifest recordings are missing")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), str):
            raise ManifestError("baseline manifest contains an invalid recording")
        if row.get("split") not in DEVELOPMENT_SPLITS:
            continue
        recording_id = str(row["id"])
        if recording_id in result:
            raise ManifestError(f"baseline manifest duplicates {recording_id!r}")
        result[recording_id] = row
    if not result:
        raise ManifestError("baseline manifest contains no development recordings")
    return resolved, result


def _gate_rows(gate: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = gate.get("recordings")
    if not isinstance(rows, list):
        raise ManifestError("transition gate recording rows are missing")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("recordingId"), str):
            raise ManifestError("transition gate contains an invalid recording")
        if row.get("split") not in DEVELOPMENT_SPLITS:
            continue
        recording_id = str(row["recordingId"])
        if recording_id in result:
            raise ManifestError(f"transition gate duplicates {recording_id!r}")
        result[recording_id] = row
    return result


def _snapshot_rows(path: str | Path) -> tuple[Path, dict[str, Mapping[str, Any]]]:
    resolved, ledger = _load_json(path, label="snapshot ledger")
    if ledger.get("kind") != SNAPSHOT_KIND:
        raise ManifestError("snapshot ledger has the wrong kind")
    rows = ledger.get("recordings")
    if not isinstance(rows, list):
        raise ManifestError("snapshot ledger recordings are missing")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("recordingId"), str):
            raise ManifestError("snapshot ledger contains an invalid recording")
        recording_id = str(row["recordingId"])
        if recording_id in result:
            raise ManifestError(f"snapshot ledger duplicates {recording_id!r}")
        result[recording_id] = row
    return resolved, result


def extract_transition_pilot(recording: Recording) -> TransitionPilotRecording:
    reaction: list[ReactionCue] = []
    terminal: list[TerminalCue] = []
    immediate: list[ImmediateResultTarget] = []
    for index, row in enumerate(recording.raw.get("rallies", [])):
        if not isinstance(row, Mapping):
            continue
        start = float(row["start"])
        end = float(row["end"])
        if row.get("receiverReactionTime") is not None and row.get("startConfidence") is not None:
            cue = float(row["receiverReactionTime"])
            reaction.append(ReactionCue(index, start, cue, cue - start, float(row["startConfidence"])))
        terminal_fields = (
            "collectiveStandDownTime", "terminalCue", "endObservability", "endConfidence"
        )
        if all(row.get(field) is not None for field in terminal_fields):
            cue = float(row["collectiveStandDownTime"])
            terminal.append(
                TerminalCue(
                    index, end, cue, cue - end, str(row["terminalCue"]),
                    str(row["endObservability"]), float(row["endConfidence"]),
                )
            )
        if isinstance(row.get("verifiedImmediateResult"), bool):
            immediate.append(ImmediateResultTarget(index, row["verifiedImmediateResult"]))
    negatives = tuple(
        HardNegativeTarget(float(row["start"]), float(row["end"]), str(row["category"]))
        for row in recording.raw.get("hardNegatives", [])
        if isinstance(row, Mapping)
    )
    return TransitionPilotRecording(recording.id, tuple(reaction), tuple(terminal), tuple(immediate), negatives)


def hard_negative_mask_for_times(
    times: np.ndarray, targets: tuple[HardNegativeTarget, ...]
) -> np.ndarray:
    values = np.asarray(times, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("times must be a finite one-dimensional array")
    result = np.zeros(len(values), dtype=np.bool_)
    for target in targets:
        result |= (values >= target.start) & (values < target.end)
    return result


def _validate_snapshot_manifest(
    manifest: DatasetManifest,
    *,
    baseline: Mapping[str, Mapping[str, Any]],
    gate: Mapping[str, Any],
    ledger_path: Path,
    ledger: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[TransitionPilotRecording]]:
    expected = set(baseline)
    actual = {row.id for row in manifest.recordings}
    if actual != expected or any(row.split not in DEVELOPMENT_SPLITS for row in manifest.recordings):
        raise ManifestError(
            "pilot manifest must contain exactly baseline train/validation rows "
            f"(missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)})"
        )
    if set(ledger) != expected:
        raise ManifestError("snapshot ledger does not contain exactly the baseline development rows")
    gate_by_id = _gate_rows(gate)
    if set(gate_by_id) != expected:
        raise ManifestError("transition gate does not contain exactly the baseline development rows")
    lineage: list[dict[str, Any]] = []
    targets: list[TransitionPilotRecording] = []
    for recording in sorted(manifest.recordings, key=lambda item: item.id):
        gated = gate_by_id[recording.id]
        ledger_row = ledger[recording.id]
        source_path = Path(str(gated.get("path"))).expanduser().resolve()
        source_sha = str(gated.get("fileSha256"))
        if not source_path.is_file() or _sha256(source_path) != source_sha:
            raise ManifestError(f"{recording.id}: draft changed after transition gate")
        if ledger_row.get("sourceDraftSha256") != source_sha:
            raise ManifestError(f"{recording.id}: snapshot source hash does not match gate")
        snapshot_file = ledger_row.get("snapshotFile")
        snapshot_sha = ledger_row.get("snapshotSha256")
        if not isinstance(snapshot_file, str) or not isinstance(snapshot_sha, str):
            raise ManifestError(f"{recording.id}: snapshot identity is missing")
        snapshot_path = (ledger_path.parent / snapshot_file).resolve()
        if not snapshot_path.is_file() or _sha256(snapshot_path) != snapshot_sha:
            raise ManifestError(f"{recording.id}: snapshot file hash mismatch")
        document = load_label_document(snapshot_path, require_complete=True, require_video=True)
        base = baseline[recording.id]
        if (
            recording.split != gated.get("split")
            or recording.split != base.get("split")
            or recording.source_group != gated.get("sourceGroup")
            or recording.source_group != base.get("sourceGroup")
            or document.recording_id != recording.id
            or document.split != recording.split
            or document.source_group != recording.source_group
            or document.payload.get("rallies", []) != recording.raw.get("rallies", [])
            or document.payload.get("hardNegatives", []) != recording.raw.get("hardNegatives", [])
            or document.payload.get("ignoredIntervals", []) != recording.raw.get("ignoredIntervals", [])
            or document.payload["recording"].get("contentSha256") != recording.content_sha256
        ):
            raise ManifestError(f"{recording.id}: snapshot, baseline, gate, and manifest lineage disagree")
        extracted = extract_transition_pilot(recording)
        targets.append(extracted)
        lineage.append(
            {
                "recordingId": recording.id,
                "sourceGroup": recording.source_group,
                "split": recording.split,
                "gatedDraftSha256": source_sha,
                "snapshotFile": str(snapshot_path),
                "snapshotSha256": snapshot_sha,
                "targets": {
                    "reaction": len(extracted.reaction_cues),
                    "terminal": len(extracted.terminal_cues),
                    "immediateResult": len(extracted.immediate_results),
                    "hardNegative": len(extracted.hard_negatives),
                },
            }
        )
    return lineage, targets


def build_transition_pilot_preflight(
    *,
    baseline_manifest_path: str | Path,
    manifest_path: str | Path,
    snapshot_ledger_path: str | Path,
    gate_path: str | Path,
    candidate: str,
) -> dict[str, Any]:
    """Validate lineage before any downstream feature preparation."""

    if candidate not in REGISTERED_EXPERIMENTS:
        raise ValueError(f"unknown transition candidate: {candidate}")
    baseline_path, baseline = _baseline_development_rows(baseline_manifest_path)
    gate_file, gate = _load_json(gate_path, label="transition gate")
    if gate.get("kind") != TRANSITION_GATE_KIND:
        raise ManifestError("transition gate has the wrong kind")
    # Candidate readiness and exact corpus coverage are checked before load_manifest.
    require_transition_candidate_ready(gate, candidate, expected_development_ids=baseline)
    ledger_path, ledger = _snapshot_rows(snapshot_ledger_path)
    manifest = load_manifest(manifest_path)
    lineage, targets = _validate_snapshot_manifest(
        manifest, baseline=baseline, gate=gate, ledger_path=ledger_path, ledger=ledger
    )
    source_groups = sorted({row.source_group for row in manifest.recordings})
    if len(source_groups) < 2:
        raise ManifestError("transition pilot requires at least two development source groups")
    return {
        "schemaVersion": 1,
        "kind": TRANSITION_PREFLIGHT_KIND,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "readOnly": True,
        "candidate": candidate,
        "featureRowsPrepared": False,
        "protectedSplitsPrepared": False,
        "testLabelsUsed": False,
        "baselineManifest": str(baseline_path),
        "baselineManifestSha256": _sha256(baseline_path),
        "manifest": str(manifest.path),
        "manifestSha256": _sha256(manifest.path),
        "snapshotLedger": str(ledger_path),
        "snapshotLedgerSha256": _sha256(ledger_path),
        "transitionGate": str(gate_file),
        "transitionGateSha256": _sha256(gate_file),
        "developmentRecordingIds": sorted(baseline),
        "developmentSourceGroups": source_groups,
        "recordingLineage": lineage,
        "targetCounts": {
            "reaction": sum(len(row.reaction_cues) for row in targets),
            "terminal": sum(len(row.terminal_cues) for row in targets),
            "immediateResult": sum(len(row.immediate_results) for row in targets),
            "hardNegative": sum(len(row.hard_negatives) for row in targets),
        },
        "nextAction": (
            "consume only this candidate's extracted development targets in nested "
            "source-group evaluation; keep protected labels sealed"
        ),
    }


def write_transition_pilot_preflight(path: str | Path, report: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite transition preflight: {destination}")
    return atomic_write_text(destination, json.dumps(report, indent=2, allow_nan=False) + "\n")


__all__ = [
    "HardNegativeTarget",
    "ImmediateResultTarget",
    "REGISTERED_EXPERIMENTS",
    "ReactionCue",
    "TRANSITION_PREFLIGHT_KIND",
    "TerminalCue",
    "TransitionPilotRecording",
    "build_transition_pilot_preflight",
    "extract_transition_pilot",
    "hard_negative_mask_for_times",
    "write_transition_pilot_preflight",
]

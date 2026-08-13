"""Read-only gates and lineage plans for transition-cue experiments.

The current pilot requests five fully cued rallies per recording.  Development
may consume only train/validation drafts; protected test labels remain sealed
until a separately frozen candidate is eligible for retrospective assessment.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .annotations import load_label_document
from .artifacts import atomic_write_text
from .label_readiness import (
    RECOMMENDED_HARD_NEGATIVE_CATEGORIES,
    RECOMMENDED_TRANSITION_RALLIES_PER_RECORDING,
    TRANSITION_FIELDS,
)
from .schema import ManifestError


TRANSITION_GATE_KIND = "volleycut-transition-label-experiment-gate"
DEVELOPMENT_SPLITS = frozenset(("train", "validation"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fully_cued(rally: Mapping[str, Any]) -> bool:
    return all(field in rally for field in TRANSITION_FIELDS)


def _stratum(rally: Mapping[str, Any]) -> str:
    tags = set(rally.get("tags", []))
    if tags & {"service-fault", "service-error"}:
        return "service-fault"
    if "ace" in tags:
        return "ace"
    if "interrupted-replay" in tags:
        return "interrupted-replay"
    return "ordinary"


def build_transition_label_gate(
    paths: Iterable[str | Path],
) -> dict[str, Any]:
    resolved = sorted({Path(path).expanduser().resolve() for path in paths})
    if not resolved:
        raise ValueError("transition label gate requires label documents")
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    for path in resolved:
        try:
            document = load_label_document(
                path, require_complete=False, require_video=False
            )
            rallies = [
                row
                for row in document.payload.get("rallies", [])
                if isinstance(row, dict)
            ]
            fully = [row for row in rallies if _fully_cued(row)]
            hard_negatives = [
                row
                for row in document.payload.get("hardNegatives", [])
                if isinstance(row, dict)
            ]
            categories = {
                str(row.get("category"))
                for row in hard_negatives
                if isinstance(row.get("category"), str)
            }
            rows.append(
                {
                    "recordingId": document.recording_id,
                    "sourceGroup": document.source_group,
                    "split": document.split,
                    "path": str(path),
                    "fileSha256": _sha256(path),
                    "rallies": len(rallies),
                    "fullyCuedRallies": len(fully),
                    "pilotTarget": RECOMMENDED_TRANSITION_RALLIES_PER_RECORDING,
                    "meetsPilotTarget": len(fully)
                    >= RECOMMENDED_TRANSITION_RALLIES_PER_RECORDING,
                    "additionalRalliesToPilotTarget": max(
                        0,
                        RECOMMENDED_TRANSITION_RALLIES_PER_RECORDING - len(fully),
                    ),
                    "fullyCuedByStratum": {
                        stratum: sum(_stratum(row) == stratum for row in fully)
                        for stratum in (
                            "ordinary",
                            "ace",
                            "service-fault",
                            "interrupted-replay",
                        )
                    },
                    "hardNegatives": len(hard_negatives),
                    "hardNegativeCategories": sorted(categories),
                    "missingRecommendedHardNegativeCategories": [
                        category
                        for category in RECOMMENDED_HARD_NEGATIVE_CATEGORIES
                        if category not in categories
                    ],
                    "additionalHardNegativesToThree": max(
                        0, 3 - len(hard_negatives)
                    ),
                }
            )
        except (ManifestError, OSError, ValueError) as error:
            invalid.append({"path": str(path), "error": str(error)})
    ids = [row["recordingId"] for row in rows]
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    development = [row for row in rows if row["split"] in DEVELOPMENT_SPLITS]
    protected = [row for row in rows if row["split"] not in DEVELOPMENT_SPLITS]
    development_ready = bool(development) and all(
        row["meetsPilotTarget"] for row in development
    )
    development_hard_negative_ready = bool(development) and all(
        row["additionalHardNegativesToThree"] == 0 for row in development
    )
    return {
        "schemaVersion": 1,
        "kind": TRANSITION_GATE_KIND,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "readOnly": True,
        "documentsScanned": len(resolved),
        "invalidDocuments": invalid,
        "duplicateRecordingIds": duplicates,
        "requirements": {
            "transitionFields": list(TRANSITION_FIELDS),
            "fullyCuedRalliesPerRecording": (
                RECOMMENDED_TRANSITION_RALLIES_PER_RECORDING
            ),
            "hardNegativesPerRecording": 3,
            "recommendedHardNegativeCategories": list(
                RECOMMENDED_HARD_NEGATIVE_CATEGORIES
            ),
        },
        "development": {
            "recordings": len(development),
            "fullyCuedRallies": sum(row["fullyCuedRallies"] for row in development),
            "pilotTargetRallies": (
                len(development) * RECOMMENDED_TRANSITION_RALLIES_PER_RECORDING
            ),
            "additionalRalliesToPilotTarget": sum(
                row["additionalRalliesToPilotTarget"] for row in development
            ),
            "additionalHardNegativesToThree": sum(
                row["additionalHardNegativesToThree"] for row in development
            ),
            "allRecordingsMeetTransitionPilot": development_ready,
            "allRecordingsMeetHardNegativeMinimum": (
                development_hard_negative_ready
            ),
        },
        "protected": {
            "recordings": len(protected),
            "fullyCuedRallies": sum(row["fullyCuedRallies"] for row in protected),
            "pilotTargetRallies": (
                len(protected) * RECOMMENDED_TRANSITION_RALLIES_PER_RECORDING
            ),
            "additionalRalliesToPilotTarget": sum(
                row["additionalRalliesToPilotTarget"] for row in protected
            ),
            "additionalHardNegativesToThree": sum(
                row["additionalHardNegativesToThree"] for row in protected
            ),
            "sealedForDevelopment": True,
        },
        "developmentExperimentReady": (
            development_ready and not invalid and not duplicates
        ),
        "developmentHardNegativeExperimentReady": (
            development_hard_negative_ready and not invalid and not duplicates
        ),
        "allRegisteredDevelopmentExperimentsReady": (
            development_ready
            and development_hard_negative_ready
            and not invalid
            and not duplicates
        ),
        "recordings": rows,
        "futureExecutionPlan": {
            "manifestLineage": (
                "freeze a new manifest from the completed label snapshots; do not "
                "attach mutable draft fields to existing frozen reports"
            ),
            "developmentInputs": "train and validation documents only",
            "protectedInputs": (
                "test documents remain unopened by development runners and may be "
                "prepared only after a predeclared development gate passes"
            ),
            "registeredCandidates": [
                {
                    "name": "reaction-supervised-serve-edge",
                    "ready": development_ready and not invalid and not duplicates,
                    "targetFields": ["receiverReactionTime", "startConfidence"],
                    "action": (
                        "cross-fit a start specialist and add fixed bounded evidence "
                        "only to SETUP->SERVE"
                    ),
                },
                {
                    "name": "stand-down-supervised-terminal-edge",
                    "ready": development_ready and not invalid and not duplicates,
                    "targetFields": [
                        "collectiveStandDownTime",
                        "terminalCue",
                        "endObservability",
                        "endConfidence",
                    ],
                    "action": (
                        "cross-fit an endpoint specialist and add fixed bounded evidence "
                        "only to LIVE->DEAD"
                    ),
                },
                {
                    "name": "verified-immediate-result-branch",
                    "ready": development_ready and not invalid and not duplicates,
                    "targetFields": ["verifiedImmediateResult", "terminalCue"],
                    "action": (
                        "replace the coarse ace/fault proxy with verified branch targets"
                    ),
                },
                {
                    "name": "hard-negative-dead-state",
                    "ready": (
                        development_hard_negative_ready
                        and not invalid
                        and not duplicates
                    ),
                    "targetFields": [],
                    "action": (
                        "use categorized walking/retrieval, celebration/huddle, "
                        "model-FP, and random-dead intervals as explicit negatives"
                    ),
                },
            ],
        },
    }


def require_transition_development_ready(report: Mapping[str, Any]) -> None:
    if report.get("kind") != TRANSITION_GATE_KIND:
        raise ValueError("transition readiness report has the wrong kind")
    if report.get("developmentExperimentReady") is not True:
        debt = report.get("development", {}).get(
            "additionalRalliesToPilotTarget", "unknown"
        )
        raise RuntimeError(
            "transition-cue experiment is waiting on development labels "
            f"(additional fully cued rallies: {debt})"
        )
    if report.get("protected", {}).get("sealedForDevelopment") is not True:
        raise RuntimeError("protected transition labels are not sealed")


def write_transition_gate(path: str | Path, report: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite transition label gate: {destination}"
        )
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    return atomic_write_text(destination, rendered)


__all__ = [
    "TRANSITION_GATE_KIND",
    "build_transition_label_gate",
    "require_transition_development_ready",
    "write_transition_gate",
]

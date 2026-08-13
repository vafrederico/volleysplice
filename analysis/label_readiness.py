from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .annotations import load_label_document
from .schema import ManifestError


TRANSITION_FIELDS = (
    "receiverReactionTime",
    "collectiveStandDownTime",
    "terminalCue",
    "endObservability",
    "startConfidence",
    "endConfidence",
    "verifiedImmediateResult",
)
RECOMMENDED_HARD_NEGATIVE_CATEGORIES = (
    "walking-ball-retrieval",
    "celebration-huddle",
    "model-false-positive",
    "random-dead-control",
)
RECOMMENDED_TRACKLET_RALLIES_PER_RECORDING = 5
RECOMMENDED_TRANSITION_RALLIES_PER_RECORDING = 5


def _coverage(labeled: int, total: int) -> dict[str, Any]:
    return {
        "labeled": labeled,
        "total": total,
        "coverage": labeled / total if total else None,
    }


def _outcome_stratum(rally: dict[str, Any]) -> str:
    tags = set(rally.get("tags", []))
    if tags & {"service-fault", "service-error"}:
        return "service-fault"
    if "ace" in tags:
        return "ace"
    if "interrupted-replay" in tags:
        return "interrupted-replay"
    return "normal-or-unclassified"


def _transition_summary(rallies: list[dict[str, Any]]) -> dict[str, Any]:
    field_counts = {
        field: _coverage(sum(rally.get(field) is not None for rally in rallies), len(rallies))
        for field in TRANSITION_FIELDS
    }
    fully_labeled = sum(
        all(rally.get(field) is not None for field in TRANSITION_FIELDS)
        for rally in rallies
    )
    by_stratum: dict[str, Any] = {}
    strata = sorted({_outcome_stratum(rally) for rally in rallies})
    for stratum in strata:
        rows = [rally for rally in rallies if _outcome_stratum(rally) == stratum]
        by_stratum[stratum] = {
            "rallies": len(rows),
            "fullyLabeled": _coverage(
                sum(
                    all(rally.get(field) is not None for field in TRANSITION_FIELDS)
                    for rally in rows
                ),
                len(rows),
            ),
            "fields": {
                field: _coverage(
                    sum(rally.get(field) is not None for rally in rows), len(rows)
                )
                for field in TRANSITION_FIELDS
            },
        }
    return {
        "rallies": len(rallies),
        "fullyLabeled": _coverage(fully_labeled, len(rallies)),
        "fields": field_counts,
        "byOutcomeStratum": by_stratum,
    }


def _tracklet_summary(rallies: list[dict[str, Any]]) -> dict[str, Any]:
    windows: Counter[str] = Counter()
    usable_windows: Counter[str] = Counter()
    states: Counter[str] = Counter()
    geometry: Counter[str] = Counter()
    teams: Counter[str] = Counter()
    court_sides: Counter[str] = Counter()
    tracklet_count = 0
    usable_tracklets = 0
    observation_count = 0
    rallies_with_tracklets = 0
    rallies_with_both_windows = 0
    ready_rally_windows = 0
    rallies_with_both_ready_windows = 0
    for rally in rallies:
        raw_tracklets = rally.get("playerTracklets", [])
        tracklets = [row for row in raw_tracklets if isinstance(row, dict)]
        if tracklets:
            rallies_with_tracklets += 1
        present_windows = {
            str(row.get("window"))
            for row in tracklets
            if row.get("window") in {"serve", "rally-end"}
        }
        if present_windows == {"serve", "rally-end"}:
            rallies_with_both_windows += 1
        ready_windows: set[str] = set()
        for window in ("serve", "rally-end"):
            window_rows = [row for row in tracklets if row.get("window") == window]
            window_usable = [
                row
                for row in window_rows
                if isinstance(row.get("observations"), list)
                and len(row["observations"]) >= 2
            ]
            if len(window_usable) >= 2:
                ready_windows.add(window)
                ready_rally_windows += 1
        if ready_windows == {"serve", "rally-end"}:
            rallies_with_both_ready_windows += 1
        for tracklet in tracklets:
            tracklet_count += 1
            window = str(tracklet.get("window"))
            windows[window] += 1
            observations = [
                row
                for row in tracklet.get("observations", [])
                if isinstance(row, dict)
            ]
            if len(observations) >= 2:
                usable_tracklets += 1
                usable_windows[window] += 1
            teams[str(tracklet.get("team"))] += 1
            court_sides[str(tracklet.get("courtSide"))] += 1
            observation_count += len(observations)
            for observation in observations:
                if "footpoint" in observation:
                    geometry["footpoint"] += 1
                if "box" in observation:
                    geometry["box"] += 1
                if isinstance(observation.get("state"), str):
                    states[observation["state"]] += 1
    return {
        "rallies": len(rallies),
        "ralliesWithTracklets": rallies_with_tracklets,
        "ralliesWithBothWindows": rallies_with_both_windows,
        "ralliesWithBothReadyWindows": rallies_with_both_ready_windows,
        "readyRallyWindows": ready_rally_windows,
        "pilotTargetRallies": RECOMMENDED_TRACKLET_RALLIES_PER_RECORDING,
        "additionalRalliesToPilotTarget": max(
            0,
            RECOMMENDED_TRACKLET_RALLIES_PER_RECORDING
            - rallies_with_both_ready_windows,
        ),
        "tracklets": tracklet_count,
        "usableTracklets": usable_tracklets,
        "observations": observation_count,
        "windows": dict(sorted(windows.items())),
        "usableWindows": dict(sorted(usable_windows.items())),
        "geometry": dict(sorted(geometry.items())),
        "states": dict(sorted(states.items())),
        "teams": dict(sorted(teams.items())),
        "courtSides": dict(sorted(court_sides.items())),
        "readinessRule": (
            "A track is usable with at least two observations; a boundary window is "
            "ready with at least two usable anonymous tracks."
        ),
    }


def _recording_readiness(path: Path) -> dict[str, Any]:
    document = load_label_document(
        path,
        require_complete=False,
        require_video=False,
    )
    geometry = document.court_geometry or {}
    corners = geometry.get("corners", {})
    net = geometry.get("netAnchors", {})
    service = geometry.get("serviceZoneAnchors", {})
    corner_count = len(corners)
    net_count = len(net)
    service_count = len(service)

    raw_negatives = document.payload.get("hardNegatives", [])
    categories = Counter(
        row.get("category")
        for row in raw_negatives
        if isinstance(row, dict) and isinstance(row.get("category"), str)
    )
    missing_categories = [
        category
        for category in RECOMMENDED_HARD_NEGATIVE_CATEGORIES
        if categories.get(category, 0) == 0
    ]
    rallies = [
        dict(row)
        for row in document.payload.get("rallies", [])
        if isinstance(row, dict)
    ]
    transition_summary = _transition_summary(rallies)
    fully_labeled_transitions = int(
        transition_summary["fullyLabeled"]["labeled"]
    )
    return {
        "recordingId": document.recording_id,
        "path": str(path),
        "annotationStatus": document.payload.get("annotation", {}).get("status"),
        "geometry": {
            "corners": corner_count,
            "netAnchors": net_count,
            "serviceZoneAnchors": service_count,
            "totalAnchors": corner_count + net_count + service_count,
            "minimumReady": corner_count == 4,
            "fullEightAnchorReady": corner_count == 4 and net_count == 2 and service_count == 2,
            "missingCorners": [
                name
                for name in ("nearLeft", "nearRight", "farLeft", "farRight")
                if name not in corners
            ],
        },
        "hardNegatives": {
            "count": len(raw_negatives),
            "minimumThreeMet": len(raw_negatives) >= 3,
            "withinThreeToFiveTarget": 3 <= len(raw_negatives) <= 5,
            "additionalToMinimumThree": max(0, 3 - len(raw_negatives)),
            "categories": dict(sorted(categories.items())),
            "missingRecommendedCategories": missing_categories,
        },
        "transitions": {
            **transition_summary,
            "pilotTargetRallies": RECOMMENDED_TRANSITION_RALLIES_PER_RECORDING,
            "additionalRalliesToPilotTarget": max(
                0,
                RECOMMENDED_TRANSITION_RALLIES_PER_RECORDING
                - fully_labeled_transitions,
            ),
        },
        "playerTracklets": _tracklet_summary(rallies),
    }


def build_label_readiness_report(paths: Iterable[str | Path]) -> dict[str, Any]:
    resolved = sorted({Path(path).expanduser().resolve() for path in paths})
    recordings: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    for path in resolved:
        try:
            recordings.append(_recording_readiness(path))
        except (ManifestError, OSError, ValueError) as error:
            invalid.append({"path": str(path), "error": str(error)})

    ids = Counter(row["recordingId"] for row in recordings)
    duplicate_ids = sorted(recording_id for recording_id, count in ids.items() if count > 1)
    hard_categories: Counter[str] = Counter()
    all_rallies: list[dict[str, Any]] = []
    for row in recordings:
        hard_categories.update(row["hardNegatives"]["categories"])
        source = load_label_document(
            row["path"], require_complete=False, require_video=False
        ).payload
        all_rallies.extend(
            dict(rally)
            for rally in source.get("rallies", [])
            if isinstance(rally, dict)
        )

    checklist = {
        "validRecordings": len(recordings),
        "invalidDocuments": len(invalid),
        "duplicateRecordingIds": duplicate_ids,
        "geometry": {
            "minimumReadyRecordings": sum(
                row["geometry"]["minimumReady"] for row in recordings
            ),
            "fullEightAnchorReadyRecordings": sum(
                row["geometry"]["fullEightAnchorReady"] for row in recordings
            ),
            "missingCornerClicks": sum(
                len(row["geometry"]["missingCorners"]) for row in recordings
            ),
            "missingNetAnchorClicks": sum(
                2 - row["geometry"]["netAnchors"] for row in recordings
            ),
            "missingServiceZoneAnchorClicks": sum(
                2 - row["geometry"]["serviceZoneAnchors"] for row in recordings
            ),
        },
        "hardNegatives": {
            "total": sum(row["hardNegatives"]["count"] for row in recordings),
            "minimumThreeMetRecordings": sum(
                row["hardNegatives"]["minimumThreeMet"] for row in recordings
            ),
            "withinThreeToFiveTargetRecordings": sum(
                row["hardNegatives"]["withinThreeToFiveTarget"] for row in recordings
            ),
            "additionalIntervalsToMinimumThree": sum(
                row["hardNegatives"]["additionalToMinimumThree"] for row in recordings
            ),
            "categories": dict(sorted(hard_categories.items())),
            "recordingsMissingRecommendedCategory": {
                category: sum(
                    category in row["hardNegatives"]["missingRecommendedCategories"]
                    for row in recordings
                )
                for category in RECOMMENDED_HARD_NEGATIVE_CATEGORIES
            },
        },
        "transitions": {
            **_transition_summary(all_rallies),
            "pilotTargetRalliesPerRecording": (
                RECOMMENDED_TRANSITION_RALLIES_PER_RECORDING
            ),
            "recordingsMeetingPilotTarget": sum(
                row["transitions"]["fullyLabeled"]["labeled"]
                >= RECOMMENDED_TRANSITION_RALLIES_PER_RECORDING
                for row in recordings
            ),
            "additionalRalliesToPilotTarget": sum(
                row["transitions"]["additionalRalliesToPilotTarget"]
                for row in recordings
            ),
        },
        "playerTracklets": {
            **_tracklet_summary(all_rallies),
            "recordingsMeetingPilotTarget": sum(
                row["playerTracklets"]["ralliesWithBothReadyWindows"]
                >= RECOMMENDED_TRACKLET_RALLIES_PER_RECORDING
                for row in recordings
            ),
            "additionalRalliesToPilotTarget": sum(
                row["playerTracklets"]["additionalRalliesToPilotTarget"]
                for row in recordings
            ),
        },
    }
    return {
        "schemaVersion": 1,
        "kind": "volleycut-label-readiness",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "documentsScanned": len(resolved),
        "checklist": checklist,
        "recordings": recordings,
        "invalidDocuments": invalid,
    }

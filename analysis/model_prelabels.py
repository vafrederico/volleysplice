from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from .annotations import load_label_document
from .artifacts import atomic_write_text
from .schema import ManifestError


CONFIDENCE_VALUES = {"high", "medium", "low"}
METHOD_ID = "blind-gpt-5.6-sol-xhigh-audiovisual"
METHOD_ANNOTATORS = {
    "blind-gpt-5.6-sol-high-audiovisual": "GPT-5.6 Sol high (unvalidated)",
    METHOD_ID: "GPT-5.6 Sol xhigh (unvalidated)",
}


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ManifestError(f"{description} root must be an object: {path}")
    return value


def _candidate_events(
    value: Any,
    *,
    duration: float,
    recording_id: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ManifestError(f"{recording_id}: events must be an array")
    events: list[dict[str, Any]] = []
    previous_end = -1.0
    for index, row in enumerate(value):
        where = f"{recording_id}: events[{index}]"
        if not isinstance(row, dict):
            raise ManifestError(f"{where} must be an object")
        start = row.get("serveContact")
        end = row.get("rallyEnd")
        if (
            not isinstance(start, (int, float))
            or isinstance(start, bool)
            or not math.isfinite(float(start))
            or not isinstance(end, (int, float))
            or isinstance(end, bool)
            or not math.isfinite(float(end))
        ):
            raise ManifestError(f"{where} must contain finite serveContact and rallyEnd values")
        start_value = round(float(start), 3)
        end_value = round(float(end), 3)
        if start_value < 0 or end_value <= start_value or end_value > duration + 1e-6:
            raise ManifestError(f"{where} must satisfy 0 <= serveContact < rallyEnd <= duration")
        if start_value < previous_end:
            raise ManifestError(f"{recording_id}: candidate rallies must be ordered and non-overlapping")
        serve_confidence = row.get("serveConfidence")
        end_confidence = row.get("endConfidence")
        if serve_confidence not in CONFIDENCE_VALUES or end_confidence not in CONFIDENCE_VALUES:
            raise ManifestError(f"{where} confidence must be high, medium, or low")
        notes = row.get("notes", "")
        if not isinstance(notes, str):
            raise ManifestError(f"{where}.notes must be a string")
        events.append(
            {
                "start": start_value,
                "end": end_value,
                "tags": [
                    "ai-prelabel",
                    f"serve-confidence:{serve_confidence}",
                    f"end-confidence:{end_confidence}",
                ],
                "notes": notes,
            }
        )
        previous_end = end_value
    return events


def materialize_model_prelabels(
    candidates_directory: str | Path,
    tasks_directory: str | Path,
    output_directory: str | Path,
    *,
    analysis_method: str = METHOD_ID,
) -> dict[str, Any]:
    if analysis_method not in METHOD_ANNOTATORS:
        raise ManifestError(
            "unsupported model prelabel analysisMethod: "
            f"{analysis_method!r}; expected one of {sorted(METHOD_ANNOTATORS)}"
        )
    candidates_root = Path(candidates_directory).expanduser().resolve()
    tasks_root = Path(tasks_directory).expanduser().resolve()
    output_root = Path(output_directory).expanduser().resolve()
    candidate_paths = sorted(candidates_root.glob("*.candidates.json"))
    if not candidate_paths:
        raise ManifestError(f"no *.candidates.json files found in {candidates_root}")

    created = 0
    reused = 0
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for candidate_path in candidate_paths:
        candidate = _read_json(candidate_path, "model candidate")
        if candidate.get("schemaVersion") != 1:
            raise ManifestError(f"candidate schemaVersion must be 1: {candidate_path}")
        recording_id = candidate.get("recordingId")
        if (
            not isinstance(recording_id, str)
            or not recording_id
            or not all(character.isalnum() or character in "_-" for character in recording_id)
        ):
            raise ManifestError(f"candidate recordingId is invalid: {candidate_path}")
        if recording_id in seen_ids:
            raise ManifestError(f"duplicate candidate recordingId: {recording_id}")
        seen_ids.add(recording_id)
        if candidate.get("analysisMethod") != analysis_method:
            raise ManifestError(
                f"{recording_id}: analysisMethod must be {analysis_method!r}"
            )
        analyzed_at = candidate.get("analyzedAt")
        if not isinstance(analyzed_at, str) or not analyzed_at.strip():
            raise ManifestError(f"{recording_id}: analyzedAt must be a non-empty string")
        ambiguities = candidate.get("ambiguities", [])
        if not isinstance(ambiguities, list):
            raise ManifestError(f"{recording_id}: ambiguities must be an array")

        task_path = tasks_root / f"{recording_id}.labels.json"
        task = load_label_document(task_path, require_complete=False, require_video=False)
        video_path_value = candidate.get("videoPath")
        if not isinstance(video_path_value, str) or not video_path_value:
            raise ManifestError(f"{recording_id}: videoPath must be a non-empty path")
        candidate_video = Path(video_path_value).expanduser().resolve()
        if candidate_video != task.video:
            raise ManifestError(f"{recording_id}: candidate videoPath does not match its task")
        duration_value = candidate.get("durationSeconds")
        if (
            not isinstance(duration_value, (int, float))
            or isinstance(duration_value, bool)
            or not math.isfinite(float(duration_value))
            or abs(float(duration_value) - task.duration) > 0.1
        ):
            raise ManifestError(f"{recording_id}: candidate duration does not match its task")

        rallies = _candidate_events(
            candidate.get("events"),
            duration=task.duration,
            recording_id=recording_id,
        )
        payload = deepcopy(task.payload)
        payload["annotation"] = {
            "status": "in-progress",
            "annotator": METHOD_ANNOTATORS[analysis_method],
            "continuousVideoReviewed": False,
            "reviewedAt": None,
            "notes": (
                "Blind audiovisual AI prelabel. Validate every serve-contact and dead-ball "
                "boundary before marking this recording complete."
            ),
        }
        payload["rallies"] = rallies
        payload["ignoredIntervals"] = []
        payload["hardNegatives"] = []
        payload["prelabel"] = {
            "analysisMethod": analysis_method,
            "candidateFile": str(candidate_path),
            "analyzedAt": analyzed_at,
            "ambiguities": ambiguities,
        }

        output_path = output_root / f"{recording_id}.labels.json"
        rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
        if output_path.exists():
            if output_path.read_text(encoding="utf-8") != rendered:
                raise ManifestError(f"refusing to overwrite changed prelabel: {output_path}")
            reused += 1
        else:
            atomic_write_text(output_path, rendered)
            created += 1
        load_label_document(output_path, require_complete=False, require_video=False)
        items.append(
            {
                "recordingId": recording_id,
                "candidate": str(candidate_path),
                "prelabel": str(output_path),
                "rallies": len(rallies),
            }
        )

    return {
        "candidatesDirectory": str(candidates_root),
        "tasksDirectory": str(tasks_root),
        "outputDirectory": str(output_root),
        "analysisMethod": analysis_method,
        "annotator": METHOD_ANNOTATORS[analysis_method],
        "created": created,
        "reused": reused,
        "recordings": len(items),
        "rallies": sum(item["rallies"] for item in items),
        "items": items,
    }

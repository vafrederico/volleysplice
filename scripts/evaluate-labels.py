#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from analysis.evaluation import (
    Interval,
    aggregate_evaluations,
    evaluate_intervals,
    interval_iou,
)
from analysis.rallies import DetectionSettings, detect_rallies


SEGMENT_PATTERN = re.compile(r"(?P<youtube>[A-Za-z0-9_-]{11})-start(?P<start>\d+(?:\.\d+)?)-duration(?P<duration>\d+(?:\.\d+)?)\.mp4$")


@dataclass
class Recording:
    id: str
    environment: str
    source_group: str
    split: str
    duration: float
    offset: float
    truth: list[Interval]
    tags: list[list[str]]
    signals: list[dict[str, float]]
    camera_stability: float
    baseline_predictions: list[Interval]
    label_path: str


def load_recordings(
    labels_path: Path,
    analyses_root: Path,
    analysis_suffix: str = "",
) -> list[Recording]:
    if labels_path.is_dir():
        label_files = sorted(labels_path.glob("*.labels.json"))
        if not label_files:
            raise ValueError(f"label directory contains no *.labels.json files: {labels_path}")
    elif labels_path.is_file():
        label_files = [labels_path]
    else:
        raise ValueError(f"labels path does not exist: {labels_path}")

    recordings: list[Recording] = []
    for label_file in label_files:
        payload = json.loads(label_file.read_text(encoding="utf-8"))
        if payload.get("schemaVersion") != 1:
            raise ValueError(f"unsupported label schema in {label_file}")
        if payload.get("kind") == "volleycut-rally-labels":
            raw_recordings = [{**payload["recording"], "rallies": payload.get("rallies", [])}]
            full_recording = True
        else:
            raw_recordings = payload.get("recordings")
            full_recording = False
            if not isinstance(raw_recordings, list):
                raise ValueError(
                    f"labels must be a rally-label document or contain a recordings list: {label_file}"
                )

        for raw in raw_recordings:
            if full_recording:
                recording_id = str(raw["id"])
                analysis_id = recording_id.removesuffix("-full")
                offset = 0.0
                duration = float(raw["durationSeconds"])
            else:
                segment = SEGMENT_PATTERN.search(Path(str(raw.get("video", ""))).name)
                if not segment:
                    raise ValueError(
                        f"could not determine source segment for {raw.get('id', '<unknown>')}"
                    )
                recording_id = str(raw["id"])
                offset = float(segment.group("start"))
                duration = float(segment.group("duration"))
                analysis_id = f"{raw['environment']}-{segment.group('youtube')}"

            analysis_path = analyses_root / f"{analysis_id}{analysis_suffix}" / "analysis.json"
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            truth = [
                Interval(float(item["start"]), float(item["end"]))
                for item in raw.get("rallies", [])
            ]
            tags = [list(item.get("tags", [])) for item in raw.get("rallies", [])]
            predictions: list[Interval] = []
            for item in analysis.get("rallies", []):
                start = max(offset, float(item["start"]))
                end = min(offset + duration, float(item["end"]))
                if end > start:
                    predictions.append(Interval(start - offset, end - offset))
            signals = [
                {**sample, "time": float(sample["time"]) - offset}
                for sample in analysis.get("signals", [])
                if offset <= float(sample["time"]) < offset + duration
            ]
            recordings.append(Recording(
                id=recording_id,
                environment=str(raw["environment"]),
                source_group=str(raw["sourceGroup"]),
                split=str(raw.get("split", "unspecified")),
                duration=duration,
                offset=offset,
                truth=truth,
                tags=tags,
                signals=signals,
                camera_stability=float(analysis["analysis"]["cameraStability"]),
                baseline_predictions=predictions,
                label_path=str(label_file.resolve()),
            ))
    return recordings


def evaluate_recording(recording: Recording, predictions: list[Interval]) -> dict[str, Any]:
    result = evaluate_intervals(recording.truth, predictions)
    result.update({
        "id": recording.id,
        "environment": recording.environment,
        "sourceGroup": recording.source_group,
        "split": recording.split,
        "labelPath": recording.label_path,
        "segmentOffsetSeconds": recording.offset,
        "segmentDurationSeconds": recording.duration,
        "truth": [asdict(item) for item in recording.truth],
        "predictions": [asdict(item) for item in predictions],
    })
    return result


def grouped_summary(per_recording: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = sorted({str(item[key]) for item in per_recording})
    return {
        value: aggregate_evaluations([item for item in per_recording if str(item[key]) == value])
        for value in values
    }


def truth_slice_summary(recordings: list[Recording], evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    slices: dict[str, list[dict[str, Any]]] = {
        "under3Seconds": [],
        "atLeast3Seconds": [],
        "taggedAceOrServiceFault": [],
        "untagged": [],
    }
    for recording, evaluation in zip(recordings, evaluations, strict=True):
        matched_truth = {int(item["truthIndex"]) for item in evaluation["matches"]}
        for index, truth in enumerate(recording.truth):
            overlaps = [
                max(0.0, min(truth.end, prediction.end) - max(truth.start, prediction.start))
                for prediction in recording.baseline_predictions
            ]
            best_ious = [interval_iou(truth, prediction) for prediction in recording.baseline_predictions]
            detail = {
                "durationSeconds": truth.end - truth.start,
                "anyOverlap": max(overlaps, default=0) > 0,
                "coverage": max(overlaps, default=0) / (truth.end - truth.start),
                "strictMatch": index in matched_truth,
                "bestIoU": max(best_ious, default=0),
            }
            slices["under3Seconds" if truth.end - truth.start < 3 else "atLeast3Seconds"].append(detail)
            slices["taggedAceOrServiceFault" if recording.tags[index] else "untagged"].append(detail)

    summaries: dict[str, Any] = {}
    for name, values in slices.items():
        if not values:
            summaries[name] = {"rallies": 0}
            continue
        summaries[name] = {
            "rallies": len(values),
            "anyOverlapRate": sum(item["anyOverlap"] for item in values) / len(values),
            "meanCoverage": sum(float(item["coverage"]) for item in values) / len(values),
            "strictMatchRate": sum(item["strictMatch"] for item in values) / len(values),
            "meanBestIoU": sum(float(item["bestIoU"]) for item in values) / len(values),
        }
    return summaries


def predictions_for_settings(recording: Recording, settings: DetectionSettings) -> list[Interval]:
    return [
        Interval(float(item["start"]), float(item["end"]))
        for item in detect_rallies(
            recording.signals,
            recording.duration,
            recording.camera_stability,
            settings,
        )
    ]


def setting_grid() -> list[DetectionSettings]:
    settings: list[DetectionSettings] = []
    for high in (0.4, 0.5, 0.6, 0.7):
        for low in (0.2, 0.3, 0.4, 0.5, 0.6):
            if low >= high:
                continue
            for gap, minimum, lead, tail in itertools.product(
                (0.25, 0.75, 1.5, 2.5),
                (0.5, 1.5, 3.0),
                (0.0, 0.25, 0.75),
                (0.0, 0.5, 1.0),
            ):
                settings.append(DetectionSettings(
                    high_threshold=high,
                    low_threshold=low,
                    max_gap_seconds=gap,
                    min_rally_seconds=minimum,
                    onset_lead_seconds=lead,
                    ending_tail_seconds=tail,
                ))
    return settings


def parameter_search(recordings: list[Recording]) -> dict[str, Any]:
    configurations = setting_grid()
    per_configuration: list[list[dict[str, Any]]] = []
    aggregate: list[dict[str, Any]] = []
    for settings in configurations:
        evaluations = [
            evaluate_intervals(recording.truth, predictions_for_settings(recording, settings))
            for recording in recordings
        ]
        per_configuration.append(evaluations)
        aggregate.append(aggregate_evaluations(evaluations))

    def selection_key(index: int, result: dict[str, Any]) -> tuple[float, ...]:
        settings = configurations[index]
        return (
            float(result["timeIoU"]),
            float(result["eventF1"]),
            float(result["liveTimeRecall"]),
            settings.min_rally_seconds,
        )

    qualified = [index for index, result in enumerate(aggregate) if result["liveTimeRecall"] >= 0.9]
    if not qualified:
        qualified = list(range(len(configurations)))
    best_index = max(qualified, key=lambda index: selection_key(index, aggregate[index]))

    fold_results: list[dict[str, Any]] = []
    held_out_evaluations: list[dict[str, Any]] = []
    for source_group in sorted({recording.source_group for recording in recordings}):
        train_indexes = [index for index, item in enumerate(recordings) if item.source_group != source_group]
        test_indexes = [index for index, item in enumerate(recordings) if item.source_group == source_group]
        candidates: list[tuple[int, dict[str, Any]]] = []
        for index, evaluations in enumerate(per_configuration):
            training = aggregate_evaluations([evaluations[item] for item in train_indexes])
            if training["liveTimeRecall"] >= 0.9:
                candidates.append((index, training))
        if not candidates:
            candidates = [
                (index, aggregate_evaluations([evaluations[item] for item in train_indexes]))
                for index, evaluations in enumerate(per_configuration)
            ]
        selected_index, training = max(candidates, key=lambda item: selection_key(item[0], item[1]))
        held_out = aggregate_evaluations([
            per_configuration[selected_index][item] for item in test_indexes
        ])
        held_out_evaluations.extend(per_configuration[selected_index][item] for item in test_indexes)
        fold_results.append({
            "heldOutSourceGroup": source_group,
            "settings": asdict(configurations[selected_index]),
            "training": training,
            "heldOut": held_out,
        })

    ablation_settings = {
        "currentRedecoded": DetectionSettings(),
        "noEndingTail": DetectionSettings(ending_tail_seconds=0),
        "shortGap": DetectionSettings(max_gap_seconds=0.75),
        "shortMinimum": DetectionSettings(min_rally_seconds=0.5),
        "highSeedShortGap": DetectionSettings(high_threshold=0.7, max_gap_seconds=0.75),
        "selectedCombined": configurations[best_index],
    }
    ablations = {}
    for name, settings in ablation_settings.items():
        evaluations = [
            evaluate_intervals(recording.truth, predictions_for_settings(recording, settings))
            for recording in recordings
        ]
        ablations[name] = {
            "settings": asdict(settings),
            "aggregate": aggregate_evaluations(evaluations),
        }

    return {
        "gridConfigurations": len(configurations),
        "selectionRule": "Require >=90% training live-time recall, then maximize time IoU and event F1.",
        "tieBreak": "Prefer the existing 3-second minimum when metrics are identical; minimum duration alone was immaterial on this pilot.",
        "sameLabelExploratoryBest": {
            "warning": "Selected and scored on the same nine segments; this is an optimistic diagnostic, not an accuracy estimate.",
            "settings": asdict(configurations[best_index]),
            "aggregate": aggregate[best_index],
        },
        "leaveOneSourceGroupOut": {
            "aggregate": aggregate_evaluations(held_out_evaluations),
            "folds": fold_results,
        },
        "ablations": ablations,
    }


def build_parser() -> argparse.ArgumentParser:
    configured_root = Path(os.environ.get("VOLLEYCUT_DATA_ROOT", "data")).expanduser()
    parser = argparse.ArgumentParser(description="Evaluate generated analysis intervals against a labeled segment manifest.")
    parser.add_argument("--labels", type=Path, required=True, help="Gold label manifest built from completed labels")
    parser.add_argument("--analyses-root", type=Path, default=configured_root / "analyses")
    parser.add_argument(
        "--analysis-suffix",
        default="",
        help="Suffix appended to mapped analysis IDs, for example -v2",
    )
    parser.add_argument("--output", type=Path, help="Optional immutable JSON report destination")
    parser.add_argument("--parameter-search", action="store_true", help="Run a diagnostic decoder grid and source-group CV")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.analysis_suffix and not re.fullmatch(
        r"-[A-Za-z0-9][A-Za-z0-9_-]{0,19}",
        arguments.analysis_suffix,
    ):
        raise ValueError("--analysis-suffix must be empty or a safe hyphen-prefixed suffix")
    recordings = load_recordings(
        arguments.labels.resolve(),
        arguments.analyses_root.resolve(),
        arguments.analysis_suffix,
    )
    baseline = [evaluate_recording(item, item.baseline_predictions) for item in recordings]
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "createdAt": datetime.now(UTC).isoformat(),
        "labels": str(arguments.labels.resolve()),
        "analysesRoot": str(arguments.analyses_root.resolve()),
        "analysisSuffix": arguments.analysis_suffix,
        "matching": {"minimumIntervalIoU": 0.5},
        "baseline": {
            "aggregate": aggregate_evaluations(baseline),
            "byEnvironment": grouped_summary(baseline, "environment"),
            "bySplit": grouped_summary(baseline, "split"),
            "truthSlices": truth_slice_summary(recordings, baseline),
            "recordings": baseline,
        },
    }
    if arguments.parameter_search:
        report["parameterSearch"] = parameter_search(recordings)

    rendered = json.dumps(report, indent=2) + "\n"
    if arguments.output:
        output = arguments.output.resolve()
        if output.exists():
            raise ValueError(f"report already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
        print(output)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

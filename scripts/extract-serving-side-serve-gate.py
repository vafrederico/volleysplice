#!/usr/bin/env python3
"""Run both frozen production serve heads at every serving-side candidate anchor."""

from __future__ import annotations

import argparse
import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.features import FeatureSequence, VideoMetadata, contextualize
from analysis.production_serve_gate import (
    ANCHOR_WINDOW_SECONDS,
    GATE_AGGREGATION,
    GATE_VERSION,
    HYBRID_GATE_AGGREGATION,
    HYBRID_GATE_VERSION,
    ProductionRallyInterval,
    anchor_evidence,
    dual_head_prediction,
    gate_fingerprint,
    hybrid_gate_fingerprint,
    hybrid_gate_prediction,
    infer_head,
    load_production_serve_head,
    merge_production_rally_intervals,
    production_rally_anchor_evidence,
    sha256,
)
from analysis.serving_side_exclusions import (
    load_source_quality_exclusions,
    samples_touch_source_exclusion,
)


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
DEFAULT_REPORT = (
    ROOT
    / "reports/serving-side/serving-side-existing-label-variants-full-nas-v2.json"
)
DEFAULT_FEATURE_CACHE = Path(
    "/mnt/freenas/volleycut/intake-2026-08-13/experiments/"
    "environment-specialists-v2/features/audiovisual-audio-normalized-v3"
)
DEFAULT_FEEDBACK_ROOT = Path("/mnt/freenas/volleycut/model-feedback")
DEFAULT_ALL_LABELS_MODEL = Path("prod/public/runtime/model-1ca43e38eefc.json")
DEFAULT_PREVIOUS_MODEL = Path("prod/public/runtime/model-9c92b8e9333f.json")
DEFAULT_ALL_LABELS_ANALYSES = Path("/mnt/freenas/volleycut/intake-2026-08-13/analyses")
DEFAULT_PREVIOUS_ANALYSES = Path(
    "/mnt/freenas/volleycut/labeling-v1-2026-08-09-no-beach-2026-08-12/analyses"
)
DEFAULT_OUTPUT = ROOT / "features/serving-side-serve-gate-v2/all-reviewed.json"
DEFAULT_SOURCE_EXCLUSIONS = (
    ROOT
    / "reports/serving-side/serving-side-source-quality-exclusions-v1.json"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _decode_array(value: Any, label: str) -> np.ndarray:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not an encoded array")
    data_type = value.get("dataType")
    shape = value.get("shape")
    if (
        value.get("encoding") != "base64"
        or value.get("byteOrder") != "little-endian"
        or data_type not in {"float32", "float64"}
        or not isinstance(shape, list)
        or not shape
        or not all(isinstance(item, int) and item >= 0 for item in shape)
        or not isinstance(value.get("data"), str)
    ):
        raise ValueError(f"{label} has an invalid encoded-array contract")
    raw = base64.b64decode(value["data"], validate=True)
    dtype = "<f4" if data_type == "float32" else "<f8"
    expected = int(np.prod(shape))
    result = np.frombuffer(raw, dtype=dtype)
    if result.size != expected:
        raise ValueError(f"{label} byte length does not match its shape")
    return result.reshape(shape).copy()


def _feedback_by_recording(
    feedback_root: Path, recording_ids: set[str]
) -> dict[str, tuple[Path, Mapping[str, Any]]]:
    result: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for path in sorted(feedback_root.glob("*/bundle.json")):
        payload = _load(path)
        source = payload.get("source")
        file = source.get("file") if isinstance(source, Mapping) else None
        name = file.get("name") if isinstance(file, Mapping) else None
        if not isinstance(name, str):
            continue
        stem = Path(name).stem
        matches = [item for item in recording_ids if item == stem or item.endswith(stem)]
        if len(matches) == 1:
            result[matches[0]] = (path, payload)
    return result


def _feature_sequence(
    recording_id: str,
    duration: float,
    cache_root: Path,
    feedback: Mapping[str, tuple[Path, Mapping[str, Any]]],
) -> tuple[FeatureSequence, dict[str, str]]:
    caches = sorted(cache_root.glob(f"{recording_id}-*.npz"))
    if len(caches) == 1:
        source_path = caches[0]
        with np.load(source_path, allow_pickle=False) as cached:
            times = np.ascontiguousarray(cached["times"], dtype=np.float64)
            values = np.ascontiguousarray(cached["values"], dtype=np.float32)
            names = tuple(str(item) for item in cached["names"])
        source_kind = "canonical-feature-cache"
    elif len(caches) > 1:
        raise ValueError(f"multiple canonical feature caches found for {recording_id}")
    else:
        source = feedback.get(recording_id)
        if source is None:
            raise FileNotFoundError(f"no base features found for {recording_id}")
        source_path, payload = source
        features = payload.get("features")
        if not isinstance(features, Mapping) or not isinstance(
            features.get("names"), list
        ):
            raise ValueError(f"model feedback has no base features: {source_path}")
        times = np.ascontiguousarray(
            _decode_array(features.get("timestamps"), "feature timestamps"),
            dtype=np.float64,
        )
        values = np.ascontiguousarray(
            _decode_array(features.get("values"), "feature values"),
            dtype=np.float32,
        )
        names = tuple(str(item) for item in features["names"])
        source_kind = "model-feedback-feature-matrix"
    if (
        times.ndim != 1
        or values.shape != (len(times), len(names))
        or not len(times)
        or not np.isfinite(times).all()
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"base features are malformed for {recording_id}")
    sequence = FeatureSequence(
        times=times,
        values=values,
        names=names,
        metadata=VideoMetadata(
            duration=duration,
            width=192,
            height=108,
            fps=4.0,
            frame_count=len(times),
            has_audio=True,
        ),
    )
    return sequence, {
        "kind": source_kind,
        "path": str(source_path.resolve()),
        "sha256": sha256(source_path),
    }


def _production_rally_intervals(
    recording_id: str,
    file: Mapping[str, Any],
    all_labels_analyses: Path,
    previous_analyses: Path,
) -> tuple[tuple[ProductionRallyInterval, ...], list[dict[str, str]]]:
    label_path_value = file.get("path")
    if not isinstance(label_path_value, str) or not label_path_value:
        raise ValueError(f"label source path is unavailable for {recording_id}")
    label_path = Path(label_path_value).expanduser().resolve()
    if label_path.name.endswith(".model-feedback.json"):
        payload = _load(label_path)
        inference = payload.get("initialInference")
        ranges = inference.get("ranges") if isinstance(inference, Mapping) else None
        if not isinstance(ranges, list) or not all(
            isinstance(item, Mapping) for item in ranges
        ):
            raise ValueError(f"production ensemble ranges are unavailable for {recording_id}")
        intervals: list[ProductionRallyInterval] = []
        for item in ranges:
            start = item.get("start")
            end = item.get("end")
            agreement = item.get("agreement")
            if (
                not isinstance(start, (int, float))
                or not isinstance(end, (int, float))
                or agreement
                not in {
                    "both-models",
                    "all-labels-v2-only",
                    "previous-production-only",
                }
                or float(end) <= float(start)
            ):
                raise ValueError(
                    f"production ensemble range is malformed for {recording_id}"
                )
            intervals.append(
                ProductionRallyInterval(float(start), float(end), str(agreement))
            )
        return tuple(intervals), [
            {
                "kind": "model-feedback-production-ensemble",
                "path": str(label_path),
                "sha256": sha256(label_path),
            }
        ]

    all_labels_path = (
        all_labels_analyses / f"model-1ca43e38eefc--{recording_id}" / "analysis.json"
    ).resolve()
    previous_path = (
        previous_analyses / f"model-9c92b8e9333f--{recording_id}" / "analysis.json"
    ).resolve()
    all_labels = _load(all_labels_path)
    previous = _load(previous_path)
    for label, payload in (("all-labels V2", all_labels), ("previous", previous)):
        if payload.get("recordingId") != recording_id or not isinstance(
            payload.get("rallies"), list
        ):
            raise ValueError(f"{label} production analysis is invalid for {recording_id}")
    intervals = merge_production_rally_intervals(
        all_labels["rallies"], previous["rallies"]
    )
    return intervals, [
        {
            "kind": "all-labels-v2-analysis",
            "path": str(all_labels_path),
            "sha256": sha256(all_labels_path),
        },
        {
            "kind": "previous-production-analysis",
            "path": str(previous_path),
            "sha256": sha256(previous_path),
        },
    ]


def extract(args: argparse.Namespace) -> Mapping[str, Any]:
    report_path = args.serving_report.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite serve-gate evidence: {output_path}")
    report = _load(report_path)
    source_exclusions_path = args.source_exclusions.resolve()
    source_exclusions = load_source_quality_exclusions(source_exclusions_path)
    rallies = report.get("rallies")
    labels = report.get("labels")
    files = labels.get("files") if isinstance(labels, Mapping) else None
    if not isinstance(rallies, list) or not isinstance(files, list):
        raise ValueError("serving-side report has an invalid schema")
    recording_ids = {
        str(item.get("recordingId")) for item in files if isinstance(item, Mapping)
    }
    metadata = {
        str(item["recordingId"]): item
        for item in files
        if isinstance(item, Mapping) and str(item.get("recordingId")) in recording_ids
    }
    heads = (
        load_production_serve_head(args.all_labels_model),
        load_production_serve_head(args.previous_model),
    )
    if (
        heads[0].feature_config != heads[1].feature_config
        or heads[0].feature_names != heads[1].feature_names
    ):
        raise ValueError("production serve heads use different feature contracts")
    feedback = _feedback_by_recording(args.feedback_root.resolve(), recording_ids)
    rows: list[dict[str, Any]] = []
    feature_sources: list[dict[str, str]] = []
    rally_sources: list[dict[str, Any]] = []
    for recording_id in sorted(recording_ids):
        file = metadata[recording_id]
        duration = float(file["durationSeconds"])
        sequence, feature_source = _feature_sequence(
            recording_id,
            duration,
            args.feature_cache.resolve(),
            feedback,
        )
        contextual_values, contextual_names = contextualize(
            sequence, heads[0].feature_config
        )
        if contextual_names != heads[0].feature_names:
            raise ValueError(f"feature signature differs for {recording_id}")
        outputs = [
            infer_head(
                head,
                sequence.times,
                contextual_values,
                duration=duration,
            )
            for head in heads
        ]
        production_intervals, production_sources = _production_rally_intervals(
            recording_id,
            file,
            args.all_labels_analyses.resolve(),
            args.previous_analyses.resolve(),
        )
        selected = [
            rally
            for rally in rallies
            if isinstance(rally, Mapping) and rally.get("recordingId") == recording_id
            and not samples_touch_source_exclusion(
                source_exclusions,
                recording_id,
                [float(rally["start"])],
            )
        ]
        for rally in selected:
            anchor = float(rally["start"])
            evidence = [
                anchor_evidence(
                    head,
                    sequence.times,
                    probabilities,
                    detections,
                    anchor,
                )
                for head, (probabilities, detections) in zip(
                    heads, outputs, strict=True
                )
            ]
            rally_evidence = production_rally_anchor_evidence(
                production_intervals, anchor
            )
            prediction, decision_source, review_recommended = hybrid_gate_prediction(
                evidence, rally_evidence
            )
            rows.append(
                {
                    "rallyId": rally["rallyId"],
                    "recordingId": recording_id,
                    "serveAnchor": anchor,
                    "headPrediction": dual_head_prediction(evidence),
                    "prediction": prediction,
                    "decisionSource": decision_source,
                    "reviewRecommended": review_recommended,
                    "heads": {
                        "allLabelsV2": evidence[0].to_dict(anchor),
                        "previousProduction": evidence[1].to_dict(anchor),
                    },
                    "productionRally": rally_evidence.to_dict(),
                }
            )
        feature_sources.append({"recordingId": recording_id, **feature_source})
        rally_sources.append(
            {"recordingId": recording_id, "sources": production_sources}
        )
        print(f"{recording_id}: {len(selected)} candidates", flush=True)
    source_quality_excluded = len(rallies) - len(rows)
    prediction_counts = {
        decision: sum(row["prediction"] == decision for row in rows)
        for decision in ("serve", "not-serve")
    }
    decision_source_counts = {
        source: sum(row["decisionSource"] == source for row in rows)
        for source in ("serve-head", "production-rally-recovery", "none")
    }
    payload = {
        "schemaVersion": 1,
        "kind": "volleycut-serving-side-production-serve-gate-v2",
        "createdAt": datetime.now(UTC).isoformat(),
        "gateFingerprint": hybrid_gate_fingerprint(heads),
        "serveHeadGateFingerprint": gate_fingerprint(heads),
        "gate": {
            "version": HYBRID_GATE_VERSION,
            "anchorWindowSeconds": ANCHOR_WINDOW_SECONDS,
            "aggregation": HYBRID_GATE_AGGREGATION,
            "serveHeadVersion": GATE_VERSION,
            "serveHeadAggregation": GATE_AGGREGATION,
            "reviewRequiredForRallyRecovery": True,
            "meaning": "serve when either production serve head reaches its deployed threshold inside the source-aligned anchor window, or recover for review when the anchor is contained in a both-model production rally interval",
        },
        "models": {
            "allLabelsV2": {
                "modelId": heads[0].model_id,
                "bundleSha256": heads[0].bundle_sha256,
                "threshold": heads[0].decoder.threshold,
            },
            "previousProduction": {
                "modelId": heads[1].model_id,
                "bundleSha256": heads[1].bundle_sha256,
                "threshold": heads[1].decoder.threshold,
            },
        },
        "featurePipeline": {
            "featureConfig": heads[0].feature_config.to_dict(),
            "contextualFeatureNames": list(heads[0].feature_names),
            "sources": feature_sources,
        },
        "productionRallyPipeline": {
            "algorithm": "overlap-union-disagreement-v1",
            "recoveryRule": "anchor-contained-and-both-models",
            "sources": rally_sources,
        },
        "counts": {
            "rows": len(rows),
            "recordings": len(recording_ids),
            "sourceQualityExcluded": source_quality_excluded,
            **prediction_counts,
            "serveHeadPasses": decision_source_counts["serve-head"],
            "productionRallyRecoveries": decision_source_counts[
                "production-rally-recovery"
            ],
            "noServeEvidence": decision_source_counts["none"],
        },
        "rows": rows,
        "sources": {
            "servingSideReport": {
                "path": str(report_path),
                "sha256": sha256(report_path),
            },
            "sourceQualityExclusions": {
                "path": str(source_exclusions_path),
                "sha256": sha256(source_exclusions_path),
            },
            "productionModels": {
                "allLabelsV2": str(args.all_labels_model.resolve()),
                "previousProduction": str(args.previous_model.resolve()),
            },
            "implementation": [
                {
                    "path": str(path.relative_to(REPOSITORY_ROOT)),
                    "sha256": sha256(path),
                }
                for path in (
                    REPOSITORY_ROOT / "analysis/production_serve_gate.py",
                    Path(__file__).resolve(),
                )
            ],
        },
        "dataPolicy": {
            "labelsUsedForGateSelection": True,
            "protectedTestUsedForGateSelection": True,
            "thresholds": "unchanged deployed thresholds from each production serve head",
            "selectionScope": "assisted serving-side review workflow; chosen after all-video current-label diagnostics and not a production rally-model promotion",
            "productionRallyRecovery": "requires anchor containment in a both-model interval and always requests human review",
            "coverage": "every candidate in the serving-side report outside source-quality exclusions, including unclear rows",
        },
    }
    atomic_write_text(output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"wrote {output_path} ({len(rows)} rows)")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--feedback-root", type=Path, default=DEFAULT_FEEDBACK_ROOT)
    parser.add_argument(
        "--all-labels-analyses", type=Path, default=DEFAULT_ALL_LABELS_ANALYSES
    )
    parser.add_argument(
        "--previous-analyses", type=Path, default=DEFAULT_PREVIOUS_ANALYSES
    )
    parser.add_argument(
        "--source-exclusions", type=Path, default=DEFAULT_SOURCE_EXCLUSIONS
    )
    parser.add_argument("--all-labels-model", type=Path, default=DEFAULT_ALL_LABELS_MODEL)
    parser.add_argument("--previous-model", type=Path, default=DEFAULT_PREVIOUS_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


if __name__ == "__main__":
    extract(_parser().parse_args())

#!/usr/bin/env python3
"""Train and evaluate a nested-LOO rare-event ranker over the full candidate union."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.side_switch_continuity import (
    apply_veto,
    build_verifier_rows,
    select_veto_threshold,
)
from analysis.side_switch_full_union_ranker import (
    DERIVED_FEATURE_NAMES,
    UnionDecoderSettings,
    add_derived_features,
    decode_ranked_candidates,
    predict_classifier,
)
from analysis.side_switch_full_video import (
    event_metric_counts,
    monotonic_interval_match,
)
from analysis.side_switch_peak_cleanup import combine_soft_context
from analysis.side_switch_production_state import STATE_GATE_FEATURE_NAMES
from analysis.side_switch_v3 import V3Event, average_precision
from analysis.side_switch_v5 import VISUAL_FEATURE_NAMES
from analysis.side_switch_v6 import fit_model, grouped_cross_fit, matrix_for


ROOT = Path("/mnt/freenas/volleycut/labeling-v1-2026-08-09")
REPORTS = ROOT / "reports/side-switch"
DEFAULT_FEATURES = REPORTS / "side-switch-full-union-v5-state-features-v1.json"
DEFAULT_FULL_AUDIT = REPORTS / "side-switch-full-video-marker-evaluation-2026-08-21-r2.json"
DEFAULT_WINNER = ROOT / "models/side-switch-v5-peak-cleanup-v1/model.json"
DEFAULT_MODEL = ROOT / "models/side-switch-full-union-ranker-v1/model.json"
DEFAULT_EVALUATION = REPORTS / "side-switch-full-union-ranker-v1-evaluation.json"
EXPECTED_SHA256 = {
    "features": "9763cb3e5cd9baada64f4bf54f06140dcff5068bb8cd74a1d485c677d1e6c551",
    "fullAudit": "142d6617aed7f20b51cdcdfc9b3e61c8c76beaf79a31fc2d7ee54fb0fdf27b89",
    "winner": "82e64c69564d17c3dfbea9c5f4a276cdd161a0c400507d3df7589c019e44ee3f",
}
PRIMARY_FEATURE_NAMES = (*VISUAL_FEATURE_NAMES, *STATE_GATE_FEATURE_NAMES)
FEATURE_GROUPS = {
    "frozen-primary32": PRIMARY_FEATURE_NAMES,
    "union-native34": (*PRIMARY_FEATURE_NAMES, *DERIVED_FEATURE_NAMES),
}
L2_GRID = (0.1, 1.0, 10.0)
DECODER_GRID = tuple(
    UnionDecoderSettings(
        minimum_index_separation=index_separation,
        minimum_time_separation_seconds=time_separation,
        free_predictions_per_recording=6,
        count_penalty_logit=penalty,
    )
    for index_separation, time_separation in (
        (0, 0.0),
        (2, 0.0),
        (0, 8.0),
        (0, 14.0),
        (0, 20.0),
    )
    for penalty in (0.0, 0.5)
)
PRIMARY_PADDING_SECONDS = 4.0
THRESHOLD_QUANTILES = 65


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


def _events(rows: Sequence[Mapping[str, Any]], labels: Mapping[str, int]) -> list[V3Event]:
    ordinal_by_id: dict[str, int] = {}
    for recording_id in sorted({str(row["recordingId"]) for row in rows}):
        recording_rows = sorted(
            (row for row in rows if str(row["recordingId"]) == recording_id),
            key=lambda value: (
                float(value["transitionTime"]),
                str(value["eventId"]),
            ),
        )
        ordinal_by_id.update(
            {str(row["eventId"]): index for index, row in enumerate(recording_rows, 1)}
        )
    result: list[V3Event] = []
    for row in rows:
        recording_id = str(row["recordingId"])
        event_id = str(row["eventId"])
        result.append(
            V3Event(
                event_id=event_id,
                recording_id=recording_id,
                role="opened-development",
                gap_order=ordinal_by_id[event_id],
                label=int(labels[event_id]),
                row=row,
            )
        )
    return result


def _proposal(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "eventId": str(row["eventId"]),
        "recordingId": str(row["recordingId"]),
        "kind": str(row["kind"]),
        "gapStart": float(row["gapStart"]),
        "gapEnd": float(row["gapEnd"]),
        "transitionTime": float(row["transitionTime"]),
        "score": float(row["score"]) if row.get("score") is not None else None,
    }


def _candidate_labels(
    rows: Sequence[Mapping[str, Any]],
    markers: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, int], dict[str, Any]]:
    labels = {str(row["eventId"]): 0 for row in rows}
    audit: dict[str, Any] = {}
    for recording_id, recording_markers in markers.items():
        recording_rows = [
            row for row in rows if str(row["recordingId"]) == recording_id
        ]
        proposals = [_proposal(row) for row in recording_rows]
        match = monotonic_interval_match(
            proposals, recording_markers, PRIMARY_PADDING_SECONDS
        )
        for pair in match.pairs:
            labels[str(recording_rows[pair.proposal_index]["eventId"])] = 1
        audit[recording_id] = {
            "candidates": len(recording_rows),
            "humanEvents": len(recording_markers),
            "positiveCandidates": len(match.pairs),
            "uncoveredHumanTimes": [
                float(recording_markers[index]["time"])
                for index in match.unmatched_marker_indices
            ],
        }
    return labels, audit


def _evaluate(
    rows: Sequence[Mapping[str, Any]],
    predictions: np.ndarray,
    markers: Mapping[str, Sequence[Mapping[str, Any]]],
    padding: float,
    *,
    include_inventory: bool = False,
) -> dict[str, Any]:
    selected = np.asarray(predictions, dtype=bool)
    if selected.shape != (len(rows),):
        raise ValueError("evaluation predictions are not aligned")
    per_video: dict[str, Any] = {}
    for recording_id, recording_markers in markers.items():
        proposals = [
            _proposal(row)
            for index, row in enumerate(rows)
            if selected[index] and str(row["recordingId"]) == recording_id
        ]
        proposals.sort(key=lambda value: (value["transitionTime"], value["eventId"]))
        match = monotonic_interval_match(proposals, recording_markers, padding)
        counts = event_metric_counts(match)
        per_video[recording_id] = {
            "humanEvents": len(recording_markers),
            "proposals": len(proposals),
            **counts,
            "matchedEventIds": [
                proposals[pair.proposal_index]["eventId"] for pair in match.pairs
            ],
            "falsePositiveEventIds": [
                proposals[index]["eventId"]
                for index in match.unmatched_proposal_indices
            ],
            "missedHumanTimes": [
                float(recording_markers[index]["time"])
                for index in match.unmatched_marker_indices
            ],
        }
        if include_inventory:
            per_video[recording_id]["proposalInventory"] = proposals
    true_positives = sum(int(value["truePositives"]) for value in per_video.values())
    false_positives = sum(int(value["falsePositives"]) for value in per_video.values())
    false_negatives = sum(int(value["falseNegatives"]) for value in per_video.values())
    proposals = true_positives + false_positives
    human_events = true_positives + false_negatives
    precision = true_positives / proposals if proposals else 0.0
    recall = true_positives / human_events if human_events else 0.0
    return {
        "paddingSeconds": padding,
        "recordings": len(per_video),
        "humanEvents": human_events,
        "proposals": proposals,
        "truePositives": true_positives,
        "falsePositives": false_positives,
        "falseNegatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "macroPerVideoPrecision": float(
            np.mean([float(value["precision"] or 0.0) for value in per_video.values()])
        ),
        "macroPerVideoRecall": float(
            np.mean([float(value["recall"] or 0.0) for value in per_video.values()])
        ),
        "averagePerVideo": {
            "proposals": proposals / len(per_video),
            "truePositives": true_positives / len(per_video),
            "falsePositives": false_positives / len(per_video),
            "falseNegatives": false_negatives / len(per_video),
        },
        "byRecording": per_video,
    }


def _thresholds(scores: np.ndarray) -> tuple[float, ...]:
    values = np.asarray(scores, dtype=np.float64)
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, THRESHOLD_QUANTILES))
    above = min(1.0, math.nextafter(float(np.max(values)), math.inf))
    return tuple(sorted({above, *(float(value) for value in quantiles)}, reverse=True))


def _selection_rank(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = candidate["metrics"]
    decoder = candidate["decoder"]
    complexity = sum(
        (
            int(decoder["minimumIndexSeparation"]) > 0,
            float(decoder["minimumTimeSeparationSeconds"]) > 0,
            float(decoder["countPenaltyLogit"]) > 0,
        )
    )
    return (
        float(metrics["f1"]),
        float(metrics["precision"]),
        float(metrics["recall"]),
        -float(metrics["proposals"]),
        -float(complexity),
        float(candidate["threshold"]),
        -float(decoder["countPenaltyLogit"]),
        -float(decoder["minimumTimeSeparationSeconds"]),
    )


def _select_decoder(
    rows: Sequence[Mapping[str, Any]],
    scores: np.ndarray,
    markers: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for settings in DECODER_GRID:
        for threshold in _thresholds(scores):
            predictions = decode_ranked_candidates(rows, scores, threshold, settings)
            candidate = {
                "threshold": threshold,
                "decoder": settings.to_dict(),
                "metrics": _evaluate(
                    rows, predictions, markers, PRIMARY_PADDING_SECONDS
                ),
            }
            if best is None or _selection_rank(candidate) > _selection_rank(best):
                best = candidate
    if best is None:
        raise AssertionError("decoder selection produced no result")
    return best


def _settings(payload: Mapping[str, Any]) -> UnionDecoderSettings:
    return UnionDecoderSettings(
        minimum_index_separation=int(payload["minimumIndexSeparation"]),
        minimum_time_separation_seconds=float(
            payload["minimumTimeSeparationSeconds"]
        ),
        free_predictions_per_recording=int(payload["freePredictionsPerRecording"]),
        count_penalty_logit=float(payload["countPenaltyLogit"]),
    )


def _variant_rank(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        *_selection_rank(candidate),
        float(candidate["rowAveragePrecision"] or 0.0),
        -float(candidate["l2"]),
        -float(len(candidate["featureNames"])),
    )


def _select_ranker(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, int],
    markers: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    events = _events(rows, labels)
    leaderboard: list[dict[str, Any]] = []
    scores_by_key: dict[tuple[str, float], np.ndarray] = {}
    for group_name, feature_names in FEATURE_GROUPS.items():
        for l2 in L2_GRID:
            scores, crossfit = grouped_cross_fit(events, l2, feature_names)
            scores_by_key[(group_name, l2)] = scores
            selected = _select_decoder(rows, scores, markers)
            leaderboard.append(
                {
                    "featureGroup": group_name,
                    "featureNames": list(feature_names),
                    "l2": l2,
                    "rowAveragePrecision": average_precision(
                        np.asarray([labels[str(row["eventId"])] for row in rows]),
                        scores,
                    ),
                    "crossFit": crossfit,
                    **selected,
                }
            )
    leaderboard.sort(key=_variant_rank, reverse=True)
    selected = leaderboard[0]
    scores = scores_by_key[(str(selected["featureGroup"]), float(selected["l2"]))]
    return selected, scores, leaderboard


def _selected_ids(rows: Sequence[Mapping[str, Any]], predictions: np.ndarray) -> set[str]:
    return {
        str(row["eventId"])
        for row, selected in zip(rows, predictions, strict=True)
        if selected
    }


def _continuity_rows(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, int],
    selected_ids: set[str],
) -> list[Any]:
    result = []
    for recording_id in sorted({str(row["recordingId"]) for row in rows}):
        recording_rows = [
            row for row in rows if str(row["recordingId"]) == recording_id
        ]
        result.extend(build_verifier_rows(recording_rows, labels, selected_ids))
    return result


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "features": args.features.expanduser().resolve(),
        "fullAudit": args.full_audit.expanduser().resolve(),
        "winner": args.winner.expanduser().resolve(),
    }
    model_path = args.model.expanduser().resolve()
    evaluation_path = args.evaluation.expanduser().resolve()
    if model_path.exists() or evaluation_path.exists():
        raise FileExistsError("refusing to overwrite full-union ranker artifacts")
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if args.enforce_source_hash and hashes != EXPECTED_SHA256:
        raise ValueError(f"full-union ranker source identity changed: {hashes}")
    features = _load(paths["features"])
    full_audit = _load(paths["fullAudit"])
    winner = _load(paths["winner"])
    rows = [add_derived_features(row) for row in features["rows"]]
    recording_ids = tuple(features["scope"]["recordingIds"])
    markers = {
        recording_id: [
            {"time": float(value)}
            for value in full_audit["scope"]["humanEventsByRecording"][recording_id]
        ]
        for recording_id in recording_ids
    }
    labels, label_audit = _candidate_labels(rows, markers)
    if sum(labels.values()) != 46:
        raise ValueError("full-union positive candidate count changed")

    outer_scores = np.full(len(rows), np.nan)
    outer_predictions = np.zeros(len(rows), dtype=bool)
    outer_veto_predictions = np.zeros(len(rows), dtype=bool)
    outer_folds: list[dict[str, Any]] = []
    for fold_number, held_id in enumerate(recording_ids, start=1):
        fit_indexes = [
            index for index, row in enumerate(rows) if str(row["recordingId"]) != held_id
        ]
        held_indexes = [
            index for index, row in enumerate(rows) if str(row["recordingId"]) == held_id
        ]
        fit_rows = [rows[index] for index in fit_indexes]
        held_rows = [rows[index] for index in held_indexes]
        fit_markers = {key: value for key, value in markers.items() if key != held_id}
        print(
            f"[{fold_number}/{len(recording_ids)}] nested LOO held={held_id} "
            f"fit={len(fit_rows)} held={len(held_rows)}",
            flush=True,
        )
        selected, fit_scores, leaderboard = _select_ranker(
            fit_rows, labels, fit_markers
        )
        decoder = _settings(selected["decoder"])
        fit_predictions = decode_ranked_candidates(
            fit_rows, fit_scores, float(selected["threshold"]), decoder
        )
        feature_names = tuple(selected["featureNames"])
        fit_events = _events(fit_rows, labels)
        held_events = _events(held_rows, labels)
        classifier = fit_model(fit_events, float(selected["l2"]), feature_names)
        held_scores = classifier.predict_proba(matrix_for(held_events, feature_names))
        held_predictions = decode_ranked_candidates(
            held_rows, held_scores, float(selected["threshold"]), decoder
        )
        outer_scores[held_indexes] = held_scores
        outer_predictions[held_indexes] = held_predictions

        fit_selected_ids = _selected_ids(fit_rows, fit_predictions)
        fit_verifier = _continuity_rows(fit_rows, labels, fit_selected_ids)
        veto = select_veto_threshold(
            fit_verifier,
            signal="player",
            minimum_quality=0.0,
            marker_count=sum(map(len, fit_markers.values())),
            minimum_true_positive_retention=0.90,
        )
        held_selected_ids = _selected_ids(held_rows, held_predictions)
        held_verifier = _continuity_rows(held_rows, labels, held_selected_ids)
        retained_ids = apply_veto(
            held_verifier,
            signal="player",
            threshold=float(veto["threshold"]),
            minimum_quality=float(veto["minimumQuality"]),
        )
        outer_veto_predictions[held_indexes] = np.asarray(
            [str(row["eventId"]) in retained_ids for row in held_rows]
        )
        outer_folds.append(
            {
                "heldRecordingId": held_id,
                "fitRows": len(fit_rows),
                "heldRows": len(held_rows),
                "selected": {
                    key: selected[key]
                    for key in (
                        "featureGroup",
                        "featureNames",
                        "l2",
                        "rowAveragePrecision",
                        "threshold",
                        "decoder",
                        "metrics",
                    )
                },
                "leaderboard": [
                    {
                        key: value
                        for key, value in candidate.items()
                        if key != "crossFit"
                    }
                    for candidate in leaderboard
                ],
                "continuityVeto": {
                    key: value
                    for key, value in veto.items()
                    if key != "retainedEventIds"
                },
            }
        )
    if not np.isfinite(outer_scores).all():
        raise ValueError("outer LOO left candidates unscored")

    selected_full, full_oof_scores, full_leaderboard = _select_ranker(
        rows, labels, markers
    )
    full_decoder = _settings(selected_full["decoder"])
    full_oof_predictions = decode_ranked_candidates(
        rows,
        full_oof_scores,
        float(selected_full["threshold"]),
        full_decoder,
    )
    full_selected_ids = _selected_ids(rows, full_oof_predictions)
    full_verifier = _continuity_rows(rows, labels, full_selected_ids)
    full_veto = select_veto_threshold(
        full_verifier,
        signal="player",
        minimum_quality=0.0,
        marker_count=sum(map(len, markers.values())),
        minimum_true_positive_retention=0.90,
    )
    final_events = _events(rows, labels)
    final_classifier = fit_model(
        final_events, float(selected_full["l2"]), tuple(selected_full["featureNames"])
    )

    frozen_primary = predict_classifier(winner["primaryClassifier"], rows)
    frozen_context = predict_classifier(winner["productionContextClassifier"], rows)
    frozen_context = np.asarray(
        [
            score
            if row["productionStateContext"][
                "eligibleForFrozenProductionContextHead"
            ]
            else 0.5
            for row, score in zip(rows, frozen_context, strict=True)
        ]
    )
    frozen_combined = combine_soft_context(
        frozen_primary,
        frozen_context,
        float(winner["decoder"]["productionContextWeight"]),
    )
    frozen_settings = UnionDecoderSettings(
        minimum_index_separation=int(winner["decoder"]["minimumGapSeparation"]),
        minimum_time_separation_seconds=float(
            winner["decoder"]["minimumTimeSeparationSeconds"]
        ),
        free_predictions_per_recording=int(
            winner["decoder"]["freePredictionsPerRecording"]
        ),
        count_penalty_logit=float(winner["decoder"]["countPenaltyLogit"]),
    )
    frozen_predictions = decode_ranked_candidates(
        rows, frozen_combined, float(winner["threshold"]), frozen_settings
    )

    label_vector = np.asarray([labels[str(row["eventId"])] for row in rows])
    development_metrics = {
        "frozenWinnerExpandedUnion": {
            "rowAveragePrecision": average_precision(label_vector, frozen_combined),
            "strict": _evaluate(rows, frozen_predictions, markers, 0.0),
            "primary": _evaluate(
                rows,
                frozen_predictions,
                markers,
                PRIMARY_PADDING_SECONDS,
                include_inventory=True,
            ),
            "decoderTranslation": (
                "minimumGapSeparation is applied to chronological full-union candidate "
                "ordinals; internal context logits are neutral"
            ),
        },
        "nestedOuterLoo": {
            "rowAveragePrecision": average_precision(label_vector, outer_scores),
            "withoutContinuity": {
                "strict": _evaluate(rows, outer_predictions, markers, 0.0),
                "primary": _evaluate(
                    rows,
                    outer_predictions,
                    markers,
                    PRIMARY_PADDING_SECONDS,
                    include_inventory=True,
                ),
            },
            "withContinuityVeto": {
                "strict": _evaluate(rows, outer_veto_predictions, markers, 0.0),
                "primary": _evaluate(
                    rows,
                    outer_veto_predictions,
                    markers,
                    PRIMARY_PADDING_SECONDS,
                    include_inventory=True,
                ),
            },
        },
    }
    selection_counts = {
        "featureGroups": dict(
            Counter(str(fold["selected"]["featureGroup"]) for fold in outer_folds)
        ),
        "l2": dict(Counter(str(fold["selected"]["l2"]) for fold in outer_folds)),
        "decoder": dict(
            Counter(
                json.dumps(fold["selected"]["decoder"], sort_keys=True)
                for fold in outer_folds
            )
        ),
    }
    created_at = datetime.now(UTC).isoformat()
    model_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-full-union-ranker-v1",
        "createdAt": created_at,
        "status": "research-only-opened-development",
        "classifier": final_classifier.to_dict(),
        "threshold": float(selected_full["threshold"]),
        "decoder": selected_full["decoder"],
        "continuityVeto": {
            "signal": "player",
            "threshold": float(full_veto["threshold"]),
            "minimumQuality": float(full_veto["minimumQuality"]),
            "minimumTruePositiveRetention": float(
                full_veto["minimumTruePositiveRetention"]
            ),
        },
        "selection": {
            "featureGroup": selected_full["featureGroup"],
            "l2": selected_full["l2"],
            "rowAveragePrecision": selected_full["rowAveragePrecision"],
            "developmentMetrics": selected_full["metrics"],
            "primaryPaddingSeconds": PRIMARY_PADDING_SECONDS,
            "nestedOuterSelectionCounts": selection_counts,
        },
        "candidateContract": features["profile"],
        "sources": {
            name: {"path": str(paths[name]), "sha256": hashes[name]}
            for name in paths
        },
    }
    evaluation_payload = {
        "schemaVersion": 1,
        "kind": "volleycut-side-switch-full-union-ranker-evaluation-v1",
        "createdAt": created_at,
        "scope": {
            **features["scope"],
            "humanMarkers": sum(map(len, markers.values())),
            "positiveCandidateLabels": int(np.sum(label_vector)),
            "status": "opened-development-only",
        },
        "labelAssignment": {
            "rule": "one-to-one monotonic maximum-cardinality match at +/-4 seconds, then minimum anchor distance",
            "byRecording": label_audit,
        },
        "selectionProtocol": {
            "outer": "leave one recording out from all ranker and decoder selection",
            "inner": "grouped leave-one-recording-out scores on the ten outer-fit recordings",
            "featureGroups": {
                name: list(values) for name, values in FEATURE_GROUPS.items()
            },
            "l2Grid": list(L2_GRID),
            "decoderGrid": [value.to_dict() for value in DECODER_GRID],
            "thresholdQuantiles": THRESHOLD_QUANTILES,
            "ranking": "pooled +/-4-second event F1, then precision, recall, fewer proposals, lower complexity",
            "continuity": "player-margin veto fitted only on each outer-fit scope with >=90% candidate-TP retention",
        },
        "metrics": development_metrics,
        "outerFolds": outer_folds,
        "fullDevelopmentSelection": {
            "selected": {
                key: selected_full[key]
                for key in (
                    "featureGroup",
                    "featureNames",
                    "l2",
                    "rowAveragePrecision",
                    "threshold",
                    "decoder",
                    "metrics",
                )
            },
            "leaderboard": [
                {
                    key: value
                    for key, value in candidate.items()
                    if key != "crossFit"
                }
                for candidate in full_leaderboard
            ],
            "continuityVeto": {
                key: value
                for key, value in full_veto.items()
                if key != "retainedEventIds"
            },
        },
        "sources": model_payload["sources"],
        "limitations": [
            "The 11 exhaustively reviewed recordings are opened development, not an untouched test set.",
            "Four human markers remain outside the candidate union at +/-4 seconds and cannot be recovered by this ranker.",
            "Candidate-label assignment chooses one representative proposal per covered marker and is not frame-accurate switch segmentation.",
            "The frozen winner's gap-index cleanup is translated to chronological full-union candidate ordinals for its expanded-union diagnostic.",
        ],
    }
    model_path.parent.mkdir(parents=True, exist_ok=False)
    atomic_write_text(model_path, json.dumps(model_payload, indent=2, allow_nan=False) + "\n")
    atomic_write_text(
        evaluation_path,
        json.dumps(evaluation_payload, indent=2, allow_nan=False) + "\n",
    )
    return model_payload, evaluation_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--full-audit", type=Path, default=DEFAULT_FULL_AUDIT)
    parser.add_argument("--winner", type=Path, default=DEFAULT_WINNER)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument(
        "--enforce-source-hash", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> None:
    model, evaluation = run(_parser().parse_args())
    print(
        json.dumps(
            {
                "selection": model["selection"],
                "nestedOuterLoo": evaluation["metrics"]["nestedOuterLoo"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

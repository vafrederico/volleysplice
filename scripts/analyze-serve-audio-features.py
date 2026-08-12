#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.artifacts import atomic_write_text
from analysis.features import cached_features
from analysis.model import RALLY_LIVE_TASK, LogisticModel, ModelError, load_model
from analysis.pipeline import _manifest_digest
from analysis.schema import Interval, Recording, load_manifest, mask_for_times


DEFAULT_FEATURES = (
    "audio_band_80_250_snr_flux",
    "audio_noise_normalized_flux",
    "audio_rms_novelty",
    "audio_band_4000_7800_snr_flux",
    "audio_snr",
    "audio_spectral_flux",
)


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Return tie-aware Mann-Whitney AUC without an sklearn dependency."""
    values = np.concatenate((positive, negative)).astype(np.float64, copy=False)
    labels = np.concatenate(
        (np.ones(len(positive), dtype=np.bool_), np.zeros(len(negative), dtype=np.bool_))
    )
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    left = 0
    while left < len(values):
        right = left + 1
        while right < len(values) and sorted_values[right] == sorted_values[left]:
            right += 1
        ranks[order[left:right]] = 0.5 * (left + right - 1) + 1.0
        left = right
    positive_rank_sum = float(np.sum(ranks[labels]))
    count_positive = len(positive)
    count_negative = len(negative)
    return (
        positive_rank_sum - count_positive * (count_positive + 1) / 2.0
    ) / (count_positive * count_negative)


def _hard_negative_mask(
    times: np.ndarray,
    rallies: Sequence[Interval],
    *,
    preparation_start_seconds: float,
    preparation_end_seconds: float,
    in_rally_start_seconds: float,
) -> np.ndarray:
    mask = np.zeros(len(times), dtype=np.bool_)
    for rally in rallies:
        mask |= (times >= rally.start - preparation_start_seconds) & (
            times < rally.start - preparation_end_seconds
        )
        mask |= (times >= rally.start + in_rally_start_seconds) & (
            times < rally.end
        )
    return mask


def _recording_values(
    recording: Recording,
    model: LogisticModel,
    cache_dir: Path,
    selected_features: Sequence[str],
    *,
    preparation_start_seconds: float,
    preparation_end_seconds: float,
    in_rally_start_seconds: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    sequence = cached_features(
        recording.id,
        recording.video,
        model.feature_config,
        recording.roi,
        cache_dir,
        content_sha256=recording.content_sha256,
    )
    indexes = {name: index for index, name in enumerate(sequence.names)}
    missing = sorted(set(selected_features) - indexes.keys())
    if missing:
        raise ModelError(f"cached feature signature is missing: {missing}")
    valid = mask_for_times(sequence.times, recording.ignored_intervals)
    hard_negative = _hard_negative_mask(
        sequence.times,
        recording.rallies,
        preparation_start_seconds=preparation_start_seconds,
        preparation_end_seconds=preparation_end_seconds,
        in_rally_start_seconds=in_rally_start_seconds,
    ) & valid
    contact_rows = np.asarray(
        [
            int(np.argmin(np.abs(sequence.times - rally.start)))
            for rally in recording.rallies
        ],
        dtype=np.int64,
    )
    contact_rows = contact_rows[valid[contact_rows]]
    hard_negative[contact_rows] = False
    return (
        {
            name: sequence.values[contact_rows, indexes[name]].astype(
                np.float64, copy=False
            )
            for name in selected_features
        },
        {
            name: sequence.values[hard_negative, indexes[name]].astype(
                np.float64, copy=False
            )
            for name in selected_features
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure raw audio-feature separation between the nearest sampled serve contact "
            "and pre-serve/in-rally hard negatives."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feature", action="append", dest="features")
    parser.add_argument("--preparation-start-seconds", type=float, default=4.0)
    parser.add_argument("--preparation-end-seconds", type=float, default=1.0)
    parser.add_argument("--in-rally-start-seconds", type=float, default=2.0)
    arguments = parser.parse_args()
    destination = arguments.output.expanduser().resolve()
    if destination.exists():
        raise ModelError(f"analysis output already exists: {destination}")
    if not (
        arguments.preparation_start_seconds > arguments.preparation_end_seconds >= 0
        and arguments.in_rally_start_seconds >= 0
    ):
        raise ValueError("hard-negative time windows are invalid")
    model = load_model(arguments.model)
    if model.prediction_task != RALLY_LIVE_TASK:
        raise ModelError("feature-analysis model must predict rally-live state")
    manifest = load_manifest(arguments.manifest)
    manifest_sha256 = _manifest_digest(manifest)
    if model.training_summary.get("manifestSha256") != manifest_sha256:
        raise ModelError("model differs from the immutable feature-analysis manifest")
    selected_features = tuple(arguments.features or DEFAULT_FEATURES)
    if not selected_features or len(set(selected_features)) != len(selected_features):
        raise ValueError("features must be a non-empty unique list")

    rows: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for recording in manifest.recordings:
        if recording.split not in {"train", "validation"}:
            continue
        positives, negatives = _recording_values(
            recording,
            model,
            arguments.cache_dir.expanduser().resolve(),
            selected_features,
            preparation_start_seconds=arguments.preparation_start_seconds,
            preparation_end_seconds=arguments.preparation_end_seconds,
            in_rally_start_seconds=arguments.in_rally_start_seconds,
        )
        rows[recording.id] = {"positive": positives, "negative": negatives}

    results: dict[str, object] = {}
    for feature in selected_features:
        train_recordings = [
            recording for recording in manifest.recordings if recording.split == "train"
        ]
        validation_recordings = [
            recording for recording in manifest.recordings if recording.split == "validation"
        ]

        def combined_values(
            recordings: Sequence[Recording],
        ) -> tuple[np.ndarray, np.ndarray]:
            positive = np.concatenate(
                [rows[item.id]["positive"][feature] for item in recordings]
            )
            negative = np.concatenate(
                [rows[item.id]["negative"][feature] for item in recordings]
            )
            if len(positive) == 0 or len(negative) == 0:
                raise ModelError(f"feature analysis has an empty class for {feature}")
            return positive, negative

        def combined_auc(recordings: Sequence[Recording]) -> float:
            return _auc(*combined_values(recordings))

        train_groups = sorted({item.source_group for item in train_recordings})
        group_auc = {
            group: combined_auc(
                [item for item in train_recordings if item.source_group == group]
            )
            for group in train_groups
        }
        train_positive, train_negative = combined_values(train_recordings)
        validation_positive, validation_negative = combined_values(
            validation_recordings
        )
        results[feature] = {
            "trainAuc": _auc(train_positive, train_negative),
            "trainContacts": len(train_positive),
            "trainHardNegatives": len(train_negative),
            "minimumTrainSourceGroupAuc": min(group_auc.values()),
            "trainSourceGroupAuc": group_auc,
            "validationAuc": _auc(validation_positive, validation_negative),
            "validationContacts": len(validation_positive),
            "validationHardNegatives": len(validation_negative),
        }

    payload = {
        "schemaVersion": 1,
        "analysis": "serve-audio-contact-hard-negative-auc-v1",
        "dataset": manifest.name,
        "manifestSha256": manifest_sha256,
        "modelSha256": model.artifact_sha256,
        "featureVersion": model.feature_version,
        "positiveDefinition": "nearest cached sample to each gold rally start",
        "hardNegativeDefinition": {
            "preparationSecondsBeforeServe": [
                arguments.preparation_start_seconds,
                arguments.preparation_end_seconds,
            ],
            "inRallySecondsAfterServe": arguments.in_rally_start_seconds,
        },
        "features": results,
        "limitations": [
            "AUC measures raw feature ranking, not serve-model average precision.",
            "Validation is descriptive tuning evidence and the test split is not read.",
        ],
    }
    atomic_write_text(
        destination, json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

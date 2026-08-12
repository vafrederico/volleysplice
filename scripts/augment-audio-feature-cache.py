#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.config import (
    NOISE_NORMALIZED_AUDIO_FEATURE_SET,
    FeatureConfig,
    feature_version_for_config,
)
from analysis.features import (
    FeatureSequence,
    VideoMetadata,
    _audio_feature_names,
    _audio_features_from_samples,
    _cache_key,
    _decode_audio_samples,
    _valid_cached_sequence,
    feature_names,
)
from analysis.model import RALLY_LIVE_TASK, ModelError, load_model
from analysis.pipeline import _manifest_digest
from analysis.schema import load_manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_id(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def _cache_path(
    directory: Path,
    recording_id: str,
    video: Path,
    config: FeatureConfig,
    roi: tuple[float, float, float, float] | None,
    digest: str,
) -> Path:
    return directory / (
        f"{_safe_id(recording_id)}-{_cache_key(video, config, roi, digest)}.npz"
    )


def _load_sequence(path: Path) -> FeatureSequence:
    try:
        with np.load(path, allow_pickle=False) as cached:
            return FeatureSequence(
                times=cached["times"].astype(np.float64, copy=False),
                values=cached["values"].astype(np.float32, copy=False),
                names=tuple(str(item) for item in cached["names"]),
                metadata=VideoMetadata(**json.loads(str(cached["metadata_json"].item()))),
            )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ModelError(f"cannot load source cache {path}: {error}") from error


def _write_sequence(path: Path, sequence: FeatureSequence) -> None:
    if path.exists():
        raise ModelError(f"refusing to overwrite enhanced cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            times=sequence.times,
            values=sequence.values,
            names=np.asarray(sequence.names),
            metadata_json=json.dumps(
                asdict(sequence.metadata), sort_keys=True, allow_nan=False
            ),
        )
        if path.exists():
            raise ModelError(f"refusing to overwrite enhanced cache: {path}")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse a frozen audiovisual-v2 visual cache while appending the v3 "
            "noise-normalized spectral audio channels."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--source-cache-dir", required=True, type=Path)
    parser.add_argument("--output-cache-dir", required=True, type=Path)
    arguments = parser.parse_args()

    model = load_model(arguments.base_model)
    if model.prediction_task != RALLY_LIVE_TASK:
        raise ModelError("base model must predict rally-live state")
    base_config = model.feature_config
    if base_config.audio_feature_set != "legacy-v2" or not base_config.use_audio:
        raise ModelError("base model must use the legacy-v2 audio feature set")
    enhanced_config = replace(
        base_config, audio_feature_set=NOISE_NORMALIZED_AUDIO_FEATURE_SET
    )
    enhanced_config.validate()
    manifest = load_manifest(arguments.manifest)
    if model.training_summary.get("manifestSha256") != _manifest_digest(manifest):
        raise ModelError("base model differs from the immutable cache manifest")
    source_dir = arguments.source_cache_dir.expanduser().resolve()
    output_dir = arguments.output_cache_dir.expanduser().resolve()
    if (
        output_dir == source_dir
        or output_dir in source_dir.parents
        or source_dir in output_dir.parents
    ):
        raise ModelError("source and output cache directories must be separate")
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_audio_count = len(_audio_feature_names(base_config))
    expected_base_names = feature_names(base_config)
    expected_enhanced_names = feature_names(enhanced_config)
    if expected_enhanced_names[: len(expected_base_names)] != expected_base_names:
        raise ModelError("enhanced audio signature is not an append-only extension")

    outputs: dict[str, dict[str, object]] = {}
    for index, recording in enumerate(manifest.recordings, start=1):
        if recording.content_sha256 is None:
            raise ModelError(f"recording digest is unavailable: {recording.id}")
        source = _cache_path(
            source_dir,
            recording.id,
            recording.video,
            base_config,
            recording.roi,
            recording.content_sha256,
        )
        if not source.is_file():
            raise ModelError(f"source cache is missing: {source}")
        sequence = _load_sequence(source)
        if not _valid_cached_sequence(sequence, base_config):
            raise ModelError(f"source cache signature is invalid: {source}")
        destination = _cache_path(
            output_dir,
            recording.id,
            recording.video,
            enhanced_config,
            recording.roi,
            recording.content_sha256,
        )
        if destination.exists():
            existing = _load_sequence(destination)
            if not _valid_cached_sequence(existing, enhanced_config):
                raise ModelError(f"existing enhanced cache is invalid: {destination}")
            print(
                f"[{index}/{len(manifest.recordings)}] reusing audio cache: {recording.id}",
                file=sys.stderr,
                flush=True,
            )
            outputs[recording.id] = {
                "path": str(destination),
                "sha256": _sha256_file(destination),
                "rows": len(existing.times),
                "features": len(existing.names),
            }
            continue
        print(
            f"[{index}/{len(manifest.recordings)}] decoding audio: {recording.id}",
            file=sys.stderr,
            flush=True,
        )
        samples, available = _decode_audio_samples(
            recording.video, sequence.metadata, enhanced_config.audio_sample_rate
        )
        enhanced_audio = _audio_features_from_samples(
            samples,
            sequence.times,
            enhanced_config,
            available=available,
        )
        extra_audio = enhanced_audio[:, legacy_audio_count:]
        enhanced = FeatureSequence(
            times=sequence.times,
            values=np.column_stack((sequence.values, extra_audio)).astype(
                np.float32, copy=False
            ),
            names=expected_enhanced_names,
            metadata=sequence.metadata,
        )
        if not _valid_cached_sequence(enhanced, enhanced_config):
            raise ModelError(f"generated cache signature is invalid: {recording.id}")
        _write_sequence(destination, enhanced)
        outputs[recording.id] = {
            "path": str(destination),
            "sha256": _sha256_file(destination),
            "rows": len(enhanced.times),
            "features": len(enhanced.names),
        }

    print(
        json.dumps(
            {
                "featureVersion": feature_version_for_config(enhanced_config),
                "featureConfig": enhanced_config.to_dict(),
                "recordings": outputs,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

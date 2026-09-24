"""Attach hash-bound, label-blind recognition caches to unchanged supervision.

The source dataset, sanitized extraction manifest and feature index must describe
the same eighteen development recordings. This module never opens label files.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from .mobile_visual_features import (EMBEDDING_DIMENSION, TOKEN_COUNT, QUALITY_NAMES,
                                     align_mobile_features, load_mobile_visual_cache, normalize_roi)

if TYPE_CHECKING:
    from .recognition_temporal_model import RecognitionConfig

TIERS = {"exact": "exactRows", "draft": "draftRows", "coverage": "coverageRows"}
TIER_COUNTS = {"exact": 8, "draft": 3, "coverage": 7}
PLAYER_FEATURE_VERSION = "continuous-player-court-motion-v1"
_SIDE_STATISTICS = ("count", "confidence", "center_x", "center_y", "spread_x", "spread_y", "pair_distance",
                    "torso_area", "motion_coverage", "speed_mean", "speed_p90", "moving_fraction", "flow_x",
                    "flow_y", "direction_coherence", "speed_rise_fraction", "speed_fall_fraction")
PLAYER_FEATURE_NAMES = ("player_detector_ran", "player_tracks_available", "player_observation_age_seconds",
    "player_detection_saturated", "player_rejected_roi_fraction", "player_net_geometry_supplied",
    "player_motion_available", "player_scene_jump", "player_camera_speed", "player_camera_fit_residual",
    "player_matched_fraction", "player_both_sides_visible", "player_both_sides_moving",
    "player_reaction_synchrony", "player_standdown_synchrony",
    *(f"player_{side}_{name}" for side in ("near", "far") for name in _SIDE_STATISTICS))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _path(value: Any) -> Path:
    _require(isinstance(value, (str, Path)) and bool(str(value)), "missing artifact path")
    return Path(value).expanduser().resolve()


def _digest(value: Any) -> str:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "invalid full SHA-256")
    return value


def _verify_file(path: Any, expected_sha: Any) -> Path:
    resolved = _path(path)
    _require(_sha(resolved) == _digest(expected_sha), f"artifact hash mismatch: {resolved}")
    return resolved


def _same_reference(reference: Mapping[str, Any], path: Path, digest: str, description: str) -> None:
    _require(isinstance(reference, Mapping) and _path(reference.get("path")) == path
             and reference.get("sha256") == digest, f"{description} lineage mismatch")


def _unique_rows(rows: Any, description: str) -> dict[str, dict[str, Any]]:
    _require(isinstance(rows, list) and len(rows) == 18, f"{description} must contain exactly 18 recordings")
    result = {}
    for row in rows:
        _require(isinstance(row, dict), f"invalid {description} row")
        identifier = row.get("id", row.get("recordingId"))
        _require(isinstance(identifier, str) and identifier and identifier not in result,
                 f"missing/duplicate {description} recording")
        result[identifier] = row
    return result


def _runtime_dependencies(index: Mapping[str, Any]) -> None:
    """Bind imported helper bytes through the registered feature-index artifact.

    This lets feature arms bind new loading dependencies without modifying a
    runner whose AV/DINO arms have already been registered and are executing.
    """
    dependencies = index.get("runtimeDependencies")
    expected = Path(__file__).with_name("mobile_visual_features.py").resolve()
    _require(isinstance(dependencies, list) and len(dependencies) == 1,
             "feature index must bind the mobile visual runtime helper and archived source")
    dependency = dependencies[0]
    _require(isinstance(dependency, dict) and _path(dependency.get("path")) == expected,
             "unexpected recognition runtime dependency")
    actual = _verify_file(dependency.get("path"), dependency.get("sha256"))
    archive = _verify_file(dependency.get("archivePath"), dependency.get("sha256"))
    _require(actual != archive, "runtime dependency archive must be a separate source snapshot")


def _manifest_contract(data, feature_manifest: Path, source_manifest: Path):
    source_path, index_path = _path(source_manifest), _path(feature_manifest)
    source_digest, index_digest = _sha(source_path), _sha(index_path)
    source, index = _read(source_path), _read(index_path)
    _require(index.get("schemaVersion") == 1, "unsupported feature index schema")
    _runtime_dependencies(index)
    manifest_reference = index.get("manifest")
    if isinstance(manifest_reference, str):
        manifest_reference = {"path": manifest_reference, "sha256": index.get("manifestSha256")}
    _require(isinstance(manifest_reference, dict), "feature index lacks sanitized manifest association")
    extraction_path = _verify_file(manifest_reference.get("path"), manifest_reference.get("sha256"))
    extraction_digest = manifest_reference["sha256"]
    extraction = _read(extraction_path)
    _require(extraction.get("schemaVersion") == 1
             and extraction.get("kind") == "label-free-recognition-extraction-input-v1"
             and extraction.get("labelsUsed") is False and extraction.get("protectedTestOpened") is False
             and extraction.get("beachIncluded") is False, "invalid sanitized extraction manifest contract")
    _same_reference(extraction.get("sourceManifest"), source_path, source_digest, "source manifest")
    extraction_rows = _unique_rows(extraction.get("records"), "extraction manifest")
    entries = _unique_rows(index.get("records"), "feature index")
    source_rows = {}
    seen_data = set()
    for tier, key in TIERS.items():
        rows, loaded = source.get(key), data.get(tier)
        _require(isinstance(rows, list) and len(rows) == TIER_COUNTS[tier]
                 and isinstance(loaded, (tuple, list)) and len(loaded) == len(rows), f"{tier} inventory changed")
        _require([row.example.id for row in loaded] == [row["id"] for row in rows], f"{tier} recording order changed")
        for raw, supervised in zip(rows, loaded, strict=True):
            identifier, example = raw["id"], supervised.example
            _require(identifier not in source_rows and identifier not in seen_data, "duplicate dataset recording")
            seen_data.add(identifier)
            source_rows[identifier] = raw
            _require(supervised.tier == tier and example.group == raw["sourceGroup"]
                     and example.environment == raw["environment"], f"loaded source group/tier differs: {identifier}")
            _require(raw["split"] in ("train", "validation") and raw["environment"] in ("indoor", "grass")
                     and raw["sourceGroup"] not in source.get("protectedSourceGroups", []), "protected/nondevelopment source")
            _require(abs(float(example.duration) - float(raw["durationSeconds"])) < .011, "loaded duration differs")
            _require(example.values.shape == (len(example.times), 104) and np.isfinite(example.values).all(), "expected finite unchanged AV104 inputs")
            _require(np.asarray(example.times).ndim == 1 and len(example.times) > 0
                     and np.isfinite(example.times).all() and np.all(np.diff(example.times) > 0), "invalid AV timestamps")
    _require(len(source_rows) == 18 and set(source_rows) == set(extraction_rows) == set(entries), "feature/source inventory mismatch")
    _require(len({row["sourceGroup"] for row in source["exactRows"]}) == 4, "exact development group inventory changed")
    for identifier, raw in source_rows.items():
        clean = extraction_rows[identifier]
        for key in ("sourceGroup", "split", "environment", "contentSha256"):
            _require(clean.get(key) == raw.get(key), f"sanitized {key} differs: {identifier}")
        _digest(raw["contentSha256"])
        _require(_path(clean.get("video")) == _path(raw["video"])
                 and normalize_roi(clean.get("roi")) == normalize_roi(raw.get("roi")), "sanitized video/ROI differs")
        _require(abs(float(clean.get("durationSeconds", -1)) - float(raw["durationSeconds"])) < .000001, "sanitized duration differs")
        av = raw["featureCaches"]["audiovisual"]
        _same_reference(clean.get("avCache"), _path(av["path"]), av["sha256"], "sanitized AV cache")
    return source_rows, entries, index, extraction_path, extraction_digest, (source_path, source_digest, index_path, index_digest)


def _load_av_times(source, example):
    reference = source["featureCaches"]["audiovisual"]
    path = _verify_file(reference["path"], reference["sha256"])
    with np.load(path, allow_pickle=False) as cache:
        times = np.asarray(cache["times"], np.float64)
        values = np.asarray(cache["values"])
    _require(values.shape == (len(times), 104) and np.isfinite(values).all()
             and np.array_equal(times, example.times), "loaded AV times/cache association differs")
    return times


def _mobile_features(entry, source, example, index, times):
    _require(entry.get("trainingEligible", True) is True, "partial mobile feature index is not trainable")
    _require(entry.get("recordingId", entry.get("id")) == example.id
             and entry.get("sourceGroup") == example.group and entry.get("split") == source["split"]
             and entry.get("recordingContentSha256") == source["contentSha256"], "mobile index source association differs")
    path = _verify_file(entry.get("cachePath", entry.get("path")), entry.get("cacheSha256", entry.get("sha256")))
    cache = load_mobile_visual_cache(path)
    metadata, identity = cache.metadata, cache.metadata.get("identity", {})
    _require(metadata.get("partialVideo") is False and metadata.get("trainingEligible", True) is True
             and identity.get("maximumExtractionSeconds", "missing") is None, "partial mobile cache is not trainable")
    _require(metadata.get("recordingId") == example.id and identity.get("recordingId") == example.id
             and metadata.get("recordingContentSha256") == source["contentSha256"]
             and identity.get("recordingContentSha256") == source["contentSha256"]
             and _path(metadata.get("sourceVideoPath")) == _path(source["video"])
             and normalize_roi(identity.get("roi")) == normalize_roi(source.get("roi")), "mobile cache source/ROI association differs")
    _require(metadata.get("qualityNames") == list(QUALITY_NAMES)
             and identity.get("config") == index.get("config") and identity.get("backbone") == index.get("backbone"),
             "mobile feature config/model/quality schema differs")
    duration = float(metadata.get("extractedDurationSeconds", -1))
    _require(abs(duration - example.duration) < .011 and abs(float(metadata["video"]["duration"]) - duration) < .000001,
             "mobile full duration differs")
    expected_grid = np.arange(int(np.ceil(duration * 2 - 1e-9)), dtype=np.float64) / 2
    expected_grid = expected_grid[expected_grid < duration]
    _require(np.array_equal(cache.timestamps, expected_grid), "mobile full timestamp coverage differs")
    with np.load(path, allow_pickle=False) as raw:
        _require(raw["tokens"].dtype == np.float16, "mobile storage must be float16")
    _digest(identity.get("extractorSourceSha256"))
    _require(_sha(Path(__file__).with_name("mobile_visual_features.py")) == identity["extractorSourceSha256"],
             "mobile loading/alignment helper source differs from cache-bound extractor")
    _digest(identity.get("backbone", {}).get("checkpointSha256"))
    signature = {key: identity.get(key) for key in ("config", "backbone", "extractorSourceSha256", "opencv", "opencvBuildSha256")}
    _require(all(value is not None for value in signature.values()), "mobile recipe provenance missing")
    aligned = align_mobile_features(cache, times)
    _require(np.all(aligned["available"] == 1), "mobile full cache leaves AV ticks uncovered")
    result = np.concatenate((aligned["tokens"].reshape(len(times), -1), aligned["quality"],
                             aligned["feature_age_seconds"][:, None], aligned["available"][:, None]), axis=1)
    return result, signature


def _player_features(entry, source, example, index, times, extraction_path, extraction_digest):
    _require(entry.get("trainingEligible") is True, "partial player feature index is not trainable")
    path = _verify_file(entry.get("path", entry.get("cachePath")), entry.get("sha256", entry.get("cacheSha256")))
    with np.load(path, allow_pickle=False) as cache:
        timestamps, values = np.asarray(cache["times"], np.float64), np.asarray(cache["values"], np.float32)
        names = tuple(str(name) for name in cache["names"])
        metadata = json.loads(str(cache["metadata_json"].item()))
    _require(isinstance(metadata, dict) and metadata.get("schemaVersion") == 1
             and metadata.get("featureVersion") == PLAYER_FEATURE_VERSION
             and metadata.get("labelsUsed") is False and metadata.get("labelIndependent") is True,
             "player cache schema/label-blind contract differs")
    _require(metadata.get("trainingEligible") is True and metadata.get("fullRecording") is True
             and metadata.get("pilotMaximumSeconds", "missing") is None, "partial player cache is not trainable")
    _require(metadata.get("recordingId") == example.id and metadata.get("sourceGroup") == example.group
             and metadata.get("environment") == source["environment"], "player cache source group differs")
    _same_reference(metadata.get("manifest"), extraction_path, extraction_digest, "player extraction manifest")
    _same_reference(metadata.get("sourceVideo"), _path(source["video"]), source["contentSha256"], "player source video")
    av = source["featureCaches"]["audiovisual"]
    _same_reference(metadata.get("avCache"), _path(av["path"]), av["sha256"], "player AV cache")
    _require(normalize_roi(metadata.get("roi")) == normalize_roi(source.get("roi")), "player ROI differs")
    _require(names == PLAYER_FEATURE_NAMES and index.get("featureNames") == list(PLAYER_FEATURE_NAMES)
             and index.get("featureDimension") == len(PLAYER_FEATURE_NAMES)
             and index.get("featureVersion") == PLAYER_FEATURE_VERSION, "player feature-name schema differs")
    _require(np.array_equal(timestamps, times) and values.shape == (len(times), len(PLAYER_FEATURE_NAMES))
             and np.isfinite(values).all(), "player features nonfinite or not on exact AV timestamps")
    alignment = metadata.get("alignment", {})
    _require(alignment.get("avTimesExact") is True
             and alignment.get("timesSha256") == hashlib.sha256(times.astype("<f8").tobytes()).hexdigest(), "player AV timestamp provenance differs")
    _require(abs(float(metadata["media"]["stream"]["duration"]) - example.duration) < .011, "player full duration differs")
    source_code = metadata.get("sourceCode")
    _require(isinstance(source_code, dict) and source_code, "player extractor source provenance missing")
    for value in source_code.values():
        _digest(value)
    signature = {key: metadata.get(key) for key in ("featureVersion", "config", "detector", "detectorSelectionOverride", "decoder", "sourceCode", "runtime")}
    _require(all(value is not None for value in signature.values()), "player recipe provenance missing")
    return values, signature


def attach_features(data, feature_manifest: Path, config: RecognitionConfig, source_manifest: Path):
    """Return new row/example objects; preserve supervision and original AV values."""
    config.validate()
    _require(config.family in ("player", "mobile"), "attach_features supports player/mobile families only")
    expected_scalars = 8 if config.family == "mobile" else len(PLAYER_FEATURE_NAMES)
    _require(config.scalar_dimension == expected_scalars, "recognition scalar dimension differs from feature contract")
    if config.family == "mobile":
        _require(config.token_count == TOKEN_COUNT and config.token_dimension == EMBEDDING_DIMENSION,
                 "mobile token geometry differs")
    sources, entries, index, extraction_path, extraction_digest, bound = _manifest_contract(data, feature_manifest, source_manifest)
    result, common_signature = dict(data), None
    for tier in TIERS:
        new_rows = []
        for supervised in data[tier]:
            example = supervised.example
            source, entry = sources[example.id], entries[example.id]
            times = _load_av_times(source, example)
            if config.family == "mobile":
                extra, signature = _mobile_features(entry, source, example, index, times)
            else:
                extra, signature = _player_features(entry, source, example, index, times, extraction_path, extraction_digest)
            if common_signature is None:
                common_signature = signature
            _require(signature == common_signature, f"mixed feature code/model/config/runtime recipes: {example.id}")
            values = np.ascontiguousarray(np.concatenate((example.values, extra), axis=1), dtype=np.float32)
            _require(values.shape == (len(times), config.input_dimension) and np.isfinite(values).all(), "invalid attached feature matrix")
            new_rows.append(replace(supervised, example=replace(example, values=values)))
        result[tier] = tuple(new_rows) if isinstance(data[tier], tuple) else new_rows
    source_path, source_digest, index_path, index_digest = bound
    _require(_sha(source_path) == source_digest and _sha(index_path) == index_digest
             and _sha(extraction_path) == extraction_digest, "input manifest changed while attaching features")
    _runtime_dependencies(index)
    return result

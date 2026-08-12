from __future__ import annotations

from collections import defaultdict
from typing import Sequence


FEATURE_FAMILY_KINDS = {
    "legacy_appearance": "legacy",
    "legacy_frame_difference": "legacy",
    "legacy_optical_flow": "legacy",
    "camera_quality": "added",
    "player_motion": "added",
    "formation_change_proxy": "added",
    "audio_level": "added",
    "audio_onset": "added",
    "audio_cadence": "added",
    "audio_normalized": "added",
    "audio_frequency": "added",
}


def base_feature_name(contextual_name: str) -> str:
    return contextual_name.split("/", 1)[-1]


def feature_family(name: str) -> str:
    base = base_feature_name(name)
    if base == "quality_gated_player_motion":
        return "player_motion"
    if base.startswith(("luma_", "saturation_")) or base in {
        "edge_density",
        "sharpness",
    }:
        return "legacy_appearance"
    if base.startswith("diff_"):
        return "legacy_frame_difference"
    if base.startswith("flow_"):
        return "legacy_optical_flow"
    if base.startswith(
        (
            "focus_",
            "blur_",
            "dark_",
            "bright_",
            "low_texture_",
            "occlusion_",
            "visibility_",
            "camera_shift_",
        )
    ):
        return "camera_quality"
    if base.startswith("audio_"):
        if base.startswith("audio_band_"):
            return "audio_frequency"
        if base in {
            "audio_noise_removed_broadband",
            "audio_noise_normalized_flux",
        }:
            return "audio_normalized"
        if base in {
            "audio_available",
            "audio_rms",
            "audio_peak",
            "audio_peak_to_rms",
            "audio_noise_floor",
            "audio_snr",
        }:
            return "audio_level"
        if base in {
            "audio_spectral_flux",
            "audio_rms_novelty",
            "audio_onset_strength",
            "audio_contact_like_transient",
        }:
            return "audio_onset"
        return "audio_cadence"
    if base in {
        "player_motion_spatial_entropy",
        "player_motion_centroid_x",
        "player_motion_centroid_y",
        "player_motion_spread_x",
        "player_motion_spread_y",
        "receiving_formation_change_proxy",
    }:
        return "formation_change_proxy"
    if base.startswith("player_motion_") or base == "synchronized_stand_down":
        return "player_motion"
    raise ValueError(f"unassigned feature family for {name!r}")


def grouped_feature_indexes(names: Sequence[str]) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(names):
        grouped[feature_family(name)].append(index)
    result = {family: tuple(indexes) for family, indexes in sorted(grouped.items())}
    expected = set(range(len(names)))
    covered = {index for indexes in result.values() for index in indexes}
    if covered != expected:
        raise ValueError("feature-family mapping does not cover every feature exactly once")
    return result


def grouped_base_features(names: Sequence[str]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for name in names:
        grouped[feature_family(name)].add(base_feature_name(name))
    return {
        family: tuple(sorted(base_names))
        for family, base_names in sorted(grouped.items())
    }

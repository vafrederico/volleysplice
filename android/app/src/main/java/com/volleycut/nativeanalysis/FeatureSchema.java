package com.volleycut.nativeanalysis;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

final class FeatureSchema {
    static final int ANALYSIS_FPS = 4;
    static final int ANALYSIS_WIDTH = 192;
    static final int ANALYSIS_HEIGHT = 108;
    static final int BENCHMARK_SOURCE_FRAME_LIMIT = 1_000;

    static final List<String> FRAME = buildFrameNames();
    static final List<String> TEMPORAL = List.of(
            "player_motion_onset",
            "player_motion_collapse",
            "synchronized_stand_down",
            "receiving_formation_change_proxy"
    );
    static final List<String> AUDIO = List.of(
            "audio_available",
            "audio_rms",
            "audio_peak",
            "audio_peak_to_rms",
            "audio_noise_floor",
            "audio_snr",
            "audio_spectral_flux",
            "audio_rms_novelty",
            "audio_onset_strength",
            "audio_contact_like_transient",
            "audio_onset_cadence",
            "audio_cadence_collapse",
            "audio_seconds_since_transient",
            "audio_noise_removed_broadband",
            "audio_noise_normalized_flux",
            "audio_band_80_250_snr",
            "audio_band_80_250_snr_flux",
            "audio_band_250_500_snr",
            "audio_band_250_500_snr_flux",
            "audio_band_500_1000_snr",
            "audio_band_500_1000_snr_flux",
            "audio_band_1000_2000_snr",
            "audio_band_1000_2000_snr_flux",
            "audio_band_2000_4000_snr",
            "audio_band_2000_4000_snr_flux",
            "audio_band_4000_7800_snr",
            "audio_band_4000_7800_snr_flux"
    );
    static final List<String> BASE = buildBaseNames();
    static final Set<String> ABSOLUTE = new HashSet<>(Arrays.asList(
            "audio_available",
            "focus_quality",
            "blur_probability",
            "occlusion_fraction",
            "visibility_quality",
            "camera_shift_response"
    ));
    static final int[] CONTEXT_OFFSETS_SECONDS = {-2, -1, 0, 1, 2};

    private FeatureSchema() {}

    private static List<String> buildFrameNames() {
        ArrayList<String> names = new ArrayList<>(List.of(
                "luma_mean", "luma_std", "saturation_mean", "saturation_std",
                "edge_density", "sharpness"
        ));
        appendGrid(names, "luma_grid_");
        names.addAll(List.of("diff_mean", "diff_std", "diff_p90", "diff_active_fraction"));
        appendGrid(names, "diff_grid_");
        names.addAll(List.of(
                "focus_quality", "blur_probability", "dark_fraction", "bright_fraction",
                "low_texture_fraction", "occlusion_fraction", "visibility_quality",
                "camera_shift_x", "camera_shift_y", "camera_shift_magnitude",
                "camera_shift_response", "flow_mean", "flow_p90", "flow_active_fraction",
                "flow_median_x", "flow_median_y"
        ));
        appendGrid(names, "flow_grid_");
        names.addAll(List.of(
                "player_motion_mean", "player_motion_p90", "player_motion_active_fraction",
                "player_motion_active_zone_fraction", "player_motion_spatial_entropy",
                "player_motion_centroid_x", "player_motion_centroid_y",
                "player_motion_spread_x", "player_motion_spread_y",
                "player_motion_coherence", "quality_gated_player_motion"
        ));
        appendGrid(names, "player_motion_grid_");
        return List.copyOf(names);
    }

    private static void appendGrid(List<String> names, String prefix) {
        for (int index = 0; index < 9; index++) names.add(prefix + index);
    }

    private static List<String> buildBaseNames() {
        ArrayList<String> names = new ArrayList<>(FRAME.size() + TEMPORAL.size() + AUDIO.size());
        names.addAll(FRAME);
        names.addAll(TEMPORAL);
        names.addAll(AUDIO);
        return List.copyOf(names);
    }

    static List<String> contextualNames() {
        ArrayList<String> names = new ArrayList<>(BASE.size() * CONTEXT_OFFSETS_SECONDS.length);
        for (int offset : CONTEXT_OFFSETS_SECONDS) {
            String prefix = "t" + (offset >= 0 ? "+" : "") + offset + "s/";
            for (String name : BASE) names.add(prefix + name);
        }
        return names;
    }
}

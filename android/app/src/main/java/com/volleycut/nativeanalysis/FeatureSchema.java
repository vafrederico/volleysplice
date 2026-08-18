package com.volleycut.nativeanalysis;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

final class FeatureSchema {
    static final String ALL_LABELS_V2_MODEL_ID = "model-1ca43e38eefc";
    static final String PREVIOUS_PRODUCTION_MODEL_ID = "model-9c92b8e9333f";
    static final String SUPPRESSION_MODEL_ID = "suppression-overlap-exclusion-retrained";
    static final String SUPPRESSION_ARTIFACT_SHA256 =
            "39eddf58163901930434ea422a802686ae921ae1e8fe23c20a3c12e5f453da93";
    static final String SUPPRESSION_WEIGHTS_SHA256 =
            "a943749b69c98a1bc926f8efc9fe60c67c1226534fe09a892c632519217aa3bb";
    static final String SUPPRESSION_DECODER_VERSION = "held-production-suppression-decoder-v1";
    static final String SUPPRESSION_POLICY_CONTRACT_VERSION = "suppression-policy-v1";
    static final String ALL_LABELS_V2_BUNDLE_SHA256 =
            "d2c2c11e8fed8b6c6ad77d244b613e81d5bab101939a8f57be5166b45ebca78f";
    static final String PREVIOUS_PRODUCTION_BUNDLE_SHA256 =
            "d8cc42f70bc10576a5e03251b05981ceeee1a61a15c61cc5dfb68dd631e6f90d";
    static final String ENSEMBLE_ALGORITHM_VERSION = "overlap-union-disagreement-v1";
    static final String MODEL_ID = "ensemble-" + ENSEMBLE_ALGORITHM_VERSION + "-"
            + ALL_LABELS_V2_BUNDLE_SHA256 + "-" + PREVIOUS_PRODUCTION_BUNDLE_SHA256;
    static final int ANALYSIS_FPS = 4;
    static final int ANALYSIS_WIDTH = 192;
    static final int ANALYSIS_HEIGHT = 108;
    static final int BENCHMARK_SOURCE_FRAME_LIMIT = 1_000;
    static final int FULL_SOURCE_FRAME_LIMIT = Integer.MAX_VALUE;

    static String modelAsset(String modelId) {
        return modelId + ".json";
    }

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

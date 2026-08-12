#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analysis.artifacts import atomic_write_text
from analysis.dead_ball_experiment import (
    _prediction_inputs as dead_ball_inputs,
    _predictions_for as dead_ball_predictions,
    _validate_model_triplet as validate_dead_ball_triplet,
)
from analysis.dead_state_experiment import (
    _global_dead_rally_decoder,
    _predictions_for as dead_state_predictions,
    _validate_model_triplet as validate_dead_state_triplet,
)
from analysis.decoder import DecodedInterval, decode_probabilities
from analysis.features import camera_warnings, write_preview
from analysis.model import LogisticModel, ModelError, load_model
from analysis.pipeline import PreparedRecording, _manifest_digest, prepare_recording
from analysis.schema import Recording, load_manifest
from analysis.serve_evidence_experiment import (
    _stack_peak_features,
    _validate_models as validate_peak_models,
)
from analysis.serve_experiment import _effective_analysis_fps
from analysis.stacked_serve_experiment import (
    _stack_prepared,
    _validate_stacked_models,
)
from analysis.version import __version__


V4_RALLY = "full-audiovisual-v2-final"
V4_SERVE = "serve-specialist-audiovisual-v4"
STACK_CONTROL = "full-audiovisual-serve-prob-stack-v1-control"
V5_RALLY = "full-audiovisual-audio-normalized-v3"
V5_SERVE = "serve-specialist-audio-normalized-v5"

STACKED_MODELS = (
    "full-audiovisual-serve-prob-stack-v1-exploratory",
    "full-audiovisual-serve-peak-window-v1-exploratory",
)
V4_DEAD_BALL_MODELS = (
    "dead-ball-specialist-audiovisual-v1",
    "dead-ball-specialist-audiovisual-v2",
)
V5_DEAD_BALL_MODELS = (
    "dead-ball-specialist-audio-normalized-v2-legacy-only",
    "dead-ball-specialist-audio-normalized-v3",
    "dead-ball-specialist-audio-normalized-v4-no-legacy",
    "dead-ball-specialist-audio-normalized-v5-new-only",
)
DEAD_STATE_MODELS = (
    "dead-state-global-audio-normalized-v1-full-final",
    "dead-state-global-audio-normalized-v2-full-final",
    "dead-state-transition-audio-normalized-v0-legacy-only",
    "dead-state-transition-audio-normalized-v1-full",
    "dead-state-transition-audio-normalized-v2-no-legacy",
    "dead-state-transition-audio-normalized-v3-legacy-only-final",
    "dead-state-transition-audio-normalized-v4-full-final",
    "dead-state-transition-audio-normalized-v5-no-legacy-final",
)


ITERATIONS: dict[str, tuple[str, str]] = {
    STACKED_MODELS[0]: (
        "Stacked rally · continuous serve probability",
        "Exploratory stacking iteration: appends the frozen v4 specialist's continuous serve probability to the 450 audiovisual rally inputs before decoding a new rally head.",
    ),
    STACKED_MODELS[1]: (
        "Stacked rally · ±2 s serve-peak evidence",
        "Exploratory stacking iteration: appends four features describing validation-frozen serve peaks within ±2 seconds. Its recorded validation decision did not promote it.",
    ),
    V4_DEAD_BALL_MODELS[0]: (
        "End detector · audiovisual v1 no-op",
        "First audiovisual dead-ball boundary iteration. Validation selected the exact v4 rally/serve no-op, so this timeline intentionally duplicates its upstream pair.",
    ),
    V4_DEAD_BALL_MODELS[1]: (
        "End detector · audiovisual v2 safe refine",
        "Second audiovisual dead-ball iteration: reuses the v1 boundary head but applies the validation-selected safe end refinement to v4 serve rescues.",
    ),
    V5_DEAD_BALL_MODELS[0]: (
        "End detector · legacy audio only",
        "Matched enhanced-feature endpoint ablation retaining visual plus legacy audio while removing the new normalized-band audio channels.",
    ),
    V5_DEAD_BALL_MODELS[1]: (
        "End detector · full normalized audio",
        "Endpoint iteration using the full enhanced visual, legacy-audio, and noise-normalized frequency-band audio feature set.",
    ),
    V5_DEAD_BALL_MODELS[2]: (
        "End detector · normalized audio, no legacy",
        "Endpoint ablation removing legacy audio while retaining visual and the new noise-normalized frequency-band audio features.",
    ),
    V5_DEAD_BALL_MODELS[3]: (
        "End detector · normalized-band audio only",
        "Endpoint attribution iteration using only the new normalized-band audio inputs. Validation selected a no-op composition, so its timeline duplicates the upstream pair.",
    ),
    DEAD_STATE_MODELS[0]: (
        "Dead state · global v1 refinement",
        "First global dead-time-state iteration, used only to refine the ends of the enhanced v5 rally/serve composition.",
    ),
    DEAD_STATE_MODELS[1]: (
        "Dead state · global v2 inverse rally",
        "Second global dead-time iteration shown through its validation-selected direct rally control: the rally probability is one minus predicted dead-time probability.",
    ),
    DEAD_STATE_MODELS[2]: (
        "Dead transition · legacy audio v0",
        "Local end-transition iteration using visual plus legacy audio features; scores outside the trained end window are not global dead-time probabilities.",
    ),
    DEAD_STATE_MODELS[3]: (
        "Dead transition · full audio v1",
        "Local end-transition iteration using the full enhanced visual and old+new audio feature set to refine v5-composed endpoints.",
    ),
    DEAD_STATE_MODELS[4]: (
        "Dead transition · no legacy audio v2",
        "Local end-transition iteration using visual plus normalized frequency-band audio while removing the legacy audio channels.",
    ),
    DEAD_STATE_MODELS[5]: (
        "Dead transition · legacy audio v3 final",
        "Finalized legacy-audio transition artifact. It preserves the v0 learned head and selection as immutable final provenance.",
    ),
    DEAD_STATE_MODELS[6]: (
        "Dead transition · full audio v4 final",
        "Finalized full-audio transition artifact. It preserves the v1 learned head and validation-selected refinement as immutable final provenance.",
    ),
    DEAD_STATE_MODELS[7]: (
        "Dead transition · no legacy audio v5 final",
        "Finalized no-legacy transition artifact. It preserves the v2 learned head and validation-selected refinement as immutable final provenance.",
    ),
}


def _run_id(model: LogisticModel, recording_id: str) -> str:
    if not model.artifact_sha256:
        raise ModelError("specialist model has no immutable artifact hash")
    value = f"model-{model.artifact_sha256[:12]}--{recording_id}"
    if len(value) > 80:
        raise ModelError(f"specialist dashboard id is too long: {value}")
    return value


def _model_row(path: Path, model: LogisticModel) -> dict[str, Any]:
    return {"version": path.name, "sha256": model.artifact_sha256}


def _write_analysis(
    recording: Recording,
    model_path: Path,
    model: LogisticModel,
    base_path: Path,
    base: LogisticModel,
    prepared: PreparedRecording,
    intervals: Sequence[DecodedInterval],
    output_root: Path,
    *,
    method: str,
    selection: dict[str, Any],
    dependencies: dict[str, Any],
) -> str:
    run_id = _run_id(model, recording.id)
    destination = output_root / run_id
    analysis_path = destination / "analysis.json"
    preview_path = destination / "court-preview.jpg"
    label, description = ITERATIONS[model_path.name]
    expected_models = {
        "specialist": _model_row(model_path, model),
        "rally": _model_row(base_path, base),
        **dependencies["models"],
    }
    if destination.exists():
        if (
            not analysis_path.is_file()
            or not preview_path.is_file()
            or preview_path.stat().st_size == 0
        ):
            raise ModelError(f"incomplete specialist output exists: {destination}")
        try:
            existing = json.loads(analysis_path.read_text(encoding="utf-8"))
            existing_analysis = existing["analysis"]
            existing_source = existing["source"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ModelError(
                f"invalid existing specialist output {analysis_path}: {error}"
            ) from error
        expected = (
            existing.get("id") == run_id
            and existing.get("recordingId") == recording.id
            and existing_source.get("filename") == recording.video.name
            and existing_source.get("contentSha256") == recording.content_sha256
            and existing_analysis.get("method") == method
            and existing_analysis.get("modelVersion") == model_path.name
            and existing_analysis.get("modelSha256") == model.artifact_sha256
            and existing_analysis.get("variantLabel") == f"Trained model · {label}"
            and existing_analysis.get("variantDescription") == description
            and existing_analysis.get("models") == expected_models
            and existing_analysis.get("selection") == selection
            and isinstance(existing.get("rallies"), list)
        )
        if not expected:
            raise ModelError(f"specialist output is bound to another artifact: {destination}")
        return "skipped"

    metadata = prepared.sequence.metadata
    warnings = camera_warnings(metadata, recording.capture)
    warnings.append("experimental model iteration retained for timeline comparison")
    roi = recording.roi
    roi_payload = (
        {"x": roi[0], "y": roi[1], "width": roi[2], "height": roi[3]}
        if roi is not None
        else None
    )
    rows = [
        {
            "id": f"R{index:03d}",
            "start": max(0.0, round(float(interval.start), 3)),
            "end": min(metadata.duration, round(float(interval.end), 3)),
            "confidence": round(
                max(0.0, min(1.0, float(interval.confidence))), 5
            ),
            "included": True,
        }
        for index, interval in enumerate(intervals, start=1)
        if interval.end > interval.start
    ]
    payload = {
        "schemaVersion": 1,
        "id": run_id,
        "recordingId": recording.id,
        "title": recording.id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "source": {
            "filename": recording.video.name,
            "contentSha256": recording.content_sha256,
            **metadata.to_dict(),
        },
        "assets": {"courtPreviewPath": "court-preview.jpg"},
        "analysis": {
            "method": method,
            "modelVersion": model_path.name,
            "modelSha256": model.artifact_sha256,
            "variantLabel": f"Trained model · {label}",
            "variantDescription": description,
            "producer": f"volleycut-analysis/{__version__}",
            "analysisFps": base.feature_config.analysis_fps,
            "models": expected_models,
            "selection": selection,
            "warnings": warnings,
            "court": {
                "source": "manual-roi" if roi is not None else "full-frame-fallback",
                "roi": roi_payload,
            },
        },
        "rallies": rows,
    }
    destination.mkdir(parents=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".court-preview-", suffix=".jpg", dir=destination
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    reference_preview = (
        output_root
        / f"model-full-percentile-v1--{recording.id}"
        / "court-preview.jpg"
    )
    try:
        if reference_preview.is_file():
            shutil.copyfile(reference_preview, temporary)
        else:
            write_preview(recording.video, temporary, recording.roi)
        temporary.replace(preview_path)
        atomic_write_text(
            analysis_path, json.dumps(payload, indent=2, allow_nan=False) + "\n"
        )
    except Exception:
        temporary.unlink(missing_ok=True)
        analysis_path.unlink(missing_ok=True)
        preview_path.unlink(missing_ok=True)
        try:
            destination.rmdir()
        except OSError:
            pass
        raise
    return "created"


def _decode(model: LogisticModel, prepared: Any) -> tuple[DecodedInterval, ...]:
    intervals, _ = decode_probabilities(
        prepared.sequence.times,
        model.predict(prepared.contextual_values),
        prepared.sequence.metadata.duration,
        model.decoder,
        _effective_analysis_fps(prepared),
    )
    return tuple(intervals)


def _prepare_inference_recording(
    recording: Recording, model: LogisticModel, cache_dir: Path
) -> PreparedRecording:
    prepared = prepare_recording(recording, model.feature_config, cache_dir)
    # Prediction must be independent of gold rally and ignored-interval labels.
    return replace(
        prepared,
        recording=replace(recording, rallies=(), ignored_intervals=()),
        labels=np.zeros(len(prepared.sequence.times), dtype=np.float32),
        sample_mask=np.ones(len(prepared.sequence.times), dtype=np.bool_),
    )


def main() -> int:
    data_root = Path(
        os.environ.get("VOLLEYCUT_DATA_ROOT", "data")
    ).expanduser().resolve()
    workspace = Path(
        os.environ.get(
            "VOLLEYCUT_LABELING_WORKSPACE",
            data_root / "labeling-v1-2026-08-09",
        )
    ).expanduser().resolve()
    parser = argparse.ArgumentParser(
        description="Materialize stacked, endpoint, and dead-state model timelines."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=workspace / "manifests" / "full-gold-v1.json",
    )
    parser.add_argument("--models-root", type=Path, default=workspace / "models")
    parser.add_argument("--v4-cache", type=Path, default=workspace / "features" / "audiovisual-v2")
    parser.add_argument(
        "--v5-cache",
        type=Path,
        default=workspace / "features" / "audiovisual-audio-normalized-v3",
    )
    parser.add_argument("--output-root", type=Path, default=data_root / "analyses")
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()

    manifest = load_manifest(arguments.manifest)
    manifest_sha256 = _manifest_digest(manifest)
    recordings = list(manifest.recordings)
    if arguments.limit is not None:
        if arguments.limit < 1:
            raise ValueError("--limit must be positive")
        recordings = recordings[: arguments.limit]
    models_root = arguments.models_root.expanduser().resolve()
    output_root = arguments.output_root.expanduser().resolve()

    def load(name: str) -> tuple[Path, LogisticModel]:
        path = models_root / name
        return path, load_model(path)

    v4_path, v4 = load(V4_RALLY)
    v4_serve_path, v4_serve = load(V4_SERVE)
    control_path, control = load(STACK_CONTROL)
    v5_path, v5 = load(V5_RALLY)
    v5_serve_path, v5_serve = load(V5_SERVE)
    created = skipped = 0

    stack_models = [load(name) for name in STACKED_MODELS]
    _validate_stacked_models(
        v4, v4_serve, control, stack_models[0][1], manifest_sha256=manifest_sha256
    )
    peak_decoder, _ = validate_peak_models(
        v4, v4_serve, control, stack_models[1][1], manifest_sha256=manifest_sha256
    )
    v4_dead_models = [load(name) for name in V4_DEAD_BALL_MODELS]
    v4_dead_configs = [
        validate_dead_ball_triplet(v4, v4_serve, model, manifest_sha256=manifest_sha256)
        for _, model in v4_dead_models
    ]
    v5_dead_models = [load(name) for name in V5_DEAD_BALL_MODELS]
    v5_dead_configs = [
        validate_dead_ball_triplet(v5, v5_serve, model, manifest_sha256=manifest_sha256)
        for _, model in v5_dead_models
    ]
    state_models = [load(name) for name in DEAD_STATE_MODELS]
    state_configs = [
        validate_dead_state_triplet(v5, v5_serve, model, manifest_sha256=manifest_sha256)
        for _, model in state_models
    ]

    for index, recording in enumerate(recordings, start=1):
        print(f"Preparing specialist timelines {index}/{len(recordings)}: {recording.id}", flush=True)
        prepared_v4 = _prepare_inference_recording(recording, v4, arguments.v4_cache)
        probability_stacked = _stack_prepared([prepared_v4], v4_serve)[0]
        peak_stacked = _stack_peak_features([prepared_v4], v4_serve, peak_decoder)[0]
        for (model_path, model), prepared, transform in (
            (stack_models[0], probability_stacked, "continuous-serve-probability"),
            (stack_models[1], peak_stacked, "decoded-serve-peaks-plus-minus-2-seconds"),
        ):
            status = _write_analysis(
                recording,
                model_path,
                model,
                v4_path,
                v4,
                prepared_v4,
                _decode(model, prepared),
                output_root,
                method="court-motion-temporal-logistic+stacked-serve-feature-v1",
                selection={"transform": transform, "decoder": model.decoder.to_dict()},
                dependencies={
                    "cacheDir": arguments.v4_cache,
                    "models": {
                        "serve": _model_row(v4_serve_path, v4_serve),
                        "control": _model_row(control_path, control),
                    },
                },
            )
            created += status == "created"
            skipped += status == "skipped"

        for (model_path, model), config in zip(
            v4_dead_models, v4_dead_configs, strict=True
        ):
            serve_decoder, serve_composition, _, selected_decoder, selected, _ = config
            prediction = dead_ball_predictions(
                [
                    dead_ball_inputs(
                        prepared_v4,
                        v4,
                        v4_serve,
                        model,
                        serve_decoder,
                        serve_composition,
                    )
                ],
                selected_decoder,
                selected,
            )[0]
            status = _write_analysis(
                recording, model_path, model, v4_path, v4, prepared_v4, prediction.candidate, output_root,
                method="court-motion-temporal-logistic+dead-ball-refinement-v1",
                selection={
                    "deadBallDecoder": selected_decoder.to_dict(),
                    "composition": selected.to_dict(),
                },
                dependencies={
                    "cacheDir": arguments.v4_cache,
                    "models": {"serve": _model_row(v4_serve_path, v4_serve)},
                },
            )
            created += status == "created"
            skipped += status == "skipped"

        prepared_v5 = _prepare_inference_recording(recording, v5, arguments.v5_cache)
        for (model_path, model), config in zip(
            v5_dead_models, v5_dead_configs, strict=True
        ):
            serve_decoder, serve_composition, _, selected_decoder, selected, _ = config
            prediction = dead_ball_predictions(
                [
                    dead_ball_inputs(
                        prepared_v5,
                        v5,
                        v5_serve,
                        model,
                        serve_decoder,
                        serve_composition,
                    )
                ],
                selected_decoder,
                selected,
            )[0]
            status = _write_analysis(
                recording, model_path, model, v5_path, v5, prepared_v5, prediction.candidate, output_root,
                method="court-motion-temporal-logistic+dead-ball-refinement-v1",
                selection={
                    "deadBallDecoder": selected_decoder.to_dict(),
                    "composition": selected.to_dict(),
                },
                dependencies={
                    "cacheDir": arguments.v5_cache,
                    "models": {"serve": _model_row(v5_serve_path, v5_serve)},
                },
            )
            created += status == "created"
            skipped += status == "skipped"

        for (model_path, model), config in zip(
            state_models, state_configs, strict=True
        ):
            serve_decoder, serve_composition, state_decoder, refinement = config
            inverse = _global_dead_rally_decoder(model)
            if model_path.name == DEAD_STATE_MODELS[1]:
                if inverse is None:
                    raise ModelError("global dead-state v2 lacks its inverse rally decoder")
                inverse_decoder, metadata = inverse
                intervals, _ = decode_probabilities(
                    prepared_v5.sequence.times,
                    1.0 - model.predict(prepared_v5.contextual_values),
                    prepared_v5.sequence.metadata.duration,
                    inverse_decoder,
                    _effective_analysis_fps(prepared_v5),
                )
                selected_intervals = tuple(intervals)
                selection = {
                    "mode": "one-minus-dead-probability",
                    "decoder": inverse_decoder.to_dict(),
                    "selectedOn": metadata.get("selectedOn", "validation"),
                }
                method = "court-motion-temporal-logistic+inverse-dead-state-rally-v1"
            else:
                inputs = dead_ball_inputs(
                    prepared_v5,
                    v5,
                    v5_serve,
                    model,
                    serve_decoder,
                    serve_composition,
                )
                prediction = dead_state_predictions(
                    [inputs], state_decoder, refinement
                )[0]
                selected_intervals = prediction.candidate
                selection = {
                    "deadStateDecoder": state_decoder.to_dict(),
                    "refinement": refinement.to_dict(),
                }
                method = "court-motion-temporal-logistic+dead-state-refinement-v1"
            status = _write_analysis(
                recording, model_path, model, v5_path, v5, prepared_v5, selected_intervals, output_root,
                method=method,
                selection=selection,
                dependencies={
                    "cacheDir": arguments.v5_cache,
                    "models": {"serve": _model_row(v5_serve_path, v5_serve)},
                },
            )
            created += status == "created"
            skipped += status == "skipped"

    print(
        json.dumps(
            {
                "manifest": manifest.name,
                "recordings": len(recordings),
                "modelArtifacts": len(ITERATIONS),
                "created": created,
                "skipped": skipped,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

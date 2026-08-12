from __future__ import annotations

import argparse
import importlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from .artifacts import atomic_write_text
from .config import DecoderConfig, FeatureConfig, TrainingConfig
from .media import X264_PRESETS, NormalizationError, normalize_video
from .schema import ManifestError, load_manifest, manifest_warnings
from .version import __version__


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, allow_nan=False))


def _roi(value: str) -> tuple[float, float, float, float]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height") from error
    if len(parsed) != 4:
        raise argparse.ArgumentTypeError("ROI must contain four comma-separated numbers")
    x, y, width, height = parsed
    if (
        not all(math.isfinite(item) for item in parsed)
        or x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > 1
        or y + height > 1
    ):
        raise argparse.ArgumentTypeError("ROI must be a normalized rectangle inside the frame")
    return parsed  # type: ignore[return-value]


def _default_cache() -> Path:
    return Path("data/features")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis",
        description="Train and evaluate VolleyCut's local rally-detection baseline.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="check local analysis dependencies")
    subparsers.add_parser("smoke", help="run a synthetic train/save/load/decode test")

    validate = subparsers.add_parser("validate", help="validate annotations and split provenance")
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument(
        "--allow-missing-videos",
        action="store_true",
        help="validate the schema before the referenced videos arrive",
    )

    inspect = subparsers.add_parser("inspect", help="inspect capture metadata and basic compatibility")
    inspect.add_argument("--video", required=True, type=Path)

    normalize = subparsers.add_parser("normalize", help="create a constant-frame-rate analysis master")
    normalize.add_argument("--video", required=True, type=Path)
    normalize.add_argument("--output", required=True, type=Path)
    normalize.add_argument("--fps", type=float, default=30.0)
    normalize.add_argument("--max-width", type=int, default=1920)
    normalize.add_argument("--crf", type=int, default=20)
    normalize.add_argument("--preset", choices=sorted(X264_PRESETS), default="medium")
    normalize.add_argument("--threads", type=int, default=2)
    normalize.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="source start time in seconds (default: beginning)",
    )
    normalize.add_argument(
        "--duration",
        type=float,
        help="optional excerpt duration in seconds",
    )

    train = subparsers.add_parser("train", help="extract features and train a model")
    train.add_argument("--manifest", required=True, type=Path)
    train.add_argument("--model", required=True, type=Path, help="new model artifact directory")
    train.add_argument("--cache-dir", type=Path, default=_default_cache())
    train.add_argument("--analysis-fps", type=float, default=4.0)
    train.add_argument("--resize-width", type=int, default=192)
    train.add_argument("--resize-height", type=int, default=108)
    train.add_argument("--no-optical-flow", action="store_true")
    train.add_argument("--no-advanced-visual", action="store_true")
    train.add_argument("--no-audio", action="store_true")
    train.add_argument("--audio-sample-rate", type=int, default=16000)
    train.add_argument(
        "--sequence-normalization",
        choices=("none", "percentile-rank"),
        default="percentile-rank",
    )
    train.add_argument("--epochs", type=int, default=180)
    train.add_argument("--batch-size", type=int, default=2048)
    train.add_argument("--learning-rate", type=float, default=0.02)
    train.add_argument("--seed", type=int, default=7)

    train_serve = subparsers.add_parser(
        "train-serve",
        help="train a serve-contact specialist and select rally composition on validation",
    )
    train_serve.add_argument("--manifest", required=True, type=Path)
    train_serve.add_argument("--rally-model", required=True, type=Path)
    train_serve.add_argument("--model", required=True, type=Path, help="new serve model directory")
    train_serve.add_argument("--cache-dir", type=Path, default=_default_cache())
    train_serve.add_argument("--output", type=Path, help="new validation experiment report")
    train_serve.add_argument("--target-radius", type=float, default=1.0)
    train_serve.add_argument("--epochs", type=int, default=180)
    train_serve.add_argument("--batch-size", type=int, default=2048)
    train_serve.add_argument("--learning-rate", type=float, default=0.02)
    train_serve.add_argument("--seed", type=int, default=7)

    train_stacked = subparsers.add_parser(
        "train-rally-with-serve",
        help="train a rally head with cross-fitted serve probability as an added feature",
    )
    train_stacked.add_argument("--manifest", required=True, type=Path)
    train_stacked.add_argument("--baseline-rally-model", required=True, type=Path)
    train_stacked.add_argument("--serve-model", required=True, type=Path)
    train_stacked.add_argument("--control-model", required=True, type=Path)
    train_stacked.add_argument("--model", required=True, type=Path)
    train_stacked.add_argument("--cache-dir", type=Path, default=_default_cache())
    train_stacked.add_argument("--output", type=Path)
    train_stacked.add_argument("--epochs", type=int, default=180)
    train_stacked.add_argument("--batch-size", type=int, default=2048)
    train_stacked.add_argument("--learning-rate", type=float, default=0.02)
    train_stacked.add_argument("--seed", type=int, default=7)

    train_dead_ball = subparsers.add_parser(
        "train-dead-ball",
        help="train a rally-end specialist and select serve-gated closure on validation",
    )
    train_dead_ball.add_argument("--manifest", required=True, type=Path)
    train_dead_ball.add_argument("--rally-model", required=True, type=Path)
    train_dead_ball.add_argument("--serve-model", required=True, type=Path)
    train_dead_ball.add_argument("--model", required=True, type=Path)
    train_dead_ball.add_argument("--cache-dir", type=Path, default=_default_cache())
    train_dead_ball.add_argument("--output", type=Path)
    train_dead_ball.add_argument("--target-radius", type=float, default=0.5)
    train_dead_ball.add_argument("--epochs", type=int, default=120)
    train_dead_ball.add_argument("--batch-size", type=int, default=2048)
    train_dead_ball.add_argument("--learning-rate", type=float, default=0.02)
    train_dead_ball.add_argument("--seed", type=int, default=7)

    infer = subparsers.add_parser("infer", help="detect rallies in one continuous video")
    infer.add_argument("--model", required=True, type=Path)
    infer.add_argument(
        "--serve-model",
        type=Path,
        help="optional paired serve-contact specialist with frozen composition metadata",
    )
    infer.add_argument("--video", required=True, type=Path)
    infer.add_argument("--output", required=True, type=Path, help="new analysis directory")
    infer.add_argument("--roi", type=_roi, help="normalized x,y,width,height court rectangle")
    infer.add_argument("--title")

    evaluate = subparsers.add_parser("evaluate", help="evaluate an immutable manifest split")
    evaluate.add_argument("--manifest", required=True, type=Path)
    evaluate.add_argument("--model", required=True, type=Path)
    evaluate.add_argument("--cache-dir", type=Path, default=_default_cache())
    evaluate.add_argument("--split", choices=("validation", "test", "challenge"), default="test")
    evaluate.add_argument("--output", type=Path)

    evaluate_serve = subparsers.add_parser(
        "evaluate-serve", help="compare a frozen rally+serve composition on one manifest split"
    )
    evaluate_serve.add_argument("--manifest", required=True, type=Path)
    evaluate_serve.add_argument("--rally-model", required=True, type=Path)
    evaluate_serve.add_argument("--serve-model", required=True, type=Path)
    evaluate_serve.add_argument("--cache-dir", type=Path, default=_default_cache())
    evaluate_serve.add_argument(
        "--split", choices=("validation", "test", "challenge"), default="test"
    )
    evaluate_serve.add_argument("--output", type=Path)

    evaluate_stacked = subparsers.add_parser(
        "evaluate-rally-with-serve",
        help="evaluate a rally head that consumes a frozen serve probability feature",
    )
    evaluate_stacked.add_argument("--manifest", required=True, type=Path)
    evaluate_stacked.add_argument("--baseline-rally-model", required=True, type=Path)
    evaluate_stacked.add_argument("--serve-model", required=True, type=Path)
    evaluate_stacked.add_argument("--control-model", required=True, type=Path)
    evaluate_stacked.add_argument("--model", required=True, type=Path)
    evaluate_stacked.add_argument("--cache-dir", type=Path, default=_default_cache())
    evaluate_stacked.add_argument(
        "--split", choices=("validation", "test", "challenge"), default="test"
    )
    evaluate_stacked.add_argument("--output", type=Path)

    evaluate_dead_ball = subparsers.add_parser(
        "evaluate-dead-ball",
        help="evaluate a frozen rally, serve, and dead-ball model triplet",
    )
    evaluate_dead_ball.add_argument("--manifest", required=True, type=Path)
    evaluate_dead_ball.add_argument("--rally-model", required=True, type=Path)
    evaluate_dead_ball.add_argument("--serve-model", required=True, type=Path)
    evaluate_dead_ball.add_argument("--model", required=True, type=Path)
    evaluate_dead_ball.add_argument("--cache-dir", type=Path, default=_default_cache())
    evaluate_dead_ball.add_argument(
        "--split", choices=("validation", "test", "challenge"), default="test"
    )
    evaluate_dead_ball.add_argument("--output", type=Path)

    initialize = subparsers.add_parser("init-manifest", help="write an annotation manifest template")
    initialize.add_argument("--output", required=True, type=Path)

    prepare_label = subparsers.add_parser(
        "prepare-label",
        help="create a resumable rally-label document for one normalized video",
    )
    prepare_label.add_argument("--video", required=True, type=Path)
    prepare_label.add_argument("--output", required=True, type=Path)
    prepare_label.add_argument("--id", required=True)
    prepare_label.add_argument("--source-group", required=True)
    prepare_label.add_argument("--split", choices=sorted(("train", "validation", "test", "challenge")), required=True)
    prepare_label.add_argument("--environment", choices=sorted(("indoor", "beach", "grass", "broadcast", "unknown")), required=True)
    prepare_label.add_argument("--players-per-team", type=int)
    prepare_label.add_argument("--target-points", type=int)
    prepare_label.add_argument("--format-name")
    prepare_label.add_argument("--roi", type=_roi)
    prepare_label.add_argument("--position", default="centered-behind-endline")
    prepare_label.add_argument("--height-meters", type=float)

    validate_labels = subparsers.add_parser(
        "validate-labels",
        help="validate a rally-label document and its referenced video",
    )
    validate_labels.add_argument("--labels", required=True, type=Path)
    validate_labels.add_argument("--allow-incomplete", action="store_true")
    validate_labels.add_argument("--allow-missing-video", action="store_true")

    build_manifest = subparsers.add_parser(
        "build-manifest",
        help="combine completed *.labels.json files into a training manifest",
    )
    build_manifest.add_argument("--labels-dir", required=True, type=Path)
    build_manifest.add_argument("--output", required=True, type=Path)
    build_manifest.add_argument("--name", required=True)

    freeze_labels = subparsers.add_parser(
        "freeze-labels",
        help="freeze reviewed label drafts as an immutable completed snapshot",
    )
    freeze_labels.add_argument("--labels-dir", required=True, type=Path)
    freeze_labels.add_argument("--output-dir", required=True, type=Path)
    freeze_labels.add_argument("--annotator", required=True)
    freeze_labels.add_argument(
        "--drop-touching-duplicate-tails",
        action="store_true",
        help=(
            "drop only zero-gap second intervals whose tags and notes exactly duplicate the "
            "preceding rally, recording every removal in the snapshot ledger"
        ),
    )

    prepare_workspace = subparsers.add_parser(
        "prepare-labeling-workspace",
        help="create or resume normalized full-video labeling tasks",
    )
    prepare_workspace.add_argument("--plan", required=True, type=Path)
    prepare_workspace.add_argument("--workspace", required=True, type=Path)
    prepare_workspace.add_argument("--fps", type=float, default=30.0)
    prepare_workspace.add_argument("--max-width", type=int, default=960)
    prepare_workspace.add_argument("--crf", type=int, default=24)
    prepare_workspace.add_argument("--preset", choices=sorted(X264_PRESETS), default="ultrafast")
    prepare_workspace.add_argument("--threads", type=int, default=1)

    import_prelabels = subparsers.add_parser(
        "import-model-prelabels",
        help="validate blind audiovisual candidates and create isolated review drafts",
    )
    import_prelabels.add_argument("--candidates-dir", required=True, type=Path)
    import_prelabels.add_argument("--tasks-dir", required=True, type=Path)
    import_prelabels.add_argument("--output-dir", required=True, type=Path)
    return parser


def _doctor() -> tuple[dict[str, Any], int]:
    def imported(module_name: str) -> dict[str, Any]:
        try:
            module = importlib.import_module(module_name)
            return {"available": True, "version": str(getattr(module, "__version__", "unknown"))}
        except Exception as error:  # dependency diagnostics must survive binary/ABI failures
            return {"available": False, "error": f"{type(error).__name__}: {error}"}

    python_supported = sys.version_info >= (3, 10)
    dependencies = {
        "python": {"available": python_supported, "version": sys.version.split()[0], "minimum": "3.10"},
        "numpy": imported("numpy"),
        "opencv": imported("cv2"),
        "ffmpeg": {"available": shutil.which("ffmpeg") is not None},
        "ffprobe": {"available": shutil.which("ffprobe") is not None},
    }
    required = all(
        dependencies[name]["available"]
        for name in ("python", "numpy", "opencv", "ffmpeg", "ffprobe")
    )
    result = {
        "ready": bool(required),
        "dependencies": dependencies,
        "note": "ffmpeg/ffprobe are required by the default audiovisual feature extractor",
    }
    return result, 0 if required else 1


def _manifest_template() -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "name": "volleycut-feasibility",
        "annotationPolicy": {
            "id": "serve-contact-to-dead-ball-v1",
            "rallyStart": "serve-ball contact",
            "rallyEnd": "first instant live play has ended",
            "intervalConvention": "half-open [start,end) seconds on the normalized video",
        },
        "recordings": [
            {
                "id": "match-001-set-01",
                "video": "../videos/indoor/match-001-set-01.mp4",
                "split": "train",
                "sourceGroup": "match-001",
                "environment": "indoor",
                "game": {
                    "playersPerTeam": 6,
                    "targetPoints": 25,
                    "format": "indoor 6v6",
                    "scoringRule": "rally scoring",
                },
                "consent": {"analyze": True, "train": True},
                "capture": {
                    "position": "centered-behind-endline",
                    "stationary": True,
                    "fullCourtVisible": True,
                    "serviceAreasVisible": True,
                    "heightMeters": None,
                },
                "roi": {"x": 0.05, "y": 0.12, "width": 0.9, "height": 0.86},
                "rallies": [
                    {"start": 12.4, "end": 21.75},
                    {"start": 36.1, "end": 44.0},
                ],
            }
        ],
    }


def run(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "doctor":
            result, exit_code = _doctor()
            _json(result)
            return exit_code
        if arguments.command == "smoke":
            from .pipeline import smoke_test

            result = smoke_test()
            _json(result)
            return 0 if result["passed"] else 1
        if arguments.command == "validate":
            from .features import camera_warnings, probe_video

            manifest = load_manifest(
                arguments.manifest,
                require_videos=not arguments.allow_missing_videos,
            )
            warnings = manifest_warnings(manifest)
            if not arguments.allow_missing_videos:
                for recording in manifest.recordings:
                    metadata = probe_video(recording.video)
                    if recording.rallies and recording.rallies[-1].end > metadata.duration + 1e-6:
                        raise ManifestError(
                            f"{recording.id}: final annotation ends after video duration "
                            f"({recording.rallies[-1].end:.3f}s > {metadata.duration:.3f}s)"
                        )
                    for warning in camera_warnings(metadata, recording.capture):
                        message = f"{recording.id}: {warning}"
                        if message not in warnings:
                            warnings.append(message)
            _json(
                {
                    "valid": True,
                    "name": manifest.name,
                    "recordings": len(manifest.recordings),
                    "splits": {
                        split: len(manifest.for_split(split))
                        for split in ("train", "validation", "test", "challenge")
                    },
                    "warnings": warnings,
                }
            )
            return 0
        if arguments.command == "inspect":
            from .features import camera_warnings, probe_video

            metadata = probe_video(arguments.video)
            warnings = camera_warnings(metadata)
            _json({"video": str(arguments.video.resolve()), "metadata": metadata.to_dict(), "warnings": warnings})
            return 0
        if arguments.command == "normalize":
            _json(
                normalize_video(
                    arguments.video,
                    arguments.output,
                    fps=arguments.fps,
                    max_width=arguments.max_width,
                    crf=arguments.crf,
                    start_seconds=arguments.start,
                    duration_seconds=arguments.duration,
                    preset=arguments.preset,
                    threads=arguments.threads,
                )
            )
            return 0
        if arguments.command == "train":
            from .pipeline import train_dataset

            features = FeatureConfig(
                analysis_fps=arguments.analysis_fps,
                resize_width=arguments.resize_width,
                resize_height=arguments.resize_height,
                use_optical_flow=not arguments.no_optical_flow,
                use_advanced_visual=not arguments.no_advanced_visual,
                use_audio=not arguments.no_audio,
                audio_sample_rate=arguments.audio_sample_rate,
                sequence_normalization=arguments.sequence_normalization,
            )
            training = TrainingConfig(
                epochs=arguments.epochs,
                batch_size=arguments.batch_size,
                learning_rate=arguments.learning_rate,
                seed=arguments.seed,
            )
            result = train_dataset(
                arguments.manifest,
                arguments.model,
                arguments.cache_dir,
                feature_config=features,
                training_config=training,
                decoder_config=DecoderConfig(),
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            _json(result)
            return 0
        if arguments.command == "train-serve":
            from .serve_experiment import train_serve_dataset

            result = train_serve_dataset(
                arguments.manifest,
                arguments.rally_model,
                arguments.model,
                arguments.cache_dir,
                target_radius_seconds=arguments.target_radius,
                training_config=TrainingConfig(
                    epochs=arguments.epochs,
                    batch_size=arguments.batch_size,
                    learning_rate=arguments.learning_rate,
                    seed=arguments.seed,
                ),
                output_path=arguments.output,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            _json(
                {
                    "model": str(arguments.model.expanduser().resolve()),
                    "serveSpottingAt1Second": result["serveSpotting"]["at1Second"],
                    "baseline": result["baseline"]["aggregate"],
                    "composed": result["composed"]["aggregate"],
                    "delta": result["delta"],
                }
            )
            return 0
        if arguments.command == "train-rally-with-serve":
            from .stacked_serve_experiment import train_stacked_rally_dataset

            result = train_stacked_rally_dataset(
                arguments.manifest,
                arguments.baseline_rally_model,
                arguments.serve_model,
                arguments.control_model,
                arguments.model,
                arguments.cache_dir,
                training_config=TrainingConfig(
                    epochs=arguments.epochs,
                    batch_size=arguments.batch_size,
                    learning_rate=arguments.learning_rate,
                    seed=arguments.seed,
                ),
                output_path=arguments.output,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            _json(
                {
                    "controlModel": str(arguments.control_model.expanduser().resolve()),
                    "stackedModel": str(arguments.model.expanduser().resolve()),
                    "stackedFeature": result["stackedFeature"],
                    "frozenBaseline": result["frozenBaseline"]["aggregate"],
                    "retrainedControl": result["retrainedControl"]["aggregate"],
                    "stackedRally": result["stackedRally"]["aggregate"],
                    "deltaVsRetrainedControl": result["deltaVsRetrainedControl"],
                }
            )
            return 0
        if arguments.command == "train-dead-ball":
            from .dead_ball_experiment import train_dead_ball_dataset

            result = train_dead_ball_dataset(
                arguments.manifest,
                arguments.rally_model,
                arguments.serve_model,
                arguments.model,
                arguments.cache_dir,
                target_radius_seconds=arguments.target_radius,
                training_config=TrainingConfig(
                    epochs=arguments.epochs,
                    batch_size=arguments.batch_size,
                    learning_rate=arguments.learning_rate,
                    seed=arguments.seed,
                ),
                output_path=arguments.output,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            _json(
                {
                    "model": str(arguments.model.expanduser().resolve()),
                    "endSpottingAt1Second": result["endSpotting"]["at1Second"],
                    "v4Composition": result["v4Composition"]["aggregate"],
                    "selected": result["selected"]["aggregate"],
                    "deltaSelectedVsV4": result["deltaSelectedVsV4"],
                }
            )
            return 0
        if arguments.command == "infer":
            from .pipeline import infer_video

            result = infer_video(
                arguments.video,
                arguments.model,
                arguments.output,
                roi=arguments.roi,
                title=arguments.title,
                serve_model_path=arguments.serve_model,
            )
            _json(
                {
                    "analysis": str((arguments.output / "analysis.json").resolve()),
                    "rallies": len(result["rallies"]),
                    "warnings": result["analysis"]["warnings"],
                }
            )
            return 0
        if arguments.command == "evaluate":
            from .pipeline import evaluate_dataset

            result = evaluate_dataset(
                arguments.manifest,
                arguments.model,
                arguments.cache_dir,
                split=arguments.split,
                output_path=arguments.output,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            _json(result["aggregate"])
            return 0
        if arguments.command == "evaluate-serve":
            from .serve_experiment import evaluate_serve_dataset

            result = evaluate_serve_dataset(
                arguments.manifest,
                arguments.rally_model,
                arguments.serve_model,
                arguments.cache_dir,
                split=arguments.split,
                output_path=arguments.output,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            _json(
                {
                    "baseline": result["baseline"]["aggregate"],
                    "composed": result["composed"]["aggregate"],
                    "delta": result["delta"],
                }
            )
            return 0
        if arguments.command == "evaluate-rally-with-serve":
            from .stacked_serve_experiment import evaluate_stacked_rally_dataset

            result = evaluate_stacked_rally_dataset(
                arguments.manifest,
                arguments.baseline_rally_model,
                arguments.serve_model,
                arguments.control_model,
                arguments.model,
                arguments.cache_dir,
                split=arguments.split,
                output_path=arguments.output,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            _json(
                {
                    "frozenBaseline": result["frozenBaseline"]["aggregate"],
                    "retrainedControl": result["retrainedControl"]["aggregate"],
                    "stackedFrozenDecoder": result["stackedFrozenDecoder"]["aggregate"],
                    "stackedRally": result["stackedRally"]["aggregate"],
                    "stackedServeFeatureClamped": result["stackedServeFeatureClamped"][
                        "aggregate"
                    ],
                    "deltaVsRetrainedControl": result["deltaVsRetrainedControl"],
                }
            )
            return 0
        if arguments.command == "evaluate-dead-ball":
            from .dead_ball_experiment import evaluate_dead_ball_dataset

            result = evaluate_dead_ball_dataset(
                arguments.manifest,
                arguments.rally_model,
                arguments.serve_model,
                arguments.model,
                arguments.cache_dir,
                split=arguments.split,
                output_path=arguments.output,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            _json(
                {
                    "endSpottingAt1Second": result["endSpotting"]["at1Second"],
                    "v4Composition": result["v4Composition"]["aggregate"],
                    "selected": result["selected"]["aggregate"],
                    "deltaSelectedVsV4": result["deltaSelectedVsV4"],
                }
            )
            return 0
        if arguments.command == "init-manifest":
            destination = arguments.output.expanduser().resolve()
            if destination.exists():
                raise ManifestError(f"refusing to overwrite existing file: {destination}")
            atomic_write_text(
                destination,
                json.dumps(_manifest_template(), indent=2, allow_nan=False) + "\n",
            )
            _json({"created": str(destination)})
            return 0
        if arguments.command == "prepare-label":
            from .annotations import create_label_draft

            payload = create_label_draft(
                arguments.video,
                arguments.output,
                recording_id=arguments.id,
                source_group=arguments.source_group,
                split=arguments.split,
                environment=arguments.environment,
                players_per_team=arguments.players_per_team,
                target_points=arguments.target_points,
                format_name=arguments.format_name,
                roi=arguments.roi,
                capture={
                    "position": arguments.position,
                    "stationary": True,
                    "fullCourtVisible": True,
                    "serviceAreasVisible": True,
                    "heightMeters": arguments.height_meters,
                },
            )
            _json(
                {
                    "created": str(arguments.output.expanduser().resolve()),
                    "recording": payload["recording"]["id"],
                    "video": payload["recording"]["videoFilename"],
                    "durationSeconds": payload["recording"]["durationSeconds"],
                }
            )
            return 0
        if arguments.command == "validate-labels":
            from .annotations import load_label_document

            document = load_label_document(
                arguments.labels,
                require_complete=not arguments.allow_incomplete,
                require_video=not arguments.allow_missing_video,
            )
            _json(
                {
                    "valid": True,
                    "recording": document.recording_id,
                    "durationSeconds": document.duration,
                    "rallies": len(document.rallies),
                    "ignoredIntervals": len(document.ignored_intervals),
                    "hardNegatives": len(document.hard_negatives),
                    "warnings": list(document.warnings),
                }
            )
            return 0
        if arguments.command == "build-manifest":
            from .annotations import build_manifest_from_labels

            label_paths = sorted(arguments.labels_dir.expanduser().resolve().glob("*.labels.json"))
            payload = build_manifest_from_labels(
                label_paths,
                arguments.output,
                name=arguments.name,
            )
            _json(
                {
                    "created": str(arguments.output.expanduser().resolve()),
                    "recordings": len(payload["recordings"]),
                }
            )
            return 0
        if arguments.command == "freeze-labels":
            from .annotations import freeze_label_snapshot

            label_paths = sorted(
                arguments.labels_dir.expanduser().resolve().glob("*.labels.json")
            )
            documents = freeze_label_snapshot(
                label_paths,
                arguments.output_dir,
                annotator=arguments.annotator,
                drop_touching_duplicate_tails=arguments.drop_touching_duplicate_tails,
            )
            _json(
                {
                    "created": str(arguments.output_dir.expanduser().resolve()),
                    "recordings": len(documents),
                    "rallies": sum(len(document.rallies) for document in documents),
                    "annotator": arguments.annotator,
                }
            )
            return 0
        if arguments.command == "prepare-labeling-workspace":
            from .labeling_workspace import prepare_labeling_workspace

            result = prepare_labeling_workspace(
                arguments.plan,
                arguments.workspace,
                fps=arguments.fps,
                max_width=arguments.max_width,
                crf=arguments.crf,
                preset=arguments.preset,
                threads=arguments.threads,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            _json(result)
            return 0
        if arguments.command == "import-model-prelabels":
            from .model_prelabels import materialize_model_prelabels

            _json(
                materialize_model_prelabels(
                    arguments.candidates_dir,
                    arguments.tasks_dir,
                    arguments.output_dir,
                )
            )
            return 0
    except (ManifestError, NormalizationError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 2

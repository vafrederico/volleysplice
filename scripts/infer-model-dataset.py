#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from analysis.model import RALLY_LIVE_TASK, LogisticModel, ModelError, load_model
from analysis.pipeline import infer_video
from analysis.schema import load_manifest
from analysis.serve_experiment import _validate_model_pair


SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_existing_output(
    destination: Path,
    *,
    recording_id: str,
    source_filename: str,
    content_sha256: str | None,
    primary_path: Path,
    primary: LogisticModel,
    serve_path: Path | None,
    serve: LogisticModel | None,
    variant_label: str | None,
    variant_description: str | None,
) -> None:
    analysis_path = destination / "analysis.json"
    preview_path = destination / "court-preview.jpg"
    if (
        not destination.is_dir()
        or not analysis_path.is_file()
        or not preview_path.is_file()
        or preview_path.stat().st_size == 0
    ):
        raise ModelError(f"incomplete inference destination exists: {destination}")
    try:
        payload = json.loads(analysis_path.read_text(encoding="utf-8"))
        analysis = payload["analysis"]
        source = payload["source"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ModelError(f"invalid existing inference output {analysis_path}: {error}") from error
    explicit_recording = payload.get("recordingId")
    if explicit_recording is not None and explicit_recording != recording_id:
        raise ModelError(f"existing inference output has another recording id: {analysis_path}")
    recorded_version = analysis.get("modelVersion")
    if recorded_version is not None and recorded_version != primary_path.name:
        raise ModelError(f"existing inference output has another model version: {analysis_path}")
    valid = (
        payload.get("id") == destination.name
        and source.get("filename") == source_filename
        and (
            source.get("contentSha256") == content_sha256
            # The three original visual-only dashboard runs predate explicit
            # source-content provenance. Keep those immutable legacy outputs
            # loadable while requiring it for every newly materialized run.
            or (
                analysis.get("modelVersion") is None
                and source.get("contentSha256") is None
                and serve is None
            )
        )
        and analysis.get("modelSha256") == primary.artifact_sha256
        and analysis.get("featureConfig") == primary.feature_config.to_dict()
        and analysis.get("decoder") == primary.decoder.to_dict()
        and analysis.get("method")
        == (
            "court-motion-temporal-logistic+serve-specialist-v1"
            if serve is not None
            else "court-motion-temporal-logistic-v0"
        )
        and analysis.get("variantLabel") == variant_label
        and analysis.get("variantDescription") == variant_description
    )
    if serve is not None:
        assert serve_path is not None
        valid = valid and analysis.get("models") == {
            "rally": {
                "version": primary_path.name,
                "sha256": primary.artifact_sha256,
            },
            "serve": {
                "version": serve_path.name,
                "sha256": serve.artifact_sha256,
            },
        }
    else:
        valid = valid and analysis.get("models") is None
    if not valid:
        raise ModelError(f"existing inference output is bound to different inputs: {analysis_path}")


def main() -> int:
    data_root = Path(os.environ.get("VOLLEYCUT_DATA_ROOT", "data")).expanduser().resolve()
    workspace = Path(
        os.environ.get(
            "VOLLEYCUT_LABELING_WORKSPACE",
            data_root / "labeling-v1-2026-08-09",
        )
    ).expanduser().resolve()
    parser = argparse.ArgumentParser(
        description="Run one immutable trained model over every recording in a manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=workspace / "manifests" / "full-gold-v1.json",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--serve-model", type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="reuse immutable feature caches instead of decoding every video",
    )
    parser.add_argument(
        "--run-version",
        help="dashboard-safe output version; defaults to the primary model directory name",
    )
    parser.add_argument("--variant-label")
    parser.add_argument("--variant-description")
    parser.add_argument(
        "--include-signals",
        action="store_true",
        help="include per-sample probability signals (omitted by default for dashboard batches)",
    )
    parser.add_argument("--output-root", type=Path, default=data_root / "analyses")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    serve_model_path = args.serve_model.expanduser().resolve() if args.serve_model else None
    cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir else None
    output_root = args.output_root.expanduser().resolve()
    model_version = args.run_version or model_path.name
    if not SAFE_ID.fullmatch(model_version) or not (model_path / "model.json").is_file():
        raise ValueError(f"Model directory is invalid: {model_path}")
    if serve_model_path is not None and not (serve_model_path / "model.json").is_file():
        raise ValueError(f"Serve model directory is invalid: {serve_model_path}")
    if cache_dir is not None and not cache_dir.is_dir():
        raise ValueError(f"Feature cache directory is invalid: {cache_dir}")
    manifest = load_manifest(manifest_path)
    recordings = list(manifest.recordings)
    primary_model = load_model(model_path)
    if primary_model.prediction_task != RALLY_LIVE_TASK:
        raise ModelError("dataset inference primary model must predict rally-live state")
    serve_model = load_model(serve_model_path) if serve_model_path is not None else None
    if serve_model is not None:
        _validate_model_pair(primary_model, serve_model)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        recordings = recordings[: args.limit]

    completed = skipped = 0
    for row in recordings:
        recording_id = row.id
        roi = row.roi
        if not SAFE_ID.fullmatch(recording_id) or roi is None:
            raise ValueError(f"Recording is missing a safe id, video, or ROI: {recording_id}")
        video_path = row.video
        destination = output_root / f"model-{model_version}--{recording_id}"
        if len(destination.name) > 80:
            raise ValueError(
                "Dashboard analysis id exceeds 80 characters; pass a shorter "
                f"--run-version for {model_version}"
            )
        if destination.exists():
            _validate_existing_output(
                destination,
                recording_id=recording_id,
                source_filename=video_path.name,
                content_sha256=row.content_sha256,
                primary_path=model_path,
                primary=primary_model,
                serve_path=serve_model_path,
                serve=serve_model,
                variant_label=args.variant_label,
                variant_description=args.variant_description,
            )
            print(f"Skipping existing {model_version} inference for {recording_id}")
            skipped += 1
            continue
        print(f"Inferring {model_version} on {recording_id}", flush=True)
        existing_preview = (
            output_root
            / f"model-full-percentile-v1--{recording_id}"
            / "court-preview.jpg"
        )
        result = infer_video(
            video_path,
            model_path,
            destination,
            roi=roi,
            title=recording_id,
            capture=row.capture,
            serve_model_path=serve_model_path,
            cache_dir=cache_dir,
            recording_id=recording_id,
            content_sha256=row.content_sha256,
            variant_label=args.variant_label,
            variant_description=args.variant_description,
            include_signals=args.include_signals,
            preview_source=(existing_preview if existing_preview.is_file() else None),
        )
        print(
            json.dumps(
                {
                    "analysis": str((destination / "analysis.json").resolve()),
                    "rallies": len(result["rallies"]),
                    "warnings": result["analysis"]["warnings"],
                },
                indent=2,
                allow_nan=False,
            )
        )
        completed += 1
    print(f"Inference complete: {completed} created, {skipped} already present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

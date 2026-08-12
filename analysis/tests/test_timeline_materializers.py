from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from analysis.model import ModelError


ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DIRECT = _load_script("infer_model_dataset", "infer-model-dataset.py")
SPECIALIST = _load_script(
    "infer_specialist_model_dataset", "infer-specialist-model-dataset.py"
)


class _Serializable:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def to_dict(self) -> dict[str, object]:
        return self.payload


def _model(artifact: str) -> SimpleNamespace:
    return SimpleNamespace(
        artifact_sha256=artifact,
        feature_config=_Serializable({"analysis_fps": 4.0}),
        decoder=_Serializable({"enter_threshold": 0.5}),
    )


class TimelineMaterializerTests(unittest.TestCase):
    def test_direct_resume_binds_content_and_rejects_paired_output(self) -> None:
        primary = _model("primary-sha")
        with tempfile.TemporaryDirectory(prefix="timeline-direct-") as directory:
            destination = Path(directory) / "model-primary--recording"
            destination.mkdir()
            (destination / "court-preview.jpg").write_bytes(b"jpeg")
            payload = {
                "id": destination.name,
                "recordingId": "recording",
                "source": {
                    "filename": "recording.mp4",
                    "contentSha256": "video-sha",
                },
                "analysis": {
                    "method": "court-motion-temporal-logistic-v0",
                    "modelVersion": "primary",
                    "modelSha256": "primary-sha",
                    "featureConfig": {"analysis_fps": 4.0},
                    "decoder": {"enter_threshold": 0.5},
                    "variantLabel": "Trained model · primary",
                    "variantDescription": "Primary iteration.",
                },
            }
            (destination / "analysis.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            DIRECT._validate_existing_output(
                destination,
                recording_id="recording",
                source_filename="recording.mp4",
                content_sha256="video-sha",
                primary_path=Path("primary"),
                primary=primary,
                serve_path=None,
                serve=None,
                variant_label="Trained model · primary",
                variant_description="Primary iteration.",
            )

            payload["analysis"]["models"] = {
                "rally": {"version": "primary", "sha256": "primary-sha"},
                "serve": {"version": "serve", "sha256": "serve-sha"},
            }
            (destination / "analysis.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with self.assertRaisesRegex(ModelError, "different inputs"):
                DIRECT._validate_existing_output(
                    destination,
                    recording_id="recording",
                    source_filename="recording.mp4",
                    content_sha256="video-sha",
                    primary_path=Path("primary"),
                    primary=primary,
                    serve_path=None,
                    serve=None,
                    variant_label="Trained model · primary",
                    variant_description="Primary iteration.",
                )

    def test_specialist_resume_binds_dependencies_and_selection(self) -> None:
        specialist_name = SPECIALIST.STACKED_MODELS[0]
        specialist = _model("specialist-sha")
        base = _model("base-sha")
        serve = _model("serve-sha")
        recording = SimpleNamespace(
            id="recording",
            video=Path("recording.mp4"),
            content_sha256="video-sha",
        )
        selection = {"transform": "continuous-serve-probability"}
        dependencies = {
            "models": {
                "serve": {"version": "serve", "sha256": "serve-sha"}
            }
        }
        label, description = SPECIALIST.ITERATIONS[specialist_name]
        with tempfile.TemporaryDirectory(prefix="timeline-specialist-") as directory:
            output_root = Path(directory)
            run_id = SPECIALIST._run_id(specialist, recording.id)
            destination = output_root / run_id
            destination.mkdir()
            (destination / "court-preview.jpg").write_bytes(b"jpeg")
            payload = {
                "id": run_id,
                "recordingId": recording.id,
                "source": {
                    "filename": recording.video.name,
                    "contentSha256": recording.content_sha256,
                },
                "analysis": {
                    "method": "specialist-method",
                    "modelVersion": specialist_name,
                    "modelSha256": specialist.artifact_sha256,
                    "variantLabel": f"Trained model · {label}",
                    "variantDescription": description,
                    "models": {
                        "specialist": {
                            "version": specialist_name,
                            "sha256": "specialist-sha",
                        },
                        "rally": {"version": "base", "sha256": "base-sha"},
                        "serve": {"version": "serve", "sha256": "serve-sha"},
                    },
                    "selection": selection,
                },
                "rallies": [],
            }
            (destination / "analysis.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            status = SPECIALIST._write_analysis(
                recording,
                Path(specialist_name),
                specialist,
                Path("base"),
                base,
                SimpleNamespace(),
                (),
                output_root,
                method="specialist-method",
                selection=selection,
                dependencies=dependencies,
            )
            self.assertEqual(status, "skipped")

            payload["analysis"]["selection"] = {"transform": "changed"}
            (destination / "analysis.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with self.assertRaisesRegex(ModelError, "another artifact"):
                SPECIALIST._write_analysis(
                    recording,
                    Path(specialist_name),
                    specialist,
                    Path("base"),
                    base,
                    SimpleNamespace(),
                    (),
                    output_root,
                    method="specialist-method",
                    selection=selection,
                    dependencies=dependencies,
                )


if __name__ == "__main__":
    unittest.main()

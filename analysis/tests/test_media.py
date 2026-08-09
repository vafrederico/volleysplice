from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.media import normalize_video
from analysis.schema import ManifestError, load_manifest


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(FFMPEG and FFPROBE, "ffmpeg and ffprobe are required")
class VideoNormalizationTests(unittest.TestCase):
    def test_interrupted_normalization_removes_temporary_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="volleycut-normalize-interrupt-test-") as directory:
            root = Path(directory)
            source = root / "source.mkv"
            source.write_bytes(b"synthetic-source")
            output = root / "normalized.mp4"

            with patch("analysis.media.subprocess.run", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    normalize_video(source, output, preset="ultrafast", threads=1)

            self.assertFalse(output.exists())
            temporary_outputs = [
                path for path in root.iterdir() if path.name.startswith(".normalized-")
            ]
            self.assertEqual(temporary_outputs, [])

    def test_odd_width_is_normalized_to_even_dimensions_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="volleycut-normalize-test-") as directory:
            root = Path(directory)
            source = root / "odd-source.mkv"
            output = root / "normalized.mp4"
            subprocess.run(
                [
                    str(FFMPEG),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=321x241:rate=5",
                    "-t",
                    "2.0",
                    "-an",
                    "-c:v",
                    "ffv1",
                    "-pix_fmt",
                    "yuv444p",
                    str(source),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            source_probe = subprocess.run(
                [
                    str(FFPROBE),
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "json",
                    str(source),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            source_stream = json.loads(source_probe.stdout)["streams"][0]
            self.assertEqual(source_stream["width"], 321)
            self.assertEqual(source_stream["height"], 241)

            result = normalize_video(
                source,
                output,
                fps=10.0,
                max_width=640,
                crf=30,
                start_seconds=0.4,
                duration_seconds=0.8,
                preset="fast",
                threads=1,
            )

            probe = subprocess.run(
                [
                    str(FFPROBE),
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,pix_fmt,avg_frame_rate",
                    "-of",
                    "json",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            stream = json.loads(probe.stdout)["streams"][0]
            self.assertEqual(stream["width"], 320)
            self.assertEqual(stream["width"] % 2, 0)
            self.assertEqual(stream["height"] % 2, 0)
            self.assertEqual(stream["pix_fmt"], "yuv420p")
            self.assertEqual(stream["avg_frame_rate"], "10/1")

            provenance_path = Path(result["provenance"])
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            self.assertEqual(Path(result["video"]), output)
            self.assertEqual(provenance["schemaVersion"], 1)
            self.assertEqual(provenance["source"]["filename"], source.name)
            self.assertEqual(provenance["source"]["sizeBytes"], source.stat().st_size)
            self.assertEqual(provenance["source"]["sha256"], sha256(source))
            self.assertEqual(provenance["segment"]["sourceStartSeconds"], 0.4)
            self.assertEqual(provenance["segment"]["requestedDurationSeconds"], 0.8)
            self.assertEqual(provenance["normalized"]["filename"], output.name)
            self.assertEqual(provenance["normalized"]["sizeBytes"], output.stat().st_size)
            self.assertEqual(provenance["normalized"]["sha256"], sha256(output))
            self.assertEqual(provenance["normalized"]["fps"], 10.0)
            self.assertEqual(provenance["normalized"]["maxWidth"], 640)
            self.assertEqual(provenance["normalized"]["crf"], 30)
            self.assertEqual(provenance["normalized"]["preset"], "fast")
            self.assertEqual(provenance["normalized"]["threads"], 1)

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "name": "normalization-integrity-test",
                        "annotationPolicy": {"id": "serve-contact-to-dead-ball-v1"},
                        "recordings": [
                            {
                                "id": "normalized-clip",
                                "video": output.name,
                                "split": "test",
                                "sourceGroup": "normalized-source",
                                "environment": "beach",
                                "consent": {"analyze": True, "train": False},
                                "capture": {},
                                "rallies": [],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = load_manifest(manifest_path)
            self.assertEqual(manifest.recordings[0].content_sha256, sha256(output))

            tampered = bytearray(output.read_bytes())
            tampered[-1] ^= 0x01
            output.write_bytes(tampered)
            self.assertEqual(output.stat().st_size, provenance["normalized"]["sizeBytes"])
            with self.assertRaisesRegex(ManifestError, "SHA-256.*does not match"):
                load_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main()

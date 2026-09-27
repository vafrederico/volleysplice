"""Privacy scanning must distinguish numeric weights from ONNX text fields."""
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("audit-branch-privacy.py").resolve()
AUDIT = runpy.run_path(str(SCRIPT))
PUBLIC_PATH = "prod/public/runtime/rally-models/f1/encoder.onnx"
MARKER = "private-fixture-marker"


def varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def field(number, value):
    if isinstance(value, int):
        return varint(number << 3) + varint(value)
    if isinstance(value, str):
        value = value.encode()
    return varint((number << 3) | 2) + varint(len(value)) + value


def tensor(raw=None, data_type=1, dimensions=(1,), extra=b""):
    data = field(1, b"".join(varint(d) for d in dimensions)) + field(2, data_type)
    return data + (field(9, raw) if raw is not None else b"") + extra


def model(tensors=(), graph_extra=b"", extra=b""):
    graph = field(2, "publication-fixture")
    graph += b"".join(field(5, t) for t in tensors) + graph_extra
    return field(1, 8) + field(7, graph) + field(8, field(2, 17)) + extra


def entry(value):
    return field(1, "fixture") + field(2, value)


def findings(data, path=PUBLIC_PATH):
    text, errors = AUDIT["audit_text"](data, path)
    return errors + AUDIT["categories"](text, {MARKER}, path)


class PublicOnnxPrivacyTest(unittest.TestCase):
    def test_numeric_raw_bytes_do_not_become_drive_path_findings(self):
        coincidental_float_bytes = (chr(90) + ":/x").encode()
        data = model([tensor(coincidental_float_bytes), tensor(coincidental_float_bytes * 2, 7)])
        self.assertEqual([], findings(data))
        # The exception applies to neither arbitrary ONNX files nor other binary files.
        for path in ("private/encoder.onnx", PUBLIC_PATH + ".backup", "model.bin"):
            self.assertIn("private-drive", findings(data, path))

    def test_all_textual_locations_still_detect_ledger_identities(self):
        locations = [
            model(extra=field(14, entry(MARKER))),
            model(extra=field(6, MARKER)),
            model(graph_extra=field(16, entry(MARKER))),
            model([tensor(b"\0" * 4, extra=field(8, MARKER))]),
            model([tensor(b"\0" * 4, extra=field(12, MARKER))]),
            model([tensor(b"\0" * 4, extra=field(16, entry(MARKER)))]),
            model(graph_extra=field(1, field(5, field(4, MARKER)))),
            model([tensor(data_type=8, extra=field(6, MARKER))]),
            model(extra=field(500, MARKER)),
        ]
        for index, data in enumerate(locations):
            with self.subTest(location=index):
                self.assertIn("ledger-identity", findings(data))

    def test_generic_private_paths_and_external_tensor_locations_remain_visible(self):
        private_location = chr(90) + ":/fixture/model.data"
        for data in (
            model(extra=field(14, entry(private_location))),
            model([tensor(extra=field(13, entry(private_location)) + field(14, 1))]),
        ):
            self.assertIn("private-drive", findings(data))

    def test_unknown_and_string_tensor_types_do_not_mask_raw_bytes(self):
        payload = (chr(90) + ":/x").encode()
        for data_type in (8, 99):
            self.assertIn("private-drive", findings(model([tensor(payload, data_type)])))

    def test_nested_constant_tensor_is_scanned_by_the_same_rules(self):
        raw = (chr(90) + ":/x").encode()
        constant = field(4, "Constant") + field(5, field(5, tensor(raw)))
        self.assertEqual([], findings(model(graph_extra=field(1, constant))))
        leaking_constant = constant + field(6, MARKER)
        self.assertIn("ledger-identity", findings(model(graph_extra=field(1, leaking_constant))))

    def test_truncation_invalid_framing_and_conflicting_storage_fail_closed(self):
        valid = model([tensor(b"\0" * 4)])
        invalid = [
            valid[:-1], b"\xff" * 12, b"", valid + b"\0", field(7, b""),
            model([tensor(b"\0" * 3)]),
            model([tensor(b"\0" * 4, dimensions=(2,))]),
            model([tensor(b"\0" * 4, dimensions=(1 << 63,))]),
            model([tensor(b"\0" * 4, extra=field(2, 1))]),
            model([tensor(b"\0" * 4, extra=field(9, b"\0" * 4))]),
            model([tensor(b"\0" * 4, extra=field(13, entry(MARKER)))]),
        ]
        for index, data in enumerate(invalid):
            with self.subTest(case=index):
                self.assertIn("invalid-public-onnx", findings(data))

    def test_generic_non_onnx_text_behavior_is_unchanged(self):
        text = "ordinary text\n" + MARKER
        scan_text, errors = AUDIT["audit_text"](text.encode(), "source.txt")
        self.assertEqual(text, scan_text)
        self.assertEqual([], errors)
        self.assertEqual(["ledger-identity"], findings(text.encode(), "source.txt"))

    def test_history_finds_removed_metadata_but_ignores_numeric_coincidence(self):
        with tempfile.TemporaryDirectory(prefix="onnx-privacy-") as directory:
            root = Path(directory)

            def git(*args):
                return subprocess.check_output(
                    ["git", "-c", "user.name=Privacy Test", "-c", "user.email=test@example.invalid", *args],
                    cwd=root, stderr=subprocess.DEVNULL,
                ).decode().strip()

            git("init")
            git("commit", "--allow-empty", "-m", "Baseline")
            baseline = git("rev-parse", "HEAD")
            source = root / PUBLIC_PATH
            source.parent.mkdir(parents=True)
            raw = (chr(90) + ":/x").encode()
            source.write_bytes(model([tensor(raw)], extra=field(14, entry(MARKER))))
            git("add", ".")
            git("commit", "-m", "Metadata fixture")
            source.write_bytes(model([tensor(raw)]))
            git("add", ".")
            git("commit", "-m", "Remove metadata fixture")
            ledger = root / "ledger.json"
            ledger.write_text(json.dumps({"denyTokens": [MARKER]}))
            output = root / "audit.json"
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--base", baseline, "--revision", "HEAD",
                 "--history", "--ledger", str(ledger), "--output", str(output)],
                cwd=root, capture_output=True, text=True,
            )
            self.assertEqual(1, result.returncode, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual([], report["workingTreeFindings"])
            self.assertEqual(1, len(report["historicalObjectsWithFindings"]))
            self.assertEqual(["ledger-identity"], report["historicalObjectsWithFindings"][0]["categories"])


if __name__ == "__main__":
    unittest.main()

"""Capture the physical launcher briefly, then return to verified parking.

Never fetch the Home accessibility tree: it can stall until Auto-Lock engages.
All HTTP reads are bounded; no passcode, lock, unlock or Auto-Lock changes occur.
The JPEG still requires visual inspection before accepting the installed icon.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lab",
        type=Path,
        default=Path(os.environ.get("VOLLEYCUT_IOS_LAB_ROOT", "artifacts/ios-lab")),
    )
    parser.add_argument("--url", default="http://127.0.0.1:18100")
    parser.add_argument("--mjpeg", default="http://127.0.0.1:19100/")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("lab_wda", args.lab / "scripts/wda-client.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class BoundedWDA(module.WDA):
        def request(self, path, body=None):
            request = urllib.request.Request(self.base + path,
                data=None if body is None else json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=3) as response:
                result = json.load(response)
            if isinstance(result.get("value"), dict) and result["value"].get("error"):
                raise RuntimeError(result["value"])
            return result

    wda = BoundedWDA(args.url)
    wda.connect()
    if wda.request("/wda/locked")["value"]:
        raise RuntimeError("Unlock the iPad and open iOS Lab Parking before this check")
    result = {"visualAcceptance": "pending JPEG inspection"}
    try:
        wda.activate(module.PARKING)
        source = wda.command("/source")
        if "Idle timer: disabled" not in source:
            raise RuntimeError("Parking did not disable the idle timer")
        (args.output / "parking-before.xml").write_text(source, encoding="utf-8")
        began = time.monotonic()
        wda.request("/wda/homescreen", {})
        # Keep the latest complete frame during a short settling interval.
        # Stream observation is read-only and never asks SpringBoard for source.
        latest, buffer = None, b""
        with urllib.request.urlopen(args.mjpeg, timeout=2) as response:
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                block = response.read1(16384)
                if not block:
                    break
                buffer += block
                while True:
                    start, end = buffer.find(b"\xff\xd8"), buffer.find(b"\xff\xd9")
                    if not 0 <= start < end:
                        break
                    latest, buffer = buffer[start:end + 2], buffer[end + 2:]
                if len(buffer) > 8 * 1024 * 1024:
                    raise RuntimeError("Malformed MJPEG stream")
        if latest is None:
            raise RuntimeError("No complete Home frame")
        (args.output / "launcher.jpg").write_bytes(latest)
        result["homeCaptureSeconds"] = round(time.monotonic() - began, 3)
    except Exception as error:
        result["failure"] = str(error)
        raise
    finally:
        try:
            wda.activate(module.PARKING)
            source = wda.command("/source")
            if "Idle timer: disabled" not in source:
                raise RuntimeError("Parking idle timer is not disabled")
            (args.output / "parking-after.xml").write_text(source, encoding="utf-8")
            result["teardown"] = "parking foreground; idle timer disabled"
        except Exception as error:
            result["parkingFailure"] = str(error)
            if "failure" not in result:
                raise
        finally:
            (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result))


if __name__ == "__main__":
    main()

"""Run iPhone Duo simulator tests and folded/unfolded screenshot captures on CI.

Designed for GitHub-hosted Apple Silicon runners (runs-on: xcode-27) where the
iOS 27.1 / 27.2 iPhone Duo simulator runtimes are preinstalled and duoctl drives
hinge posture, interface rotation, and active-display screenshots headlessly.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import time
import zlib


BUNDLE_ID = "com.volleysplice.VolleySplice"
DUOCTL_COMMIT = "c6ff3e961a9858fb061907448e702d7d1617dc4a"
DUOCTL_URL = f"https://github.com/skarol/duoctl/archive/{DUOCTL_COMMIT}.tar.gz"
DUOCTL_SHA256 = "1bf6f0793ae77796c7a20cc59df4b9920daaed378855bd4fc3862cbe54699464"
PREFERRED_XCODES = ("Xcode_27.2.app", "Xcode_27.1.app", "Xcode_27.3_beta.app", "Xcode.app")

POSTURE_PHASES = [
    {
        "id": "folded-portrait",
        "title": "Folded · Portrait (iPhone UI)",
        "hinge": "close",
        "orientation": "portrait",
        "test": "VolleySpliceUITests/IPhoneDuoUITests/testFoldedPortraitUsesPhoneInterface",
        "screens": [
            ("setup", ["--parity-skip-tour"]),
            ("rallies", ["--parity-editor", "--parity-skip-tour", "--parity-stage=rallies"]),
            ("editor", ["--parity-editor", "--parity-skip-tour", "--parity-stage=editor"]),
            ("export", ["--parity-editor", "--parity-skip-tour", "--parity-stage=export"]),
        ],
    },
    {
        "id": "folded-landscape",
        "title": "Folded · Landscape (iPhone landscape UI)",
        "hinge": "close",
        "orientation": "landscape",
        "test": "VolleySpliceUITests/IPhoneDuoUITests/testFoldedLandscapeUsesPhoneLandscapeInterface",
        "screens": [
            ("rallies", ["--parity-editor", "--parity-skip-tour", "--parity-stage=rallies"]),
            ("editor", ["--parity-editor", "--parity-skip-tour", "--parity-stage=editor"]),
        ],
    },
    {
        "id": "unfolded-landscape",
        "title": "Unfolded · Landscape (iPad UI)",
        "hinge": "open",
        "orientation": "landscape",
        "test": "VolleySpliceUITests/IPhoneDuoUITests/testUnfoldedLandscapeUsesIPadInterface",
        "screens": [
            ("setup", ["--parity-skip-tour"]),
            ("rallies", ["--parity-editor", "--parity-skip-tour", "--parity-stage=rallies"]),
            ("editor", ["--parity-editor", "--parity-skip-tour", "--parity-stage=editor"]),
            ("export", ["--parity-editor", "--parity-skip-tour", "--parity-stage=export"]),
        ],
    },
    {
        "id": "unfolded-portrait",
        "title": "Unfolded · Portrait (iPad portrait UI)",
        "hinge": "open",
        "orientation": "portrait",
        "test": "VolleySpliceUITests/IPhoneDuoUITests/testUnfoldedPortraitUsesIPadPortraitInterface",
        "screens": [
            ("rallies", ["--parity-editor", "--parity-skip-tour", "--parity-stage=rallies"]),
            ("editor", ["--parity-editor", "--parity-skip-tour", "--parity-stage=editor"]),
        ],
    },
]


def parse_version_tuple(version: str) -> tuple[int, ...]:
    parts = [int(part) for part in re.findall(r"\d+", version)]
    return tuple(parts) if parts else (0,)


def select_xcode(requested: str, applications_dir: Path = Path("/Applications")) -> Path:
    if requested and requested != "auto":
        path = Path(requested)
        developer = path / "Contents/Developer" if path.suffix == ".app" else path
        if not developer.is_dir():
            raise ValueError(f"Requested Xcode developer directory not found: {developer}")
        return developer
    for name in PREFERRED_XCODES:
        candidate = applications_dir / name / "Contents/Developer"
        if candidate.is_dir():
            return candidate
    fallback = sorted(applications_dir.glob("Xcode_27*.app"), reverse=True)
    for app in fallback:
        candidate = app / "Contents/Developer"
        if candidate.is_dir():
            return candidate
    raise ValueError(f"No Xcode 27 installation found under {applications_dir}")


def select_duo_runtime(runtimes_payload: dict, devicetypes_payload: dict) -> tuple[str, str, str]:
    """Return (device_type_id, runtime_id, runtime_name) for iPhone Duo.

    iOS 27.2 ships separate standard and iPhone Duo runtimes with the same
    display version; inspect supportedDeviceTypes rather than matching by name.
    """
    device_types = devicetypes_payload.get("devicetypes", [])
    duo_types = [
        item for item in device_types
        if "iphone duo" in item.get("name", "").lower()
        or "iphone-duo" in item.get("identifier", "").lower()
    ]
    if not duo_types:
        raise ValueError("No iPhone Duo simulator device type found in xcrun simctl list devicetypes")
    duo_type_id = duo_types[0]["identifier"]
    matching_runtimes = []
    for runtime in runtimes_payload.get("runtimes", []):
        if not runtime.get("isAvailable", False):
            continue
        if runtime.get("platform", "").lower() not in ("", "ios") and "ios" not in runtime.get("name", "").lower():
            continue
        supported = runtime.get("supportedDeviceTypes", [])
        supported_ids = {entry.get("identifier") for entry in supported if isinstance(entry, dict)}
        if duo_type_id in supported_ids:
            matching_runtimes.append(runtime)
    if not matching_runtimes:
        raise ValueError(f"No available simulator runtime supports {duo_type_id}")
    matching_runtimes.sort(
        key=lambda item: (
            parse_version_tuple(str(item.get("version", ""))),
            str(item.get("buildversion", "")),
        ),
        reverse=True,
    )
    chosen = matching_runtimes[0]
    return duo_type_id, chosen["identifier"], chosen.get("name", chosen["identifier"])


def verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}: {actual} != {expected}")


def install_duoctl(install_root: Path, archive_path: Path | None = None) -> Path:
    binary = install_root / f"duoctl-{DUOCTL_COMMIT}/bin/duoctl"
    if binary.is_file():
        return binary
    install_root.mkdir(parents=True, exist_ok=True)
    archive = archive_path or (install_root / "duoctl.tar.gz")
    if archive_path is None:
        subprocess.run(
            ["curl", "--fail", "--location", "--retry", "3", "--silent", "--show-error", DUOCTL_URL, "-o", str(archive)],
            check=True,
        )
    verify_sha256(archive, DUOCTL_SHA256)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(install_root)
    if not binary.is_file():
        candidates = list(install_root.glob("*/bin/duoctl"))
        if not candidates:
            raise ValueError(f"duoctl executable not found under {install_root}")
        binary = candidates[0]
    binary.chmod(0o755)
    return binary


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def validate_png(path: Path) -> dict:
    """Validate PNG structure, dimensions, and verify the frame is not solid black."""
    data = path.read_bytes()
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path.name}: not a valid PNG file")
    offset = 8
    width = height = bit_depth = color_type = interlace = 0
    idat_parts = []
    while offset + 8 <= len(data):
        length, chunk_type = struct.unpack(">I4s", data[offset:offset + 8])
        chunk_data = data[offset + 8:offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", chunk_data)
        elif chunk_type == b"IDAT":
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            break
    if width <= 0 or height <= 0:
        raise ValueError(f"{path.name}: invalid PNG dimensions {width}x{height}")
    if not idat_parts:
        raise ValueError(f"{path.name}: missing IDAT pixel data")

    non_black_detected = None
    channels_by_type = {0: 1, 2: 3, 4: 2, 6: 4}
    color_channels_by_type = {0: 1, 2: 3, 4: 1, 6: 3}
    if bit_depth == 8 and interlace == 0 and color_type in channels_by_type:
        raw = zlib.decompress(b"".join(idat_parts))
        bpp = channels_by_type[color_type]
        color_channels = color_channels_by_type[color_type]
        stride = width * bpp
        prev = bytearray(stride)
        pos = 0
        non_black_detected = False
        # Sample rows across the full frame for speed while unfiltering sequentially.
        for _ in range(height):
            if pos + 1 + stride > len(raw):
                break
            filter_type = raw[pos]
            pos += 1
            row = bytearray(raw[pos:pos + stride])
            pos += stride
            if filter_type == 1:
                for i in range(bpp, stride):
                    row[i] = (row[i] + row[i - bpp]) & 0xFF
            elif filter_type == 2:
                for i in range(stride):
                    row[i] = (row[i] + prev[i]) & 0xFF
            elif filter_type == 3:
                for i in range(stride):
                    left = row[i - bpp] if i >= bpp else 0
                    row[i] = (row[i] + ((left + prev[i]) >> 1)) & 0xFF
            elif filter_type == 4:
                for i in range(stride):
                    left = row[i - bpp] if i >= bpp else 0
                    up = prev[i]
                    up_left = prev[i - bpp] if i >= bpp else 0
                    row[i] = (row[i] + _paeth(left, up, up_left)) & 0xFF
            if not non_black_detected:
                for px in range(0, stride, bpp):
                    if any(row[px + ch] > 8 for ch in range(color_channels)):
                        non_black_detected = True
                        break
            prev = row
        if not non_black_detected:
            raise ValueError(f"{path.name}: captured frame appears completely black ({width}x{height})")

    return {
        "file": path.name,
        "width": width,
        "height": height,
        "bytes": len(data),
        "non_black": non_black_detected,
    }


def build_gallery_html(report: dict) -> str:
    sections = []
    for phase in report.get("phases", []):
        status = phase.get("test_status", "skipped")
        badge_bg = "#1f6f50" if status == "passed" else "#b3261e" if status == "failed" else "#65766f"
        state_json = html.escape(json.dumps(phase.get("duoctl_state", {}), sort_keys=True))
        cards = []
        for shot in phase.get("screenshots", []):
            file_name = html.escape(shot["file"])
            label = html.escape(shot.get("label", shot["file"]))
            dims = f"{shot.get('width', '?')} × {shot.get('height', '?')} px"
            cards.append(
                f"<figure class='card'>"
                f"<a href='{file_name}'><img src='{file_name}' alt='{label}' loading='lazy'></a>"
                f"<figcaption><strong>{label}</strong><span>{dims}</span></figcaption>"
                f"</figure>"
            )
        sections.append(
            f"<section>"
            f"<div class='phase-head'><h2>{html.escape(phase['title'])}</h2>"
            f"<span class='badge' style='background:{badge_bg}'>{html.escape(status)}</span></div>"
            f"<pre class='state'>{state_json}</pre>"
            f"<div class='grid'>{''.join(cards)}</div>"
            f"</section>"
        )
    live = report.get("live_transition", {})
    if live.get("screenshots"):
        cards = []
        for shot in live["screenshots"]:
            file_name = html.escape(shot["file"])
            label = html.escape(shot.get("label", shot["file"]))
            dims = f"{shot.get('width', '?')} × {shot.get('height', '?')} px"
            cards.append(
                f"<figure class='card'>"
                f"<a href='{file_name}'><img src='{file_name}' alt='{label}' loading='lazy'></a>"
                f"<figcaption><strong>{label}</strong><span>{dims}</span></figcaption>"
                f"</figure>"
            )
        sections.append(
            f"<section>"
            f"<div class='phase-head'><h2>Live Fold → Unfold → Refold (Single Process)</h2>"
            f"<span class='badge' style='background:#1f6f50'>captured</span></div>"
            f"<div class='grid'>{''.join(cards)}</div>"
            f"</section>"
        )
    meta = html.escape(
        f"Xcode: {report.get('xcode_version', 'unknown')} | "
        f"Runtime: {report.get('runtime_name', 'unknown')} ({report.get('runtime_id', '')})"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VolleySplice · iPhone Duo Simulator Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; background: #f8f7ee; color: #17382e; }}
  h1 {{ margin: 0 0 6px; }}
  .meta {{ color: #65766f; font-size: 13px; margin-bottom: 24px; font-family: ui-monospace, monospace; }}
  section {{ margin-bottom: 32px; padding: 16px; border: 1px solid #c8d6ca; border-radius: 10px; background: #fff; }}
  .phase-head {{ display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }}
  .phase-head h2 {{ margin: 0; font-size: 18px; }}
  .badge {{ color: #fff; padding: 3px 9px; border-radius: 999px; font-size: 12px; font-weight: 600; text-transform: uppercase; }}
  pre.state {{ background: #f3f5f1; padding: 8px 10px; border-radius: 6px; font-size: 11px; overflow-x: auto; margin: 0 0 14px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }}
  .card {{ margin: 0; border: 1px solid #d9e2da; border-radius: 8px; overflow: hidden; background: #fafaf6; }}
  .card img {{ display: block; width: 100%; height: auto; background: #111; }}
  .card figcaption {{ padding: 8px 10px; display: flex; justify-content: space-between; gap: 8px; font-size: 12px; }}
  .card figcaption span {{ color: #65766f; font-family: ui-monospace, monospace; }}
</style>
</head>
<body>
  <h1>VolleySplice · iPhone Duo Simulator Verification</h1>
  <div class="meta">{meta}</div>
  {''.join(sections)}
</body>
</html>
"""


def run_cmd(cmd: list[str], env: dict[str, str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        env=env,
        check=check,
        capture_output=capture,
        text=True,
    )


def duoctl_set_posture(duoctl: Path, udid: str, hinge: str, orientation: str, env: dict[str, str]) -> dict:
    run_cmd([str(duoctl), "-d", udid, hinge], env=env)
    time.sleep(0.8)
    run_cmd([str(duoctl), "-d", udid, "rotate", orientation], env=env)
    time.sleep(0.8)
    state_out = run_cmd([str(duoctl), "-d", udid, "state"], env=env, capture=True).stdout.strip()
    print("duoctl state:", state_out, flush=True)
    try:
        return json.loads(state_out)
    except json.JSONDecodeError:
        return {"raw": state_out}


def capture_screen(duoctl: Path, udid: str, launch_args: list[str], output_png: Path, env: dict[str, str], settle_seconds: float = 2.5) -> dict:
    subprocess.run(["xcrun", "simctl", "terminate", udid, BUNDLE_ID], env=env, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run_cmd(["xcrun", "simctl", "launch", udid, BUNDLE_ID, *launch_args], env=env)
    time.sleep(settle_seconds)
    run_cmd([str(duoctl), "-d", udid, "screenshot", str(output_png)], env=env)
    return validate_png(output_png)


def run_pipeline(args: argparse.Namespace) -> int:
    developer_dir = select_xcode(args.xcode)
    env = os.environ.copy()
    env["DEVELOPER_DIR"] = str(developer_dir)

    xcode_version = run_cmd(["xcodebuild", "-version"], env=env, capture=True).stdout.strip().replace("\n", " / ")
    runtimes_payload = json.loads(run_cmd(["xcrun", "simctl", "list", "runtimes", "-j"], env=env, capture=True).stdout)
    devicetypes_payload = json.loads(run_cmd(["xcrun", "simctl", "list", "devicetypes", "-j"], env=env, capture=True).stdout)
    duo_type_id, runtime_id, runtime_name = select_duo_runtime(runtimes_payload, devicetypes_payload)
    print(f"Selected {duo_type_id} on {runtime_name} ({runtime_id})", flush=True)

    duoctl = install_duoctl(args.tools_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    derived_data = args.derived_data

    udid = run_cmd(
        ["xcrun", "simctl", "create", f"volleycut-ci-duo-{int(time.time())}", duo_type_id, runtime_id],
        env=env,
        capture=True,
    ).stdout.strip()
    env["DUOCTL_UDID"] = udid

    report = {
        "xcode_developer_dir": str(developer_dir),
        "xcode_version": xcode_version,
        "device_type_id": duo_type_id,
        "runtime_id": runtime_id,
        "runtime_name": runtime_name,
        "simulator_udid": udid,
        "phases": [],
        "live_transition": {"screenshots": []},
        "hosted_tests_status": "skipped",
    }
    failures: list[str] = []

    try:
        run_cmd(["xcrun", "simctl", "boot", udid], env=env)
        run_cmd(["xcrun", "simctl", "bootstatus", udid, "-b"], env=env)

        build_cmd = [
            "xcodebuild",
            "-project", str(args.ios_source / "VolleySplice.xcodeproj"),
            "-scheme", "VolleySplice",
            "-configuration", "Debug",
            "-sdk", "iphonesimulator",
            "-destination", f"id={udid}",
            "-derivedDataPath", str(derived_data),
            "-jobs", str(max(2, os.cpu_count() or 4)),
            "ARCHS=arm64",
            "ONLY_ACTIVE_ARCH=YES",
            "CODE_SIGNING_ALLOWED=NO",
            f"VOLLEYCUT_IOS_OPENCV_FRAMEWORK_ROOT={args.opencv_root}",
            "build-for-testing",
        ]
        build_log = logs_dir / "build-for-testing.log"
        with build_log.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(build_cmd, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            print(build_log.read_text(encoding="utf-8", errors="replace")[-8000:], file=sys.stderr)
            raise subprocess.CalledProcessError(proc.returncode, build_cmd)

        app_bundle = derived_data / "Build/Products/Debug-iphonesimulator/VolleySplice.app"
        if not app_bundle.is_dir():
            raise ValueError(f"Built app not found at {app_bundle}")
        run_cmd(["xcrun", "simctl", "install", udid, str(app_bundle)], env=env)

        for phase in POSTURE_PHASES:
            phase_id = phase["id"]
            print(f"\n=== Phase: {phase['title']} ===", flush=True)
            state = duoctl_set_posture(duoctl, udid, phase["hinge"], phase["orientation"], env)
            phase_entry = {
                "id": phase_id,
                "title": phase["title"],
                "duoctl_state": state,
                "test": phase["test"],
                "test_status": "passed",
                "screenshots": [],
            }

            result_bundle = logs_dir / f"{phase_id}.xcresult"
            shutil.rmtree(result_bundle, ignore_errors=True)
            test_cmd = [
                "xcodebuild",
                "-project", str(args.ios_source / "VolleySplice.xcodeproj"),
                "-scheme", "VolleySplice",
                "-configuration", "Debug",
                "-destination", f"id={udid}",
                "-derivedDataPath", str(derived_data),
                "-parallel-testing-enabled", "NO",
                "-resultBundlePath", str(result_bundle),
                f"-only-testing:{phase['test']}",
                "test-without-building",
            ]
            test_log = logs_dir / f"{phase_id}-xctest.log"
            with test_log.open("w", encoding="utf-8") as handle:
                proc = subprocess.run(test_cmd, env=env, stdout=handle, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                phase_entry["test_status"] = "failed"
                failures.append(f"UI test failed in {phase_id}: {phase['test']}")
                print(test_log.read_text(encoding="utf-8", errors="replace")[-4000:], file=sys.stderr)

            # Re-assert posture in case XCTest reset anything before host screenshots.
            duoctl_set_posture(duoctl, udid, phase["hinge"], phase["orientation"], env)
            for screen_name, launch_args in phase["screens"]:
                out_png = args.output_dir / f"{phase_id}-{screen_name}.png"
                try:
                    meta = capture_screen(duoctl, udid, launch_args, out_png, env)
                    meta["label"] = f"{phase_id} · {screen_name}"
                    phase_entry["screenshots"].append(meta)
                except Exception as error:
                    failures.append(f"Screenshot failed for {phase_id}-{screen_name}: {error}")
                    print(f"ERROR capturing {phase_id}-{screen_name}: {error}", file=sys.stderr)
            report["phases"].append(phase_entry)

        # Live single-process fold -> unfold -> refold check on the Editor screen.
        print("\n=== Phase: Live Fold -> Unfold -> Refold ===", flush=True)
        try:
            duoctl_set_posture(duoctl, udid, "close", "portrait", env)
            subprocess.run(["xcrun", "simctl", "terminate", udid, BUNDLE_ID], env=env, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            run_cmd(
                ["xcrun", "simctl", "launch", udid, BUNDLE_ID, "--parity-editor", "--parity-skip-tour", "--parity-stage=editor"],
                env=env,
            )
            time.sleep(2.5)
            steps = [
                ("live-01-folded-editor.png", "Live 1 · Folded (start)", None),
                ("live-02-unfolded-editor.png", "Live 2 · Unfolded (same process)", "open"),
                ("live-03-refolded-editor.png", "Live 3 · Refolded (same process)", "close"),
            ]
            captured = []
            for filename, label, action in steps:
                if action is not None:
                    run_cmd([str(duoctl), "-d", udid, action], env=env)
                    time.sleep(2.5)
                out_png = args.output_dir / filename
                run_cmd([str(duoctl), "-d", udid, "screenshot", str(out_png)], env=env)
                meta = validate_png(out_png)
                meta["label"] = label
                captured.append(meta)
            report["live_transition"]["screenshots"] = captured
            if len(captured) == 3 and (captured[0]["width"], captured[0]["height"]) == (captured[1]["width"], captured[1]["height"]):
                failures.append("Live unfold screenshot dimensions did not change from folded display")
        except Exception as error:
            failures.append(f"Live fold/unfold transition failed: {error}")
            print(f"ERROR in live transition: {error}", file=sys.stderr)

        if args.run_hosted_tests:
            print("\n=== Phase: Hosted VolleySpliceTests ===", flush=True)
            hosted_log = logs_dir / "hosted-tests.log"
            hosted_cmd = [
                "xcodebuild",
                "-project", str(args.ios_source / "VolleySplice.xcodeproj"),
                "-scheme", "VolleySplice",
                "-configuration", "Debug",
                "-destination", f"id={udid}",
                "-derivedDataPath", str(derived_data),
                "-parallel-testing-enabled", "NO",
                "-only-testing:VolleySpliceTests",
                "test-without-building",
            ]
            with hosted_log.open("w", encoding="utf-8") as handle:
                proc = subprocess.run(hosted_cmd, env=env, stdout=handle, stderr=subprocess.STDOUT)
            report["hosted_tests_status"] = "passed" if proc.returncode == 0 else "failed"
            if proc.returncode != 0:
                failures.append("Hosted VolleySpliceTests reported failures")
    finally:
        subprocess.run(["xcrun", "simctl", "shutdown", udid], env=env, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["xcrun", "simctl", "delete", udid], env=env, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    report["failures"] = failures
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "index.html").write_text(build_gallery_html(report), encoding="utf-8")
    if failures:
        print("\niPhone Duo CI failures:\n- " + "\n- ".join(failures), file=sys.stderr)
        return 1
    print("\nAll iPhone Duo UI checks and screenshot captures passed.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ios-source", type=Path, required=True, help="Prepared iOS source root containing VolleySplice.xcodeproj")
    parser.add_argument("--opencv-root", type=Path, required=True, help="Directory containing arm64-simulator opencv2.framework")
    parser.add_argument("--output-dir", type=Path, required=True, help="Artifact output directory for screenshots, index.html, report.json")
    parser.add_argument("--derived-data", type=Path, required=True, help="xcodebuild DerivedData path")
    parser.add_argument("--tools-dir", type=Path, required=True, help="Staging directory for duoctl")
    parser.add_argument("--xcode", default="auto", help="Xcode .app path, Developer dir, or 'auto'")
    parser.add_argument("--run-hosted-tests", action="store_true", help="Also run VolleySpliceTests on the iPhone Duo simulator")
    args = parser.parse_args()
    sys.exit(run_pipeline(args))


if __name__ == "__main__":
    main()

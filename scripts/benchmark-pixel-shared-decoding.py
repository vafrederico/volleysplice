"""Run the optimized, frozen Distilled Large pipeline on an authorized Pixel.

Inputs and output roots resolve through the private ledger and local environment.
Installation and source hash verification are separate, untimed preparation steps.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
from private_ledger import private_value


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--phase", choices=("short", "full"), required=True)
    args = parser.parse_args()
    indexed = private_value("private-reference-0223")
    parent = Path(os.environ.get("VOLLEYCUT_DEVICE_ARTIFACT_ROOT", indexed))
    if not parent.is_absolute() or not parent.is_dir():
        raise ValueError("Configure the indexed artifact root for this host")
    root = parent / "pixel-shared-video-decoding"
    folder = root / (args.phase + "-shared")
    if folder.exists():
        raise FileExistsError("Preserve the previous run")
    if args.phase == "full" and not read(root / "short-parity.json")["exact"]:
        raise ValueError("Short output qualification required before the full run")
    source = read(root / "sources.private.json")["pilot" if args.phase == "short" else "full"]
    assert source["verified"]
    package = "com.volleycut.nativeanalysis.pipelinebenchmark"

    def adb(*command, timeout=30):
        return subprocess.run([args.adb, "-s", args.serial, *command], capture_output=True,
                              text=True, check=True, timeout=timeout).stdout.strip()

    assert not args.serial.startswith("emulator-")
    assert adb("shell", "getprop", "ro.product.model") == "Pixel 10 Pro"
    assert adb("shell", "getprop", "ro.product.cpu.abi") == "arm64-v8a"

    def observation():
        power = adb("shell", "dumpsys", "power")
        policy = adb("shell", "dumpsys", "window", "policy")
        thermal = adb("shell", "dumpsys", "thermalservice")
        battery = adb("shell", "dumpsys", "battery")
        activity = adb("shell", "dumpsys", "activity", "activities")
        status = re.search(r"^Thermal Status: (\d+)", thermal, re.M)
        def number(key):
            found = re.search(r"^\s*" + key + r": (\d+)", battery, re.M)
            return int(found.group(1)) if found else None
        awake = any(line.strip() in ("mWakefulness=Awake", "mWakefulness=1") for line in power.splitlines())
        locked = bool(re.search(r"^\s*(?:showing|inputRestricted)=true\s*$", policy, re.M))
        return dict(unixSeconds=time.time(), awake=awake, locked=locked,
            benchmarkForeground=any(package in line for line in activity.splitlines() if "topResumedActivity=" in line),
            thermalStatus=int(status.group(1)) if status else None,
            batteryLevel=number("level"), batteryTemperatureTenthsC=number("temperature"),
            charging=[line.strip() for line in battery.splitlines() if "powered:" in line])

    before = observation()
    if before["locked"] or not before["awake"]:
        raise RuntimeError("Unlock the phone before starting a timed run")
    folder.mkdir()
    state = dict(schema="pixel-shared-decoding-execution-v1", status="running", phase=args.phase,
        artifactIndex="private-reference-0223", recordingIndex="recording-044",
        sourceSha256=source["sha256"], sourceBytes=source["bytes"], seconds=source["seconds"],
        model="Frozen highest-recall Distilled MobileNetV3-Large + TCN FP32",
        startedUnixSeconds=time.time(), environment=dict(model="Pixel 10 Pro", abi="arm64-v8a",
            api=adb("shell", "getprop", "ro.build.version.sdk"), transport="Tailscale ADB"),
        protocol="One fresh-process shared-decoder run, caches bypassed, both score specialists included. Source verification, model/APK transfer and result downloads are outside device timing.")
    state_path = root / (args.phase + "-execution.json")
    write(state_path, state)
    observations, stop = [before], threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                observations.append(observation())
            except Exception as error:
                observations.append(dict(unixSeconds=time.time(), error=type(error).__name__))
            write(folder / "observations.json", observations)
            stop.wait(20)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        command = [sys.executable, str(REPO / "scripts/benchmark-pixel-complete-pipeline.py"),
            "--adb", args.adb, "--serial", args.serial, "--graphs", str(parent / "graphs-recall"),
            "--output", str(folder), "--source-uri", source["sourceUri"], "--seconds", str(source["seconds"]),
            "--runs", "1", "--families", "mobile-large", "--transfer-models", "--shared-decoding"]
        if args.phase == "short":
            command.append("--capture-pixel-hashes")
        print(json.dumps(dict(phase=args.phase, status="starting")), flush=True)
        with (folder / "host.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = read(folder / "result.json")
        row = result["results"][0]
        assert result["status"] == "complete" and result["device"] == "Pixel 10 Pro"
        assert result["cacheMode"] == "bypass" and row["neural"]["sharedDecoding"]
        assert row["servingSideReady"] and row["sideSwitchReady"]
        state.update(status="measured", totalMs=row["totalMs"], rallyCount=row["rallyCount"],
                     finishedUnixSeconds=time.time())
        print(json.dumps(dict(phase=args.phase, status="measured", totalMs=row["totalMs"],
                              rallyCount=row["rallyCount"], stagesMs=row["stagesMs"])), flush=True)
    except BaseException as error:
        state.update(status="failed", error=type(error).__name__)
        raise
    finally:
        stop.set()
        monitor_thread.join(35)
        write(state_path, state)
    spec = importlib.util.spec_from_file_location("shared_parity", REPO / "scripts/validate-shared-video-decoding.py")
    parity_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parity_module)
    reference = parent / ("pixel-visual-retest/runs/repaired-1" if args.phase == "short" else "visual-repair/full-recall")
    parity = parity_module.compare(reference, folder)
    write(root / (args.phase + "-parity.json"), parity)
    state["outputParity"] = parity["exact"]
    state["allObservedAwakeAndUnlocked"] = all(o.get("awake") and not o.get("locked") for o in observations)
    state["status"] = "complete" if parity["exact"] and state["allObservedAwakeAndUnlocked"] else "needs-review"
    write(state_path, state)
    print(json.dumps(dict(phase=args.phase, status=state["status"], exact=parity["exact"],
        differences=[key for key, value in parity["checks"].items() if not value])), flush=True)
    if state["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

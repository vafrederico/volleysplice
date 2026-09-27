"""Sequential emulator qualification of experimental shared AV/neural decoding.

Uses ledger-indexed frozen inputs, one fresh process per run, and exact parity
gates. Heavy emulator runs must not overlap builds, browser tests, or training.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
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


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Memory(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong) for name in ("totalPhys", "availPhys", "totalPage", "availPage",
                                               "totalVirtual", "availVirtual", "availExtended")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adb", required=True)
    p.add_argument("--serial", required=True)
    p.add_argument("--phase", choices=("short", "full"), required=True)
    args = p.parse_args()
    if not args.serial.startswith("emulator-"):
        raise ValueError("Emulator-only protocol")
    indexed = private_value("private-reference-0223")
    parent = Path(os.environ.get("VOLLEYCUT_EMULATOR_ARTIFACT_ROOT", indexed))
    if not parent.is_absolute() or not parent.is_dir():
        raise ValueError("Configure the indexed artifact root through local environment")
    root = parent / "shared-video-decoding"
    state_path = root / (args.phase + "-execution.json")
    if state_path.exists():
        raise FileExistsError("Preserve previous run evidence")
    if args.phase == "full" and not read(root / "short-parity.json")["exact"]:
        raise ValueError("Short-clip qualification must pass first")
    prepared = read(parent / "native-area-full-video/prepared.private.json")
    if args.phase == "short":
        plan = read(parent / "native-area-optimization/runs/native-1/pipeline-plan.json")
        uri, seconds = plan["sourceUri"], 120
        expected = read(parent / "emulator-regression/source-receipt.json")
        expected_bytes = expected["bytes"]
    else:
        uri, seconds = prepared["sourceUri"], prepared["seconds"]
        expected = dict(sha256=prepared["sourceSha256"])
        expected_bytes = prepared["sourceSizeBytes"]
    package = "com.volleycut.nativeanalysis.pipelinebenchmark"

    def adb(*command, timeout=120, check=True):
        return subprocess.run([args.adb, "-s", args.serial, *command],
            capture_output=True, text=True, timeout=timeout, check=check).stdout.strip()

    assert adb("shell", "getprop", "ro.product.cpu.abi") == "x86_64"
    assert adb("shell", "getprop", "sys.boot_completed") == "1"
    assert adb("shell", "getenforce") == "Permissive"
    adb("shell", "svc", "power", "stayon", "true")
    adb("shell", "input", "keyevent", "224")
    adb("shell", "input", "keyevent", "82")
    # Resolve MediaStore's actual path privately and verify it outside timing.
    catalog = adb("shell", "content", "query", "--uri", uri, "--projection", "_data:_size:width:height")
    import re
    match = re.search(r"_data=(.*?), _size=(\d+)", catalog)
    if not match or int(match.group(2)) != expected_bytes:
        raise ValueError("Source identity/size mismatch")
    assert "width=1920" in catalog and "height=1080" in catalog
    assert adb("shell", "sha256sum", match.group(1), timeout=600).split()[0] == expected["sha256"]
    apk = root / "emulator-pipeline/pipeline-build/outputs/apk/debug/app-debug.apk"
    adb("install", "-r", str(apk), timeout=300)
    for permission in ("android.permission.READ_MEDIA_VIDEO", "android.permission.READ_MEDIA_AUDIO"):
        adb("shell", "pm", "grant", package, permission)
    state = dict(schema="shared-video-execution-v1", phase=args.phase, status="running",
        artifactIndex="private-reference-0223", recordingIndex="recording-044", seconds=seconds,
        sourceSha256=expected["sha256"], sourceBytes=expected_bytes, apkSha256=digest(apk), runs=[],
        protocol="Fresh process, bypassed caches, FP32 frozen highest-recall Distilled Large; sequential runs; no discarded warmup. Short runs capture pixel hashes. Full reference is the saved independent-decoder run.",
        environment=dict(abi="x86_64", api=adb("shell", "getprop", "ro.build.version.sdk"),
            selinux="Permissive", guestMemory=adb("shell", "head", "-n", "3", "/proc/meminfo").splitlines()))
    write(state_path, state)
    variants = ["short-independent", "short-shared"] if args.phase == "short" else ["full-shared"]
    try:
        for variant in variants:
            folder = root / variant
            folder.mkdir()
            row = dict(variant=variant, status="running", startedUnixSeconds=time.time())
            state["runs"].append(row)
            write(state_path, state)
            stop, observations = threading.Event(), []

            def monitor():
                while not stop.is_set():
                    try:
                        power = adb("shell", "dumpsys", "power", timeout=20)
                        memory = Memory()
                        memory.length = ctypes.sizeof(memory)
                        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
                        observations.append(dict(unixSeconds=time.time(),
                            wakefulness=[line.strip() for line in power.splitlines() if "mWakefulness=" in line],
                            hostAvailableMemoryMB=memory.availPhys / 1024**2))
                    except Exception as error:
                        observations.append(dict(unixSeconds=time.time(), error=type(error).__name__))
                    write(folder / "observations.json", observations)
                    stop.wait(15)

            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            try:
                command = [sys.executable, str(REPO / "scripts/benchmark-pixel-complete-pipeline.py"),
                    "--adb", args.adb, "--serial", args.serial, "--graphs", str(parent / "graphs-recall"),
                    "--output", str(folder), "--source-uri", uri, "--seconds", str(seconds),
                    "--runs", "1", "--families", "mobile-large", "--transfer-models"]
                if variant.endswith("shared"):
                    command.append("--shared-decoding")
                if args.phase == "short":
                    command.append("--capture-pixel-hashes")
                print(json.dumps(dict(variant=variant, status="starting")), flush=True)
                with (folder / "host.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                result = read(folder / "result.json")
                measured = result["results"][0]
                assert result["status"] == "complete" and result["cacheMode"] == "bypass"
                assert measured["servingSideReady"] and measured["sideSwitchReady"]
                row.update(status="complete", totalMs=measured["totalMs"], rallyCount=measured["rallyCount"],
                           finishedUnixSeconds=time.time(), resultSha256=digest(folder / "result.json"))
                print(json.dumps(row), flush=True)
            finally:
                stop.set()
                thread.join(30)
                write(state_path, state)
        module_spec = importlib.util.spec_from_file_location("shared_parity", REPO / "scripts/validate-shared-video-decoding.py")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        reference = root / "short-independent" if args.phase == "short" else parent / "native-area-full-video/native-recall"
        parity = module.compare(reference, root / variants[-1], args.phase == "short")
        write(root / (args.phase + "-parity.json"), parity)
        assert parity["exact"], "Output mismatch; inspect parity artifact before proceeding"
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error=type(error).__name__)
        raise
    finally:
        write(state_path, state)


if __name__ == "__main__":
    main()

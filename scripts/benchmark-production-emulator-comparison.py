"""Run indexed, frozen full-video production preprocessing controls sequentially.

Requires the previously qualified NAS-backed x86_64 emulator to be running.
Exact private inputs are resolved exclusively through the external ledger.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
from private_ledger import private_value


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", required=True)
    parser.add_argument("--serial", required=True)
    args = parser.parse_args()
    if not args.serial.startswith("emulator-"):
        raise ValueError("This protocol is emulator-only")
    indexed_root = private_value("private-reference-0223")
    parent = Path(os.environ.get("VOLLEYCUT_EMULATOR_ARTIFACT_ROOT", indexed_root))
    if not parent.is_absolute() or not parent.is_dir():
        raise ValueError("Configure the indexed artifact root for this host through the local environment")
    root = parent / "production-emulator-full-comparison"
    root.mkdir(exist_ok=True)
    if (root / "execution.json").exists():
        raise FileExistsError("Preserve the existing experiment; do not overwrite runs")
    package = "com.volleycut.nativeanalysis.pipelinebenchmark"

    def adb(*command, timeout=120, check=True):
        return subprocess.run([args.adb, "-s", args.serial, *command],
                              capture_output=True, text=True, timeout=timeout, check=check).stdout.strip()

    for _ in range(180):
        if adb("shell", "getprop", "sys.boot_completed", timeout=10, check=False) == "1":
            break
        time.sleep(2)
    else:
        raise TimeoutError("Emulator boot failed")
    assert adb("shell", "getprop", "ro.product.cpu.abi") == "x86_64"
    assert adb("shell", "getprop", "ro.build.type") == "userdebug"
    if adb("shell", "getenforce") != "Permissive":
        adb("root")
        time.sleep(3)
        adb("shell", "setenforce", "0")
        adb("unroot")
        time.sleep(3)
    assert adb("shell", "getenforce") == "Permissive"
    adb("shell", "svc", "power", "stayon", "true")
    adb("shell", "input", "keyevent", "224")
    adb("shell", "input", "keyevent", "82")

    prepared = json.loads((parent / "native-area-full-video/prepared.private.json").read_text())
    expected = json.loads((parent / "benchmark-contract.json").read_text())["sourceIdentities"]["full"]
    assert prepared["sourceSha256"] == expected["sha256"]
    assert int(adb("shell", "stat", "-c", "%s", prepared["sourcePath"])) == expected["sizeBytes"]
    assert adb("shell", "sha256sum", prepared["sourcePath"], timeout=600).split()[0] == expected["sha256"]
    catalog = adb("shell", "content", "query", "--uri", prepared["sourceUri"],
                  "--projection", "_data:_size:width:height")
    assert str(expected["sizeBytes"]) in catalog and "width=1920" in catalog and "height=1080" in catalog
    assert prepared["sourcePath"].replace("/sdcard/", "/storage/emulated/0/") in catalog

    apks = {
        "original-point": parent / "emulator-regression/baseline/baseline-x86_64.apk",
        "repaired-java": parent / "emulator-regression/repaired/repaired-x86_64.apk",
        "optimized-native": parent / "native-area-optimization/emulator-pipeline/pipeline-build/outputs/apk/debug/app-debug.apk",
    }
    historical = json.loads((REPO / "docs/research/android-emulator-visual-regression.json").read_text())
    full = json.loads((parent / "native-area-full-video/prepared.private.json").read_text())
    hashes = {name: digest(path) for name, path in apks.items()}
    assert hashes["original-point"] == historical["identity"]["apks"]["baseline"]["sha256"]
    assert hashes["repaired-java"] == historical["identity"]["apks"]["repaired"]["sha256"]
    assert hashes["optimized-native"] == full["apkSha256"]
    assets = {}
    for name, path in apks.items():
        with zipfile.ZipFile(path) as archive:
            assets[name] = {n: hashlib.sha256(archive.read(n)).hexdigest()
                            for n in archive.namelist() if n.startswith("assets/") and not n.endswith("/")}
    assert assets["original-point"] == assets["repaired-java"] == assets["optimized-native"]
    state = {
        "schema": "production-emulator-full-comparison-v1", "status": "running",
        "artifactIndex": "private-reference-0223", "recordingIndex": "recording-044",
        "sourceSha256": expected["sha256"], "sourceBytes": expected["sizeBytes"],
        "seconds": prepared["seconds"], "apkHashes": hashes, "identicalPackagedAssets": True,
        "assetHashes": assets["optimized-native"], "order": list(apks), "runs": [],
        "protocol": "One fresh-process full-video production-only run per version, caches bypassed; no discarded pipeline warmup. Versions run sequentially; filesystem and JIT warmth not controlled.",
        "environment": {"abi": "x86_64", "buildType": "userdebug", "selinux": "Permissive",
                        "api": adb("shell", "getprop", "ro.build.version.sdk"),
                        "guestMemory": adb("shell", "head", "-n", "3", "/proc/meminfo").splitlines()},
    }
    write(root / "execution.json", state)

    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ("totalPhys", "availPhys", "totalPage", "availPage", "totalVirtual", "availVirtual", "availExtended")]

    try:
        for variant, apk in apks.items():
            folder = root / variant
            folder.mkdir()
            adb("install", "-r", str(apk), timeout=300)
            for permission in ("android.permission.READ_MEDIA_VIDEO", "android.permission.READ_MEDIA_AUDIO"):
                adb("shell", "pm", "grant", package, permission)
            row = {"variant": variant, "status": "running", "startedUnixSeconds": time.time()}
            state["runs"].append(row)
            write(root / "execution.json", state)
            stop = threading.Event()
            observations = []

            def monitor():
                while not stop.is_set():
                    try:
                        power = adb("shell", "dumpsys", "power", timeout=20)
                        memory = Memory()
                        memory.length = ctypes.sizeof(memory)
                        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
                        observations.append({"unixSeconds": time.time(),
                            "wakefulness": [line.strip() for line in power.splitlines() if "mWakefulness=" in line],
                            "hostAvailableMemoryMB": memory.availPhys / 1024 ** 2,
                            "hostMemoryLoadPercent": memory.load})
                    except Exception as error:
                        observations.append({"unixSeconds": time.time(), "error": type(error).__name__})
                    write(folder / "observations.json", observations)
                    stop.wait(15)

            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            try:
                command = [sys.executable, str(REPO / "scripts/benchmark-pixel-complete-pipeline.py"),
                    "--adb", args.adb, "--serial", args.serial, "--graphs", str(parent / "graphs-recall"),
                    "--output", str(folder), "--source-uri", prepared["sourceUri"],
                    "--seconds", str(prepared["seconds"]), "--runs", "1", "--families", "production"]
                print(json.dumps({"variant": variant, "status": "starting"}), flush=True)
                with (folder / "host.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                result = json.loads((folder / "result.json").read_text())
                measured = result["results"][0]
                assert result["status"] == "complete" and result["cacheMode"] == "bypass"
                assert measured["sampleRows"] == 4245 and "neural" not in measured
                assert measured["servingSideReady"] and measured["sideSwitchReady"]
                assert measured["roi"] == [0, 0, 1, 1]
                row.update(status="complete", totalMs=measured["totalMs"],
                           rallyCount=measured["rallyCount"], finishedUnixSeconds=time.time(),
                           resultSha256=digest(folder / "result.json"))
                print(json.dumps(row), flush=True)
            finally:
                stop.set()
                thread.join(timeout=30)
                write(folder / "observations.json", observations)
                write(root / "execution.json", state)
        state["status"] = "complete"
    except Exception as error:
        state.update(status="failed", error=type(error).__name__)
        raise
    finally:
        write(root / "execution.json", state)


if __name__ == "__main__":
    main()

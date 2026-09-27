"""Measure the normal Android app's optional paired variants on the authorized emulator.

Source mappings and APK locations are supplied by an external preparation receipt.
No model fitting, calibration, or threshold selection occurs here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
from private_ledger import private_value


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--mode", choices=("single", "paired"), required=True)
    parser.add_argument("--variant", choices=("recall", "f1"), default="recall")
    parser.add_argument("--scope", choices=("short", "full"), default="short")
    parser.add_argument("--cache", choices=("bypass", "refresh", "use"), default="bypass")
    args = parser.parse_args()
    if not args.serial.startswith("emulator-") or not re.fullmatch(r"[a-z0-9-]+", args.run):
        raise ValueError("Use the authorized emulator and a nonidentifying run label")
    indexed = private_value("private-reference-0223")
    parent = Path(os.environ.get("VOLLEYCUT_DEVICE_ARTIFACT_ROOT", indexed))
    if not parent.is_absolute() or not parent.is_dir():
        raise ValueError("Configure the indexed artifact root for this host")
    root = parent / "paired-variant-experiment"
    prepared = read(root / "prepared.private.json")
    source = prepared["sources"][args.scope]
    if not source["verified"]:
        raise ValueError("Verify source bytes before timing")
    if args.scope == "full" and not read(root / "short-qualification.json")["passed"]:
        raise ValueError("Qualify both short outputs before the full run")
    folder = root / args.run
    folder.mkdir(exist_ok=False)
    package = "com.volleycut.nativeanalysis.debug"

    def adb(*command, timeout=60):
        return subprocess.run([args.adb, "-s", args.serial, *command], capture_output=True,
                              text=True, check=True, timeout=timeout).stdout.strip()

    if adb("shell", "getprop", "ro.product.cpu.abi") != "x86_64":
        raise ValueError("This protocol is emulator-only")
    apk = Path(prepared["apk"])
    actual = hashlib.sha256(apk.read_bytes()).hexdigest()
    if actual != prepared["apkSha256"]:
        raise ValueError("The installed experiment APK must stay frozen during measurements")
    adb("shell", "am", "force-stop", package)
    state = dict(schema="paired-neural-variant-experiment-v1", status="running",
                 recordingIndex="recording-044", artifactIndex="private-reference-0223",
                 mode=args.mode, variant=args.variant, scope=args.scope, cacheMode=args.cache,
                 sourceSha256=source["sha256"], sourceBytes=source["bytes"], seconds=source["seconds"],
                 apkSha256=actual, startedUnixSeconds=time.time(),
                 environment=dict(abi="x86_64", api=adb("shell", "getprop", "ro.build.version.sdk"),
                                  selinux=adb("shell", "getenforce")),
                 protocol="Sequential fresh-process observations. File/model caches and host scheduling are uncontrolled. No training. Device readiness excludes diagnostics and project serialization, which are reported separately.")
    write(folder / "execution.json", state)
    command = [args.adb, "-s", args.serial, "shell", "am", "instrument", "-w", "-r",
               "-e", "class", "com.volleycut.nativeanalysis.PairedNeuralIntegrationTest#indexedVideoBenchmark",
               "-e", "neuralVideoUri", source["uri"], "-e", "neuralVideoSeconds", str(source["seconds"]),
               "-e", "pairedMode", args.mode, "-e", "neuralRallyModel", f"distilled-large-{args.variant}-v1",
               "-e", "neuralCacheMode", args.cache,
               package + ".test/androidx.test.runner.AndroidJUnitRunner"]
    try:
        with (folder / "instrumentation.private.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
            while process.poll() is None:
                time.sleep(15)
                elapsed = time.time() - state["startedUnixSeconds"]
                if elapsed > 7200:
                    adb("shell", "am", "force-stop", package)
                    process.terminate()
                    raise TimeoutError("Instrumented experiment exceeded two hours")
                print(json.dumps(dict(run=args.run, status="running", elapsedSeconds=round(elapsed))), flush=True)
            if process.returncode:
                raise RuntimeError("Instrumentation command failed; inspect the private receipt")
        log_text = (folder / "instrumentation.private.log").read_text(encoding="utf-8")
        if "OK (1 test)" not in log_text or "FAILURES!!!" in log_text:
            raise RuntimeError("The instrumented experiment did not pass")
        result = json.loads(adb("exec-out", "run-as", package, "cat",
                               "files/integration-validation/paired-production-result.json"))
        write(folder / "result.json", result)
        state.update(status="complete", allReadyMs=result["allReadyMs"], finishedUnixSeconds=time.time())
        print(json.dumps(dict(run=args.run, status="complete", allReadySeconds=result["allReadyMs"] / 1000)), flush=True)
    except BaseException as error:
        state.update(status="failed", error=type(error).__name__)
        raise
    finally:
        write(folder / "execution.json", state)


if __name__ == "__main__":
    main()

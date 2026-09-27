#!/usr/bin/env python3
"""Pause only the extraction coordinator before it can race a bounded cache job."""
from analysis.private_ledger import private_value
import argparse
import hashlib
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--coordinator-pid", type=int, required=True)
    parser.add_argument("--precompute-output", type=Path, required=True)
    args = parser.parse_args()
    pid = args.coordinator_pid
    command_path = Path(f"/proc/{pid}/cmdline")
    expected = "scripts/prepare-neural-dino-transfer.py"
    command = command_path.read_bytes().replace(b"\0", b" ").decode()
    if expected not in command or "--workers 2" not in command or "--worker-id" in command:
        raise ValueError("PID is not the declared two-worker coordinator")
    report = {"createdAt": datetime.now(timezone.utc).isoformat(), "coordinatorPid": pid,
              "verifiedCoordinatorCommand": command, "paused": False,
              "planConcurrencyClarification": "Frozen extraction-plan.json maximumWorkers=2 refers to original coordinator only. One independent cache-only precompute gives actual maximum extraction concurrency3; numerical extraction recipe is unchanged.",
              "guard": "If main reaches Pixel194010307 before cache-only helper finishes, SIGSTOP only coordinator; current children finish normally. Resume coordinator when cache-only outcome exists.",
              "precomputeOutput": str(args.precompute_output),
              "scriptSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    paused = False
    try:
        while command_path.exists() and not (args.precompute_output/"outcome.json").exists():
            if list((args.root/"extraction-logs").glob(private_value('private-reference-0099'))) and not paused:
                os.kill(pid, signal.SIGSTOP)
                paused = True
                report["paused"] = True
                report["pausedAt"] = datetime.now(timezone.utc).isoformat()
                print("Paused coordinator before pending final Pixel source; active extraction children continue", flush=True)
            time.sleep(5)
        report["helperOutcomeExists"] = (args.precompute_output/"outcome.json").exists()
    finally:
        if paused and command_path.exists():
            os.kill(pid, signal.SIGCONT)
            report["resumedAt"] = datetime.now(timezone.utc).isoformat()
        report["finishedAt"] = datetime.now(timezone.utc).isoformat()
        path = args.precompute_output/"scheduling-guard.json"
        with path.open("x") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(json.dumps({"guardComplete": True, "paused": report["paused"], "artifact": str(path)}), flush=True)


if __name__ == "__main__":
    main()

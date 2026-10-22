"""Run the pinned wrapper and require fresh, nonempty, entirely passing JVM tests."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET


REPOSITORY = Path(__file__).resolve().parents[1]
MANDATORY_TASKS = (
    ":app:prepareNeuralAssets", ":app:prepareServingAssets",
    ":app:verifySuppressionAsset", ":app:verifyServingSideAsset",
    ":app:verifySideSwitchAsset", ":app:testDebugUnitTest",
)


class GateFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateFailure(message)


def fresh_file(path: Path, started_ns: int) -> None:
    require(path.is_file() and path.stat().st_size > 0, f"Missing or empty evidence: {path.name}")
    require(path.stat().st_mtime_ns >= started_ns, f"Stale evidence: {path.name}")


def verify_tasks(path: Path, started_ns: int, run_id: str) -> None:
    fresh_file(path, started_ns)
    receipts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    require(len(receipts) == len(MANDATORY_TASKS), "Missing or duplicate mandatory task outcomes")
    require({row["task"] for row in receipts} == set(MANDATORY_TASKS), "Missing mandatory task outcomes")
    for row in receipts:
        require(row.get("runId") == run_id, "Task outcome belongs to a different run")
        require(row.get("executed") is True and row.get("didWork") is True
                and all(row.get(key) is False for key in ("skipped", "noSource", "upToDate", "failed"))
                and row.get("skipMessage") is None,
                f"Mandatory task did not execute successfully: {row['task']}")


def verify_reports(directory: Path, started_ns: int) -> dict:
    reports = sorted(directory.glob("*.xml"))
    require(bool(reports), "No fresh JVM XML reports were produced")
    total = 0
    for report in reports:
        fresh_file(report, started_ns)
        suite = ET.parse(report).getroot()
        require(suite.tag == "testsuite", "Unexpected JVM XML report format")
        count = int(suite.attrib["tests"])
        require(count > 0, "Empty JVM test suite")
        require(all(int(suite.attrib[key]) == 0 for key in ("skipped", "errors", "failures")),
                "JVM report contains skipped, errored, or failed tests")
        cases = suite.findall("testcase")
        require(len(cases) == count, "JVM report count does not match executed test cases")
        require(not any(node.tag in ("skipped", "failure", "error") for node in suite.iter()),
                "JVM test case contains a skipped, failure, or error result")
        timestamp = datetime.fromisoformat(suite.attrib["timestamp"].replace("Z", "+00:00"))
        require(timestamp.tzinfo is not None, "JVM report timestamp has no timezone")
        # Gradle serializes timestamps to milliseconds, while the marker has ns.
        require(int(timestamp.timestamp() * 1_000_000_000) >= started_ns - 1_000_000,
                "JVM report contains a stale suite timestamp")
        require(timestamp <= datetime.now(timezone.utc), "JVM report timestamp is in the future")
        total += count
    return {"tests": total, "suites": len(reports), "skipped": 0, "errors": 0, "failures": 0}


def run_gate(project: Path, wrapper: Path) -> dict:
    output = project / "build" / "regression-gate"
    output.mkdir(parents=True, exist_ok=True)
    # Never accept, delete, or reuse an earlier run's report directory.
    run_directory = Path(tempfile.mkdtemp(prefix="jvm-", dir=output)).resolve()
    run_id = uuid.uuid4().hex
    environment = os.environ.copy()
    environment.update(VOLLEYCUT_REGRESSION_RUN_DIR=str(run_directory), VOLLEYCUT_REGRESSION_RUN_ID=run_id)
    for key in ("GITHUB_TOKEN", "FORGEJO_TOKEN", "GITEA_TOKEN"):
        environment.pop(key, None)
    command = [str(wrapper), "--no-daemon", "--stacktrace", "--console=plain", "--max-workers=1",
               "--no-build-cache", "--no-configuration-cache", "--init-script",
               str(REPOSITORY / "android" / "scripts" / "regression-gate.init.gradle"),
               ":app:prepareNeuralAssets", "--rerun", ":app:prepareServingAssets", "--rerun",
               ":app:testDebugUnitTest", "--rerun"]
    started_ns = time.time_ns()
    with (run_directory / "gradle.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=project, env=environment, stdout=log, stderr=subprocess.STDOUT)
    print((run_directory / "gradle.log").read_text(encoding="utf-8"), end="", flush=True)
    require(result.returncode == 0, f"Gradle failed with exit code {result.returncode}")
    verify_tasks(run_directory / "tasks.jsonl", started_ns, run_id)
    summary = verify_reports(run_directory / "junit", started_ns)
    (run_directory / "summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    print("Verified fresh Android JVM gate: " + json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    try:
        android = REPOSITORY / "android"
        run_gate(android, android / ("gradlew.bat" if os.name == "nt" else "gradlew"))
    except (GateFailure, OSError, ValueError, KeyError, ET.ParseError) as error:
        raise SystemExit(f"Android regression gate failed: {error}")

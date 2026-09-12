"""Physical iPad queue acceptance; creates disposable jobs, preserves the reference project.

Run with the lab's existing pymobiledevice3 Python. UI mutations go through WDA;
read-only AFC ledger observations prove progress while the parking app is active.
Only VolleySplice is terminated for the restart test. The VM/WDA stay running.
"""
import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time

from pymobiledevice3.lockdown import create_using_usbmux
from pymobiledevice3.services.house_arrest import HouseArrestService

spec = importlib.util.spec_from_file_location("editor_smoke", Path(__file__).with_name("editor-smoke.py"))
editor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(editor)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--recording", required=True)
parser.add_argument("--project", required=True, help="Existing reference project button ID")
parser.add_argument("--end-seconds", type=float, default=1100, help="Uncached window long enough for interruption checks")
parser.add_argument("--output", type=Path, default=Path("E:/wslmac/artifacts/volleysplice/queue-smoke"))
args = parser.parse_args()
config = SimpleNamespace(lab=Path("E:/wslmac"), url="http://127.0.0.1:18100", mjpeg="http://127.0.0.1:19100/",
                         bundle="com.vafrederico.VolleySplice", project=args.project, output=args.output)
test = editor.EditorSmoke(config)
created = []
original_scoring = None
failed = False

async def read_jobs():
    async with await create_using_usbmux(serial="<ios-device-udid>") as lockdown:
        async with await HouseArrestService.create(lockdown=lockdown, bundle_id=config.bundle) as afc:
            # AFC stats the path before opening it. An atomic progress-ledger
            # replacement between those calls can make that read too short.
            # Retry read-only observations; a persistently invalid file fails.
            for attempt in range(5):
                data = await afc.get_file_contents("/Library/Application Support/ProcessingJobs/queue.json")
                try:
                    return json.loads(data)["jobs"]
                except json.JSONDecodeError:
                    if attempt == 4:
                        raise
                    await asyncio.sleep(0.2)

def jobs():
    value = asyncio.run(read_jobs())
    (test.output / "latest-ledger.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    return value

def job(job_id):
    return next(value for value in jobs() if value["id"] == job_id)

def wait_job(job_id, states, seconds=30):
    deadline = time.monotonic() + seconds
    last_print = 0
    while time.monotonic() < deadline:
        value = job(job_id)
        if time.monotonic() - last_print > 10:
            print(json.dumps({"job": job_id, "state": value["state"], "progress": value["progress"], "detail": value["detail"]}), flush=True)
            last_print = time.monotonic()
        if value["state"] in states:
            return value
        if value["state"] == "failed":
            raise AssertionError(value)
        time.sleep(1)
    raise TimeoutError(f"Job did not reach {states}: {value}")

def set_end(value):
    field = test.element("gameEnd")
    test.command(f"/element/{field}/clear", {})
    test.command(f"/element/{field}/value", {"value": f"{value:g}\n"})

def show_queue():
    test.source(refresh=True)
    if not test.nodes("queueDone"):
        test.click("processingQueueBar" if test.nodes("processingQueueBar") else "processingQueue")

try:
    test.wda.connect()
    test.park()
    settings = test.command("/appium/settings")
    test.original_settings = {key: settings[key] for key in ("waitForIdleTimeout", "animationCoolOffTimeout")}
    test.command("/appium/settings", {"settings": {"waitForIdleTimeout": 0, "animationCoolOffTimeout": 0}})
    test.wda.activate(config.bundle)
    test.source(refresh=True)
    if test.nodes("settingsDone"):
        test.click("settingsDone")
    if test.nodes("backToProjects"):
        test.click("backToProjects")
    baseline = jobs()
    assert not any(value["state"] in ("queued", "running", "cancelling") for value in baseline), "Existing jobs must finish first"
    baseline_ids = {value["id"] for value in baseline}
    test.click("refreshRecordings")
    test.click("recording-" + args.recording)
    test.wait(lambda: bool(test.nodes("gameEnd")), "Recording setup did not open")
    original_scoring = test.value("prepareScore")
    set_end(args.end_seconds)
    if test.value("prepareScore") == "1":
        test.click("prepareScore")
    test.click("analyzeRecording")
    created = [value["id"] for value in jobs() if value["id"] not in baseline_ids]
    assert len(created) == 1
    first = created[0]
    assert job(first)["analysis"]["endMs"] == round(args.end_seconds * 1000)
    assert job(first)["analysis"]["prepareScores"] is False
    wait_job(first, {"running"})
    set_end(30)
    test.click("analyzeRecording")
    created = [value["id"] for value in jobs() if value["id"] not in baseline_ids]
    assert len(created) == 2
    second = next(value for value in created if value != first)
    assert job(second)["state"] == "queued", "Second decode overlapped the first"
    show_queue()
    test.click("cancelJob-" + second)
    assert job(second)["state"] == "cancelled"
    assert job(first)["state"] == "running"
    test.click("cancelJob-" + first)
    wait_job(first, {"cancelled"})
    test.record("serial-queue-and-cancellation", jobs=created)
    test.click("resumeJob-" + first)
    resumed = wait_job(first, {"running"})
    assert resumed["attempt"] == 2
    test.click("queueDone")
    test.click("backToProjects") if test.nodes("backToProjects") else None
    if test.value("prepareScore") != original_scoring:
        test.click("prepareScore")
    test.choose_project()
    test.capture("editing-reference-during-analysis")
    assert job(first)["state"] == "running"
    test.park()
    before = job(first)
    deadline = time.monotonic() + 40
    while True:
        time.sleep(2)
        after = job(first)
        if after["state"] == "completed" or after["progress"] > before["progress"]:
            break
        assert after["state"] == "running", after
        assert time.monotonic() < deadline, (before, after)
    test.record("background-progress", before=before["progress"], after=after["progress"])
    assert after["state"] == "running", "Background completion passed; restart test needs a longer uncached window"
    test.command("/wda/apps/terminate", {"bundleId": config.bundle})
    test.wda.activate(config.bundle)
    recovered = wait_job(first, {"interrupted"})
    assert recovered["attempt"] == 2
    time.sleep(2)
    assert job(first)["state"] == "interrupted", "Restart silently resumed work"
    show_queue()
    test.capture("recovered-interruption")
    test.click("resumeJob-" + first)
    wait_job(first, {"running"})
    test.park()
    complete = wait_job(first, {"completed"}, seconds=900)
    assert complete["attempt"] == 3 and complete["outputNames"]
    assert job(second)["state"] == "cancelled"
    test.record("restart-and-explicit-resume", job=first, outputs=complete["outputNames"])
except Exception as error:
    failed = True
    test.results.append({"failure": str(error), "createdJobs": created})
    try:
        test.capture("failure")
    except Exception:
        pass
    raise
finally:
    try:
        if failed and created:
            try:
                pending = [value["id"] for value in jobs() if value["id"] in created and value["state"] in ("queued", "running", "cancelling")]
                if pending:
                    test.wda.activate(config.bundle)
                    show_queue()
                    for value in pending:
                        test.click("cancelJob-" + value)
                        wait_job(value, {"cancelled", "failed", "completed"})
            except Exception as cleanup_error:
                test.results.append({"jobCleanupError": str(cleanup_error), "createdJobs": created})
        if failed and original_scoring is not None:
            try:
                test.wda.activate(config.bundle)
                test.source(refresh=True)
                if test.nodes("queueDone"):
                    test.click("queueDone")
                if test.nodes("backToProjects"):
                    test.click("backToProjects")
                if not test.nodes("prepareScore"):
                    test.click("recording-" + args.recording)
                    test.wait(lambda: bool(test.nodes("prepareScore")), "Recording setup did not reopen")
                if test.value("prepareScore") != original_scoring:
                    test.click("prepareScore")
            except Exception as cleanup_error:
                test.results.append({"setupRestoreError": str(cleanup_error)})
        test.park()
        test.results.append({"teardown": "parking foreground; idle timer disabled"})
    finally:
        try:
            if test.original_settings is not None:
                test.command("/appium/settings", {"settings": test.original_settings})
        except Exception as cleanup_error:
            test.results.append({"settingsRestoreError": str(cleanup_error)})
            if not failed:
                raise
        finally:
            (test.output / "results.json").write_text(json.dumps(test.results, indent=2), encoding="utf-8")

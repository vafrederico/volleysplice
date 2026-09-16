"""Serial physical-iPad editor checks using the existing Windows lab WDA client.

Run against a disposable known-video project. Changes are restored through Undo
or the original control values. --restart additionally verifies disk persistence
across terminating/relaunching only VolleySplice. No media is decoded on the Mac.
Every run begins with fresh WDA status/source and always returns to idle-disabled
parking, including assertion failures. The script never shares to another app.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET


class EditorSmoke:
    def __init__(self, args):
        self.args = args
        spec = importlib.util.spec_from_file_location("lab_wda", args.lab / "scripts/wda-client.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.wda = self.module.WDA(args.url)
        self.output = args.output or args.lab / "artifacts/volleysplice/editor-smoke"
        self.output.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.project_button = args.project
        self.original_settings = None
        self.tree = None
        self.tree_time = 0.0
        self.playback_element = None
        self.pending_sliders = {}
        self.pending_chapter = None
        self.settings_baseline = {}

    def command(self, path, body=None):
        began = time.monotonic()
        try:
            try:
                return self.wda.command(path, body)
            except TimeoutError:
                if path != "/source" or body is not None:
                    raise
                print(json.dumps({"retry": "read-only source timeout", "attempt": 2}), flush=True)
                return self.wda.command(path, body)
        finally:
            elapsed = time.monotonic() - began
            if body is not None and path not in ("/element", "/elements"):
                self.tree = None
            if elapsed >= 2:
                print(json.dumps({"command": path, "seconds": round(elapsed, 2)}), flush=True)

    def source(self, refresh=False):
        if refresh or self.tree is None:
            self.tree = ET.fromstring(self.command("/source"))
            self.tree_time = time.monotonic()
        return self.tree

    def nodes(self, name=None, tree=None):
        tree = tree if tree is not None else self.source()
        return [node for node in tree.iter() if name is None or node.get("name") == name]

    def node(self, name, scroll=False, require_visible=True):
        observed = self.nodes(name)
        visible = [node for node in observed if node.get("visible") == "true"]
        if observed and not visible and scroll:
            # Native scrollTo can perform 25 internal scrolls. Instead make at
            # most twelve single scrolls, grounded in the observed target and
            # enclosing scroll-view coordinates, refreshing after each one.
            for _ in range(12):
                tree = self.source()
                parents = {child: parent for parent in tree.iter() for child in parent}
                targets = self.nodes(name)
                if not targets:
                    break
                target = targets[0]
                parent = parents.get(target)
                while parent is not None and not (parent.get("type") in (
                        "XCUIElementTypeScrollView", "XCUIElementTypeTable", "XCUIElementTypeCollectionView")
                        and parent.get("visible") == "true"):
                    parent = parents.get(parent)
                if parent is None:
                    break
                direction = "up" if float(target.get("y", 0)) < float(parent.get("y", 0)) else "down"
                # A scroll view's accessibility frame can extend beyond the
                # screen. Native scroll then starts on the Home indicator.
                # Bound the gesture to the observed visible interior instead.
                app = next(node for node in tree.iter() if node.get("type") == "XCUIElementTypeApplication")
                left = max(float(parent.get("x")), float(app.get("x")) + 20)
                right = min(float(parent.get("x")) + float(parent.get("width")), float(app.get("x")) + float(app.get("width")) - 20)
                top = max(float(parent.get("y")) + 20, float(app.get("y")) + 80)
                bottom = min(float(parent.get("y")) + float(parent.get("height")) - 20, float(app.get("y")) + float(app.get("height")) - 80)
                if right <= left or bottom - top < 40:
                    break
                x = (left + right) / 2
                start = bottom if direction == "down" else top
                end = start + (top - bottom) * (0.6 if direction == "down" else -0.6)
                self.command("/actions", {"actions": [{"type": "pointer", "id": "scroll", "parameters": {"pointerType": "touch"}, "actions": [
                    {"type": "pointerMove", "duration": 0, "origin": "viewport", "x": x, "y": start},
                    {"type": "pointerDown", "button": 0},
                    {"type": "pointerMove", "duration": 450, "origin": "viewport", "x": x, "y": end},
                    {"type": "pointerUp", "button": 0}]}]})
                visible = [node for node in self.nodes(name) if node.get("visible") == "true"]
                if visible:
                    break
        if observed and not require_visible:
            return (visible or observed)[0]
        if not visible:
            raise AssertionError(f"No currently visible accessibility element: {name}")
        return visible[0]

    def element(self, name):
        # Refresh old observations before acting, but reuse the immediately
        # preceding assertion's snapshot. Never scroll from a read/poll method.
        if time.monotonic() - self.tree_time > 2:
            self.source(refresh=True)
        observed = self.node(name, scroll=True)
        if observed.get("enabled") != "true":
            raise AssertionError(f"Control is disabled: {name}")
        # System Files can expose a hidden search-cancel element before its
        # visible Cancel button with the same name. Resolve the observed type
        # and visibility, rather than clicking the first identifier match.
        predicate = (f"name == {json.dumps(name, ensure_ascii=False)} AND "
                     f"type == {json.dumps(observed.get('type'))} AND visible == true")
        found = self.command("/element", {"using": "predicate string", "value": predicate})
        return found.get("ELEMENT") or found["element-6066-11e4-a52e-4f735466cecf"]

    def click(self, name):
        print(json.dumps({"action": "click", "control": name}), flush=True)
        element = self.element(name)
        self.command(f"/element/{element}/click", {})

    def label(self, name):
        node = self.node(name)
        return node.get("label", node.get("value", ""))

    def value(self, name):
        return self.node(name, require_visible=False).get("value", "")

    def wait(self, predicate, message, seconds=15):
        print(json.dumps({"waiting": message}), flush=True)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                self.source(refresh=True)
                if predicate():
                    return
            except (AssertionError, StopIteration):
                pass
            time.sleep(0.25)
        raise TimeoutError(message)

    def wait_attribute(self, element, attribute, predicate, message, seconds=15):
        # The element was resolved from a fresh stable editor observation before
        # Play. Read only that element while the 100 ms playback clock changes;
        # a full XML hierarchy can fail to settle even with short idle waits.
        print(json.dumps({"waiting": message, "attribute": attribute}), flush=True)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            value = self.command(f"/element/{element}/attribute/{attribute}")
            if predicate(value):
                return value
            time.sleep(0.25)
        raise TimeoutError(message)

    def capture(self, name):
        tree = self.source(refresh=time.monotonic() - self.tree_time > 2)
        (self.output / f"{name}.xml").write_text(ET.tostring(tree, encoding="unicode"), encoding="utf-8")
        with urllib.request.urlopen(self.args.mjpeg, timeout=15) as response:
            buffer = b""
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and len(buffer) < 4 * 1024 * 1024:
                block = response.read1(16384)
                if not block:
                    break
                buffer += block
                start, end = buffer.find(b"\xff\xd8"), buffer.find(b"\xff\xd9")
                if 0 <= start < end:
                    (self.output / f"{name}.jpg").write_bytes(buffer[start:end + 2])
                    return
        raise RuntimeError("No complete MJPEG frame")

    def record(self, name, **details):
        result = {"test": name, "result": "passed", **details}
        self.results.append(result)
        print(json.dumps(result), flush=True)

    def choose_project(self):
        self.wait(lambda: bool(self.nodes("refreshRecordings")), "Project chooser did not open")
        self.click("refreshRecordings")
        candidates = [node.get("name") for node in self.nodes()
                      if node.get("type") == "XCUIElementTypeButton"
                      and node.get("name", "").startswith("project-")
                      and node.get("visible") == "true"]
        if self.project_button is None:
            if len(candidates) != 1:
                raise AssertionError(f"Specify --project with an observed project-button ID; found {candidates}")
            self.project_button = candidates[0]
        self.node(self.project_button, scroll=True)
        self.click(self.project_button)
        self.wait(lambda: bool(self.nodes("currentRallyRange")), "Project did not reopen", seconds=30)

    def slider_fraction(self, name):
        raw = self.value(name)
        # SwiftUI exposes actual bound values here (e.g. "3,500" ms),
        # not the normalized fraction expected by XCTest's set-value route.
        matched = re.search(r"[-+]?\d+(?:\.\d+)?", raw.replace(",", "").replace("\u202f", "").replace("\u00a0", ""))
        if not matched:
            raise AssertionError(f"Slider has no numeric accessibility value: {name}={raw!r}")
        value = float(matched.group())
        if "%" in raw:
            value /= 100
        elif name in ("beforePadding", "afterPadding", "joinGap"):
            value /= 10_000
        if not 0 <= value <= 1:
            raise AssertionError(f"Invalid normalized slider value: {name}={raw!r}")
        return value

    def set_slider(self, name, fraction):
        # Verified installed WDA FBElementCommands.handleSetValue uses a 0..1
        # fraction with XCTest adjustToNormalizedSliderPosition for sliders.
        element = self.element(name)
        self.command(f"/element/{element}/value", {"value": str(fraction)})

    def restore_slider(self, name, original):
        if any(node.get("type") == "XCUIElementTypeKeyboard" for node in self.nodes()):
            self.command("/wda/keyboard/dismiss", {})
        # Native slider dragging is deliberately tested for changes, but XCTest
        # does not reliably hit a precise step. Use the editor's exact seconds
        # field for restoration, then verify the resulting bound slider value.
        field = self.element(name + "Seconds")
        self.command(f"/element/{field}/clear", {})
        self.command(f"/element/{field}/value", {"value": f"{original * 10:.1f}\n"})
        self.wait(lambda: abs(self.slider_fraction(name) - original) < 0.0001,
                  f"{name} did not restore exact {round(original * 10_000)} milliseconds")

    def remember_slider(self, name, original):
        self.pending_sliders[name] = original
        (self.output / "pending-slider-restoration.json").write_text(json.dumps(self.pending_sliders, indent=2), encoding="utf-8")

    def forget_slider(self, name):
        self.pending_sliders.pop(name, None)
        (self.output / "pending-slider-restoration.json").write_text(json.dumps(self.pending_sliders, indent=2), encoding="utf-8")

    def assert_settings_unchanged(self, stage):
        self.click("editorSettings")
        self.wait(lambda: bool(self.nodes("beforePadding")), "Settings did not open for invariant check")
        observed = {name: self.slider_fraction(name) for name in self.settings_baseline}
        mismatches = {name: round(value * 10_000) for name, value in observed.items()
                      if abs(value - self.settings_baseline[name]) >= 0.0001}
        if mismatches:
            for name in mismatches:
                self.remember_slider(name, self.settings_baseline[name])
            raise AssertionError(f"Settings changed after {stage}: observed {mismatches}; expected "
                                 f"{ {name: round(self.settings_baseline[name] * 10_000) for name in mismatches} }")
        self.click("settingsDone")
        self.record("settings-unchanged", stage=stage, milliseconds={name: round(value * 10_000) for name, value in observed.items()})

    def park(self):
        self.wda.connect()
        self.wda.activate(self.module.PARKING)
        if "Idle timer: disabled" not in self.command("/source"):
            raise AssertionError("Parking did not report its idle timer disabled")

    def run(self):
        self.wda.connect()
        (self.output / "preflight.xml").write_text(self.command("/source"), encoding="utf-8")
        self.park()
        try:
            # The editor's playback clock prevents XCTest's default 10-second
            # idle wait. Assert observed UI state explicitly after every action.
            settings = self.command("/appium/settings")
            self.original_settings = {key: settings[key] for key in ("waitForIdleTimeout", "animationCoolOffTimeout")}
            self.command("/appium/settings", {"settings": {"waitForIdleTimeout": 0, "animationCoolOffTimeout": 0}})
            if self.args.start_at_settings and self.args.restart:
                self.command("/wda/apps/terminate", {"bundleId": self.args.bundle})
            if self.args.start_at_settings:
                self.command("/orientation", {"orientation": "PORTRAIT"})
            self.wda.activate(self.args.bundle)
            self.tree = None
            if not self.nodes("currentRallyRange") and not (self.args.start_at_settings and self.nodes("beforePadding")):
                self.choose_project()
            if not self.args.start_at_settings:
                for orientation in ("PORTRAIT", "LANDSCAPE"):
                    self.command("/orientation", {"orientation": orientation})
                    self.wait(lambda: self.command("/orientation") == orientation, "Orientation did not change")
                    self.wait(lambda: bool(self.nodes("playPause")), "Editor controls missing after rotation")
                    self.capture(orientation.lower())
                    self.record("orientation", orientation=orientation, layout=self.value("exportSummary"))

                if self.label("playPause") == "Pause":
                    self.click("playPause")
                self.click("clip-" + self.args.clip)
                baseline_range = self.label("currentRallyRange")
                baseline_keep = self.value("keepRally")
                for action, expected in (("removeRally", "removed"), ("keepRally", "included")):
                    self.click(action)
                    self.wait(lambda: self.value("keepRally") == expected, f"{action} did not change effective state")
                    self.capture(action)
                    self.click("undoEdit")
                    self.wait(lambda: self.value("keepRally") == baseline_keep, "Undo did not restore original inclusion")
                self.record("keep-remove-undo", restoredState=baseline_keep)

                self.click("nudgeStartEarlier")
                self.wait(lambda: self.label("currentRallyRange") != baseline_range, "Core nudge did not change timestamp")
                nudged = self.label("currentRallyRange")
                self.click("undoEdit")
                self.wait(lambda: self.label("currentRallyRange") == baseline_range, "Core reset did not restore exact timestamp")
                self.record("core-nudge-undo", before=baseline_range, changed=nudged)

                original_rate = self.label("playbackRate")
                target_rate = "4×" if "2×" in original_rate else "2×"
                self.click("playbackRate")
                self.wait(lambda: any(node.get("visible") == "true" for node in self.nodes(target_rate)), "Speed menu did not open")
                (self.output / "speed-menu.xml").write_text(ET.tostring(self.source(), encoding="unicode"), encoding="utf-8")
                self.click(target_rate)
                self.wait(lambda: target_rate in self.label("playbackRate"), "Playback rate was not set")
                self.playback_element = self.element("playPause")
                clock_element = self.element("playheadTime")
                self.command(f"/element/{self.playback_element}/click", {})
                self.wait_attribute(self.playback_element, "label", lambda value: value == "Pause", "Video did not start")
                started = self.command(f"/element/{clock_element}/attribute/label")
                self.wait_attribute(clock_element, "label", lambda value: value and value != started, "Source clock did not advance")
                self.command(f"/element/{self.playback_element}/click", {})
                self.wait_attribute(self.playback_element, "label", lambda value: value == "Play", "Video did not pause")
                self.playback_element = None
                self.click("undoEdit")
                self.wait(lambda: self.label("playbackRate") == original_rate, "Undo did not restore playback rate")
                self.record("play-pause-rate", restoredRate=original_rate)

            if not self.nodes("beforePadding"):
                self.click("editorSettings")
            self.wait(lambda: bool(self.nodes("beforePadding")), "Settings did not open")
            for control, milliseconds in (("beforePadding", self.args.recover_before_ms), ("joinGap", self.args.recover_join_ms)):
                if milliseconds is not None:
                    recovered = milliseconds / 10_000
                    self.remember_slider(control, recovered)
                    self.restore_slider(control, recovered)
                    self.forget_slider(control)
                    self.record("recover-setting", control=control, milliseconds=milliseconds)
            settings_results = []
            for name in ("beforePadding", "afterPadding", "joinGap"):
                original = self.slider_fraction(name)
                self.settings_baseline[name] = original
                target = 0.4 if abs(original - 0.3) < 0.02 else 0.3
                self.remember_slider(name, original)
                drag_tested = not self.args.join_only or name == "joinGap"
                if drag_tested:
                    self.set_slider(name, target)
                    self.wait(lambda: abs(self.slider_fraction(name) - original) > 0.02, f"{name} did not change")
                changed = self.slider_fraction(name)
                self.restore_slider(name, original)
                self.forget_slider(name)
                settings_results.append({"control": name, "original": original, "changed": changed, "sliderDragTested": drag_tested})
            self.capture("settings")
            self.click("settingsDone")
            self.record("padding-and-join", values=settings_results, sliderDragScope="join-only" if self.args.join_only else "all")

            was_scoring = self.value("scoreTracking")
            if was_scoring == "0":
                self.click("scoreTracking")
            self.wait(lambda: bool(self.nodes("addNearServe")), "Score controls did not appear")
            original_markers = {node.get("name") for node in self.nodes() if node.get("name", "").startswith("serve-")}
            self.click("addNearServe")
            self.wait(lambda: len({node.get("name") for node in self.nodes() if node.get("name", "").startswith("serve-")} - original_markers) == 1,
                      "Manual serve marker did not appear")
            self.capture("score-marker")
            self.click("undoEdit")
            self.wait(lambda: {node.get("name") for node in self.nodes() if node.get("name", "").startswith("serve-")} == original_markers,
                      "Undo did not remove new serve marker")
            if was_scoring == "0":
                self.click("scoreTracking")
            self.record("score-marker-undo", originalScoreToggle=was_scoring)
            self.assert_settings_unchanged("score marker Undo")

            # Persist a harmless display/playback setting. This verifies a real
            # changed value after reopening, without leaving label edits behind.
            original_preview = self.value("finalCutPreview")
            self.click("finalCutPreview")
            changed_preview = self.value("finalCutPreview")
            assert changed_preview != original_preview
            self.click("exportTab")
            self.wait(lambda: bool(self.nodes("chapterRallyNumbers")), "Chapter controls did not appear")
            original_chapter = self.value("chapterRallyNumbers")
            self.pending_chapter = original_chapter
            (self.output / "pending-chapter-restoration.json").write_text(json.dumps(original_chapter), encoding="utf-8")
            self.click("chapterRallyNumbers")
            self.wait(lambda: self.value("chapterRallyNumbers") != original_chapter, "Chapter preference did not change")
            changed_chapter = self.value("chapterRallyNumbers")
            time.sleep(0.6)  # Root's documented 300 ms save debounce.
            self.click("backToProjects")
            time.sleep(0.6)
            if self.args.restart:
                self.command("/wda/apps/terminate", {"bundleId": self.args.bundle})
                self.wda.activate(self.args.bundle)
            self.tree = None
            self.choose_project()
            self.wait(lambda: self.value("finalCutPreview") == changed_preview, "Changed final-cut preference did not persist")
            self.assert_settings_unchanged("project restart" if self.args.restart else "project reopen")
            self.click("finalCutPreview")
            self.wait(lambda: self.value("finalCutPreview") == original_preview, "Final-cut preference did not restore")
            self.click("exportTab")
            self.wait(lambda: self.value("chapterRallyNumbers") == changed_chapter, "Chapter preference did not persist")
            self.record("project-persistence", processRestart=self.args.restart, project=self.project_button,
                        changedChapterRallyNumbers=changed_chapter)

            self.wait(lambda: bool(self.nodes("exportProject")), "Export workspace did not open")
            self.click("exportProject")
            self.wait(lambda: any(node.get("label") in ("Close", "Cancel", "Done") and node.get("type") == "XCUIElementTypeButton"
                                 and node.get("visible") == "true" for node in self.nodes()), "Project share sheet did not open", seconds=45)
            self.capture("project-export")
            close = next(node for node in self.nodes() if node.get("label") in ("Close", "Cancel", "Done")
                         and node.get("type") == "XCUIElementTypeButton" and node.get("visible") == "true")
            self.click(close.get("name") or close.get("label"))
            self.record("project-export", scope="Local feedback file prepared with changed chapter preference; share sheet opened, nothing sent",
                        expectedIncludeRallyNumber=changed_chapter == "1")
            self.click("chapterRallyNumbers")
            self.wait(lambda: self.value("chapterRallyNumbers") == original_chapter, "Chapter preference did not restore")
            self.pending_chapter = None
            (self.output / "pending-chapter-restoration.json").write_text("null", encoding="utf-8")
            self.record("chapter-preference-restored", original=original_chapter)
            self.click("reviewTab")
            self.assert_settings_unchanged("project export")
            self.capture("restored-editor")
        except Exception as error:
            self.results.append({"result": "failed", "error": str(error)})
            if self.playback_element is not None:
                try:
                    if self.command(f"/element/{self.playback_element}/attribute/label") == "Pause":
                        self.command(f"/element/{self.playback_element}/click", {})
                except Exception as pause_error:
                    self.results.append({"pauseError": str(pause_error)})
            try:
                self.capture("failure")
            except Exception as capture_error:
                self.results.append({"captureError": str(capture_error)})
            for name, original in list(self.pending_sliders.items()):
                try:
                    self.restore_slider(name, original)
                    self.forget_slider(name)
                    self.results.append({"restoredAfterFailure": name, "milliseconds": round(original * 10_000)})
                except Exception as restore_error:
                    self.results.append({"sliderRestoreError": str(restore_error), "control": name,
                                         "originalMilliseconds": round(original * 10_000)})
            if self.pending_chapter is not None:
                try:
                    if not self.nodes("exportTab"):
                        self.choose_project()
                    if not self.nodes("chapterRallyNumbers"):
                        self.click("exportTab")
                    if self.value("chapterRallyNumbers") != self.pending_chapter:
                        self.click("chapterRallyNumbers")
                    self.wait(lambda: self.value("chapterRallyNumbers") == self.pending_chapter, "Chapter restore failed")
                    self.pending_chapter = None
                    (self.output / "pending-chapter-restoration.json").write_text("null", encoding="utf-8")
                except Exception as restore_error:
                    self.results.append({"chapterRestoreError": str(restore_error), "originalValue": self.pending_chapter})
            raise
        finally:
            failed = sys.exc_info()[0] is not None
            try:
                self.park()
                self.results.append({"teardown": "parking foreground; idle timer disabled"})
            except Exception as parking_error:
                self.results.append({"parkingError": str(parking_error)})
                if not failed:
                    raise
            finally:
                try:
                    if self.original_settings is not None:
                        self.command("/appium/settings", {"settings": self.original_settings})
                except Exception as settings_error:
                    self.results.append({"settingsRestoreError": str(settings_error)})
                    if not failed:
                        raise
                finally:
                    (self.output / "results.json").write_text(json.dumps(self.results, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lab",
        type=Path,
        default=Path(os.environ.get("VOLLEYCUT_IOS_LAB_ROOT", "artifacts/ios-lab")),
    )
    parser.add_argument("--url", default="http://127.0.0.1:18100")
    parser.add_argument("--mjpeg", default="http://127.0.0.1:19100/")
    parser.add_argument("--bundle", default="com.volleysplice.VolleySplice")
    parser.add_argument("--project", help="Exact observed project-button ID, required when several projects are listed")
    parser.add_argument("--clip", default="R001", help="Retainable fixture clip with room to nudge its core start")
    parser.add_argument("--restart", action="store_true", help="Also terminate/relaunch VolleySplice during persistence check")
    parser.add_argument("--start-at-settings", action="store_true", help="Resume with settings, score, persistence and export; skip proven orientation/edit/playback checks")
    parser.add_argument("--recover-before-ms", type=int, help="Explicitly restore this known pre-failure Before padding before testing (0..10000 in 100 ms steps)")
    parser.add_argument("--recover-join-ms", type=int, help="Explicitly restore this known pre-failure join threshold before testing (0..10000 in 100 ms steps)")
    parser.add_argument("--join-only", action="store_true", help="Drag Join once then restore exactly to exercise Undo regression; only numeric restoration for Before/After")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for flag, value in (("--recover-before-ms", args.recover_before_ms), ("--recover-join-ms", args.recover_join_ms)):
        if value is not None and (not 0 <= value <= 10_000 or value % 100):
            parser.error(f"{flag} must be between 0 and 10000 in 100 ms steps")
    EditorSmoke(args).run()


if __name__ == "__main__":
    main()

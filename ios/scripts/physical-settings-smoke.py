"""Physical iPad display/settings persistence checks; no project edits or exports.

Uses the installed app and existing Windows lab WDA. Example:
  python physical-settings-smoke.py --project project-OBSERVED.volleyproject.json
Add --layout --udid DEVICE_UDID to test player/both sidebars persistence using
read-only AFC preference evidence (requires pymobiledevice3 in this Python).
Without --project, setup scale checks still run; an already-open editor also
gets initial marker/awarded-count checks, but is not guessed after relaunch.

Drags the real scale slider to 60/125, verifies sharing across setup/editor and
process relaunch, then restores the original effective scale and override state.
Optional layout checks restore original effective saved sizes through UI drags.
Missing preference keys may become explicit defaults; no app-container writes
are made. A pending-restore.json journal survives interruption: use --recover
PATH to restore a previous run instead of performing more checks. Every run
attempts idle-disabled parking in finally, even after failures. Never run beside
another device operator. Screenshots require human inspection; this is not an
inference-accuracy or performance benchmark. Builds/install are out of scope.
"""
import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import plistlib
import re
import sys
import time

spec = importlib.util.spec_from_file_location("physical_parity", Path(__file__).with_name("physical-parity-smoke.py"))
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


class SettingsSmoke(parity.PhysicalParitySmoke):
    LAYOUT = {
        "resizeVideo": ("desktop_editor_layout.desktopReviewPlayerHeight", 320.0, "height"),
        "resizeLeftSidebar": ("desktop_editor_layout.leftSidebarWidth", 292.0, "x"),
        "resizeRightSidebar": ("desktop_editor_layout.rightSidebarWidth", 292.0, "x"),
    }

    def __init__(self, args):
        super().__init__(args)
        self.pending = json.loads(args.recover.read_text(encoding="utf-8")) if args.recover else {}
        if self.pending.get("project"):
            self.project_button = self.pending["project"]
        self.settings_values = None
        if args.recover:
            self.journal()

    def journal(self):
        (self.output / "pending-restore.json").write_text(json.dumps(self.pending, indent=2), encoding="utf-8")

    def scale_state(self):
        percent = int(re.search(r"\d+", self.label("uiScalePercent")).group())
        return {"percent": percent, "override": self.node("uiScaleDefault").get("enabled") == "true"}

    def settings_open(self):
        self.source(refresh=True)
        if self.nodes("uiScalePercent") or self.visible("settingsDone"):
            self.reveal_scale()
            return
        for name in ("editorSettings", "setupSettings"):
            if self.nodes(name):
                self.click(name)
                self.wait(lambda: bool(self.visible("uiScalePercent")), "UI size setting did not appear")
                return
        raise AssertionError("Expected setup/editor Settings; another presentation is open")

    def reveal_scale(self):
        self.source(refresh=True)
        if self.visible("uiScalePercent"):
            return
        # A long downward gesture can dismiss an iPad sheet when its Form
        # reaches the top. Reopen Settings to return to its first section.
        self.settings_close()
        self.settings_open()

    def settings_close(self):
        self.source(refresh=True)
        if self.visible("settingsDone"):
            self.click("settingsDone")
        elif self.nodes("uiScalePercent"):
            self.click_label("Done")
        else:
            return
        self.wait(lambda: not self.visible("uiScalePercent") and not self.visible("settingsDone"), "Settings did not close")

    def drag(self, start, end):
        self.command("/actions", {"actions": [{"type": "pointer", "id": "settings-drag",
            "parameters": {"pointerType": "touch"}, "actions": [
                {"type": "pointerMove", "duration": 0, "origin": "viewport", "x": start[0], "y": start[1]},
                {"type": "pointerDown", "button": 0}, {"type": "pause", "duration": 100},
                {"type": "pointerMove", "duration": 600, "origin": "viewport", "x": end[0], "y": end[1]},
                {"type": "pointerUp", "button": 0}]}]})

    def scale_drag(self, target, restoring=False):
        # Restore can correct bounded rounding misses; acceptance must pass one
        # actual drag so repeated attempts cannot conceal touch cancellation.
        for attempt in range(4 if restoring else 1):
            self.source(refresh=True)
            current = self.scale_state()["percent"]
            if restoring and current == target and self.scale_state()["override"]:
                return
            frame = self.frame(self.node("uiScale"))
            app = self.frame(next(node for node in self.nodes() if node.get("type") == "XCUIElementTypeApplication"))
            inset = min(frame["height"] / 2, frame["width"] / 4)
            span = frame["width"] - 2 * inset
            start_x = frame["x"] + inset + (current - 60) / 65 * span
            end_x = frame["x"] + inset + (target - 60) / 65 * span
            if target in (60, 125):
                end_x = max(app["x"] + 8, frame["x"] - 18) if target == 60 else min(app["x"] + app["width"] - 8, frame["x"] + frame["width"] + 18)
            y = frame["y"] + frame["height"] / 2
            self.drag((start_x, y), (end_x, y))
            self.source(refresh=True)
            actual = self.scale_state()["percent"]
            print(json.dumps({"scaleDrag": {"from": current, "target": target, "observed": actual, "attempt": attempt + 1}}), flush=True)
            if actual == target:
                return
        raise AssertionError(f"Scale drag expected {target}%, observed {actual}%")

    def restart(self):
        self.settings_close()
        self.park()  # Flush through the real inactive/background lifecycle first.
        self.command("/wda/apps/terminate", {"bundleId": self.args.bundle})
        self.wda.activate(self.args.bundle)
        self.tree = None
        self.wait(lambda: bool(self.nodes("setupSettings")), "Relaunch did not return to setup", seconds=30)

    def counters(self, stage):
        observed = {}
        for name in ("markerCount", "awardedCount"):
            # Older SwiftUI builds inherited the pointTimeline identifier on
            # this caption. Check its actual visible wording in that build too.
            if name == "awardedCount" and not self.nodes(name):
                legacy = [node for node in self.nodes("pointTimeline")
                          if node.get("visible") == "true" and re.fullmatch(r"\d+ awarded", node.get("label", ""))]
                if len(legacy) == 1:
                    observed[name] = legacy[0].get("label")
                    continue
            self.node(name, scroll=True)
            self.source(refresh=True)
            node = self.node(name)
            observed[name] = node.get("label") or node.get("value")
            if not re.search(r"\d+", observed[name] or ""):
                raise AssertionError(f"Missing numeric {name}: {observed[name]}")
        self.capture(stage)
        self.record("ipad-counts-visible", stage=stage, counters=observed)
        return observed

    def inspect_settings(self, stage):
        """Read every setting without changing padding, policy, or thresholds."""
        if not self.visible("settingsDone"):
            return
        names = ("beforePadding", "beforePaddingSeconds", "afterPadding", "afterPaddingSeconds",
                 "joinGap", "joinGapSeconds", "reviewThreshold", "suppressionPolicy", "resetRanges")
        values, headings = {}, set()
        for page in range(9):
            self.source(refresh=True)
            for node in self.nodes():
                if node.get("visible") == "true":
                    headings.add((node.get("label") or "").casefold())
            for name in names:
                if self.visible(name):
                    node = self.node(name)
                    if node.get("enabled") != "true":
                        raise AssertionError(f"Disabled editor setting: {name}")
                    values[name] = {"value": node.get("value"), "label": node.get("label"), "enabled": True}
            self.absent("undoEdit", "ignoreReason", "suppressionBehavior", prefixes=("suppression-",))
            self.capture(f"{stage}-page-{page}")
            if all(name in values for name in names):
                break
            self.scroll_form()
        if set(values) != set(names):
            raise AssertionError(f"Missing editor settings after bounded scroll: {sorted(set(names) - set(values))}")
        missing = {"padding and gaps", "review", "automatic cleanup", "restore"} - headings
        if missing:
            raise AssertionError(f"Missing settings section headings: {sorted(missing)}")
        if self.settings_values is None:
            self.settings_values = values
        elif values != self.settings_values:
            raise AssertionError(f"Display-only changes altered setting values: baseline={self.settings_values}, current={values}")
        self.record("editor-settings-values-and-enabled", stage=stage, values=values)
        self.reveal_scale()

    def orientation_evidence(self, stage):
        counts = None
        for orientation in ("LANDSCAPE", "PORTRAIT"):
            rejection = None
            try:
                self.command("/orientation", {"orientation": orientation})
            except RuntimeError as error:
                # WDA waits for the app interface to adopt the requested
                # orientation. A landscape-only app can reject portrait.
                if orientation != "PORTRAIT" or "Unable To Rotate Device" not in str(error):
                    raise
                rejection = str(error)
            def landscape_editor():
                app = next(node for node in self.nodes() if node.get("type") == "XCUIElementTypeApplication")
                frame = self.frame(app)
                return frame["width"] > frame["height"] and bool(self.nodes("playPause"))
            self.wait(landscape_editor, "iPad editor must remain landscape, including an upright rotation request")
            suffix = "landscape" if orientation == "LANDSCAPE" else "upright-keeps-landscape"
            self.record("ipad-landscape-only", requested=orientation, reported=self.command("/orientation"), rejection=rejection)
            counts = self.counters(stage + "-" + suffix)
            self.settings_open()
            self.inspect_settings(stage + "-settings-" + suffix)
            self.settings_close()
        self.command("/orientation", {"orientation": "LANDSCAPE"})
        return counts

    def preferences(self):
        async def read():
            from pymobiledevice3.lockdown import create_using_usbmux
            from pymobiledevice3.services.house_arrest import HouseArrestService
            async with await create_using_usbmux(serial=self.args.udid) as lockdown:
                async with await HouseArrestService.create(lockdown=lockdown, bundle_id=self.args.bundle) as afc:
                    return plistlib.loads(await afc.get_file_contents(f"/Library/Preferences/{self.args.bundle}.plist"))
        return asyncio.run(read())

    def layout_position(self, name):
        self.source(refresh=True)
        self.node(name, scroll=True)
        if name == "resizeVideo":
            return self.frame(self.node("editorVideo"))["height"]
        frame = self.frame(self.node(name))
        return frame["x"] + frame["width"] / 2

    def resize(self, name, delta):
        frame = self.frame(self.node(name, scroll=True))
        start = (frame["x"] + frame["width"] / 2, frame["y"] + frame["height"] / 2)
        if name == "resizeRightSidebar":
            delta = -delta  # Moving the right divider left expands its sidebar.
        end = (start[0] + (delta if name != "resizeVideo" else 0), start[1] + (delta if name == "resizeVideo" else 0))
        self.drag(start, end)

    def layout_checks(self):
        self.command("/orientation", {"orientation": "LANDSCAPE"})
        self.wait(lambda: self.command("/orientation") == "LANDSCAPE", "Landscape not observed")
        self.settings_open()
        self.click("uiScaleDefault") if self.scale_state()["override"] else None
        self.settings_close()
        prefs = self.preferences()
        original = {name: float(prefs.get(key, default)) for name, (key, default, _) in self.LAYOUT.items()}
        # Do not replace an oversized stored preference with a clamped viewport.
        video = self.layout_position("resizeVideo")
        if (abs(video - original["resizeVideo"]) > 3
                or not 180 <= original["resizeLeftSidebar"] <= 480
                or not 220 <= original["resizeRightSidebar"] <= 520):
            raise AssertionError("Layout baseline is clamped/out of safe test range; rerun without --layout")
        self.pending["layout"] = original
        self.journal()
        targets = {}
        for name, value in original.items():
            delta = -24 if name == "resizeVideo" and value > 1100 else 24
            before = self.layout_position(name)
            self.resize(name, delta)
            movement = -delta if name == "resizeRightSidebar" else delta
            self.wait(lambda: abs(self.layout_position(name) - before - movement) <= 3, f"{name} failed to follow 24pt drag")
            targets[name] = value + delta
        self.restart()
        self.choose_current_project()
        prefs = self.preferences()
        for name, target in targets.items():
            key = self.LAYOUT[name][0]
            if abs(float(prefs.get(key, -1)) - target) > 3:
                raise AssertionError(f"Persisted {key} differs: expected {target}, got {prefs.get(key)}")
        self.capture("layout-after-relaunch")
        self.record("layout-persistence", logicalValues=targets, evidence="AFC read-only preferences after relaunch")

    def restore_layout(self):
        if self.pending.get("layout"):
            self.settings_close()
            self.command("/orientation", {"orientation": "LANDSCAPE"})
            self.choose_current_project()
            self.settings_open()
            if self.scale_state()["override"]:
                self.click("uiScaleDefault")
            self.settings_close()
            for name, original in list(self.pending["layout"].items()):
                key = self.LAYOUT[name][0]
                for _ in range(3):
                    current = float(self.preferences().get(key, self.LAYOUT[name][1]))
                    if abs(current - original) <= 1:
                        break
                    self.resize(name, original - current)
                    self.park(); self.wda.activate(self.args.bundle); self.tree = None
                if abs(float(self.preferences().get(key, self.LAYOUT[name][1])) - original) > 1:
                    raise AssertionError(f"Could not restore saved layout {key}")
                del self.pending["layout"][name]
                self.journal()

    def restore(self):
        if not self.pending:
            return
        self.wda.activate(self.args.bundle)
        self.tree = None
        layout_error = None
        try:
            self.restore_layout()
        except Exception as error:
            # Scale restoration remains independent even if a layout drag or
            # AFC read fails. Preserve the failed layout entries in the journal.
            layout_error = error
        original = self.pending.get("scale")
        if original:
            self.settings_open()
            if original["override"]:
                self.scale_drag(original["percent"], restoring=True)
            elif self.scale_state()["override"]:
                self.click("uiScaleDefault")
            self.wait(lambda: self.scale_state() == original, "Original scale/default mode did not restore")
            self.settings_close()
            self.restart()
            self.settings_open()
            if self.scale_state() != original:
                raise AssertionError("Restored scale did not persist across relaunch")
            self.settings_close()
            del self.pending["scale"]
            self.journal()
        if layout_error:
            raise RuntimeError(f"Layout restore failed (scale restore attempted independently): {layout_error}")
        self.record("original-preferences-restored")

    def run(self):
        failed = False
        try:
            self.wda.connect()
            self.park()
            observed_orientation = self.command("/orientation")
            self.original_orientation = self.pending.get("orientation") or observed_orientation
            settings = self.command("/appium/settings")
            self.original_settings = {key: settings[key] for key in ("waitForIdleTimeout", "animationCoolOffTimeout") if key in settings}
            self.command("/appium/settings", {"settings": {"waitForIdleTimeout": 0, "animationCoolOffTimeout": 0}})
            if self.args.orientation_only and not self.args.recover:
                self.command("/orientation", {"orientation": "PORTRAIT"})
                self.record("parking-upright-before-launch", reported=self.command("/orientation"))
                self.command("/wda/apps/terminate", {"bundleId": self.args.bundle})
            self.wda.activate(self.args.bundle)
            self.source(refresh=True)
            if self.args.recover:
                return
            if self.project_button:
                self.choose_current_project()
            editor = bool(self.nodes("playPause"))
            if self.args.orientation_only:
                if not editor:
                    raise AssertionError("Orientation acceptance requires an existing editor")
                frame = self.frame(next(node for node in self.nodes() if node.get("type") == "XCUIElementTypeApplication"))
                if frame["width"] <= frame["height"]:
                    raise AssertionError("Launching on an upright iPad did not enter landscape")
                self.record("ipad-upright-launch-is-landscape", frame=frame)
                self.orientation_evidence("ipad-orientation")
                return
            if editor:
                self.node("playPause", scroll=True)
                if self.label("playPause") == "Pause":
                    self.click("playPause")
            self.settings_open()
            self.pending = {"scale": self.scale_state(), "project": self.project_button, "orientation": self.original_orientation}
            self.journal()
            self.capture("original-settings")
            self.inspect_settings("original-settings")
            self.scale_drag(60)
            self.capture("settings-60")
            self.settings_close()
            baseline_counts = self.orientation_evidence("editor-60") if editor else None
            self.restart()
            self.settings_open()
            if self.scale_state() != {"percent": 60, "override": True}:
                raise AssertionError("60% did not persist into setup after restart")
            self.record("scale-60-persistence-setup")
            self.scale_drag(125)
            self.capture("settings-125")
            self.restart()
            self.settings_open()
            if self.scale_state() != {"percent": 125, "override": True}:
                raise AssertionError("125% did not persist into setup after restart")
            self.settings_close()
            if self.project_button:
                self.choose_current_project()
                self.settings_open()
                if self.scale_state()["percent"] != 125:
                    raise AssertionError("Setup scale did not apply in editor")
                self.settings_close()
                counts = self.orientation_evidence("editor-125")
                if baseline_counts and counts["markerCount"] != baseline_counts["markerCount"]:
                    raise AssertionError("Marker count changed without project edits")
                # Awarded points follow the playhead, which can reset on reopen;
                # require visibility, not equality across source-time changes.
            else:
                self.results.append({"test": "editor-reopen", "result": "skipped", "reason": "No exact --project identifier; no project guessed"})
            self.record("scale-125-persistence-and-sharing")
            if self.args.layout:
                self.layout_checks()
        except BaseException as error:
            failed = True
            self.results.append({"result": "failed", "error": str(error)})
            try:
                if self.wda.session:
                    self.capture("failure")
            except Exception as capture_error:
                self.results.append({"captureError": str(capture_error)})
            raise
        finally:
            teardown = []
            try:
                if self.wda.session:
                    self.restore()
            except Exception as error:
                teardown.append({"restoreError": str(error), "recoveryJournal": str(self.output / "pending-restore.json")})
            try:
                if self.original_orientation:
                    # Restore physical orientation in Parking, which supports
                    # portrait even though VolleySplice now rejects it.
                    self.park()
                    self.command("/orientation", {"orientation": self.original_orientation})
            except Exception as error:
                teardown.append({"orientationRestoreError": str(error)})
            try:
                if not self.wda.session:
                    self.wda.connect()
                self.park()
                self.results.append({"teardown": "parking foreground; idle timer disabled"})
            except Exception as error:
                teardown.append({"parkingError": str(error)})
            try:
                if self.original_settings is not None:
                    self.command("/appium/settings", {"settings": self.original_settings})
            except Exception as error:
                teardown.append({"wdaSettingsRestoreError": str(error)})
            self.results.extend(teardown)
            (self.output / "results.json").write_text(json.dumps(self.results, indent=2), encoding="utf-8")
            if teardown and not failed:
                raise RuntimeError(f"Teardown failed: {teardown}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--lab",
        type=Path,
        default=Path(os.environ.get("VOLLEYCUT_IOS_LAB_ROOT", "artifacts/ios-lab")),
    )
    parser.add_argument("--url", default="http://127.0.0.1:18100")
    parser.add_argument("--mjpeg", default="http://127.0.0.1:19100/")
    parser.add_argument("--bundle", default="com.volleysplice.VolleySplice")
    parser.add_argument("--project", help="Exact observed project-<filename> accessibility identifier")
    parser.add_argument("--layout", action="store_true", help="Also resize player/both sidebars; requires --project and --udid")
    parser.add_argument("--orientation-only", action="store_true", help="Verify landscape-only iPad and Settings without changing display preferences")
    parser.add_argument("--udid", help="Paired iPad UDID for read-only AFC layout preference checks")
    parser.add_argument("--recover", type=Path, help="Restore pending journal from an interrupted run, then park")
    parser.add_argument("--output", type=Path, help="New or empty evidence directory")
    args = parser.parse_args()
    if args.project and not args.project.startswith("project-"):
        parser.error("--project must be an exact observed identifier starting with project-")
    if args.layout and (not args.project or not args.udid):
        parser.error("--layout requires --project and --udid")
    if args.recover and json.loads(args.recover.read_text(encoding="utf-8")).get("layout") and not args.udid:
        parser.error("Recovering layout preferences requires --udid")
    args.output = args.output or args.lab / "artifacts/volleysplice" / time.strftime("physical-settings-%Y%m%d-%H%M%S")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be new or empty")
    SettingsSmoke(args).run()


if __name__ == "__main__":
    main()

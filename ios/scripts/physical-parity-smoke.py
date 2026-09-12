"""Current physical iPad parity observations through the existing Windows WDA.

Default checks navigate an already open editor, rotate, inspect settings and
review navigation, and capture XML/JPEG evidence. They do not change project
decisions, scores, export options, or start analysis/export. Screenshots still
need human comparison against Android; WDA timings are not app benchmarks.

--project accepts an exact observed project-<filename> button identifier. No
project is guessed. --disposable-project selects that exact project and permits
the optional --cleanup-action keep/remove test, which PERSISTS its decision.
--cleanup-resolved-count must explicitly state the expected fragment count for
that logical cleanup group; a single decision can resolve several fragments.
It does not undo or restore that decision. Use only a separately created copy.
The harness never creates a synthetic fixture or assumes canonical predictions.

Every execution resolves the live WDA session and ends in verified parking,
including failed preflight. No install/build/VM/service lifecycle is managed.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time


spec = importlib.util.spec_from_file_location(
    "editor_smoke", Path(__file__).with_name("editor-smoke.py"))
editor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(editor_module)


class PhysicalParitySmoke(editor_module.EditorSmoke):
    def __init__(self, args):
        super().__init__(args)
        self.original_orientation = None

    def visible(self, name):
        return [node for node in self.nodes(name) if node.get("visible") == "true"]

    def absent(self, *names, prefixes=()):
        found = [node.get("name", "") for node in self.nodes()
                 if node.get("name") in names
                 or any(node.get("name", "").startswith(prefix) for prefix in prefixes)]
        if found:
            raise AssertionError(f"Unexpected legacy controls: {found}")

    def click_label(self, label):
        # SwiftUI can inherit a container identifier on native menu/disclosure
        # children. Resolve their fresh visible button label, not that parent's ID.
        self.source(refresh=True)
        observed = [node for node in self.nodes() if node.get("label") == label
                    and node.get("type") == "XCUIElementTypeButton" and node.get("visible") == "true"]
        if len(observed) != 1 or observed[0].get("enabled") != "true":
            raise AssertionError(f"Expected one enabled visible button labelled {label!r}; found {len(observed)}")
        found = self.command("/element", {"using": "predicate string", "value":
            f"label == {json.dumps(label)} AND type == 'XCUIElementTypeButton' AND visible == true"})
        element = found.get("ELEMENT") or found["element-6066-11e4-a52e-4f735466cecf"]
        self.command(f"/element/{element}/click", {})

    @staticmethod
    def frame(node):
        return {key: float(node.get(key, 0)) for key in ("x", "y", "width", "height")}

    def choose_current_project(self):
        self.source(refresh=True)
        if self.visible("queueDone"):
            self.click("queueDone")
        if self.visible("settingsDone"):
            self.click("settingsDone")
        if self.visible("tutorialSkip"):
            raise AssertionError("Tutorial is open; inspect/finish it separately before this harness")
        if self.project_button:
            if not self.visible("startNewVideo"):
                self.click("projectSelector")
            self.wait(lambda: bool(self.visible("startNewVideo")), "Project menu did not open")
            if any(node.get("label") == "Open saved projects" and node.get("visible") == "true"
                   for node in self.nodes()):
                raise AssertionError("Project menu must directly list projects")
            self.reveal_saved_project()
            self.click(self.project_button)
        self.wait(lambda: bool(self.nodes("playPause")),
                  "Open an existing editor or pass its exact observed --project identifier", seconds=30)
        if self.visible("reviewTab"):
            self.click("reviewTab")
        self.record("editor-open", project=self.project_button or "already open; identity not inferred")

    def reveal_saved_project(self):
        # Native menus may expose rows only near their current viewport. Waiting
        # for a particular row before scrolling can never find an older project.
        # Start at the observed position, search upward to the first page, then
        # downward. Do not derive a tap or scroll offset from its timestamp/name.
        deadline = time.monotonic() + 120
        observed_ids = set()
        for direction in ("up", "down"):
            previous = None
            stationary = 0
            for _ in range(16):
                self.source(refresh=True)
                if self.visible(self.project_button):
                    return
                rows = [node for node in self.nodes() if node.get("visible") == "true"
                        and node.get("type") == "XCUIElementTypeButton"
                        and node.get("name", "").startswith(("project-", "recording-"))]
                observed_ids.update(node.get("name") for node in rows if node.get("name", "").startswith("project-"))
                signature = tuple((node.get("name"), round(float(node.get("y", 0)))) for node in rows)
                stationary = stationary + 1 if signature == previous else 0
                if stationary >= 2 or time.monotonic() >= deadline:
                    break
                previous = signature
                self.scroll_form(direction=direction)
        self.source(refresh=True)
        if not self.visible(self.project_button):
            raise AssertionError(f"Requested project not visible after bounded project-menu search: {self.project_button}; "
                                 f"observed project IDs: {sorted(observed_ids)}")

    def transport(self, orientation):
        self.command("/orientation", {"orientation": orientation})
        self.wait(lambda: self.command("/orientation") == orientation and bool(self.nodes("playPause")),
                  "Requested orientation and editor not observed")
        self.node("playPause", scroll=True)
        self.source(refresh=True)
        names = ("playPause", "sourceSeek", "playheadTime", "playbackRate")
        frames = {name: self.frame(self.node(name)) for name in names}
        centers = [frame["y"] + frame["height"] / 2 for frame in frames.values()]
        if max(centers) - min(centers) > 8:
            raise AssertionError(f"Transport controls are not in one row: {frames}")
        app = next(node for node in self.nodes() if node.get("type") == "XCUIElementTypeApplication")
        bounds = self.frame(app)
        for name, frame in frames.items():
            if not (bounds["x"] <= frame["x"] and bounds["y"] <= frame["y"]
                    and frame["x"] + frame["width"] <= bounds["x"] + bounds["width"] + 1
                    and frame["y"] + frame["height"] <= bounds["y"] + bounds["height"] + 1):
                raise AssertionError(f"Transport control is clipped: {name} {frame}, app={bounds}")
        self.absent("undoEdit", "ignoreReason", "prepareScores")
        if not self.nodes("timelineLegend"):
            raise AssertionError("Timeline legend is missing")
        counters = {}
        for name, label in (("reviewCleanup", "Review cleanup"), ("reviewClips", "Review clips"),
                            ("reviewServes", "Review serves")):
            node = self.node(name, require_visible=False)
            value = node.get("label", "")
            if not re.fullmatch(re.escape(label) + r" \d+", value):
                raise AssertionError(f"Review counter has unexpected copy: {name}={value!r}")
            counters[name] = value
        self.capture("editor-" + orientation.lower())
        self.record("transport-and-controls", orientation=orientation, appFrame=bounds,
                    layout=self.value("exportSummary"), frames=frames, counters=counters,
                    visualAcceptance="pending inspection of captured JPEG against Android")

    def scroll_form(self, direction="down"):
        # Settings Form and saved-project List both virtualize their rows.
        # Scroll only inside the currently observed native sheet viewport.
        tree = self.source(refresh=True)
        containers = [node for node in tree.iter() if node.get("visible") == "true"
                      and node.get("type") in ("XCUIElementTypeCollectionView", "XCUIElementTypeTable")]
        if not containers:
            containers = [node for node in tree.iter() if node.get("visible") == "true"
                          and node.get("type") == "XCUIElementTypeScrollView"]
        if not containers:
            raise AssertionError("No visible sheet scroll container")
        frame = self.frame(min(containers, key=lambda node: self.frame(node)["width"] * self.frame(node)["height"]))
        app = self.frame(next(node for node in tree.iter() if node.get("type") == "XCUIElementTypeApplication"))
        left, right = max(frame["x"], app["x"]), min(frame["x"] + frame["width"], app["x"] + app["width"])
        top, bottom = max(frame["y"], app["y"]) + 40, min(frame["y"] + frame["height"], app["y"] + app["height"]) - 60
        if right - left < 40 or bottom - top < 80:
            raise AssertionError("Sheet scroll viewport has no safe interior")
        x = right - 16
        start, end = (bottom, top) if direction == "down" else (top, bottom)
        self.command("/actions", {"actions": [{"type": "pointer", "id": "sheet-scroll",
            "parameters": {"pointerType": "touch"}, "actions": [
                {"type": "pointerMove", "duration": 0, "origin": "viewport", "x": x, "y": start},
                {"type": "pointerDown", "button": 0},
                {"type": "pointerMove", "duration": 400, "origin": "viewport", "x": x, "y": end},
                {"type": "pointerUp", "button": 0}]}]})

    def inspect_settings(self):
        self.click("editorSettings")
        self.wait(lambda: bool(self.visible("settingsDone")), "Settings did not open")
        try:
            for _ in range(6):
                self.absent("undoEdit", "ignoreReason", "suppressionBehavior", prefixes=("suppression-",))
                if self.visible("suppressionPolicy"):
                    break
                self.scroll_form()
            self.node("suppressionPolicy")
            self.capture("settings-policy")
            self.record("settings-policy-only", modified=False)
        finally:
            self.source(refresh=True)
            if self.visible("settingsDone"):
                self.click("settingsDone")
                # With WDA idle waits disabled, a dismissed native sheet can
                # leave its underlying header temporarily non-hittable.
                self.wait(lambda: bool(self.visible("reviewCleanup")), "Editor header did not return after settings")

    def review_navigation(self):
        cleanup = self.node("reviewCleanup", require_visible=False)
        if cleanup.get("enabled") == "true":
            before = cleanup.get("label")
            self.click("reviewCleanup")
            self.wait(lambda: bool(self.nodes("cleanupSelection")), "Cleanup did not select a suggestion")
            selected = self.value("cleanupSelection")
            self.node("keepRally", scroll=True)
            self.capture("cleanup-selected")
            if self.args.cleanup_action:
                action = "keepRally" if self.args.cleanup_action == "keep" else "removeRally"
                expected = "included" if self.args.cleanup_action == "keep" else "removed"
                count = int(re.search(r"\d+$", before).group())
                resolved_count = self.args.cleanup_resolved_count
                if resolved_count > count:
                    raise AssertionError(f"Expected resolved fragment count {resolved_count} exceeds pending count {count}")
                remaining_count = count - resolved_count
                # Write intent before the persistent decision so interrupted runs
                # leave an explicit record of which disposable project changed.
                intent = {"project": self.project_button, "suggestion": selected, "action": action,
                          "pendingBefore": count, "expectedResolvedFragments": resolved_count,
                          "expectedPendingAfter": remaining_count}
                (self.output / "persistent-cleanup-intent.json").write_text(json.dumps(intent, indent=2), encoding="utf-8")
                self.click(action)
                self.wait(lambda: self.value("keepRally") == expected
                          and self.node("reviewCleanup", require_visible=False).get("label") == f"Review cleanup {remaining_count}",
                          f"Cleanup decision did not resolve the expected {resolved_count} logical-group fragments")
                self.capture("cleanup-" + self.args.cleanup_action)
                self.record("cleanup-decision", **intent, restored=False)
            else:
                if self.node("reviewCleanup", require_visible=False).get("label") != before:
                    raise AssertionError("Navigation alone changed unresolved cleanup count")
                self.record("cleanup-navigation", selected=selected, decisionChanged=False)
        elif self.args.cleanup_action:
            raise AssertionError("Disposable project has no unresolved cleanup to test")
        else:
            self.results.append({"test": "cleanup-navigation", "result": "skipped", "reason": "No unresolved cleanup"})

        serves = self.node("reviewServes", require_visible=False)
        if serves.get("enabled") == "true":
            self.click("reviewServes")
            self.wait(lambda: bool(self.visible("selectedServeSide")), "Serve review did not reveal selected-side controls")
            # Do not scroll the marker list to manufacture successful auto-reveal.
            markers = [node for node in self.nodes() if node.get("name", "").startswith("serve-")
                       and node.get("visible") == "true"]
            if not markers:
                raise AssertionError("Serve review left every marker row offscreen")
            self.capture("serve-review")
            self.record("serve-review-navigation", visibleMarkerIds=[node.get("name") for node in markers],
                        selectionAndGlyphVisualAcceptance="pending JPEG/XML inspection", decisionChanged=False)
        else:
            self.results.append({"test": "serve-review-navigation", "result": "skipped", "reason": "No unresolved enabled serves"})

    def run(self):
        try:
            self.wda.connect()
            (self.output / "preflight.xml").write_text(self.command("/source"), encoding="utf-8")
            self.park()
            self.original_orientation = self.command("/orientation")
            settings = self.command("/appium/settings")
            self.original_settings = {key: settings[key] for key in ("waitForIdleTimeout", "animationCoolOffTimeout") if key in settings}
            self.command("/appium/settings", {"settings": {"waitForIdleTimeout": 0, "animationCoolOffTimeout": 0}})
            self.wda.activate(self.args.bundle)
            self.tree = None
            self.choose_current_project()
            self.node("playPause", scroll=True)
            if self.label("playPause") == "Pause":
                self.click("playPause")
                self.record("paused-existing-playback", scope="Playback stays paused after the run")
            for orientation in self.args.orientations:
                self.transport(orientation)
            self.inspect_settings()
            self.review_navigation()
        except Exception as error:
            self.results.append({"result": "failed", "error": str(error)})
            try:
                if self.wda.session:
                    self.capture("failure")
            except Exception as capture_error:
                self.results.append({"captureError": str(capture_error)})
            raise
        finally:
            failed = sys.exc_info()[0] is not None
            teardown_errors = []
            if self.original_orientation:
                try:
                    self.command("/orientation", {"orientation": self.original_orientation})
                except Exception as error:
                    teardown_errors.append({"orientationRestoreError": str(error)})
            try:
                self.park()
                self.results.append({"teardown": "parking foreground; idle timer disabled"})
            except Exception as error:
                teardown_errors.append({"parkingError": str(error)})
            if self.original_settings is not None:
                try:
                    self.command("/appium/settings", {"settings": self.original_settings})
                except Exception as error:
                    teardown_errors.append({"settingsRestoreError": str(error)})
            self.results.extend(teardown_errors)
            (self.output / "results.json").write_text(json.dumps(self.results, indent=2), encoding="utf-8")
            if teardown_errors and not failed:
                raise RuntimeError(f"Teardown failed: {teardown_errors}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lab",
        type=Path,
        default=Path(os.environ.get("VOLLEYCUT_IOS_LAB_ROOT", "artifacts/ios-lab")),
    )
    parser.add_argument("--url", default="http://127.0.0.1:18100")
    parser.add_argument("--mjpeg", default="http://127.0.0.1:19100/")
    parser.add_argument("--bundle", default="com.vafrederico.VolleySplice")
    projects = parser.add_mutually_exclusive_group()
    projects.add_argument("--project", help="Exact observed project-<filename> identifier; otherwise inspect already-open editor")
    projects.add_argument("--disposable-project", help="Exact observed disposable project identifier; authorizes persistent cleanup decision")
    parser.add_argument("--cleanup-action", choices=("keep", "remove"), help="Resolve one logical cleanup group on --disposable-project; not restored")
    parser.add_argument("--cleanup-resolved-count", type=int, help="Required with --cleanup-action: expected number of pending fragments resolved by its logical-group decision (>=1)")
    parser.add_argument("--orientations", nargs="+", choices=("PORTRAIT", "LANDSCAPE"), default=["PORTRAIT", "LANDSCAPE"])
    parser.add_argument("--output", type=Path, help="Fresh evidence directory; default timestamped lab artifact directory")
    args = parser.parse_args()
    if args.cleanup_action and not args.disposable_project:
        parser.error("--cleanup-action requires --disposable-project with an exact observed ID")
    if args.cleanup_action and args.cleanup_resolved_count is None:
        parser.error("--cleanup-action requires explicit --cleanup-resolved-count from the selected logical group")
    if args.cleanup_resolved_count is not None and args.cleanup_resolved_count < 1:
        parser.error("--cleanup-resolved-count must be at least 1")
    if args.cleanup_resolved_count is not None and not args.cleanup_action:
        parser.error("--cleanup-resolved-count requires --cleanup-action")
    args.project = args.disposable_project or args.project
    if args.project and not args.project.startswith("project-"):
        parser.error("Project must be the exact accessibility identifier starting with project-")
    args.output = args.output or args.lab / "artifacts/volleysplice" / time.strftime("physical-parity-%Y%m%d-%H%M%S")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be new or empty so earlier evidence is preserved")
    PhysicalParitySmoke(args).run()


if __name__ == "__main__":
    main()

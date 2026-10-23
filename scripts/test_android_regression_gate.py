"""Exercise report rejection and real pinned-Gradle false-success scenarios."""
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest


spec = importlib.util.spec_from_file_location(
    "android_regression_gate", Path(__file__).with_name("android-regression-gate.py")
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class JvmReportGateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.started_ns = time.time_ns() - 2_000_000_000
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.report = self.directory / "TEST-example.xml"

    def write_report(self, *, tests=2, skipped=0, failures=0, errors=0, cases=None, timestamp=None):
        self.report.write_text(
            f'<testsuite tests="{tests}" skipped="{skipped}" failures="{failures}" '
            f'errors="{errors}" timestamp="{timestamp or self.timestamp}">'
            + (cases if cases is not None else '<testcase name="first"/><testcase name="second"/>')
            + '</testsuite>', encoding="utf-8"
        )

    def test_positive_execution_count(self):
        self.write_report()
        self.assertEqual(gate.verify_reports(self.directory, self.started_ns)["tests"], 2)

    def test_absent_empty_and_zero_test_reports(self):
        with self.assertRaises(gate.GateFailure):
            gate.verify_reports(self.directory, self.started_ns)
        self.report.touch()
        with self.assertRaises(gate.GateFailure):
            gate.verify_reports(self.directory, self.started_ns)
        self.write_report(tests=0, cases="")
        with self.assertRaises(gate.GateFailure):
            gate.verify_reports(self.directory, self.started_ns)

    def test_stale_file_and_recopied_stale_suite(self):
        self.write_report()
        os.utime(self.report, ns=(self.started_ns - 1, self.started_ns - 1))
        with self.assertRaisesRegex(gate.GateFailure, "Stale evidence"):
            gate.verify_reports(self.directory, self.started_ns)
        self.write_report(timestamp="2000-01-01T00:00:00Z")
        with self.assertRaisesRegex(gate.GateFailure, "stale suite timestamp"):
            gate.verify_reports(self.directory, self.started_ns)

    def test_skipped_failed_and_errored_summaries(self):
        for field in ("skipped", "failures", "errors"):
            with self.subTest(field=field):
                self.write_report(**{field: 1})
                with self.assertRaises(gate.GateFailure):
                    gate.verify_reports(self.directory, self.started_ns)

    def test_case_outcomes_cannot_hide_behind_zero_summaries(self):
        for outcome in ("skipped", "failure", "error"):
            with self.subTest(outcome=outcome):
                self.write_report(tests=1, cases=f'<testcase name="bad"><{outcome}/></testcase>')
                with self.assertRaises(gate.GateFailure):
                    gate.verify_reports(self.directory, self.started_ns)

    def test_summary_without_executed_cases_is_rejected(self):
        self.write_report(tests=99)
        with self.assertRaises(gate.GateFailure):
            gate.verify_reports(self.directory, self.started_ns)

    def test_malformed_report_is_rejected(self):
        self.report.write_text("<testsuite>", encoding="utf-8")
        with self.assertRaises(gate.ET.ParseError):
            gate.verify_reports(self.directory, self.started_ns)


class TaskEvidenceGateTest(unittest.TestCase):
    def test_invalid_worker_limits_fail_before_launching_gradle(self):
        from unittest.mock import patch
        for value in ("0", "9", "-1", "two", "", "２"):
            with self.subTest(value=value), patch.dict(os.environ, VOLLEYCUT_GRADLE_WORKERS=value):
                with self.assertRaisesRegex(gate.GateFailure, "integer from 1 to 8"):
                    gate.run_gate(Path("unused"), Path("unused"))

    def test_missing_stale_duplicate_or_skipped_task_outcomes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tasks.jsonl"
            started_ns = time.time_ns() - 2_000_000_000
            rows = [dict(runId="new-run", task=task, executed=True, didWork=True,
                         skipped=False, noSource=False, upToDate=False, failed=False, skipMessage=None)
                    for task in gate.MANDATORY_TASKS]

            def write(receipts):
                path.write_text("\n".join(json.dumps(row) for row in receipts), encoding="utf-8")

            with self.assertRaises(gate.GateFailure):
                gate.verify_tasks(path, started_ns, "new-run")
            write(rows)
            gate.verify_tasks(path, started_ns, "new-run")
            for receipts in (rows[:-1], rows + [rows[0]], [dict(row, runId="previous-run") for row in rows]):
                write(receipts)
                with self.assertRaises(gate.GateFailure):
                    gate.verify_tasks(path, started_ns, "new-run")
            for mutation in ({"skipped": True}, {"noSource": True}, {"upToDate": True},
                             {"skipMessage": "FROM-CACHE"}, {"didWork": False}, {"failed": True}):
                write([dict(rows[0], **mutation), *rows[1:]])
                with self.assertRaises(gate.GateFailure):
                    gate.verify_tasks(path, started_ns, "new-run")
            write(rows)
            os.utime(path, ns=(started_ns - 1, started_ns - 1))
            with self.assertRaises(gate.GateFailure):
                gate.verify_tasks(path, started_ns, "new-run")


class PinnedGradleFalseSuccessTest(unittest.TestCase):
    def test_cached_compilation_still_runs_tests_and_all_asset_checks(self):
        parent = gate.REPOSITORY / "android" / "build" / "regression-probes"
        parent.mkdir(parents=True, exist_ok=True)
        wrapper = gate.REPOSITORY / "android" / ("gradlew.bat" if os.name == "nt" else "gradlew")
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            project = Path(temporary).resolve()
            (project / "settings.gradle").write_text("rootProject.name = 'cached-gate-probe'\ninclude ':app'\n")
            (project / "gradle.properties").write_text("org.gradle.jvmargs=-Xmx256m\n")
            app = project / "app"
            sources = app / "src" / "test" / "java"
            sources.mkdir(parents=True)
            (sources / "FreshExecutionTest.java").write_text('''
import org.junit.Test;
import static org.junit.Assert.assertTrue;
public class FreshExecutionTest {
    @Test public void executes() { assertTrue(System.nanoTime() > 0); }
}
''')
            (app / "build.gradle").write_text('''
plugins { id 'java' }
repositories { mavenCentral() }
dependencies { testImplementation 'junit:junit:4.13.2' }
['prepareNeuralAssets', 'prepareServingAssets', 'verifySuppressionAsset',
 'verifyServingSideAsset', 'verifySideSwitchAsset'].each { name ->
    tasks.register(name) {
        outputs.file(layout.buildDirectory.file(name + '.txt'))
        doLast { outputs.files.singleFile.text = 'verified again' }
    }
}
tasks.register('testDebugUnitTest', Test) {
    testClassesDirs = sourceSets.test.output.classesDirs
    classpath = sourceSets.test.runtimeClasspath
    useJUnit()
    dependsOn 'verifySuppressionAsset', 'verifyServingSideAsset', 'verifySideSwitchAsset'
}
''')
            self.assertEqual(gate.run_gate(project, wrapper)["tests"], 1)
            # Remove only this temporary project's outputs, retaining Gradle's
            # build cache. The next run must restore compilation, not test results.
            self.assertTrue(app.resolve().is_relative_to(parent.resolve()))
            shutil.rmtree(app / "build")
            self.assertEqual(gate.run_gate(project, wrapper)["tests"], 1)
            logs = [(path / "gradle.log").read_text(encoding="utf-8")
                    for path in sorted((project / "build" / "regression-gate").glob("jvm-*"),
                                       key=lambda path: path.stat().st_mtime_ns)]
            self.assertIn(":app:compileTestJava FROM-CACHE", logs[-1])
            for path in (project / "build" / "regression-gate").glob("jvm-*"):
                receipts = [json.loads(row) for row in (path / "tasks.jsonl").read_text().splitlines()]
                self.assertEqual(len(receipts), 6)
                self.assertTrue(all(row["didWork"] and not row["skipped"] for row in receipts))

    def test_real_no_source_disabled_and_excluded_tasks_fail_the_gate(self):
        # Tiny Gradle projects reproduce successful Gradle exit codes;
        # they do not load Android plugins, dependencies, sources, or native builds.
        parent = gate.REPOSITORY / "android" / "build" / "regression-probes"
        parent.mkdir(parents=True, exist_ok=True)
        wrapper = gate.REPOSITORY / "android" / ("gradlew.bat" if os.name == "nt" else "gradlew")
        scenarios = {
            "no-source": ("", "Mandatory task did not execute successfully"),
            "disabled-test": ("tasks.named('testDebugUnitTest') { enabled = false }", "Mandatory task did not execute successfully"),
            "disabled-preparation": ("tasks.named('prepareNeuralAssets') { enabled = false }", "Mandatory task did not execute successfully"),
            "excluded-preparation": ("gradle.startParameter.setExcludedTaskNames([':app:prepareServingAssets'])", "Missing or duplicate mandatory task outcomes"),
        }
        for name, (configuration, expected) in scenarios.items():
            with self.subTest(scenario=name), tempfile.TemporaryDirectory(dir=parent) as temporary:
                project = Path(temporary).resolve()
                self.assertTrue(project.is_relative_to(parent.resolve()))
                (project / "settings.gradle").write_text("rootProject.name = 'gate-probe'\ninclude ':app'\n")
                (project / "gradle.properties").write_text("org.gradle.jvmargs=-Xmx256m\n")
                app = project / "app"
                app.mkdir()
                (app / "build.gradle").write_text("""
plugins { id 'java' }
['prepareNeuralAssets', 'prepareServingAssets', 'verifySuppressionAsset',
 'verifyServingSideAsset', 'verifySideSwitchAsset'].each { name ->
    tasks.register(name) { doLast { logger.lifecycle('Probe asset action executed') } }
}
tasks.register('testDebugUnitTest', Test) {
    dependsOn 'verifySuppressionAsset', 'verifyServingSideAsset', 'verifySideSwitchAsset'
}
""" + configuration, encoding="utf-8")
                # Earlier default reports cannot rescue a new NO-SOURCE/disabled run.
                old = app / "build" / "test-results" / "testDebugUnitTest"
                old.mkdir(parents=True)
                (old / "TEST-previous.xml").write_text(
                    '<testsuite tests="1" skipped="0" failures="0" errors="0" timestamp="2000-01-01T00:00:00Z">'
                    '<testcase name="previous"/></testsuite>'
                )
                with self.assertRaisesRegex(gate.GateFailure, expected):
                    gate.run_gate(project, wrapper)
                log = next((project / "build" / "regression-gate").glob("jvm-*/gradle.log")).read_text(encoding="utf-8")
                self.assertIn("BUILD SUCCESSFUL", log)
                self.assertIn("NO-SOURCE" if name != "disabled-test" else "SKIPPED", log)


if __name__ == "__main__":
    unittest.main()

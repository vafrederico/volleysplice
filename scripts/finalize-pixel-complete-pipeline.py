"""Finish validation/reporting after the phone benchmark and tensor transfer end.

All locations are explicit inputs. Run this in the neural Python environment;
it performs CPU parity checks, never model training or another phone run.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def transferred(folder, report):
    rows = report.get("results", [])
    if report.get("status") != "complete" or not rows:
        return False
    for row in rows:
        if row.get("status") != "complete" or not row.get("servingSideReady") or not row.get("sideSwitchReady"):
            raise RuntimeError("A neural case lacks complete rally and score results")
        meta = row["neural"]
        for suffix, size in (("features", meta["featureRows"] * meta["featureDimension"] * 4),
                             ("probabilities", meta["featureRows"] * 4 * 4)):
            path = folder / (row["id"] + "-" + suffix + ".f32")
            if not path.exists() or path.stat().st_size != size:
                return False
    return True


def main():
    parser = argparse.ArgumentParser()
    for name in ("root", "production", "neural", "graphs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--recording-index", required=True)
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--mobile", type=Path)
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args()
    status_path = args.root / "full-finalization-status.json"

    def status(value, **extra):
        payload = {"status": value, **extra}
        status_path.write_text(json.dumps(payload, indent=2), encoding="utf8")
        print(json.dumps(payload), flush=True)

    scripts = Path(__file__).resolve().parent

    def run(script, *arguments):
        with (args.root / (script + ".log")).open("w", encoding="utf8") as log:
            subprocess.run([sys.executable, str(scripts / script), *map(str, arguments)],
                           check=True, stdout=log, stderr=subprocess.STDOUT)

    try:
        status("waiting-for-neural-completion-and-tensor-transfer")
        deadline = time.monotonic() + 7200
        ready_once = False
        while time.monotonic() < deadline:
            try:
                report = read(args.neural / "result.json")
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(10)
                continue
            if report.get("status") == "failed":
                raise RuntimeError("Phone benchmark reported failure; retain partial results")
            if transferred(args.neural, report):
                if ready_once:
                    break
                ready_once = True
            else:
                ready_once = False
            time.sleep(10)
        else:
            raise TimeoutError("Waiting for full benchmark results exceeded two hours")
        status("validating-native-temporal-inference-and-decoder")
        run("validate-pixel-complete-pipeline.py", "--results", args.neural, "--graphs", args.graphs)
        parity = read(args.neural / "temporal-decoder-parity.json")
        if parity.get("passed") is not True or len(parity.get("checks", [])) != len(report['results']):
            raise RuntimeError("Every successful neural case needs a parity check")
        inputs=["--input", args.production, "--input", args.neural]
        if args.mobile:
            run("validate-pixel-complete-pipeline.py", "--results", args.mobile, "--graphs", args.graphs)
            mobile_parity=read(args.mobile/'temporal-decoder-parity.json')
            if mobile_parity.get('passed') is not True or len(mobile_parity['checks'])!=1:
                raise RuntimeError('Full Mobile parity required')
            parity['checks'].extend(mobile_parity['checks'])
            inputs.extend(['--input',args.mobile])
        run("combine-pixel-pipeline-results.py", *inputs, "--output", args.output)
        run("summarize-pixel-complete-pipeline.py", "--results", args.output, "--output", args.output,
            "--recording-index", args.recording_index, "--source-index", args.source_index)
        pilot = (args.root / "complete-pipeline-report.md").read_text(encoding="utf8")
        full = (args.output / "complete-pipeline-report.md").read_text(encoding="utf8")
        notes = ("\n\nFull-recording qualification: one successful pass per pipeline, not repeated-run medians. "
                 "Failed and interrupted attempts are retained privately and excluded from these timings. "
                 "The successful production build includes the shared terminal-frame correction; the neural build "
                 "also includes the final embedding-sample correction. Both score specialists completed. "
                 "Native temporal probabilities and decoded boundaries passed comparison against desktop for "
                 "both full neural cases. Pixel/PTS/AV extraction accuracy parity remains unqualified.\n")
        comparison=''
        if args.labels:
            comparison_args=[]
            for folder in (args.production,args.neural,args.mobile):
                if folder:comparison_args.extend(['--results',folder])
            run('report-pixel-storage-and-rallies.py',*comparison_args,'--labels',args.labels,
                '--recording-index',args.recording_index,'--output',args.root/'human-comparison')
            comparison=(args.root/'human-comparison/storage-and-rallies.md').read_text()
        notes+='\nThe initial full DINO attempt failed during a whole-file token read. The successful retry maps tokens and writes diagnostic floats in bounded chunks; its timing includes this implementation.\n'
        (args.root / "report.md").write_text(pilot + "\n\n---\n\n" + full + notes+'\n\n'+comparison, encoding="utf8")
        progress_path = args.root / "progress.json"
        progress = read(progress_path)
        progress.update(status="complete FP32 pilot and full-recording benchmarks", active=None, pending=[])
        progress["completed"].extend(["Full DINO and both score specialists completed",
                                      "Both full neural temporal and decoder parity checks passed",
                                      "Combined full-video report written with private ledger indexes"])
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf8")
        summary = read(args.output / "complete-pipeline-summary.json")
        status("complete", recordingIndex=args.recording_index, sourceIndex=args.source_index,
               summary=summary["summary"], parity=parity["checks"])
    except Exception as error:
        # Exception text can contain private paths: detailed failure stays in this
        # private status file, which must not be copied into a public report.
        status("failed", errorType=type(error).__name__)
        raise


if __name__ == "__main__":
    main()

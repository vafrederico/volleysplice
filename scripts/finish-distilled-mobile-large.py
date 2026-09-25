"""Finish the registered experiment and publish only after every gate passes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    # A progress file may be in its producer's brief shared-storage write window.
    for attempt in range(6):
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            if attempt == 5:
                raise
            time.sleep(.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('experiment', 'index', 'inventory', 'baseline-experiment'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--executor-completion', type=Path)
    args = parser.parse_args()
    root, scripts = args.experiment, Path(__file__).resolve().parent
    plan = read(root / 'plan.json')
    selections = [root / 'fits' / task['variant'] / f"split-{task.get('splitSeed', task['seed'])}" / 'selection.json' for task in plan['tasks']]

    def executor_ready():
        if args.executor_completion is None:
            return True
        if not args.executor_completion.is_file():
            return False
        receipt = read(args.executor_completion)
        assert receipt['kind'] == 'distilled-large-parallel-completed-v1'
        assert receipt['status'] == 'complete' and receipt['completedFits'] == len(selections)
        assert receipt['allRegisteredFitsValidated'] is True
        assert Path(receipt['plan']['path']).resolve() == (root / 'plan.json').resolve()
        assert receipt['plan']['sha256'] == hashlib.sha256((root / 'plan.json').read_bytes()).hexdigest()
        return True

    def status(phase, **details):
        value = dict(phase=phase, **details)
        temp = root / '.completion-progress.tmp'
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(root / 'completion-progress.json')
        print(json.dumps(value), flush=True)

    def run(name, *arguments):
        with (root / (name + '.log')).open('a') as log:
            subprocess.run([sys.executable, '-B', str(scripts / name), *map(str, arguments)], check=True,
                stdout=log, stderr=subprocess.STDOUT)

    try:
        previous = -1
        while True:
            completed = sum(path.is_file() for path in selections)
            if completed != previous:
                status('waiting-for-fits', completedFits=completed, totalFits=len(selections))
                previous = completed
            if completed == len(selections) and executor_ready():
                # Parse every completion marker before starting evaluation.
                for path in selections:
                    read(path)
                break
            launch = read(root / 'launch.json')
            proc = Path('/proc') / str(launch['pid'])
            if not proc.exists() or '\nState:\tZ' in (proc / 'status').read_text():
                raise RuntimeError('Training stopped before every registered fit completed; inspect fit.log')
            time.sleep(15)
        status('selection-and-all-video-inference')
        run('evaluate-distilled-mobile-large.py', '--experiment', root, '--beach-inputs', root / 'beach-inputs/catalog.json')
        status('all-video-evaluation')
        run('evaluate-distilled-mobile-large-panels.py', '--experiment', root, '--inventory', args.inventory)
        status('reporting')
        run('report-distilled-mobile-large.py', '--experiment', root, '--baseline-experiment', args.baseline_experiment)
        status('validating-publication')
        run('publish-distilled-mobile-large-ui.py', '--experiment', root, '--index', args.index, '--validate-only')
        status('publishing')
        run('publish-distilled-mobile-large-ui.py', '--experiment', root, '--index', args.index)
        status('browser-validation')
        browser = read(root / 'browser-command.json')
        assert isinstance(browser['command'], list) and all(isinstance(v, str) for v in browser['command'])
        with (root / 'browser-validation.log').open('a') as log:
            subprocess.run(browser['command'], env={**os.environ, **browser.get('env', {})}, check=True,
                stdout=log, stderr=subprocess.STDOUT)
        status('complete', completedFits=len(selections), recordings=44, variants=2,
            report='selection-report.md', browserValidated=True)
    except Exception as error:
        status('failed', errorType=type(error).__name__, message=str(error))
        raise


if __name__ == '__main__':
    main()

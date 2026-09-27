"""Continue frozen Large evaluation/publication after registered fits finish."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read_json(path, attempts=6, delay=.2):
    """Tolerate a producer's short write/replace window on shared storage."""
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text(encoding='utf-8-sig'))
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


def write_status(path, payload):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def selection_ready(path):
    if not path.exists():
        return False
    try:
        read_json(path)
        return True
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def main():
    p=argparse.ArgumentParser()
    for key in ('experiment','study','ui-index','baseline-graphs'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args(); scripts=Path(__file__).resolve().parent
    plan=read_json(a.experiment/'plan.json')
    paths=[a.experiment/'fits'/t['variant']/f"split-{t.get('splitSeed',t['seed'])}"/'selection.json' for t in plan['tasks']]
    def status(phase,**extra):
        write_status(a.experiment/'progress.json',dict(status=phase,**extra))
        print(json.dumps(dict(status=phase,**extra)),flush=True)
    def run(script,*args):
        with (a.experiment/(script+'.log')).open('w') as log:
            subprocess.run([sys.executable,str(scripts/script),*map(str,args)],check=True,stdout=log,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+6*3600
        previous=-1
        while time.monotonic()<deadline:
            count=sum(selection_ready(path) for path in paths)
            if count!=previous:status('training-and-calibration',completedFits=count,totalFits=len(paths));previous=count
            if count==len(paths):break
            time.sleep(15)
        else:raise TimeoutError('Registered fits did not all complete')
        status('common-selection-and-all-video-inference')
        run('evaluate-mobile-large.py','--output',a.experiment)
        status('exporting-and-qualifying-native-models')
        run('prepare-pixel-mobile-large.py','--experiment',a.experiment,'--baseline-graphs',a.baseline_graphs,'--output',a.experiment/'graphs')
        status('publishing-ui-variants')
        run('publish-mobile-large-ui.py','--experiment',a.experiment,'--index',a.ui_index)
        run('report-mobile-large.py','--experiment',a.experiment)
        status('models-published-native-benchmarks-pending',completedFits=len(paths),pending=['native two-minute benchmark','native full-video benchmark','final comparative report'])
    except Exception as error:
        status('failed',errorType=type(error).__name__)
        raise


if __name__=='__main__':main()

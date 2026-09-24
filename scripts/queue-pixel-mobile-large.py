"""Windows host queue: wait for qualified models/free phone, then benchmark Large."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read_json(path, attempts=6, delay=.2):
    """Retry transient partial JSON and replace windows from other workers."""
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


def main():
    p=argparse.ArgumentParser()
    for name in ('root','experiment','graphs'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('adb','serial','pilot-video','full-video','root-wsl','experiment-wsl','neural-python-wsl','ledger-wsl'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args(); scripts=Path(__file__).resolve().parent
    read=read_json
    def status(value):
        write_status(a.root/'large-benchmark-status.json',dict(status=value))
        print(value,flush=True)
    def run(name,args):
        # Windows and WSL create NAS files with different owners. Keep each
        # invocation's log instead of reopening a log owned by another worker.
        with (a.root/(name+'.'+str(time.time_ns())+'.log')).open('x') as log:
            subprocess.run(args,check=True,stdout=log,stderr=subprocess.STDOUT)
    def research(script,*args):
        run(script,['wsl','env','PYTHONDONTWRITEBYTECODE=1','VOLLEYCUT_PRIVATE_LEDGER='+a.ledger_wsl,
            a.neural_python_wsl,'scripts/'+script,*map(str,args)])
    try:
        status('waiting-for-qualified-Large-and-completed-DINO')
        deadline=time.monotonic()+4*3600
        while time.monotonic()<deadline:
            dino=read(a.root/'full-finalization-status.json')
            training=read(a.experiment/'progress.json')
            if dino['status']=='failed' or training['status']=='failed':raise RuntimeError('Prior work failed; phone queue not started')
            if dino['status']=='complete' and training['status']=='models-published-native-benchmarks-pending':break
            time.sleep(15)
        else:raise TimeoutError('Prerequisites did not complete')
        assert read(a.graphs/'mobile-large-qualification.json')['temporalChecks']
        for scope,video,seconds,runs,index in [('120s',a.pilot_video,120,4,'private-reference-0212'),
                                            ('full',a.full_video,1061.016489,1,'private-reference-0213')]:
            folder=a.root/('pipeline-large-'+scope)
            folder_wsl=a.root_wsl+'/pipeline-large-'+scope
            result_path=folder/'result.json'
            completed=read(result_path) if result_path.exists() else None
            if completed is not None and completed.get('status')=='complete':
                # A resume revalidates saved tensors against the qualified graph;
                # it does not repeat or overwrite completed timing observations.
                plan=read(folder/'pipeline-plan.json')
                assert len(plan['cases'])==runs and len(completed['results'])==runs
                assert all(s['family']=='mobile-large' and s['seconds']==seconds for s in plan['cases'])
                assert all(r['status']=='complete' and r['servingSideReady'] and r['sideSwitchReady'] for r in completed['results'])
                status('validating-completed-native-Large-'+scope)
            else:
                if result_path.exists():raise RuntimeError('Incomplete prior phone attempt requires a new output folder')
                status('running-native-Large-'+scope)
                run('large-'+scope,[sys.executable,str(scripts/'benchmark-pixel-complete-pipeline.py'),
                    '--adb',a.adb,'--serial',a.serial,'--graphs',str(a.graphs),'--output',str(folder),
                    '--video-name',video,'--seconds',str(seconds),'--runs',str(runs),'--families','mobile-large','--transfer-models'])
            assert read(folder/'result.json')['status']=='complete'
            research('validate-pixel-complete-pipeline.py','--results',folder_wsl,'--graphs',a.experiment_wsl+'/graphs')
            research('summarize-pixel-complete-pipeline.py','--results',folder_wsl,'--output',folder_wsl,
                     '--recording-index','recording-044','--source-index',index)
        research('report-pixel-storage-and-rallies.py',
            '--results',a.root_wsl+'/pipeline-full-v2','--results',a.root_wsl+'/pipeline-full-v3',
            '--results',a.root_wsl+'/pipeline-full-v4-dino','--results',a.root_wsl+'/pipeline-large-full',
            '--labels',a.root_wsl+'/recording-044-label-snapshot.private.json',
            '--recording-index','recording-044','--output',a.root_wsl+'/human-comparison-with-large')
        research('report-mobile-large.py','--experiment',a.experiment_wsl)
        pieces=[(a.root/'report.md').read_text(encoding='utf8'),
                (a.experiment/'selection-report.md').read_text(encoding='utf8'),
                (a.root/'pipeline-large-120s/complete-pipeline-report.md').read_text(encoding='utf8'),
                (a.root/'pipeline-large-full/complete-pipeline-report.md').read_text(encoding='utf8'),
                (a.root/'human-comparison-with-large/storage-and-rallies.md').read_text(encoding='utf8')]
        (a.root/'report-with-mobile-large.md').write_text('\n\n---\n\n'.join(pieces),encoding='utf8')
        status('complete')
    except Exception as error:
        status('failed-'+type(error).__name__)
        raise


if __name__=='__main__':main()

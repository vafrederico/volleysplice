#!/usr/bin/env python3
"""Transfer a plan, run the isolated native app, and retain outputs/profiles on NAS."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--adb',required=True)
    p.add_argument('--serial',required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--collect-only',action='store_true')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    package='com.volleycut.neuralbenchmark'
    plan=json.loads(a.plan.read_text())
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',a.plan.name): raise ValueError('Unsafe plan filename')
    def adb(*args,**kwargs):
        return subprocess.run([a.adb,'-s',a.serial,*args],check=True,timeout=60,**kwargs)
    if not a.collect_only:
        remote='/data/local/tmp/volleycut-neural-benchmark/'+a.plan.name
        adb('shell','mkdir','-p','/data/local/tmp/volleycut-neural-benchmark')
        adb('push',str(a.plan),remote)
        adb('shell',f'cat {remote} | run-as {package} dd of=files/benchmark/{a.plan.name} bs=1048576')
        actual=adb('exec-out','run-as',package,'cat','files/benchmark/'+a.plan.name,capture_output=True).stdout
        if actual!=a.plan.read_bytes(): raise RuntimeError('Plan transfer mismatch')
        adb('shell','am','force-stop',package)
        grant=['-a','android.intent.action.VIEW','-d',plan['sourceUri'],'-f','0x1'] if plan.get('sourceUri') else []
        adb('shell','am','start','-W','-n',package+'/.BenchmarkActivity','--es','plan',a.plan.name,*grant)
    deadline=time.monotonic()+7200
    seen=0
    while time.monotonic()<deadline:
        raw=adb('exec-out','run-as',package,'cat','files/benchmark/result.json',capture_output=True).stdout
        result=json.loads(raw)
        if result.get('runId')!=plan['runId']:
            if result.get('status')=='failed': raise RuntimeError(result.get('error'))
            time.sleep(2); continue
        (a.output/'result.json').write_bytes(raw)
        for row in result.get('results',[])[seen:]:
            print(json.dumps({'id':row['id'],'status':row['status'],'video':{k:v for k,v in row.get('video',{}).items() if k!='sampleTimestamps'},'error':row.get('error')}),flush=True)
        seen=len(result.get('results',[]))
        if result['status'] in ('complete','failed'): break
        time.sleep(3)
    else: raise TimeoutError('Native benchmark deadline exceeded')
    files=set()
    for row in result.get('results',[]):
        files.update(x['file'] for x in row.get('outputs',[]))
        if 'profile' in row: files.add(Path(row['profile']).name)
        if 'video' in row: files.add(row['video']['tokens'])
    for name in sorted(files):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',name): raise ValueError('Unsafe output filename')
        with (a.output/name).open('wb') as f:
            adb('exec-out','run-as',package,'cat','files/benchmark/'+name,stdout=f)
    print('Collected '+str(len(files))+' outputs/profiles',flush=True)


if __name__=='__main__': main()

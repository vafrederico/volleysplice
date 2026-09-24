"""Run fresh-input FP32 pipelines, including both score specialists, on Android."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time
import uuid

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--adb',required=True);p.add_argument('--serial',required=True)
    p.add_argument('--graphs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--video-name');source.add_argument('--source-uri')
    p.add_argument('--seconds',type=float,default=120)
    p.add_argument('--runs',type=int,default=4);p.add_argument('--families',default='production,mobile,dino')
    p.add_argument('--transfer-models',action='store_true')
    p.add_argument('--full-frame',action='store_true',help='Use full-frame ROI; transferred pooling weights must match its letterbox geometry')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    package='com.volleycut.nativeanalysis.pipelinebenchmark'
    def adb(*args,check=True,**kwargs):
        return subprocess.run([a.adb,'-s',a.serial,*args],check=check,timeout=120,**kwargs)
    def transfer(path):
        name=path.name
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',name):raise ValueError('Unsafe name')
        remote='/data/local/tmp/volleycut-neural-benchmark/'+name
        adb('push',str(path),remote,capture_output=True)
        adb('shell',f'cat {remote} | run-as {package} dd of=files/benchmark/{name} bs=1048576',capture_output=True)
        # Byte equality is deliberately verified before benchmarking.
        actual=adb('exec-out','run-as',package,'cat','files/benchmark/'+name,capture_output=True).stdout
        if actual!=path.read_bytes():raise RuntimeError('Transfer mismatch '+name)
    adb('shell','run-as',package,'mkdir','-p','files/benchmark',capture_output=True)
    if a.source_uri:
        if not re.fullmatch(r'content://media/external/video/media/\d+',a.source_uri):raise ValueError('Expected MediaStore video URI')
        source_uri=a.source_uri
    else:
        catalog=adb('shell','content','query','--uri','content://media/external/video/media','--projection','_id:_display_name',capture_output=True,text=True).stdout
        ids=[re.search(r'_id=(\d+)',line).group(1) for line in catalog.splitlines() if line.split('_display_name=',1)[-1]==a.video_name]
        if len(ids)!=1:raise RuntimeError(f'MediaStore matches: {len(ids)}')
        source_uri='content://media/external/video/media/'+ids[0]
    families=a.families.split(',')
    if not set(families)<=set(('production','mobile','mobile-large','dino','specialist-smoke','tensor-io-smoke','mobile-terminal','dino-terminal')):raise ValueError('Unknown family')
    if a.transfer_models:
        for family in families:
            if family in ('production','specialist-smoke','tensor-io-smoke'):continue
            for suffix in ('-encoder-fp32.onnx','-tcn-dynamic-fp32.onnx','-pipeline.json'):
                transfer(a.graphs/(family+suffix))
        for family in ('mobile','mobile-large'):
            if family in families:transfer(a.graphs/(family+'-encoder-pool_weights.f32'))
    cases=[{'id':f'{family}-fp32-full-run{run}','family':family,'seconds':a.seconds,'fullFrame':a.full_frame,'warmup':run==0 and a.runs>1} for run in range(a.runs) for family in families]
    plan={'runId':uuid.uuid4().hex,'sourceUri':source_uri,'cases':cases}
    plan_path=a.output/'pipeline-plan.json';plan_path.write_text(json.dumps(plan,indent=2));transfer(plan_path)
    for kind in ('battery','thermalservice'):
        (a.output/(kind+'-before.txt')).write_text(adb('shell','dumpsys',kind,capture_output=True,text=True).stdout)
    adb('shell','am','force-stop',package,capture_output=True)
    adb('shell','am','start','-W','-n',package+'/com.volleycut.nativeanalysis.PipelineBenchmarkActivity',
        '-a','android.intent.action.VIEW','-d',plan['sourceUri'],'-f','0x1','--es','plan',plan_path.name,capture_output=True)
    seen=set();deadline=time.monotonic()+6*3600
    while time.monotonic()<deadline:
        raw=adb('exec-out','run-as',package,'cat','files/benchmark/pipeline-result.json',check=False,capture_output=True).stdout
        try:result=json.loads(raw)
        except ValueError:time.sleep(2);continue
        if result.get('runId')!=plan['runId']:time.sleep(2);continue
        (a.output/'result.json').write_bytes(raw)
        for row in result.get('results',[]):
            key=(row['id'],row['status'])
            if key not in seen:
                print(json.dumps({k:row.get(k) for k in ('id','status','totalMs','rallyCount','error','stagesMs')}),flush=True);seen.add(key)
        if result['status'] in ('complete','failed'):break
        alive=adb('shell','pidof',package,check=False,capture_output=True).stdout.strip()
        if not alive:
            time.sleep(2)
            if not adb('shell','pidof',package,check=False,capture_output=True).stdout.strip():
                result['status']='failed'
                result['hostError']='Benchmark process exited before final results'
                for row in result.get('results',[]):
                    if row.get('status')=='running':row.update(status='failed',error=result['hostError'])
                (a.output/'result.json').write_text(json.dumps(result,indent=2))
                (a.output/'process-exit.txt').write_bytes(adb('shell','dumpsys','activity','exit-info',package,capture_output=True).stdout)
                break
        time.sleep(3)
    else:raise TimeoutError('Full pipeline timed out')
    for row in result.get('results',[]):
        if 'neural' not in row:continue
        for suffix in ('-tokens.f32','-features.f32','-probabilities.f32'):
            with (a.output/(row['id']+suffix)).open('wb') as target:
                adb('exec-out','run-as',package,'cat','files/benchmark/'+row['id']+suffix,stdout=target)
    if result['status']!='complete':raise RuntimeError('Pipeline failed; inspect result.json')
if __name__=='__main__':main()

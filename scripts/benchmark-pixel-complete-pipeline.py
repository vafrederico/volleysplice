"""Run fresh-input FP32 pipelines, including both score specialists, on Android."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import time
import uuid

def validate_full_frame_graphs(graphs, families, width, height):
    """Reject archived cropped pooling before a run is labelled full-frame."""
    mobile_families=[family for family in families if family in ('mobile','mobile-large')]
    if not mobile_families:return None
    contract=json.loads((graphs/'input-contract.json').read_text(encoding='utf-8-sig'))
    if contract.get('roi')!={'x':0,'y':0,'width':1,'height':1}:
        raise ValueError('Full-frame graph contract requires an uncropped ROI')
    if width<=0 or height<=0:raise ValueError('Source dimensions are required for pooling validation')
    scale=min(224/width,224/height)
    resized_width=max(1,round(width*scale));resized_height=max(1,round(height*scale))
    left=(224-resized_width)//2;top=(224-resized_height)//2
    expected_box=[left/224,top/224,(left+resized_width)/224,(top+resized_height)/224]
    box=contract.get('contentBox')
    if not isinstance(box,list) or len(box)!=4 or any(
            not math.isfinite(value) or abs(value-expected)>1e-9 for value,expected in zip(box,expected_box)):
        raise ValueError('Graph pooling geometry differs from the full-frame source dimensions')
    left,top,right,bottom=box
    expected_weights=[]
    for low,high in ((0,1),(.5,1),(0,.5),(.4,.6)):
        region_top=top+low*(bottom-top);region_bottom=top+high*(bottom-top)
        area=[max(0,min((x+1)/7,right)-max(x/7,left))*
              max(0,min((y+1)/7,region_bottom)-max(y/7,region_top))
              for y in range(7) for x in range(7)]
        total=sum(area)
        if total<=0:raise ValueError('Empty pooling region')
        expected_weights.extend(value/total for value in area)
    for family in mobile_families:
        for suffix in ('-encoder-fp32.onnx','-tcn-dynamic-fp32.onnx','-pipeline.json','-encoder-pool_weights.f32'):
            path=graphs/(family+suffix)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=contract.get('hashes',{}).get(path.name):
                raise ValueError('Graph does not match frozen full-frame contract: '+path.name)
        weights=(graphs/(family+'-encoder-pool_weights.f32')).read_bytes()
        if len(weights)!=4*7*7*4 or any(not math.isfinite(value) or abs(value-expected)>1e-7
                for value,expected in zip(struct.unpack('<196f',weights),expected_weights)):
            raise ValueError('Pool weights do not match declared full-frame geometry')
    return contract

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--adb',required=True);p.add_argument('--serial',required=True)
    p.add_argument('--graphs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--video-name');source.add_argument('--source-uri')
    p.add_argument('--seconds',type=float,default=120)
    p.add_argument('--runs',type=int,default=4);p.add_argument('--families',default='production,mobile,dino')
    p.add_argument('--transfer-models',action='store_true')
    p.add_argument('--adb-transfer-timeout',type=float,default=600,help='Seconds allowed for each model or tensor transfer, outside measured inference time')
    p.add_argument('--full-frame',action='store_true',default=True,help='Compatibility flag: new runs always use the full frame; pooling weights must match its letterbox geometry')
    a=p.parse_args()
    if a.adb_transfer_timeout<=0:raise ValueError('Transfer timeout must be positive')
    a.output.mkdir(parents=True,exist_ok=True)
    package='com.volleycut.nativeanalysis.pipelinebenchmark'
    def adb(*args,check=True,**kwargs):
        return subprocess.run([a.adb,'-s',a.serial,*args],check=check,timeout=kwargs.pop('timeout',120),**kwargs)
    def transfer(path):
        name=path.name
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',name):raise ValueError('Unsafe name')
        expected=hashlib.sha256(path.read_bytes()).hexdigest()
        def remote_digest():
            result=adb('shell','run-as',package,'sha256sum','files/benchmark/'+name,check=False,capture_output=True)
            words=result.stdout.decode('ascii',errors='replace').split()
            return words[0] if result.returncode==0 and words else None
        # Frozen graphs often already exist on the device. Verify their content
        # before skipping an otherwise expensive relayed ADB transfer.
        if remote_digest()==expected:return
        remote='/data/local/tmp/volleycut-neural-benchmark/'+name
        adb('push',str(path),remote,capture_output=True,timeout=a.adb_transfer_timeout)
        adb('shell',f'cat {remote} | run-as {package} dd of=files/benchmark/{name} bs=1048576',capture_output=True)
        if remote_digest()!=expected:raise RuntimeError('Transfer SHA-256 mismatch '+name)
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
    input_contract=None
    if any(family in ('mobile','mobile-large') for family in families):
        dimensions=adb('shell','content','query','--uri',source_uri,'--projection','width:height',capture_output=True,text=True).stdout
        width=re.search(r'\bwidth=(\d+)',dimensions);height=re.search(r'\bheight=(\d+)',dimensions)
        if not width or not height:raise ValueError('Cannot verify source dimensions for regional pooling')
        input_contract=validate_full_frame_graphs(a.graphs,families,int(width.group(1)),int(height.group(1)))
        (a.output/'input-contract.json').write_text(json.dumps(input_contract,indent=2))
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
                adb('exec-out','run-as',package,'cat','files/benchmark/'+row['id']+suffix,stdout=target,timeout=a.adb_transfer_timeout)
    if result['status']!='complete':raise RuntimeError('Pipeline failed; inspect result.json')
if __name__=='__main__':main()

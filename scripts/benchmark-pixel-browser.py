#!/usr/bin/env python3
"""Loopback-only fixture host for a real Android Chrome tensor benchmark.

Use adb reverse for localhost access. No production server or desktop browser.
All generated assets and posted results live under --root on NAS.
"""
import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

PAGE = r'''<!doctype html><meta name="viewport" content="width=device-width"><title>Pixel neural browser benchmark</title>
<h1>Pixel browser benchmark</h1><pre id="status">Loading runtime…</pre><script src="/browser-runtime/ort.webgpu.min.js"></script>
<script>
(async()=>{
 const status=document.getElementById('status');
 const report={status:'running',userAgent:navigator.userAgent,secureContext:isSecureContext,isolated:crossOriginIsolated,
  hasWebGPU:!!navigator.gpu,threads:Math.min(4,navigator.hardwareConcurrency),scope:'tensor inference only; not end-to-end video analysis',results:[]};
 const save=async()=>{await fetch('/result',{method:'POST',body:JSON.stringify(report)});};
 let wakeLock;try{wakeLock=await navigator.wakeLock?.request('screen');}catch{}
 try {
  ort.env.wasm.wasmPaths='/browser-runtime/';ort.env.wasm.numThreads=report.threads;
  if(navigator.gpu){const adapter=await navigator.gpu.requestAdapter();report.adapter=adapter?.info?{vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description}:null;}
  const plan=await(await fetch('/graphs/browser-plan.json')).json();report.runId=plan.runId;
  for(const spec of plan.cases){
   status.textContent='Running '+spec.id;const row={id:spec.id,provider:spec.provider};let session;
   try{
    const data=await(await fetch('/graphs/'+spec.model)).arrayBuffer();let start=performance.now();
    session=await ort.InferenceSession.create(data,{executionProviders:[spec.provider],graphOptimizationLevel:'all'});row.loadMs=performance.now()-start;
    const feeds={};for(const input of spec.inputs){const bytes=await(await fetch('/graphs/'+input.file)).arrayBuffer();feeds[input.name]=new ort.Tensor('float32',new Float32Array(bytes),input.shape);}
    const expected=new Float32Array(await(await fetch('/graphs/'+spec.expected)).arrayBuffer());
    row.samplesMs=[];
    for(let i=-3;i<10;i++){
     start=performance.now();const outputs=await session.run(feeds);let values;
     for(const tensor of Object.values(outputs)){const read=await tensor.getData();if(!values)values=read;}
     if(i>=0)row.samplesMs.push(performance.now()-start);
     if(i===9){let sum=0,max=0,finite=true;for(let k=0;k<values.length;k++){const error=values[k]-expected[k];sum+=error*error;max=Math.max(max,Math.abs(error));finite&&=Number.isFinite(values[k]);}
      row.outputCount=values.length;row.expectedCount=expected.length;row.finite=finite;row.rmseVsOriginalFP32=Math.sqrt(sum/values.length);row.maxAbsoluteError=max;}
     for(const tensor of Object.values(outputs))tensor.dispose();
    }
    row.status='complete';for(const tensor of Object.values(feeds))tensor.dispose();
   }catch(error){row.status='failed';row.error=String(error);}
   finally{if(session)await session.release();}
   report.results.push(row);await save();
  }
  report.status='complete';
 }catch(error){report.status='failed';report.error=String(error);}
 await save();status.textContent=JSON.stringify(report,null,2);if(wakeLock)await wakeLock.release();
})();
</script>'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--port',type=int,default=13002)
    a=p.parse_args();a.root=a.root.resolve();(a.root/'browser').mkdir(exist_ok=True)
    class Handler(SimpleHTTPRequestHandler):
        extensions_map={**SimpleHTTPRequestHandler.extensions_map,'.mjs':'text/javascript','.wasm':'application/wasm'}
        def end_headers(self):
            self.send_header('Cross-Origin-Opener-Policy','same-origin');self.send_header('Cross-Origin-Embedder-Policy','require-corp')
            self.send_header('Cache-Control','no-store');super().end_headers()
        def do_GET(self):
            if self.path=='/':
                self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.end_headers();self.wfile.write(PAGE.encode());return
            if not self.path.startswith(('/graphs/','/browser-runtime/')):self.send_error(404);return
            target=Path(self.translate_path(self.path)).resolve()
            if not any(target.is_relative_to(a.root/name) for name in ('graphs','browser-runtime')):self.send_error(404);return
            super().do_GET()
        def do_POST(self):
            if self.path!='/result':self.send_error(404);return
            count=int(self.headers.get('Content-Length','0'))
            if not 0<count<2_000_000:self.send_error(400);return
            data=json.loads(self.rfile.read(count));(a.root/'browser/result.json').write_text(json.dumps(data,indent=2))
            self.send_response(200);self.end_headers();self.wfile.write(b'OK')
            print(json.dumps({'status':data.get('status'),'cases':len(data.get('results',[]))}),flush=True)
            if data.get('status') in ('complete','failed'):threading.Thread(target=self.server.shutdown,daemon=True).start()
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',a.port),functools.partial(Handler,directory=str(a.root)))
    print('Loopback phone benchmark server ready',flush=True)
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()

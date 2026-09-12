"""Physical iPad video-queue interruption and explicit-resume regression.

First run the Developer native export check to create a short fixture project.
Pass its observed project-button ID via --project. Use the lab's existing
pymobiledevice3 Python. The app must remain the same process after background
cancellation; only user-requested Resume starts the export again. Always parks.
"""
import argparse, asyncio, importlib.util, json, time
from pathlib import Path
from types import SimpleNamespace
from pymobiledevice3.lockdown import create_using_usbmux
from pymobiledevice3.services.house_arrest import HouseArrestService
spec=importlib.util.spec_from_file_location('editor',Path(__file__).with_name('editor-smoke.py'));editor=importlib.util.module_from_spec(spec);spec.loader.exec_module(editor)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project',required=True)
parser.add_argument('--output',type=Path,default=Path('E:/wslmac/artifacts/volleysplice/queued-export'))
args=parser.parse_args()
output=args.output
config=SimpleNamespace(lab=Path('E:/wslmac'),url='http://127.0.0.1:18100',mjpeg='http://127.0.0.1:19100/',output=output,project=args.project)
t=editor.EditorSmoke(config)
async def ledger():
 async with await create_using_usbmux(serial='<ios-device-udid>') as lock:
  async with await HouseArrestService.create(lockdown=lock,bundle_id='com.vafrederico.VolleySplice') as afc:
   for attempt in range(5):
    try: return json.loads(await afc.get_file_contents('/Library/Application Support/ProcessingJobs/queue.json'))['jobs']
    except json.JSONDecodeError:
     if attempt==4: raise
     await asyncio.sleep(.2)
def jobs(): return asyncio.run(ledger())
try:
 t.wda.connect();t.wda.activate('com.vafrederico.VolleySplice');t.source(refresh=True)
 if t.nodes('queueDone'):t.click('queueDone')
 if t.nodes('backToProjects'):t.click('backToProjects')
 t.choose_project();t.click('exportTab');baseline={v['id'] for v in jobs()}
 initial_pid=t.command('/wda/activeAppInfo')['pid']
 t.click('exportVideo');job=next(v for v in jobs() if v['id'] not in baseline);ident=job['id'];assert job['state']=='running',job
 t.record('export-started-before-background',job=ident,progress=job['progress'],pid=initial_pid)
 t.park();deadline=time.monotonic()+30
 while time.monotonic()<deadline:
  job=next(v for v in jobs() if v['id']==ident)
  if job['state']=='interrupted':break
  assert job['state'] in ['running','cancelling'],job
  time.sleep(.5)
 else:raise TimeoutError('Video did not interrupt safely')
 assert job['stopReason']=='backgrounded' and not job['outputNames'],job
 t.record('video-background-interruption',job=ident)
 t.wda.activate('com.vafrederico.VolleySplice');assert t.command('/wda/activeAppInfo')['pid']==initial_pid, 'App exited during background cancellation'
 t.click('processingQueueBar');t.click('resumeJob-'+ident)
 deadline=time.monotonic()+90
 while time.monotonic()<deadline:
  job=next(v for v in jobs() if v['id']==ident)
  if job['state']=='completed':break
  assert job['state'] in ['queued','running'],job
  time.sleep(1)
 else:raise TimeoutError('Video resume did not complete')
 assert job['attempt']==2 and len(job['outputNames'])==2,job
 t.record('video-foreground-resume',job=ident,outputs=job['outputNames'])
 t.capture('resumed-export-complete')
except Exception as error:
 t.results.append({'failure':str(error)})
 raise
finally:
 t.park();(output/'export-interruption-results.json').write_text(json.dumps(t.results,indent=2))

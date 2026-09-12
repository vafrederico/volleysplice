"""Physical iPad WDA checks. Always parks on success and failure; no Mac media decode."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
import urllib.request
import xml.etree.ElementTree as ET
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument('--lab', type=Path, default=Path(os.environ.get('VOLLEYCUT_IOS_LAB_ROOT', 'artifacts/ios-lab')))
parser.add_argument('--recording')
parser.add_argument('--analyze', action='store_true')
parser.add_argument('--export-check', action='store_true')
parser.add_argument('--media-check', action='store_true')
parser.add_argument('--export-only', action='store_true', help='Skip already-verified orientation/golden checks during export iterations')
args = parser.parse_args()
spec = importlib.util.spec_from_file_location('lab_wda', args.lab / 'scripts/wda-client.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
wda = module.WDA('http://127.0.0.1:18100')
output = args.lab / 'artifacts/volleysplice'
output.mkdir(exist_ok=True)
editor_spec = importlib.util.spec_from_file_location('editor_smoke', Path(__file__).with_name('editor-smoke.py'))
editor_module = importlib.util.module_from_spec(editor_spec)
editor_spec.loader.exec_module(editor_module)
ui = editor_module.EditorSmoke(SimpleNamespace(lab=args.lab, url=wda.base, mjpeg='http://127.0.0.1:19100/', output=output, project=None))
ui.wda = wda

def click(name):
    ui.tree = None
    ui.click(name)

def status():
    tree = ET.fromstring(wda.command('/source'))
    if any(e.get('name') == 'currentRallyRange' for e in tree.iter()):
        return 'Editor ready'
    return next((e.get('label', '') for e in tree.iter() if e.get('name') == 'analysisStatus'), '')

def capture(name):
    source = wda.command('/source')
    (output / (name + '.xml')).write_text(source, encoding='utf-8')
    with urllib.request.urlopen('http://127.0.0.1:19100/', timeout=15) as response:
        buffer = b''
        for _ in range(200):
            buffer += response.read(16384)
            start, end = buffer.find(b'\xff\xd8'), buffer.find(b'\xff\xd9')
            if 0 <= start < end:
                (output / (name + '.jpg')).write_bytes(buffer[start:end+2])
                break
        else: raise RuntimeError('No complete MJPEG frame')

try:
    wda.connect()
    wda.activate('com.vafrederico.VolleySplice')
    if any(e.get('name') == 'queueDone' for e in ET.fromstring(wda.command('/source')).iter()):
        click('queueDone')
    if any(e.get('name') == 'backToProjects' for e in ET.fromstring(wda.command('/source')).iter()):
        click('backToProjects')
    for orientation in ([] if args.export_only else ['PORTRAIT', 'LANDSCAPE']):
        wda.command('/orientation', {'orientation': orientation})
        assert wda.command('/orientation') == orientation
        capture(orientation.lower())
    if not any(e.get('name') == 'goldenCheck' for e in ET.fromstring(wda.command('/source')).iter()):
        click('developerChecks')
    if not args.export_only:
        click('goldenCheck')
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            text = status()
            if text.startswith('PASS:'): break
            if text == 'Operation failed': raise RuntimeError(wda.command('/source'))
            time.sleep(0.5)
        else: raise TimeoutError('Golden inference check did not finish')
        print(text, flush=True)
        capture('golden-pass')
    if args.media_check:
        click('mediaCheck')
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            text = status()
            if '4/4 fixtures passed' in text:
                assert 'all four display rotations observed: yes' in text, text
                break
            if 'FAIL ' in text: raise RuntimeError(text)
            time.sleep(1)
        else: raise TimeoutError('Android rotation fixture check timed out')
        print(text, flush=True)
        capture('rotation-fixtures')
    if args.recording:
        click('refreshRecordings')
        click('recording-' + args.recording)
        deadline = time.monotonic() + 30
        while status() != 'Set game bounds, then analyze':
            if time.monotonic() > deadline: raise TimeoutError('Recording did not open')
            time.sleep(0.5)
        capture('recording-loaded')
        if args.export_check:
            if not any(e.get('name') == 'videoExportCheck' for e in ET.fromstring(wda.command('/source')).iter()):
                click('developerChecks')
            click('videoExportCheck')
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                text = status()
                print(text, flush=True)
                if text.startswith('PASS: native export'): break
                if text == 'Operation failed': raise RuntimeError(wda.command('/source'))
                time.sleep(3)
            else: raise TimeoutError('Native video export check timed out')
            capture('native-export-check')
        if args.analyze:
            click('analyzeRecording')
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                text = status()
                print(text, flush=True)
                if 'rallies •' in text or text == 'Editor ready': break
                if text == 'Operation failed': raise RuntimeError(wda.command('/source'))
                time.sleep(5)
            else: raise TimeoutError('Native analysis timed out')
            capture('native-analysis')
            if text == 'Editor ready':
                for orientation in ['PORTRAIT', 'LANDSCAPE']:
                    wda.command('/orientation', {'orientation': orientation})
                    assert wda.command('/orientation') == orientation
                    capture('editor-' + orientation.lower())
finally:
    if wda.session:
        wda.connect()
        wda.activate(module.PARKING)
        assert 'Idle timer: disabled' in wda.command('/source')
        print('Parking foreground; idle timer disabled', flush=True)

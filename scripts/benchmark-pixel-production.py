#!/usr/bin/env python3
"""Run the installed debug native pipeline on one MediaStore video; save each raw result."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adb', required=True)
    parser.add_argument('--serial', required=True)
    parser.add_argument('--video-name', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=4)
    parser.add_argument('--seconds', type=int, default=120)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    package = 'com.volleycut.nativeanalysis.debug'

    def adb(*command, check=True):
        return subprocess.run([args.adb, '-s', args.serial, *command], check=check,
                              capture_output=True, text=True, timeout=30).stdout.strip()

    catalog = adb('shell', 'content', 'query', '--uri', 'content://media/external/video/media',
                  '--projection', '_id:_display_name')
    ids = [re.search(r'_id=(\d+)', line).group(1) for line in catalog.splitlines()
           if line.split('_display_name=', 1)[-1] == args.video_name]
    if len(ids) != 1:
        raise RuntimeError(f'Expected one MediaStore match, found {len(ids)}')
    (args.output/'installed-package.txt').write_text(adb('shell', 'dumpsys', 'package', package), encoding='utf8')
    for i in range(args.runs):
        run_id = uuid.uuid4().hex
        (args.output/f'conditions-{i}.txt').write_text(adb('shell', 'dumpsys', 'battery') + '\n' +
            adb('shell', 'dumpsys', 'thermalservice'), encoding='utf8')
        adb('shell', 'am', 'force-stop', package)
        adb('shell', 'am', 'start', '-W', '-n', package + '/com.volleycut.nativeanalysis.MainActivity', '-a', 'android.intent.action.VIEW',
            '-d', 'content://media/external/video/media/' + ids[0], '-f', '0x1',
            '--ez', 'benchmark_auto_run', 'true', '--es', 'benchmark_run_id', run_id,
            '--ei', 'benchmark_source_frame_limit', '1000000',
            '--ei', 'benchmark_codec_operating_rate', '240', '--ei', 'benchmark_codec_priority', '1',
            '--es', 'benchmark_feature_cache_mode', 'bypass', '--es', 'benchmark_stages', 'video,audio,inference',
            '--es', 'benchmark_audio_decoder_mode', 'auto',
            '--ei', 'benchmark_duration_milliseconds', str(args.seconds * 1000))
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            text = adb('exec-out', 'run-as', package, 'cat', 'files/benchmark-result.json', check=False)
            try:
                result = json.loads(text)
            except json.JSONDecodeError:
                time.sleep(1)
                continue
            if result.get('benchmarkRunId') == run_id and result.get('benchmarkStatus') in ('complete', 'failed', 'cancelled'):
                (args.output/f'run-{i}.json').write_text(json.dumps(result, indent=2), encoding='utf8')
                if result['benchmarkStatus'] != 'complete':
                    raise RuntimeError(result)
                print(json.dumps({'run': i, 'warmup': i == 0, 'totalMilliseconds': result.get('totalMilliseconds'),
                                  'stages': result.get('stageMilliseconds')}), flush=True)
                break
            time.sleep(1)
        else:
            raise TimeoutError(f'No matching result for {run_id}')


if __name__ == '__main__':
    main()

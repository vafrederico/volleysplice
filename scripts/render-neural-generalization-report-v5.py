#!/usr/bin/env python3
"""Visible deployment limits over the unchanged, lossless V4 report data."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

V4_SHA = '31c9acee117ef58ae3b736b138cc95f2dd3ac96d85aedd2a748616bc8956a0cf'
NOTICE = ('These are offline accuracy experiments, not phone qualification. '
    'DINO FP16 used CUDA; mixed INT8 used native CPU and failed the tested desktop-browser '
    'numerical parity check, so its accuracy is not established in a browser. '
    'No physical-phone latency, peak-memory or thermal measurements were made; prior graph '
    'checks do not qualify every new checkpoint or the complete video pipeline.')


def sha(value): return hashlib.sha256(value).hexdigest()


def load_v4():
    path = Path(__file__).with_name('render-neural-generalization-report-v4.py')
    if sha(path.read_bytes()) != V4_SHA: raise ValueError('Frozen V4 renderer changed')
    spec = importlib.util.spec_from_file_location('frozen_scoped_report_v4', path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def render(raw, renderer_sha):
    previous = load_v4(); html, receipt = previous.render(raw, renderer_sha)
    replace = previous.load_v3().load_v2().replace_once
    anchor = '<p class="subtle" id="precisionWarning">'
    html = replace(html, anchor,
        '<div id="runtimeQualification" class="notice" role="note">'+NOTICE+'</div>'+anchor)
    html = replace(html, 'presentationVersion:4,', 'presentationVersion:5,')
    html = replace(html, 'Scoped task view v4 / matched variants v3 / lossless columnar transport v2',
        'Deployment limits v5 / scoped task view v4 / matched variants v3 / lossless columnar transport v2')
    receipt = {**receipt, 'kind': 'deployment-disclosure-report-view-v5',
        'htmlBytes': len(html.encode()), 'v4RendererSha256': V4_SHA,
        'runtimeQualificationNotice': NOTICE,
        'presentationPolicyV5': 'One always-visible deployment qualification note; no numerical, membership, filter, transport or download changes.'}
    return html, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('input', 'output', 'receipt'): parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--allow-synthetic', action='store_true'); args = parser.parse_args()
    raw = args.input.read_bytes()
    if json.loads(raw).get('metadata', {}).get('syntheticFixture') and not args.allow_synthetic:
        raise ValueError('Synthetic fixture requires explicit flag')
    html, receipt = render(raw, sha(Path(__file__).read_bytes()))
    with args.output.open('x', encoding='utf-8', newline='\n') as stream: stream.write(html)
    receipt['output'] = {'path': str(args.output), 'sha256': sha(args.output.read_bytes())}
    with args.receipt.open('x') as stream: json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__': main()

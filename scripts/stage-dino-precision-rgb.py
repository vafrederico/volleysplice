#!/usr/bin/env python3
"""Accelerate exact ordinal sampling while proving pixel parity with seeking."""
from pathlib import Path
import argparse
import importlib.util
import json
import time
import numpy as np

HERE=Path(__file__).resolve()
SPEC=importlib.util.spec_from_file_location('precision',HERE.with_name('evaluate-dino-precision.py'))
precision=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(precision)


def main():
    import cv2
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=precision.DEFAULT)
    args=parser.parse_args()
    root=args.root
    precision.ensure_nas(root)
    parent=precision.verify(root)
    dest=root/'sequential-rgb-amendment-v1'
    dest.mkdir(parents=True,exist_ok=True)
    if not (dest/'protocol.json').exists():
        precision.write(dest/'protocol.json',{'kind':'same-pixel-sequential-ordinal-io-amendment',
            'parent':precision.ident(root/'protocol.json'),'source':precision.ident(HERE),
            'reason':'Historical per-sample seeking dominates encoder extraction; only IO mechanism changes.',
            'sampleContract':'Same min(lastFrame,max(0,round(timestamp*fps))) frame ordinal, same ROI, letterbox, normalization.',
            'gate':'All previously cached RGB frames compared byte-for-byte; first and last frame of every remaining 128-frame chunk compared byte-for-byte with historical seek decoder before publishing.',
            'labelsUsed':False,'thresholdsChanged':False})
    protocol=precision.read(dest/'protocol.json')
    if precision.sha(HERE)!=protocol['source']['sha256']:
        raise ValueError('Frozen staging code changed')
    reports=[]
    for item in parent['records']:
        src,entry=item['source'],item['dino']
        report_path=dest/(src['id']+'.json')
        if report_path.exists():
            reports.append(precision.read(report_path)); continue
        with np.load(entry['dinoPath'],allow_pickle=False) as n: ts=n['timestamps']
        cap=cv2.VideoCapture(src['video'])
        if not cap.isOpened(): raise ValueError('Cannot decode source')
        fps,total=cap.get(cv2.CAP_PROP_FPS),int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        roi=tuple(src['roi'][k] for k in ('x','y','width','height')) if src.get('roi') else None
        cursor=-1
        refs=[]; compared=0; started=time.perf_counter()
        try:
            for left in range(0,len(ts),128):
                right=min(left+128,len(ts)); rgbs=[]
                for i in range(left,right):
                    target=min(max(0,total-1),max(0,int(round(float(ts[i])*fps))))
                    while cursor<target:
                        if not cap.grab(): raise ValueError('Sequential decode ended early')
                        cursor+=1
                    ok,frame=cap.retrieve()
                    if not ok: raise ValueError('Cannot retrieve target ordinal')
                    rgbs.append(precision.letterbox_frame(precision._crop_frame(frame,roi),336))
                rgb=np.stack(rgbs)
                path=root/'rgb'/src['id']/f'{left:06d}.npy'
                if path.exists():
                    old=np.load(path,allow_pickle=False)
                    if not np.array_equal(rgb,old): raise ValueError('Sequential pixels differ from existing seek cache')
                    compared+=len(rgb)
                else:
                    sought=precision.decode_rgb(src,[left,right-1],ts)
                    if not np.array_equal(rgb[[0,-1]],sought): raise ValueError('Sequential chunk boundary pixels differ from seek decoder')
                    compared+=2
                    path.parent.mkdir(parents=True,exist_ok=True)
                    np.save(path,rgb)
                refs.append(precision.ident(path))
                print(json.dumps({'recording':src['id'],'rgbTicks':right,'total':len(ts)}),flush=True)
        finally:
            cap.release()
        report={'recordingId':src['id'],'source':precision.ident(src['video']),
                'rgbChunks':refs,'samples':len(ts),'seekParityFrames':compared,
                'allComparisonsBitExact':True,'wallSeconds':time.perf_counter()-started}
        precision.write(report_path,report); reports.append(report)
    precision.write(dest/'report.json',{'protocol':precision.ident(dest/'protocol.json'),
                    'records':reports,'passed':True})


if __name__=='__main__': main()

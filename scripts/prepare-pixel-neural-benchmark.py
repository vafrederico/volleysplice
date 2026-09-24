#!/usr/bin/env python3
"""Prepare frozen encoder/TCN portability cases, without fitting models.

Runs in the neural Python environment. All generated data goes to --output.
Encoder probes use real frames; temporal probes are synthetic shape/portability
checks and must not be reported as end-to-end video accuracy or performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.quantization import quantize_dynamic, quantize_static, CalibrationDataReader, QuantFormat, QuantType
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.mobile_visual_features import preprocess_frame, regional_pool_weights
from analysis.dinov2_embeddings import preprocess_frames
from analysis.transfer_temporal_model import model_for as dino_model
from analysis.recognition_temporal_model import RecognitionConfig, model_for as mobile_model


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study-root', type=Path, required=True)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    roi = (.03, .12, .94, .86)
    cap = cv2.VideoCapture(str(a.video)); cap.set(cv2.CAP_PROP_POS_MSEC, 30000)
    ok, frame = cap.read(); cap.release()
    if not ok: raise RuntimeError('Cannot read pilot frame')
    image, box, _ = preprocess_frame(frame, roi)
    models = [
        ('mobile-encoder', a.study_root/'2026-09-22-recognition/runtime-cpu-v1/mobile_image_encoder.onnx',
         {'image': image[None], 'pool_weights': regional_pool_weights(box[None], 7, 7)}, 'real clip frame at 30s'),
        ('dino-encoder', a.study_root/'2026-09-23-recall-distillation/dino-precision-v1/encoder-fp32.onnx',
         {'image': preprocess_frames([frame], [roi])}, 'real clip frame at 30s')]
    fits = a.study_root/'2026-09-23-recall-sweep-generalization/randomized-variants-v1/fits'
    selections = [('dino-tcn', fits/'original-medium/dino-tcn/split-3407', 30),
                  ('mobile-tcn', fits/'expanded-large/mobile-tcn/split-3407', 15)]
    provenance = []
    for name, folder, epoch in selections:
        weights = folder/f'temporal/weights-{epoch}.npz'
        done = json.loads((folder/'temporal/completed.json').read_text())
        assert digest(weights) == done['artifacts'][weights.name]
        if name == 'dino-tcn': model = dino_model('dino_tcn')
        else:
            fit = json.loads((folder/'fit-result.json').read_text())
            model = mobile_model(RecognitionConfig(**fit['config']))
        with np.load(weights, allow_pickle=False) as archive:
            model.load_state_dict({k[7:]: torch.from_numpy(archive[k].copy()) for k in archive.files if k.startswith('model::')}, strict=True)
        model.eval()
        dimension = 3944 if name == 'dino-tcn' else model.config.input_dimension
        values = np.random.default_rng(3407).normal(0, .2, (1,252,dimension)).astype('float32')
        path = a.output/f'{name}-fp32.onnx'
        torch.onnx.export(model, torch.from_numpy(values), str(path), input_names=['features'], output_names=['logits'],
                          opset_version=17, dynamo=False)
        with torch.no_grad(): expected = model(torch.from_numpy(values)).numpy()
        session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        np.testing.assert_allclose(session.run(None, {'features':values})[0], expected, atol=1e-5, rtol=1e-4)
        models.append((name, path, {'features': values}, 'synthetic standardized tensor; 252 ticks including context'))
        provenance.append({'name':name,'weights':str(weights),'sha256':digest(weights),'epoch':epoch,'seed':3407})
    cases, records = [], []
    for name, source, inputs, scope in models:
        fp32 = a.output/f'{name}-fp32.onnx'
        if source != fp32: shutil.copyfile(source, fp32)
        input_rows = []
        for key, value in inputs.items():
            file = f'{name}-{key}.f32'; value.astype('<f4').tofile(a.output/file)
            input_rows.append({'name':key,'shape':list(value.shape),'dtype':'float32','file':file})
        expected = ort.InferenceSession(str(fp32), providers=['CPUExecutionProvider']).run(None, inputs)[0]
        expected.astype('<f4').tofile(a.output/f'{name}-expected.f32')
        fp16 = a.output/f'{name}-fp16.onnx'
        onnx.save(convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True), str(fp16))
        int8 = a.output/f'{name}-int8.onnx'
        if name == 'dino-encoder':
            shutil.copyfile(a.study_root/'2026-09-23-recall-distillation/dino-precision-v1/encoder-dynamic-int8.onnx', int8)
        elif name == 'mobile-encoder':
            calibration = []
            capture = cv2.VideoCapture(str(a.video))
            for timestamp in np.linspace(0, 119, 32):
                capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp*1000))
                success, calibration_frame = capture.read()
                if not success: raise RuntimeError('Missing quantization calibration frame')
                pixels, content_box, _ = preprocess_frame(calibration_frame, roi)
                calibration.append({'image':pixels[None], 'pool_weights':regional_pool_weights(content_box[None],7,7)})
            capture.release()
            class Reader(CalibrationDataReader):
                def __init__(self): self.rows = iter(calibration)
                def get_next(self): return next(self.rows, None)
            quantize_static(str(fp32), str(int8), Reader(), quant_format=QuantFormat.QDQ,
                            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                            per_channel=True, op_types_to_quantize=['Conv','MatMul','Gemm'])
        else:
            # Deliberately labelled mixed: Conv is NOT quantized by this probe.
            quantize_dynamic(str(fp32), str(int8), per_channel=True, weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul','Gemm'])
        for precision, path in [('fp32',fp32),('fp16',fp16),('int8',int8)]:
            record = {'name':name,'precision':precision,'file':path.name,'sha256':digest(path),'bytes':path.stat().st_size,'scope':scope,
                      'precisionNote':('static QDQ INT8 Conv/MatMul/Gemm; 32 unlabelled pilot calibration frames' if name=='mobile-encoder'
                                       else 'mixed dynamic INT8 MatMul/Gemm; other ops float') if precision=='int8' else 'float32 input/output; internal '+precision}
            try:
                actual = ort.InferenceSession(str(path),providers=['CPUExecutionProvider']).run(None,inputs)[0]
                record.update(maxAbsoluteError=float(np.max(np.abs(actual-expected))),rmse=float(np.sqrt(np.mean((actual-expected)**2))),finite=bool(np.isfinite(actual).all()))
            except Exception as error: record['desktopError'] = str(error)
            records.append(record)
            for provider in ['cpu','nnapi']:
                cases.append({'id':name+'-'+precision+'-'+provider,'model':path.name,'inputs':input_rows,'provider':provider,'warmup':2,'runs':5,'scope':scope})
        print(name+' prepared',flush=True)
    (a.output/'plan.json').write_text(json.dumps({'runId':'pixel-native-portability-v1','cases':cases},indent=2))
    (a.output/'qualification.json').write_text(json.dumps({'provenance':provenance,'graphs':records,'roi':roi,'video':str(a.video)},indent=2))


if __name__ == '__main__': main()

"""Isolate native-versus-desktop feature drift with frozen temporal inference.

This is a diagnostic, not a new model, calibration, or selection run. All private
locations are supplied explicitly or resolved from the external ledger.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.neural_development import decode
from analysis import neural_generalization_inputs as inputs
from analysis.neural_recognition_fit import standardized
from analysis.recognition_temporal_model import RecognitionConfig


def read(path):
    return json.loads(Path(path).read_text())


def predict(session, values):
    result = np.zeros((len(values), 4), np.float32)
    for core in range(0, len(values), 128):
        end = min(len(values), core + 128)
        left, right = max(0, core - 62), min(len(values), end + 62)
        logits = session.run(None, {'features': values[None, left:right]})[0][0]
        result[core:end] = (1 / (1 + np.exp(-logits)))[core-left:end-left]
    return result


def analyze_case(session, values, times, duration, decoder, interval):
    p = predict(session, values)
    example = SimpleNamespace(times=times, valid=np.ones(len(times), bool), duration=duration)
    ranges = decode(example, p, decoder)
    start, end = interval
    local = (times >= start) & (times <= end)
    width = min(len(times), max(1, round(decoder['smoothing']*4)))
    smooth = np.convolve(np.pad(p[:, 0], (width//2, (width-1)//2), mode='edge'), np.ones(width)/width, mode='valid')
    return {'rallyCount': len(ranges),
            'nearbyRallies': [{'start':r.start, 'end':r.end} for r in ranges if r.end > start-5 and r.start < end+5],
            'maximumLiveProbabilityInHumanRally': float(p[local, 0].max()),
            'meanLiveProbabilityInHumanRally': float(p[local, 0].mean()),
            'maximumSmoothedLiveInHumanRally': float(smooth[local].max()),
            'humanCoreSecondsRetainedWithoutPadding': float(sum(max(0, min(end,r.end)-max(start,r.start)) for r in ranges))}, p


def run_swaps(session, native, desktop, times, duration, decoder, interval):
    blocks = {'av': slice(0,104), 'embeddings':slice(104, native.shape[1]-8), 'quality':slice(native.shape[1]-8,None)}
    output, probabilities = {}, {}
    for count in range(4):
        for replaced in itertools.combinations(blocks, count):
            key = 'native' if not replaced else 'native_with_desktop_' + '_'.join(replaced)
            value = native.copy()
            for block in replaced:
                value[:, blocks[block]] = desktop[:, blocks[block]]
            output[key], probabilities[key] = analyze_case(session, value, times, duration, decoder, interval)
    return output, probabilities


def run_narrow_swaps(session, native, desktop, times, duration, decoder, interval, names):
    groups = {
        'av_video': [i for i,name in enumerate(names) if not name.startswith('audio_')],
        'av_audio': [i for i,name in enumerate(names) if name.startswith('audio_')],
        'av_focus_blur': [names.index(name) for name in ('focus_quality','blur_probability')],
        'av_visibility': [names.index('visibility_quality')],
        'av_focus_blur_visibility': [names.index(name) for name in ('focus_quality','blur_probability','visibility_quality')],
    }
    quality = ('content_fraction','luminance_mean','luminance_std','laplacian_variance',
               'clipped_pixel_fraction','selected_pts_offset_seconds','age','available')
    groups.update({'quality_'+name: [native.shape[1]-8+i] for i,name in enumerate(quality)})
    output = {}
    for name, indexes in groups.items():
        values = native.copy()
        values[:,indexes] = desktop[:,indexes]
        output[name], _ = analyze_case(session,values,times,duration,decoder,interval)
        output[name]['meanAbsoluteDifference'] = float(abs(native[:,indexes]-desktop[:,indexes]).mean())
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ui-index', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--graphs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--recording-index', default='recording-044')
    p.add_argument('--start', type=float, default=562.4958444444444)
    p.add_argument('--end', type=float, default=568.7457444444444)
    a = p.parse_args()
    import os
    platform = 'WINDOWS' if os.name == 'nt' else 'POSIX'
    ledger = read(os.environ.get('VOLLEYCUT_PRIVATE_LEDGER_'+platform) or os.environ['VOLLEYCUT_PRIVATE_LEDGER'])
    rid = next(k for k,v in ledger['originalToAlias'].items() if v == a.recording_index)
    index = read(a.ui_index)
    entry = next(r for r in index['recordings'] if r['id'] == rid)
    recording = read(a.ui_index.parent / entry['file'])['recordings'][0]
    ui = next(r for r in recording['references'] if r['modelId'] == 'neural-mobile-tcn-fp32-high-recall')
    provenance = ui['research']['provenance']
    receipt = read(Path(provenance['source']).with_suffix('.json'))
    config = read(a.graphs / 'mobile-pipeline.json')
    assert config['weightsSha256'] == receipt['weights'][str(config['epoch'])]['sha256']
    assert hashlib.sha256(Path(receipt['weights'][str(config['epoch'])]['path']).read_bytes()).hexdigest() == config['weightsSha256']
    assert config['decoder'] == provenance['decoder'] and config['epoch'] == provenance['epoch']
    e = inputs.load_inference_examples(receipt['manifest']['path'], receipt['features']['path'], family='mobile', recording_ids=[rid])[0]
    desktop = standardized(e, np.asarray(config['mean'], np.float32), np.asarray(config['scale'], np.float32), RecognitionConfig(family='mobile',head='tcn',scalar_dimension=8))
    row = next(r for r in read(a.native / 'result.json')['results'] if r['id'].startswith('mobile-'))
    times = np.asarray(row['neural']['times'])
    np.testing.assert_array_equal(times, e.times)
    native = np.fromfile(a.native / (row['id']+'-features.f32'), dtype='<f4').reshape(desktop.shape)
    saved_native = np.fromfile(a.native / (row['id']+'-probabilities.f32'), dtype='<f4').reshape(len(times),4)
    with np.load(provenance['source']) as z:
        saved_desktop = z[f"epoch_{config['epoch']}"]
    session_options = ort.SessionOptions();session_options.intra_op_num_threads=2
    session = ort.InferenceSession(str(a.graphs/'mobile-tcn-dynamic-fp32.onnx'), sess_options=session_options, providers=['CPUExecutionProvider'])
    cases, probabilities = run_swaps(session,native,desktop,times,e.duration,config['decoder'],(a.start,a.end))
    np.testing.assert_allclose(probabilities['native'], saved_native, atol=3e-5,rtol=3e-4)
    np.testing.assert_allclose(probabilities['native_with_desktop_av_embeddings_quality'], saved_desktop,atol=3e-5,rtol=3e-4)
    local = (times>=a.start-16)&(times<=a.end+16)
    from analysis.features import feature_names
    from analysis.config import FeatureConfig, NOISE_NORMALIZED_AUDIO_FEATURE_SET
    names = feature_names(FeatureConfig(audio_feature_set=NOISE_NORMALIZED_AUDIO_FEATURE_SET))
    diff = abs(native[local]-desktop[local])
    worst = np.argsort(diff[:,:104].mean(0))[::-1][:20]
    output = {'recordingIndex':a.recording_index,'humanRally':{'start':a.start,'end':a.end},
              'modelId':ui['modelId'],'weightsSha256':config['weightsSha256'],'epoch':config['epoch'],
              'decoder':config['decoder'],'nativeAndDesktopCheckpointMatch':True,
              'nativeProbabilityParityMaxError':float(abs(probabilities['native']-saved_native).max()),
              'desktopProbabilityParityMaxError':float(abs(probabilities['native_with_desktop_av_embeddings_quality']-saved_desktop).max()),
              'cases':cases,'largestStandardizedAvDifferencesNearRally':[{'feature':names[i],'meanAbsoluteDifference':float(diff[:,i].mean()),'nativeMean':float(native[local,i].mean()),'desktopMean':float(desktop[local,i].mean())} for i in worst],
              'scope':'Counterfactual feature substitutions in one frozen temporal model; no retraining or recalibration. Not a deployment accuracy qualification.'}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'feature-counterfactuals.json').write_text(json.dumps(output,indent=2))
    narrow = run_narrow_swaps(session,native,desktop,times,e.duration,config['decoder'],(a.start,a.end),names)
    (a.output/'narrow-feature-counterfactuals.json').write_text(json.dumps(narrow,indent=2))
    np.savez_compressed(a.output/'diagnostic-tensors.private.npz', native_features=native, desktop_features=desktop, times=times,**probabilities)
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    main()

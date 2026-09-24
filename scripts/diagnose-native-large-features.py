"""Compare frozen native and desktop Large inputs without fitting or calibration."""
import importlib.util
import json
from pathlib import Path
import os
import sys
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.private_ledger import private_value
from analysis import neural_generalization_inputs as inputs
from analysis.neural_recognition_fit import standardized


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    diagnostic = module('native_feature_diagnostic', 'diagnose-native-neural-features.py')
    experiment = module('large_feature_experiment', 'experiment-mobile-large.py')
    root = Path(private_value('private-reference-0211'))
    native_root = Path(private_value('private-reference-0213'))
    destination = Path(private_value('private-reference-0217')) / 'large'
    index_path = Path(private_value('private-reference-0216'))
    ledger = diagnostic.read(os.environ.get('VOLLEYCUT_PRIVATE_LEDGER_POSIX') or os.environ['VOLLEYCUT_PRIVATE_LEDGER'])
    recording_index = 'recording-044'
    rid = next(k for k, v in ledger['originalToAlias'].items() if v == recording_index)
    index = diagnostic.read(index_path)
    entry = next(r for r in index['recordings'] if r['id'] == rid)
    recording = diagnostic.read(index_path.parent / entry['file'])['recordings'][0]
    ui = next(r for r in recording['references'] if r['modelId'] == 'neural-mobile-large-tcn-fp32-high-recall')
    provenance = ui['research']['provenance']
    config = diagnostic.read(root / 'graphs/mobile-large-pipeline.json')
    source = root / 'fits' / provenance['variant'] / f"split-{provenance['seed']}" / 'inference' / (rid + '.npz')
    receipt = diagnostic.read(source.with_suffix('.json'))
    assert receipt['output']['sha256'] == provenance['sourceSha256']
    inputs.verified(receipt['output'])
    assert config['weightsSha256'] == receipt['weights'][str(config['epoch'])]['sha256']
    assert config['decoder'] == provenance['decoder'] and config['epoch'] == provenance['epoch']
    a = SimpleNamespace(output=root)
    plan = experiment.load_plan(a)
    e, = inputs.load_inference_examples(inputs.verified(plan['inferenceManifest']), inputs.verified(plan['features']), family='av', recording_ids=[rid])
    e = experiment.attach(e, a)
    desktop = standardized(e, np.asarray(config['mean'], np.float32), np.asarray(config['scale'], np.float32), experiment.CONFIG)
    row, = diagnostic.read(native_root / 'result.json')['results']
    assert row['id'].startswith('mobile-large-')
    times = np.asarray(row['neural']['times'])
    np.testing.assert_array_equal(times, e.times)
    native = np.fromfile(native_root / (row['id'] + '-features.f32'), dtype='<f4').reshape(desktop.shape)
    saved_native = np.fromfile(native_root / (row['id'] + '-probabilities.f32'), dtype='<f4').reshape(len(times), 4)
    with np.load(source) as z:
        saved_desktop = z[f"epoch_{config['epoch']}"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(root / 'graphs/mobile-large-tcn-dynamic-fp32.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
    intervals = [(562.4958444444444, 568.7457444444444), (577.8681777777778, 582.3674666666667)]
    rallies = []
    for number, interval in enumerate(intervals, 16):
        cases, probabilities = diagnostic.run_swaps(session, native, desktop, times, e.duration, config['decoder'], interval)
        np.testing.assert_allclose(probabilities['native'], saved_native, atol=3e-5, rtol=3e-4)
        np.testing.assert_allclose(probabilities['native_with_desktop_av_embeddings_quality'], saved_desktop, atol=3e-5, rtol=3e-4)
        rallies.append(dict(humanRallyNumber=number, start=interval[0], end=interval[1], cases=cases))
    output = dict(recordingIndex=recording_index, modelId=ui['modelId'], weightsSha256=config['weightsSha256'], epoch=config['epoch'], decoder=config['decoder'], nativeAndDesktopCheckpointMatch=True,
                  nativeProbabilityParityMaxError=float(abs(probabilities['native']-saved_native).max()),
                  desktopProbabilityParityMaxError=float(abs(probabilities['native_with_desktop_av_embeddings_quality']-saved_desktop).max()),
                  rallies=rallies, scope='Frozen counterfactual feature substitutions; no fitting, calibration, or accuracy qualification.')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'feature-counterfactuals.json').write_text(json.dumps(output, indent=2))
    np.savez_compressed(destination / 'diagnostic-tensors.private.npz', native_features=native, desktop_features=desktop, times=times, **probabilities)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()

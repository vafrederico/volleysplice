#!/usr/bin/env python3
"""Bounded same-pixel CPU replay of the unchanged mixed-INT8 DINO graph."""
from pathlib import Path
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import neural_generalization_feature_inference as encoders
from analysis.neural_context_development import identity, read, write_immutable


def main():
    root = encoders.stage.environment(encoders.stage.ROOT)
    plan = encoders.verify_plan(root)
    gate = read(root/'encoder-engineering.json')
    encoders.require(gate['passed'] is True, 'Shared encoder engineering gate absent')
    staging = encoders.stage.verify_plan(root/'stage-plan.json')
    source = next(r for r in staging['records'] if r['id'] == gate['recordingId'])
    receipt = read(root/'staged'/source['id']/'receipt.json')
    registration = {'source': identity(__file__), 'encoderPlan': identity(root/'encoder-plan.json'),
                    'encoderEngineering': identity(root/'encoder-engineering.json'),
                    'stagingReceipt': identity(root/'staged'/source['id']/'receipt.json'),
                    'frameIndexes': gate['dinoFrameIndexes'], 'labelsUsed': False,
                    'selectionAllowed': False, 'batch': 1, 'threads': 2,
                    'concurrency': 'Historical fitting and CPU video staging active; timing is descriptive only.',
                    'browserQualified': False, 'physicalPhoneQualified': False}
    write_immutable(root/'int8-wrapper-plan.json', registration)
    rgb = []
    for tick in registration['frameIndexes']:
        block = np.load(encoders.verified(receipt['dinoRgbChunks'][tick//128]), mmap_mode='r')
        rgb.append(block[tick % 128])
    p = encoders.precision_module()
    images = p.normalized(np.stack(rgb))
    runtime = p.session(encoders.verified(plan['quantizedGraph']), threads=2)
    runtime.run(None, {'image': images[:1]})
    started = time.perf_counter()
    output = np.concatenate([runtime.run(None, {'image': image[None]})[0] for image in images])
    seconds = time.perf_counter()-started
    replay = np.concatenate([runtime.run(None, {'image': image[None]})[0] for image in images[:4]])
    encoders.require(output.shape == (32, 10, 384) and np.isfinite(output).all()
                     and np.array_equal(output[:4], replay), 'INT8 wrapper shape/finite/repeatability check failed')
    with np.load(encoders.verified(source['reuse']['dino']['fp32']), allow_pickle=False) as old:
        reference = old['tokens'][registration['frameIndexes']]
    report = {'plan': identity(root/'int8-wrapper-plan.json'), 'passed': True,
              'referenceDrift': p.errors(output, reference), 'secondsFor32FramesUnderConcurrency': seconds,
              'framesPerSecondUnderConcurrency': 32/seconds, 'runtime': runtime.get_providers(),
              'repeatabilityFirst4BitExact': True, 'quantizedGraphUnchanged': True,
              'precisionOutcomeSelection': False, 'browserQualified': False, 'physicalPhoneQualified': False}
    write_immutable(root/'int8-wrapper-engineering.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Execute the authorized bounded encoder gate with an explicit GPU memory cap."""
from pathlib import Path
import json
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import neural_generalization_feature_inference as encoders
from analysis.neural_context_development import identity, write_immutable


def snapshot():
    return subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu',
                           '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True).stdout.strip()


def main():
    root = encoders.stage.environment(encoders.stage.ROOT)
    registration = {'encoderPlan': identity(root/'encoder-plan.json'), 'source': identity(__file__),
                    'authorization': 'Parent bounded overlap exception:32 DINO and32 Mobile frames; sequential forwards; added allocation<=2GiB; require>=4GiB free before forward.',
                    'timingConclusionsAllowed': False, 'otherGpuWork': 'Historical registered outer refits continue.'}
    write_immutable(root/'overlap-qualification-plan.json', registration)
    before = snapshot()
    import torch
    free, total = torch.cuda.mem_get_info()
    encoders.require(free >= 4*1024**3, 'Less than4GiB free before bounded qualification')
    # Reserve headroom for CUDA context allocations outside the PyTorch allocator.
    torch.cuda.set_per_process_memory_fraction(1.75*1024**3/total)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = encoders.qualify(root)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_reserved()
    encoders.require(peak <= 1.75*1024**3, 'Bounded allocator reservation exceeded')
    receipt = {'plan': identity(root/'overlap-qualification-plan.json'), 'passed': result['passed'],
               'engineering': identity(root/'encoder-engineering.json'), 'gpuBefore': before,
               'gpuAfter': snapshot(), 'allocatorPeakReservedBytes': peak,
               'allocatorPeakAllocatedBytes': torch.cuda.max_memory_allocated(),
               'wallSecondsUnderConcurrency': time.perf_counter()-started,
               'timingConclusionsAllowed': False}
    write_immutable(root/'overlap-qualification-execution.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

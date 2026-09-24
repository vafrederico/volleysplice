#!/usr/bin/env python3
"""Release only our frozen encoder queue at a verified completed-record boundary."""
from pathlib import Path
import argparse
import json
import os
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.private_ledger import private_value
from analysis.neural_context_development import identity, read, write_immutable
from analysis.neural_generalization_inputs import require

ROOT = Path(private_value('private-reference-0089'))


def main(pid, name):
    require(name.isalnum(), 'Use a simple immutable pause name')
    folder = ROOT/'gpu-pauses-v1'/name
    proc = Path('/proc')/str(pid)
    command = proc.joinpath('cmdline').read_bytes().split(b'\0')[:-1]
    require(command[-2:] == [b'scripts/run-generalization-gpu-queue.py', b'run']
            and proc.stat().st_uid == os.getuid(), 'PID is not our registered encoder queue')
    plan = {'source': identity(__file__), 'pid': pid, 'command': [v.decode() for v in command],
            'queuePlan': identity(ROOT/'gpu-queue-plan.json'),
            'executionGrant': identity(ROOT.parent/'execution-resource-amendment-v2.json'),
            'rule': 'Stop this process group briefly; terminate only when every started GPU recording has complete immutable receipts. Otherwise resume unchanged.'}
    write_immutable(folder/'plan.json', plan)
    stopped = False
    try:
        os.kill(pid, signal.SIGSTOP)
        stopped = True
        for _ in range(200):
            if '\nState:\tT' in proc.joinpath('status').read_text():
                break
            time.sleep(.01)
        else:
            raise ValueError('Encoder queue did not reach a stopped state')
        completed = []
        pending = []
        for arm in ('fp32', 'fp16', 'mobile'):
            for directory in sorted((ROOT/'encoders'/arm).glob('*')):
                if not directory.is_dir():
                    continue
                receipt = directory/'receipt.json'
                if not receipt.exists():
                    pending.append(str(directory))
                    continue
                value = read(receipt)
                require(value['encoderPlan'] == identity(ROOT/'encoder-plan.json'), 'Receipt encoder binding differs')
                completed.append(identity(receipt))
                if arm == 'fp32' and not (ROOT/'encoders/mobile'/directory.name/'receipt.json').exists():
                    pending.append(str(ROOT/'encoders/mobile'/directory.name))
        if pending:
            write_immutable(folder/'result.json', {'plan': identity(folder/'plan.json'), 'released': False,
                                                  'resumedUnchanged': True, 'incompleteRecords': pending})
            print(json.dumps({'released': False, 'incompleteRecords': pending}), flush=True)
            return
        write_immutable(folder/'boundary.json', {'plan': identity(folder/'plan.json'),
                                                'completedReceipts': completed, 'incompleteGpuRecords': []})
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
        stopped = False
        for _ in range(500):
            if not proc.exists():
                break
            time.sleep(.01)
        require(not proc.exists(), 'Encoder queue has not exited; do not grant another encoder yet')
        write_immutable(folder/'result.json', {'plan': identity(folder/'plan.json'), 'released': True,
                                              'boundary': identity(folder/'boundary.json'),
                                              'unchangedPerRecordingOutputsReusedOnResume': True})
        print(json.dumps({'released': True, 'receipt': identity(folder/'result.json')}), flush=True)
    finally:
        if stopped and proc.exists():
            os.kill(pid, signal.SIGCONT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    main(args.pid, args.name)

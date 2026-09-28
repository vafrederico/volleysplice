"""Run the optional bounded video benchmark using an already-built simulator app.

Inputs and raw Xcode logs remain external. The JSON report uses a ledger index,
never the source filename. No model fitting or selection occurs here.
"""
import argparse
import json
import plistlib
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xctestrun', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-index', required=True)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--destination', required=True, help='Simulator UDID')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.source.is_file() or not 0 < args.seconds <= 120:
        parser.error('Provide an existing source and a duration in (0, 120]')
    if not args.source_index.startswith('recording-') or not args.source_index[10:].isdigit():
        parser.error('Source index must come from the recording ledger')
    run = plistlib.loads(args.xctestrun.read_bytes())
    target = run['VolleySpliceTests']
    target['EnvironmentVariables'].update(
        VOLLEYCUT_IOS_BENCHMARK_SOURCE=str(args.source.resolve()),
        VOLLEYCUT_IOS_BENCHMARK_SECONDS=str(args.seconds))
    target['OnlyTestIdentifiers'] = ['NeuralPipelineIntegrationTests/testOptionalVideoBenchmark']
    run = {'VolleySpliceTests': target, '__xctestrun_metadata__': run['__xctestrun_metadata__']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log = args.output.with_suffix('.private.log')
    # Preserve __TESTROOT__ resolution by placing the temporary plan beside the original.
    with tempfile.NamedTemporaryFile(suffix='.xctestrun', dir=args.xctestrun.parent, delete=False) as file:
        temporary = Path(file.name)
        file.write(plistlib.dumps(run))
    try:
        with log.open('w') as stream:
            completed = subprocess.run(['xcodebuild', 'test-without-building', '-xctestrun', str(temporary),
                '-destination', 'id=' + args.destination, '-parallel-testing-enabled', 'NO'],
                stdout=stream, stderr=subprocess.STDOUT, check=False)
        rows = []
        for line in log.read_text().splitlines():
            marker = 'NEURAL_PIPELINE_BENCHMARK '
            if marker in line:
                rows.append(json.loads(line.split(marker, 1)[1]))
        expected = {'distilled-large-recall-v1', 'distilled-large-f1-v1'}
        if completed.returncode or len(rows) != 3 or not expected.issubset({row['modelId'] for row in rows}):
            raise RuntimeError('Video benchmark did not pass all three model runs; inspect the external private log')
        report = {'schemaVersion': 1, 'sourceIndex': args.source_index,
            'scope': 'CPU simulator only; not physical-device throughput or model-accuracy qualification', 'runs': rows}
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()

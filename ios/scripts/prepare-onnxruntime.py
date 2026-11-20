"""Stage Microsoft's pinned ONNX Runtime Apple binary and Objective-C bindings."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


VERSION = '1.24.2'
RUNTIME_SHA256 = 'f7100a992d2a8135168c8afd831e6a58b465349101982aa58b3e11d36e600b54'
BINDINGS_SHA256 = 'ab102b30f15c44b08dc6fc32683abeafd216ff6bcabdf74fbac37fbb19d80eed'


def verify(path, expected):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f'SHA-256 mismatch for {path.name}: {actual}')


def stage_archive(kind, expected, destination, archives):
    name = f'pod-archive-onnxruntime-{kind}-{VERSION}.zip'
    if archives is None:
        subprocess.run(['curl', '--fail', '--location', '--retry', '3',
                        '--silent', '--show-error', f'https://download.onnxruntime.ai/{name}',
                        '--output', str(destination)], check=True)
    else:
        shutil.copyfile(archives / name, destination)
    verify(destination, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path,
                        help='Use the two official ZIPs in this directory without downloading')
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1] / 'Dependencies/ONNXRuntime'
    artifacts = package / 'Artifacts'
    if artifacts.exists():
        verify(artifacts / 'onnxruntime.zip', RUNTIME_SHA256)
        verify(artifacts / 'bindings.zip', BINDINGS_SHA256)
        print(f'ONNX Runtime {VERSION} is already staged')
        return
    with tempfile.TemporaryDirectory(dir=package) as work:
        work = Path(work)
        staged = work / 'Artifacts'
        staged.mkdir()
        stage_archive('c', RUNTIME_SHA256, staged / 'onnxruntime.zip', args.archives)
        bindings = staged / 'bindings.zip'
        stage_archive('objc', BINDINGS_SHA256, bindings, args.archives)
        with zipfile.ZipFile(bindings) as archive:
            archive.extractall(staged)
        staged.rename(artifacts)
    print(f'Staged ONNX Runtime {VERSION}')


if __name__ == '__main__':
    main()

"""Stage Microsoft's pinned ONNX Runtime Apple binary and Objective-C bindings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import zipfile
import zlib


VERSION = '1.24.2'
RUNTIME_SHA256 = 'f7100a992d2a8135168c8afd831e6a58b465349101982aa58b3e11d36e600b54'
BINDINGS_SHA256 = 'ab102b30f15c44b08dc6fc32683abeafd216ff6bcabdf74fbac37fbb19d80eed'
NUGET_URL = f'https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/{VERSION}/microsoft.ml.onnxruntime.{VERSION}.nupkg'
NUGET_ENTRY = 'runtimes/ios/native/onnxruntime.xcframework.zip'
NUGET_XCFRAMEWORK_SHA256 = 'a161fe8628ed3bf8d97915bd34063fe115f58a1b9b0f4f0baec78c62487ac3ee'
ALLOWED_RUNTIME_SHA256S = (RUNTIME_SHA256, NUGET_XCFRAMEWORK_SHA256)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str | tuple[str, ...]) -> str:
    allowed = (expected,) if isinstance(expected, str) else expected
    actual = sha256_file(path)
    if actual not in allowed:
        raise ValueError(f'SHA-256 mismatch for {path.name}: {actual}')
    return actual


def verify_bindings_tree(bindings_root: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('version') != VERSION or manifest.get('source_archive_sha256') != BINDINGS_SHA256:
        raise ValueError('Unexpected vendored ONNX Runtime bindings manifest metadata')
    files = manifest.get('files', {})
    if not files:
        raise ValueError('Vendored ONNX Runtime bindings manifest is empty')
    for rel, expected in files.items():
        if not rel.startswith('objectivec/'):
            continue
        target = bindings_root / rel
        if not target.is_file():
            raise ValueError(f'Missing vendored ONNX Runtime binding file: {rel}')
        verify(target, expected)


def stage_vendored_bindings(package: Path, staged: Path) -> None:
    vendored = package / 'Bindings'
    manifest_path = vendored / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'Missing vendored ONNX Runtime bindings manifest at {manifest_path}')
    verify_bindings_tree(vendored, manifest_path)
    shutil.copytree(vendored / 'objectivec', staged / 'objectivec')
    verify_bindings_tree(staged, manifest_path)


def curl_bytes(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    cmd = ['curl', '--fail', '--location', '--retry', '3', '--silent', '--show-error']
    if byte_range is not None:
        cmd += ['-r', f'{byte_range[0]}-{byte_range[1]}']
    cmd.append(url)
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def curl_content_length(url: str) -> int:
    headers = subprocess.run(
        ['curl', '--fail', '--location', '--retry', '3', '--silent', '--show-error', '--head', url],
        check=True, capture_output=True, text=True,
    ).stdout
    matches = re.findall(r'(?im)^content-length:\s*(\d+)\s*$', headers)
    if not matches:
        raise ValueError(f'Missing Content-Length header from {url}')
    return int(matches[-1])


def locate_zip_entry(central_directory: bytes, target_name: str) -> tuple[int, int, int]:
    """Return (compression_method, compressed_size, local_header_offset) from a ZIP central directory."""
    pos = 0
    while pos + 46 <= len(central_directory) and central_directory[pos:pos + 4] == b'PK\x01\x02':
        method = struct.unpack('<H', central_directory[pos + 10:pos + 12])[0]
        comp_size = struct.unpack('<I', central_directory[pos + 20:pos + 24])[0]
        nlen, elen, clen = struct.unpack('<HHH', central_directory[pos + 28:pos + 34])
        local_offset = struct.unpack('<I', central_directory[pos + 42:pos + 46])[0]
        name = central_directory[pos + 46:pos + 46 + nlen].decode('utf-8', 'replace')
        pos += 46 + nlen + elen + clen
        if name == target_name:
            return method, comp_size, local_offset
    raise ValueError(f'Entry {target_name} not found in archive central directory')


def extract_nuget_xcframework(destination: Path, url: str = NUGET_URL) -> None:
    """Fetch only runtimes/ios/native/onnxruntime.xcframework.zip from Microsoft's official NuGet package."""
    total_size = curl_content_length(url)
    tail_len = min(total_size, 65536)
    tail = curl_bytes(url, (total_size - tail_len, total_size - 1))
    eocd = tail.rfind(b'PK\x05\x06')
    if eocd < 0 or eocd + 22 > len(tail):
        raise ValueError('Unable to locate ZIP end-of-central-directory in NuGet package')
    cd_size, cd_offset = struct.unpack('<II', tail[eocd + 12:eocd + 20])
    central_dir = curl_bytes(url, (cd_offset, cd_offset + cd_size - 1))
    method, comp_size, local_offset = locate_zip_entry(central_dir, NUGET_ENTRY)
    local_hdr = curl_bytes(url, (local_offset, min(total_size - 1, local_offset + 511)))
    if local_hdr[:4] != b'PK\x03\x04':
        raise ValueError('Invalid ZIP local file header in NuGet package')
    lnlen, lelen = struct.unpack('<HH', local_hdr[26:30])
    data_start = local_offset + 30 + lnlen + lelen
    compressed = curl_bytes(url, (data_start, data_start + comp_size - 1))
    if method == 0:
        payload = compressed
    elif method == 8:
        payload = zlib.decompress(compressed, -15)
    else:
        raise ValueError(f'Unsupported ZIP compression method {method} in NuGet package')
    destination.write_bytes(payload)
    verify(destination, NUGET_XCFRAMEWORK_SHA256)


def stage_archive(kind: str, expected: str, destination: Path, archives: Path | None) -> None:
    name = f'pod-archive-onnxruntime-{kind}-{VERSION}.zip'
    if archives is None:
        subprocess.run(['curl', '--fail', '--location', '--retry', '3',
                        '--silent', '--show-error', f'https://download.onnxruntime.ai/{name}',
                        '--output', str(destination)], check=True)
    else:
        shutil.copyfile(archives / name, destination)
    verify(destination, expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives', type=Path,
                        help='Use the two official ZIPs in this directory without downloading')
    parser.add_argument('--prefer-nuget', action='store_true',
                        help='Download the official iOS xcframework directly from Microsoft NuGet instead of download.onnxruntime.ai')
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1] / 'Dependencies/ONNXRuntime'
    artifacts = package / 'Artifacts'
    if artifacts.exists():
        verify(artifacts / 'onnxruntime.zip', ALLOWED_RUNTIME_SHA256S)
        if (artifacts / 'bindings.zip').is_file():
            verify(artifacts / 'bindings.zip', BINDINGS_SHA256)
        else:
            verify_bindings_tree(artifacts, package / 'Bindings/manifest.json')
        print(f'ONNX Runtime {VERSION} is already staged')
        return
    with tempfile.TemporaryDirectory(dir=package) as work:
        work = Path(work)
        staged = work / 'Artifacts'
        staged.mkdir()
        if args.archives is not None:
            stage_archive('c', RUNTIME_SHA256, staged / 'onnxruntime.zip', args.archives)
            bindings = staged / 'bindings.zip'
            stage_archive('objc', BINDINGS_SHA256, bindings, args.archives)
            with zipfile.ZipFile(bindings) as archive:
                archive.extractall(staged)
        else:
            runtime_ready = False
            if not args.prefer_nuget:
                try:
                    stage_archive('c', RUNTIME_SHA256, staged / 'onnxruntime.zip', None)
                    bindings = staged / 'bindings.zip'
                    stage_archive('objc', BINDINGS_SHA256, bindings, None)
                    with zipfile.ZipFile(bindings) as archive:
                        archive.extractall(staged)
                    runtime_ready = True
                except (subprocess.CalledProcessError, ValueError) as error:
                    print(f'download.onnxruntime.ai unavailable ({error}); falling back to official Microsoft NuGet {VERSION} package and vendored MIT bindings.')
                    shutil.rmtree(staged, ignore_errors=True)
                    staged.mkdir()
            if not runtime_ready:
                extract_nuget_xcframework(staged / 'onnxruntime.zip')
                stage_vendored_bindings(package, staged)
        staged.rename(artifacts)
    print(f'Staged ONNX Runtime {VERSION}')


if __name__ == '__main__':
    main()

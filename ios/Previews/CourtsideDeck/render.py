"""Stage exact reusable app controls for a local compatible SwiftUI renderer.

Build output and dependencies stay in the explicitly supplied external work dir.
No source or sample data is uploaded. Requires Swift and a local Tokamak checkout.
"""
import argparse
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--work-dir', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--width', type=int, default=390)
args = parser.parse_args()
here = Path(__file__).resolve().parent
app = here.parents[1] / 'App' / 'CourtsideReviewControls.swift'
work = args.work_dir.resolve()
checkout = here.parents[2]
if work == checkout or checkout in work.parents:
    raise SystemExit('Use an external build/cache directory')
source = work / 'Sources' / 'CourtsidePreview'
source.mkdir(parents=True, exist_ok=True)
shutil.copyfile(here / 'Package.swift', work / 'Package.swift')
shutil.copyfile(here / 'Preview.swift', source / 'main.swift')
shutil.copyfile(app, source / app.name)
result = subprocess.run(['swift', 'run', '--package-path', str(work), '-j', '2', 'CourtsidePreview', str(args.width)], cwd=work, capture_output=True, text=True)
if result.returncode:
    print(result.stderr)
    raise SystemExit(result.returncode)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(result.stdout, encoding='utf-8')
print('Rendered local compatible SwiftUI control preview (synthetic shell).')

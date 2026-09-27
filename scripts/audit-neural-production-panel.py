#!/usr/bin/env python3
"""Independent population/gold/AV-byte continuity gate; no predictions or metrics."""
import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ADDITIONS = {'modelFamily', 'continuityReferencePanel', 'productionProtocol', 'interpretation'}


def require(value, message):
    if not value: raise ValueError(message)


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class Evidence:
    def __init__(self): self.refs = {}; self.stats = {}
    def bind(self, ref):
        p = Path(ref['path']).resolve(); before = p.stat()
        stat = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if str(p) in self.refs:
            require(self.refs[str(p)]['sha256'] == ref['sha256'] and self.stats[str(p)] == stat, 'Bound file identity/stat changed')
        else:
            value = hashlib.sha256()
            with p.open('rb') as stream:
                while block := stream.read(1024*1024): value.update(block)
            after = p.stat()
            require(stat == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                and value.hexdigest() == ref['sha256'], 'Cold source hash/stat differs')
            self.refs[str(p)] = {'path': str(p), 'sha256': ref['sha256']}; self.stats[str(p)] = stat
        return p
    def capture(self, path):
        p = Path(path).resolve(); value = hashlib.sha256(p.read_bytes()).hexdigest()
        ref = {'path': str(p), 'sha256': value}; self.bind(ref); return ref
    def document(self, ref): return json.loads(self.bind(ref).read_text())
    def closure(self, value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str): self.bind(value)
            for child in value.values(): self.closure(child)
        elif isinstance(value, list):
            for child in value: self.closure(child)
    def finish(self):
        for ref in list(self.refs.values()): self.bind(ref)


def compare_contract(panel, core, manifest, av, visual):
    require(panel['kind'] == core['kind'] == 'frozen-independent-inference-panel-v1', 'Panel schema differs')
    restored = {k: v for k, v in panel.items() if k not in ADDITIONS}
    for key in ('features', 'featureAudit', 'registrar'): restored[key] = core[key]
    require(canonical(restored) == canonical(core), 'Production changed source/gold/roles/order or panel policy')
    require(panel['modelFamily'] == 'av' and panel['inferenceUsesLabels'] is False and panel['allInferenceTicksValid'] is True,
        'Production blind AV policy differs')
    ids = [r['id'] for r in manifest['records']]
    scored = [r['id'] for r in manifest['records'] if 'evaluate' in r['eligibleRoles'] and r['scoringPolicy'] != 'none']
    require(len(ids) == len(set(ids)) == 42 and len(scored) == 34 and panel['recordingIds'] == ids
        and panel['scoredRecordingIds'] == scored, 'Expected identical42/34 ordered population')
    require([r['recordingId'] for r in av['records']] == [r['recordingId'] for r in visual['records']] == ids,
        'Feature recording order/population differs')
    av_top = {k: v for k, v in av.items() if k not in ('records', 'publicationMode', 'encoderPlan')}
    core_top = {k: v for k, v in visual.items() if k not in ('records', 'publicationMode', 'encoderPlan')}
    require(canonical(av_top) == canonical(core_top) and av['publicationMode'] == 'av'
        and av['encoderPlan'] is None and visual['publicationMode'] == 'core', 'Feature qualification/AV publication differs')
    checks = []
    for source, one, two in zip(manifest['records'], av['records'], visual['records'], strict=True):
        require(canonical({k:v for k,v in one.items() if k != 'features'})
            == canonical({k:v for k,v in two.items() if k != 'features'}), 'Feature media/source/staging lineage differs')
        require(one['sourceGroup'] == source['sourceGroup'] and one['contentSha256'] == source['contentSha256'], 'Source identity differs')
        require(one['features']['audiovisual'] == two['features']['audiovisual'], 'AV cache bytes or reference differs')
        require(one['features'].get('imageInput') == two['features'].get('imageInput'), 'Image/timing reference differs')
        checks.append({'id': source['id'], 'sourceGroup': source['sourceGroup'], 'contentSha256': source['contentSha256'],
            'audiovisual': one['features']['audiovisual'], 'imageInput': one['features'].get('imageInput')})
    return checks


def audit(panel_path, output):
    require(output.resolve().is_relative_to('/mnt/freenas') and not output.exists(), 'New NAS audit required')
    evidence = Evidence(); panel_ref = evidence.capture(panel_path); panel = evidence.document(panel_ref)
    core_ref = panel['continuityReferencePanel']; core = evidence.document(core_ref)
    manifest = evidence.document(panel['manifest']); av = evidence.document(panel['features']); visual = evidence.document(core['features'])
    checks = compare_contract(panel, core, manifest, av, visual)
    require(panel['registrar'] == evidence.capture(REPO/'scripts/register-neural-production-panel.py')
        and core['registrar'] == evidence.capture(REPO/'scripts/register-neural-generalization-panel.py'), 'Current registrars differ')
    feature_auditor = evidence.capture(REPO/'scripts/audit-neural-generalization-features.py')
    for source, mode in ((panel, 'av'), (core, 'core')):
        gate = evidence.document(source['featureAudit'])
        require(gate['kind'] == 'independent-expansion-feature-audit-v1' and gate['passed'] is True
            and gate['mode'] == mode and gate['scope'] == 'all' and gate['auditedRecordingCount'] == 42
            and gate['features'] == source['features'] and gate['inputs'] == source['manifest']
            and gate['source'] == feature_auditor and gate['inferenceMaskAlwaysAllTrue'] is True
            and gate['protectedAndPanelTeacherTargetsAbsent'] is True, 'Passing full42 feature gate differs')
        # The gate is pinned and its declared file references are checked, without opening arbitrary referenced JSON.
        evidence.closure(gate)
    production = evidence.document(panel['productionProtocol'])
    require(production['manifest'] == panel['manifest'] and production['features'] == panel['features']
        and production['labelsUsedForInference'] is False, 'Original blind production replay association differs')
    evidence.closure(checks); evidence.bind(panel['inventory']); evidence.finish()
    result = {'kind': 'independent-production-panel-continuity-audit-v1', 'passed': True, 'panel': panel_ref,
        'corePanel': core_ref, 'manifest': panel['manifest'], 'inventory': panel['inventory'],
        'productionProtocol': panel['productionProtocol'], 'featureAudits': [panel['featureAudit'], core['featureAudit']],
        'recordingCount': 42, 'scoredRecordingCount': 34, 'sameOrderedPopulationAndGold': True,
        'allAudiovisualReferencesAndBytesIdentical': True, 'checks': checks,
        'predictionOrOutcomeAccessed': False, 'auditor': evidence.capture(__file__), 'references': list(evidence.refs.values())}
    evidence.finish(); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--panel', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); audit(a.panel, a.output); print(str(a.output), flush=True)

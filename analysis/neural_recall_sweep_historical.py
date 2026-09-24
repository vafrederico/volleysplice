"""Freeze/replay historical nested sweeps and obtain all outer epochs once."""
from __future__ import annotations
from analysis.private_ledger import private_value

import importlib.util
import os
from pathlib import Path

from . import neural_development as base, neural_expanded_development as expanded
from . import neural_short_boost_transfer as historical
from . import neural_recall_sweep as sweep
from .neural_recall_sweep_adapters import NAS, bind, reference_layout, shard_from_fit
from .neural_recall_refits import identical_npz

REPO = Path(__file__).resolve().parents[1]
MODELS = ('av_tcn_short_boost', 'dino_tcn_short_boost', 'av_transformer', 'dino_transformer', 'mobile_tcn', 'distilled_mobile_tcn')
SEEDS = (3407, 1729, 20260918)


def load_script(name):
    path = REPO / 'scripts' / name
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def task_directory(output, task):
    return Path(output) / 'refits' / task['model'] / str(task['seed']) / f"outer-{task['outerIndex']}"


def prepare(output, nas_root=NAS):
    output, nas_root = Path(output), Path(nas_root)
    sweep.require(str(output.resolve()).startswith(private_value('private-reference-0060')), 'Historical outputs must use direct NAS')
    sweep.require(not (output / 'protocol.json').exists(), 'Historical protocol already frozen')
    layouts = [reference_layout(model, seed, nas_root) for model in MODELS for seed in SEEDS]
    tasks = [{**task, 'registration': layout['registration']} for layout in layouts for task in layout['missingOuterRefits']]
    source = nas_root / '2026-09-19-short-boost-transfer/manifest-pts-v1.json'
    manifest = sweep.read(source)
    exact = bind(manifest['exactManifest'])
    examples = base.load_examples(exact, False)
    sweep.require(len(examples) == 8 and len({e.group for e in examples}) == 4, 'Historical exact scope changed')
    records = [{**sweep.serial_rows([e.row([])])[0], 'timestamps': e.times.tolist(), 'valid': e.valid.tolist(),
                'environment': e.environment, 'labelPolicy': 'exact-rallies', 'reviewStatus': 'manually-reviewed'} for e in examples]
    for row in records:
        row.pop('predictions')
    sweep.write_new(output / 'records.json', {'records': records})
    code = {}
    for layout in layouts:
        registration = sweep.read(layout['registration']['path'])
        for name, reference in registration['contract']['code'].items():
            reference = reference if isinstance(reference, dict) else {'path': str(REPO / 'analysis' / name), 'sha256': reference}
            bind(reference)
            code[reference['path']] = {k: reference[k] for k in ('path', 'sha256')}
    own = ('analysis/neural_recall_sweep.py', 'analysis/neural_recall_sweep_adapters.py', 'analysis/neural_recall_sweep_historical.py',
           'analysis/neural_recall_operating_point.py', 'analysis/neural_recall_refits.py', 'scripts/evaluate-neural-recall-sweep.py',
           'scripts/audit-neural-recall-sweep.py', 'scripts/run-neural-historical-recall-sweep.py',
           'scripts/audit-neural-historical-refits.py', 'scripts/audit-neural-short-boost-intervals.py',
           'scripts/audit-neural-mobile-distillation.py', 'scripts/audit-neural-recognition.py',
           'scripts/audit-neural-context-tensors.py', 'scripts/audit-neural-short-boost-tensors.py')
    for name in own:
        reference = sweep.identity(REPO / name)
        code[reference['path']] = reference
    archive = output / 'registered-sources'
    for reference in code.values():
        path = Path(reference['path']); target = archive / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(path.read_bytes())
    protocol = {'kind': 'historical-recall-floor-sweep-v1', 'models': list(MODELS), 'seeds': list(SEEDS),
        'epochs': list(sweep.EPOCHS), 'floors': list(sweep.FLOORS), 'decoderCandidates': expanded.decoder_candidates(),
        'targetPaddingSeconds': 2, 'paddingCases': [0, 1, 2, 3], 'joinGapSeconds': 3,
        'selection': 'Same ordered192 candidates; maximumF1_padP_coreR among exact pooled inner R_core>=floor; no fallback.',
        'historicalInferencePolicy': 'Preserve original ignored-segment validity for exact historical comparability.',
        'newDeploymentInferencePolicy': 'Separate scope; external/variant inference is full timeline and never uses this historical mask.',
        'refitPolicy': 'All4outer epochs, original recipe/seed/exposure; exact weights/scores/history parity at every available original epoch. Reuse existing distilled students/features; fit only two absent outer students.',
        'source': sweep.identity(source), 'exactManifest': manifest['exactManifest'], 'records': sweep.identity(output / 'records.json'),
        'layoutsSha256': sweep.canonical(layouts), 'refitTasksSha256': sweep.canonical(tasks),
        'registrations': [layout['registration'] for layout in layouts], 'code': list(code.values()),
        'priorStudentAudit': sweep.identity(nas_root / '2026-09-23-recall-distillation/distilled-mobile-v1/audit.json'),
        'protectedTestOpened': False, 'productionPromotionAllowed': False, 'gpuRequiresSeparateResourceGrant': True}
    sweep.write_new(output / 'protocol.json', protocol)
    sweep.write_new(output / 'layouts.json', {'protocol': sweep.identity(output / 'protocol.json'), 'layouts': layouts})
    sweep.write_new(output / 'refit-plan.json', {'kind': 'historical-all-outer-epochs-refit-plan-v1',
        'protocol': sweep.identity(output / 'protocol.json'), 'tasks': tasks, 'outerFitCount': len(tasks),
        'missingEpochCount': sum(len(task['missingEpochs']) for task in tasks), 'newStudentFitCount': sum(task['requiresStudentFit'] for task in tasks)})
    make_bundle(output, complete=False)
    return protocol


def verify(output):
    output = Path(output)
    protocol = sweep.read(output / 'protocol.json')
    for reference in [protocol['source'], protocol['exactManifest'], protocol['records'], protocol['priorStudentAudit'], *protocol['registrations'], *protocol['code']]:
        bind(reference)
    for reference in protocol['code']:
        archived = output / 'registered-sources' / Path(reference['path']).relative_to(REPO)
        sweep.require(sweep.identity(archived)['sha256'] == reference['sha256'], 'Historical archived source changed')
    sweep.require(protocol['models'] == list(MODELS) and protocol['seeds'] == list(SEEDS)
                  and protocol['epochs'] == list(sweep.EPOCHS) and protocol['floors'] == list(sweep.FLOORS), 'Historical design changed')
    sweep.require(sweep.canonical(sweep.read(output / 'layouts.json')['layouts']) == protocol['layoutsSha256']
                  and sweep.canonical(sweep.read(output / 'refit-plan.json')['tasks']) == protocol['refitTasksSha256'],
                  'Historical owner graph or refit task plan changed')
    return protocol


def make_bundle(output, *, complete):
    output = Path(output); protocol = verify(output)
    layouts_doc = sweep.read(output / 'layouts.json')
    sweep.require(layouts_doc['protocol'] == sweep.identity(output / 'protocol.json'), 'Layout protocol differs')
    examples = sweep.examples_from_rows(sweep.read(output / 'records.json')['records'])
    cells, comparisons = [], []
    for layout in layouts_doc['layouts']:
        folds = []
        for fold in layout['folds']:
            outer = fold['heldSourceGroup']; inner_shards = []
            for owner in fold['innerOwners']:
                ids = [e.id for e in examples if e.group == owner['heldInnerSourceGroup']]
                for epoch in sweep.EPOCHS:
                    inner_shards.append(shard_from_fit(owner['fitDirectory'], epoch, ids,
                        additional_training_owners=owner['additionalTrainingOwners'], excluded_groups=owner['excludedSourceGroups']))
            ids = [e.id for e in examples if e.group == outer]
            if complete:
                task = {'model': layout['model'], 'seed': layout['seed'], 'outerIndex': fold['outerIndex']}
                directory = task_directory(output, task)
                parity = sweep.read(directory / 'parity.json')
                sweep.require(parity['passed'] and parity['plan'] == sweep.identity(output / 'refit-plan.json'), 'All-epoch refit parity absent')
                owners = {epoch: {'fitDirectory': str(directory / 'temporal'), 'additionalTrainingOwners': parity['additionalTrainingOwners']}
                          for epoch in sweep.EPOCHS}
            else:
                owners = {int(k): v for k, v in fold['outerEpochOwners'].items()}
            outer_shards = [shard_from_fit(owner['fitDirectory'], epoch, ids,
                additional_training_owners=owner['additionalTrainingOwners'], excluded_groups=[outer]) for epoch, owner in owners.items()]
            folds.append({'foldId': outer, 'selectionRecordingIds': [e.id for e in examples if e.group != outer],
                          'selectionScores': inner_shards, 'panels': [{'panelId': 'historical-nested-exact', 'recordingIds': ids, 'scores': outer_shards}]})
        cells.append({'model': layout['model'], 'variant': 'historical-original18', 'seed': layout['seed'],
                      'selectionDesign': 'nested-original-four-source', 'folds': folds,
                      'panels': [{'panelId': 'historical-nested-exact', 'labelPolicy': 'exact-rallies', 'reviewStatus': 'manually-reviewed',
                                  'sourcePolicy': 'source-held', 'expectedRecordingIds': [e.id for e in examples]}]})
    for model in MODELS:
        comparisons.append({'model': model, 'variant': 'historical-original18', 'selectionDesign': 'nested-original-four-source',
                            'panelId': 'historical-nested-exact', 'seeds': list(SEEDS)})
    bundle = {'kind': 'recall-floor-sweep-bundle-v1', 'protocol': sweep.identity(output / 'protocol.json'),
              'records': protocol['records'], 'code': protocol['code'], 'floors': list(sweep.FLOORS), 'epochs': list(sweep.EPOCHS),
              'targetPaddingSeconds': 2, 'joinGapSeconds': 3, 'protectedTestOpened': False, 'cells': cells, 'comparisons': comparisons}
    sweep.write_new(output / ('bundle-complete.json' if complete else 'bundle-selection.json'), bundle)


def fit_refits(output, device='cuda', *, model=None, seed=None, outer=None):
    import torch
    from .neural_recognition_fit import fit_model
    from .recognition_temporal_model import RecognitionConfig
    from .neural_recognition_inputs import attach_features
    from . import neural_mobile_distillation as distilled
    output = Path(output); verify(output)
    plan = sweep.read(output / 'refit-plan.json')
    sweep.require(plan['protocol'] == sweep.identity(output / 'protocol.json'), 'Refit plan protocol changed')
    sweep.require(not device.startswith('cuda') or os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Deterministic CUBLAS environment required')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    cache = {}
    for task in plan['tasks']:
        if (model is not None and task['model'] != model) or (seed is not None and task['seed'] != seed) or (outer is not None and task['outerIndex'] != outer):
            continue
        print(f"REFIT_SWEEP {task['model']} seed={task['seed']} outer={task['outerIndex']}", flush=True)
        registration = sweep.read(bind(task['registration'])); contract = registration['contract']
        legacy = task['model'] in ('av_tcn_short_boost', 'dino_tcn_short_boost')
        student = task['model'] == 'distilled_mobile_tcn'
        torch.set_num_threads(2 if student else 4)
        if task['model'] not in cache:
            cache.clear()
            manifest = Path(contract['source']['path']) if student else Path(contract['manifest']['path']) if not legacy else historical.MANIFEST
            dino = contract.get('dinoManifest')
            with_dino = task['model'] in ('dino_tcn_short_boost', 'dino_transformer')
            data = historical.load_data(manifest, Path(dino['path']) if dino else None, with_dino=with_dino)
            config = RecognitionConfig(family='mobile', head='tcn', scalar_dimension=8) if student else RecognitionConfig(**contract['config']) if not legacy else None
            if config and config.family == 'mobile' and not student:
                data = attach_features(data, bind(contract['featureManifest']), config, manifest)
            cache[task['model']] = data, config
        data, config = cache[task['model']]
        excluded = {task['heldSourceGroup']}; directory = task_directory(output, task)
        training_owners = []
        if student:
            if task['reuseStudentDirectory']:
                old_folder = Path(task['reuseStudentDirectory']).parent
                student_ref = sweep.identity(old_folder / 'student/completed.json')
                student_receipt = sweep.read(student_ref['path'])
                bind(student_receipt['weights'])
                index = sweep.read(old_folder / 'features.json')
                features = {row['id']: row for row in index['records']}
                train, auxiliary, _ = distilled.allowed_records(data, excluded)
                held_rows = [row for row in data['exact'] if row.example.group in excluded]
                train = distilled.attach_rows(train, features)
                auxiliary = {tier: distilled.attach_rows(rows, features) for tier, rows in auxiliary.items()}
                held = [row.example for row in distilled.attach_rows(held_rows, features)]
                training_owners = [student_ref]
            else:
                images, teachers = distilled.load_image_indexes()
                train, auxiliary, held = distilled.prepare_fit(data, excluded, images, teachers, task['seed'], directory,
                                                               registration['sha256'], device)
                training_owners = [sweep.identity(directory / 'student/completed.json')]
            fit_model(train, auxiliary, held, config, task['seed'], sweep.EPOCHS, directory / 'temporal', device, registration['sha256'], 'short_boost')
        else:
            train = [row for row in data['exact'] if row.example.group not in excluded]
            held = [row.example for row in data['exact'] if row.example.group in excluded]
            auxiliary = expanded.auxiliary_for_fold(data, 'reviewed_export', excluded)
            if legacy:
                historical.fit_model(train, auxiliary, held, 'tcn' if task['model'] == 'av_tcn_short_boost' else 'dino_tcn',
                    task['seed'], sweep.EPOCHS, directory / 'temporal', device, registration['sha256'], 'short_boost')
            else:
                fit_model(train, auxiliary, held, config, task['seed'], sweep.EPOCHS, directory / 'temporal', device, registration['sha256'], 'short_boost')
        meta = sweep.read(directory / 'temporal/completed.json')
        checks = []
        for epoch_text, owner in task['parityCheckEpochOwners'].items():
            old = Path(owner['fitDirectory']); old_meta = sweep.read(bind(owner['completed'])); epoch = int(epoch_text)
            for stem in ('weights', 'predictions'):
                name = f'{stem}-{epoch}.npz'
                bind({'path': str(old / name), 'sha256': old_meta['artifacts'][name]})
                identical_npz(directory / 'temporal' / name, old / name)
            sweep.require(meta['history'][:len(old_meta['history'])] == old_meta['history'], 'Original full numerical/exposure prefix differs')
            checks.append({'epoch': epoch, 'owner': owner['completed'], 'weightsExactlyEqual': True, 'scoresExactlyEqual': True,
                           'historyPrefixExactlyEqual': True})
        parity = {'passed': True, 'task': task, 'plan': sweep.identity(output / 'refit-plan.json'),
                  'completed': sweep.identity(directory / 'temporal/completed.json'), 'checks': checks,
                  'additionalTrainingOwners': training_owners, 'newStudentFit': task['requiresStudentFit']}
        if (directory / 'parity.json').exists():
            sweep.require(sweep.read(directory / 'parity.json') == parity, 'Completed parity receipt changed')
        else:
            sweep.write_new(directory / 'parity.json', parity)
    verify(output)

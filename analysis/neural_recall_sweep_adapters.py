"""Read-only historical-fit inventory and hash-bound score adapters for sweeps."""
from __future__ import annotations
from analysis.private_ledger import private_value

from pathlib import Path
import numpy as np

from .neural_recall_sweep import EPOCHS, canonical, identity, read, require, validate_scores

NAS = Path(private_value('private-reference-0057'))


def training_groups(metadata):
    groups = set(metadata['trainGroups'])
    for values in metadata.get('auxiliaryGroups', {}).values():
        groups.update(values)
    return groups


def bind(reference, evidence=None):
    actual = identity(reference['path'])
    require(all(actual[k] == reference[k] for k in ('path', 'sha256')), 'Artifact identity changed: ' + str(reference['path']))
    if evidence is not None:
        evidence[actual['path']] = actual
    return Path(actual['path'])


def shard_from_fit(folder, epoch, recording_ids, *, additional_training_owners=(), excluded_groups=()):
    """Describe a subset of an existing fit's saved validation archive."""
    folder = Path(folder)
    owner = read(folder / 'completed.json')
    name = f'predictions-{epoch}.npz'
    require(epoch in owner['epochs'] and name in owner['artifacts'], 'Requested saved epoch unavailable')
    archive = {'path': str(folder / name), 'sha256': owner['artifacts'][name]}
    checkpoint_name = f'weights-{epoch}.npz'
    require(checkpoint_name in owner['artifacts'], 'Score archive lacks its checkpoint')
    require(set(recording_ids) <= set(owner['validationIds']), 'Score subset is not a validation population')
    return {'epoch': epoch, 'archive': archive, 'archiveRecordingIds': owner['validationIds'],
            'recordingIds': list(recording_ids), 'checkpoint': {'path': str(folder / checkpoint_name),
            'sha256': owner['artifacts'][checkpoint_name]}, 'trainingOwners': [identity(folder / 'completed.json'),
            *list(additional_training_owners)], 'excludedSourceGroups': sorted(excluded_groups)}


def load_score_shards(examples, shards, *, evidence=None, source_policy='source-held', require_all_epochs=True):
    """Load exact score bytes and enforce every encoder/head training exclusion.

    Inference-only archives may use a separately frozen archive reference while
    retaining the checkpoint and all training-owner receipts. The bundle binds
    their exact recording inventory. Seen-source results must explicitly opt in
    to the descriptive policy and cannot be presented as source-held estimates.
    """
    require(source_policy in ('source-held', 'seen-source-descriptive'), 'Unknown inference source policy')
    by_id = {e.id: e for e in examples}
    scores = {}
    for shard in shards:
        epoch = shard['epoch']
        require(epoch in EPOCHS and shard['trainingOwners'], 'Unregistered checkpoint or missing training lineage')
        path = bind(shard['archive'], evidence)
        bind(shard['checkpoint'], evidence)
        groups = set()
        for reference in shard['trainingOwners']:
            groups.update(training_groups(read(bind(reference, evidence))))
        require(not groups & set(shard['excludedSourceGroups']), 'Declared held source leaked into encoder/head training')
        ids = shard['recordingIds']
        require(ids and len(set(ids)) == len(ids) and set(ids) <= set(by_id), 'Shard subset differs')
        require(source_policy != 'source-held' or not groups & {by_id[key].group for key in ids}, 'Panel source was used to fit encoder/head')
        with np.load(path, allow_pickle=False) as payload:
            require(set(payload.files) == set(shard['archiveRecordingIds']) and set(ids) <= set(payload.files),
                    'Score archive inventory differs')
            target = scores.setdefault(epoch, {})
            require(not set(target) & set(ids), 'Duplicate saved-score owner for a recording/checkpoint')
            target.update({key: payload[key].copy() for key in ids})
    validate_scores(examples, scores, require_all_epochs=require_all_epochs)
    return scores


def reference_layout(model, seed, nas_root=NAS):
    """Inventory original inner owners and all available original/99 outer epochs.

    This emits a plan only. Existing artifacts are never edited or backfilled.
    New refits should save all four epochs once, under the new study root, and
    exactly replay every available old checkpoint before their scores are used.
    """
    root = Path(nas_root)
    legacy = model in ('av_tcn_short_boost', 'dino_tcn_short_boost')
    distilled = model == 'distilled_mobile_tcn'
    recognition = {'av_transformer': 'av-transformer-v1', 'dino_transformer': 'dino-transformer-v1', 'mobile_tcn': 'mobile-tcn-v1'}
    require(legacy or distilled or model in recognition, 'Unsupported reference fit family')
    if legacy:
        study = root / '2026-09-19-short-boost-transfer/study'
        kind = 'tcn' if model == 'av_tcn_short_boost' else 'dino_tcn'
        fit_root = study / 'fits/reviewed_export' / kind / 'short_boost' / str(seed)
    elif distilled:
        study = root / '2026-09-23-recall-distillation/distilled-mobile-v1'
        fit_root = study / 'fits' / str(seed)
    else:
        study = root / '2026-09-22-recognition' / recognition[model]
        fit_root = study / 'fits' / str(seed)
    registration_ref = identity(study / 'preregistration.json')
    registration = read(registration_ref['path'])
    contract = registration['contract']
    require(canonical(contract) == registration['sha256'] and seed in contract['seeds']
            and tuple(contract['checkpointEpochs']) == EPOCHS, 'Historical registration/grid differs')
    groups = contract['groups']
    require(groups == sorted(set(groups)), 'Historical source-group order differs')
    folds, refits = [], []
    for outer_index, outer in enumerate(groups):
        inner = []
        for inner_index, group in enumerate(groups):
            if group == outer:
                continue
            if legacy:
                logical_index = [g for g in groups if g != outer].index(group)
                folder = fit_root / f'outer-{outer_index}' / f'inner-{logical_index}'
            else:
                a, b = sorted((outer_index, inner_index))
                folder = fit_root / f'inner-{a}-{b}'
                if distilled:
                    folder /= 'temporal'
            require((folder / 'completed.json').exists(), 'Original inner fit missing')
            inner.append({'heldInnerSourceGroup': group, 'fitDirectory': str(folder),
                          'excludedSourceGroups': sorted((outer, group)),
                          'additionalTrainingOwners': [identity(folder.parent / 'student/completed.json')] if distilled else []})
        original = fit_root / f'outer-{outer_index}'
        if legacy:
            original /= 'refit'
        elif distilled:
            original /= 'temporal'
        supplemental = root / '2026-09-23-recall-distillation/recall99-v1/refits' / model / f'seed-{seed}' / f'outer-{outer_index}'
        owners = {}
        for folder in (original, supplemental) if not distilled else (original,):
            if not (folder / 'completed.json').exists():
                continue
            meta_ref = identity(folder / 'completed.json')
            meta = read(meta_ref['path'])
            require(meta['seed'] == seed and meta['contractSha256'] == registration['sha256']
                    and outer not in training_groups(meta), 'Historical outer fit identity/exclusion differs')
            if folder == supplemental:
                parity = read(folder / 'original-parity.json')
                require(parity['passed'] and parity['weightsAndPredictionsExactlyEqual']
                        and parity['samplingHistoryExactlyEqual'] and parity['completed'] == meta_ref,
                        'Supplemental fit lacks original numerical parity')
            for epoch in meta['epochs']:
                if epoch in EPOCHS and epoch not in owners:
                    owners[epoch] = {'fitDirectory': str(folder), 'completed': meta_ref,
                                     'additionalTrainingOwners': [identity(folder.parent / 'student/completed.json')] if distilled else []}
        missing = sorted(set(EPOCHS) - set(owners))
        fold = {'outerIndex': outer_index, 'heldSourceGroup': outer, 'innerOwners': inner,
                'outerEpochOwners': {str(k): v for k, v in sorted(owners.items())}, 'missingOuterEpochs': missing}
        folds.append(fold)
        if missing:
            refits.append({'model': model, 'seed': seed, 'outerIndex': outer_index, 'heldSourceGroup': outer,
                          'saveEpochs': list(EPOCHS), 'missingEpochs': missing, 'originalContractSha256': registration['sha256'],
                          'parityCheckEpochOwners': fold['outerEpochOwners'],
                          'requiresStudentFit': distilled and not (original.parent / 'student/completed.json').exists(),
                          'reuseStudentDirectory': str(original.parent / 'student') if distilled and (original.parent / 'student/completed.json').exists() else None})
    return {'model': model, 'seed': seed, 'study': str(study), 'registration': registration_ref,
            'registrationContractSha256': registration['sha256'], 'groups': groups, 'folds': folds,
            'missingOuterRefits': refits, 'trainingPerformed': False,
            'precisionPolicy': 'FP16/INT8 replay may reuse DINO FP32 choices; do not describe transferred floors as requalified.'}

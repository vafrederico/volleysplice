"""Explicit process-local file-hash reuse; no numerical function is replaced."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path

from . import neural_recall_sweep as io
from . import neural_recall_sweep_adapters as aliases
from . import neural_generalization_inputs as inputs

BINDINGS = ['analysis.neural_recall_sweep.identity', 'analysis.neural_recall_sweep_adapters.identity',
            'analysis.neural_generalization_inputs.sha']
POLICY = {'kind': 'explicit-process-local-immutable-identity-cache-v1', 'bindings': BINDINGS,
    'keyFields': ['dev', 'ino', 'size', 'mtimeNs', 'ctimeNs'], 'coldFirstRead': True,
    'prePostStatRaceCheck': True, 'changedMetadataFullRehashThenFail': True,
    'clearExistingVerifiedCacheBetweenStages': True, 'allNumericalFunctionsUnchanged': True,
    'stageExecutionSidecarsRequired': True, 'savedSelectionBytesUnchanged': True,
    'independentColdClosureBeforeGlobalFreezeAndPublication': True,
    'laterPublicationPreflight': 'rehash gate and source plan; exact full cold-closure stat signatures',
    'coldClosurePolicy': 'actual-touched-plus-named-numerical-evidence-v1',
    'arbitraryReferencedJsonTraversal': False,
    'finalNumericalResultAuditUnchanged': True, 'persistentCacheAllowed': False}


def signature(path):
    s = Path(path).stat()
    return {'dev': s.st_dev, 'ino': s.st_ino, 'size': s.st_size,
            'mtimeNs': s.st_mtime_ns, 'ctimeNs': s.st_ctime_ns}


def cold_sha(path):
    before = signature(path); digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)
    after = signature(path)
    io.require(before == after, 'File changed while cold hashing: '+str(path))
    return digest.hexdigest(), after


class IdentityCache:
    def __init__(self):
        self.files = {}
        self.touched = set()
        self.statistics = {}
        self.active = False

    def begin(self):
        io.require(not self.active, 'Nested identity-cache stage')
        self.touched = set()
        self.statistics = {'coldHashCount': 0, 'cacheHitCount': 0, 'coldBytes': 0, 'reusedBytes': 0}
        self.active = True

    def digest(self, path):
        io.require(self.active, 'Identity cache used outside disclosed stage')
        path = Path(path); key = str(path.resolve()); current = signature(path)
        if key in self.files:
            item = self.files[key]
            if item['stat'] != current:
                # Rehash for a diagnostic, then reject even an identical-byte
                # rewrite; an immutable identity changed its metadata.
                actual, after = cold_sha(path)
                raise ValueError('Previously cached file metadata changed; cold rehash '+actual+' at '+str(after)+': '+key)
            value = item['sha256']
            self.statistics['cacheHitCount'] += 1; self.statistics['reusedBytes'] += current['size']
        else:
            value, after = cold_sha(path)
            io.require(current == after, 'File changed before cold hash: '+key)
            self.files[key] = {'path': key, 'sha256': value, 'stat': after}
            self.statistics['coldHashCount'] += 1; self.statistics['coldBytes'] += after['size']
        self.touched.add(key)
        return value

    def identity(self, path):
        # Preserve the frozen helper's caller-supplied path representation.
        return {'path': str(Path(path)), 'sha256': self.digest(path)}

    def finish(self):
        io.require(self.active, 'No active identity-cache stage')
        for key in self.touched:
            io.require(signature(key) == self.files[key]['stat'], 'Touched file changed before stage end: '+key)
        result = {'files': [dict(self.files[k]) for k in sorted(self.touched)], 'statistics': dict(self.statistics)}
        self.active = False
        return result


@contextmanager
def installed(cache):
    """Expose and restore exactly three registered hash-only bindings."""
    modules = (io, aliases, inputs)
    before = {m: {k: v for k, v in vars(m).items() if inspect.isfunction(v)} for m in modules}
    original = (io.identity, aliases.identity, inputs.sha)
    cached_identity, cached_sha = cache.identity, cache.digest
    inputs._verified.clear()
    io.identity = aliases.identity = cached_identity
    inputs.sha = cached_sha
    try:
        yield
        io.require(io.identity is cached_identity and aliases.identity is cached_identity and inputs.sha is cached_sha,
                   'Declared cache function binding changed')
        for module in modules:
            allowed = {'sha'} if module is inputs else {'identity'}
            io.require(all(vars(module)[k] is value for k, value in before[module].items() if k not in allowed),
                       'A non-hash function was replaced during cached execution')
    finally:
        io.identity, aliases.identity, inputs.sha = original
        inputs._verified.clear()

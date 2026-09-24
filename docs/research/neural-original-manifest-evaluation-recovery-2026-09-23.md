# Legacy-manifest evaluation recovery

The original 270-cell evaluator reached its first task after the complete global
162-selection freeze, then failed with `KeyError: records` before reading inference
scores or entering metric loops. Its actual exit code is 1. The original legacy
manifest has 8 `exactRows`, 3 `draftRows` and 7 `coverageRows`; all 18 recordings
were in the original training corpus. Its seven source groups must remain in the
outside-original exclusion set. No labels, features, training, selection, decoder,
padding, metric functions, protected scopes or production models change.

The separately registered and independently qualified
`analysis/neural_original_manifest_view.py` supplies a scoped, exact-path reader
view adding only `records = exactRows + draftRows + coverageRows`. The legacy
file and its original hash remain unchanged. The adapter preserves every old key,
row, type and order. Its real qualification compares all 162 task memberships
and both production memberships without opening prediction arrays or computing
accuracy. Numerical and hash bindings are guarded; reader restoration is checked.
Every successful execution carries a separate compatibility companion.

## Preserving and recovering the failure

`run-neural-manifest-view-evaluation.py register` pins the frozen original worker,
270-cell plan, all source/qualification references, exact nonstudent/student
partition, complete global gate and sidecar, failed execution/incident/snapshots,
student queue, and previous finalization contract. Registration does not score.
Root review grants each recovery or run action through an exact plan reference.

The `recover` action acquires the original evaluator's flock and verifies the
old worker is absent with actual exit 1. It checks that the only evaluation JSON
artifact across all 270 paths is the exact pinned first-start receipt, and moves
that one file to a fresh NAS archive with identical bytes. No existing destination
can be replaced. The original failure log and start snapshots remain immutable;
future appends to the canonical cell log do not alter the preserved snapshot.
Unexpected outputs or partials fail closed. This is an explicit one-time recovery,
not permission for generic overwrite or retry.

## Execution and completion

The first invocation schedules the exact 243 nonstudent cells. It calls unchanged
original `global_ready`, `panel_ready`, `ready`, `completed`, `evaluation_call` and
`execute_cell`, retaining the original plan, ROOT, argv, namespace, original
publication/queue receipts and shared flock. Only the registered reader view is
installed around `execute_cell`; its declared original numerical callables remain
the same objects. Each new cell additionally writes
`evaluation-<precision>-manifest-view-execution.json`, binding the exact original
argv, result, publication gate and original execution receipt. An existing result
cannot be accepted without both the original gates and this companion.

The later `all` scope starts only after the first scope's actual exit 0, complete
243 ledger, all 27 student inferences' actual exit 0 and independent student cold
publication audit. It validates/reuses the first 243 cells and evaluates the 27
students through the same unchanged functions. It independently accounts for all
270 cells. Completion IDs are serialized in original plan order; the append-only
execution ledger records actual dispatch order. It never edits or relies on the
original `progress.json`, whose known reuse branch can leave a stale count.

Each invocation has a new NAS operational directory and supervisor. The supervisor
records the actual `child.wait()` result, PID/boot/start identity, launch and plan;
an invalid completion body still retains the truthful actual exit code and a
separate failed completion validation. No process absence is interpreted as exit
0. The two unlaunched original-worker resume proposals are preserved and explicitly
superseded: they do not install the required manifest view.

The original CPU 18/19, nice 10, hidden CUDA, two-thread and resource policies remain.
Runtime files, logs, caches and temporary files remain on the direct NAS mount.
No concurrent writer may use the original worker flock. Any subsequent partial
failure requires inspection and a separately reviewed recovery.

## Final publication

`finalize-neural-generalization-v4.py` delegates production to unchanged V2 under
the same qualified reader and writes a production execution companion. It delegates
the final independent numerical audit to unchanged V3 under that reader, retaining
the delegated payload exactly and adding only explicit execution metadata.

Before and after final auditing/report construction, V4 requires the actual full
270-scope exit, exact ledger, prior 243 and student cold proofs, every original
cell publication/queue receipt and all 270 compatibility companions, plus the
production companion. Index rows must match the original 270 cells in declared
order, with one production document. The existing index-base → containment
`bind-index` → final-index chain remains unchanged. All V3/V2 numerical, production,
selection, ignored-range and cold publication gates continue to execute.

Report construction delegates to unchanged V3 using its stored delegated audit.
The final `inventory`, `scopes`, `series` and `tasks` canonical digest must equal
the independently audited digest. Only report metadata adds the outer audit and
compatibility disclosure. A separate report execution companion binds both the
delegated and final JSON. Renderer V5 and its real-output QA remain unchanged.

The focused regression suite covers preservation/rejection of partial artifacts,
truthful actual exit despite an invalid completion, missing companions, all 270
counts including trailing reused cells, exact index membership, unchanged delegated
audit fields, and unchanged report data. Registration, archive recovery, scoring
and final publication remain separately reviewed actions.

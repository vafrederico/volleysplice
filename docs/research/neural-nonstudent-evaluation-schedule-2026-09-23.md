# Nonstudent evaluation scheduling amendment

This execution-only amendment allows the 243 nonstudent task/precision cells to
run after the complete 162-selection freeze and original independent cold proof,
while student full-panel inference finishes. It changes no training, labels,
selection, model, checkpoint, threshold, padding, numerical function, tolerance,
metric, comparison population, or result bytes. No new outcome was opened while
designing it. The frozen original evaluation plan still contains all 270 cells.

The partition is derived from the source-pinned task documents, in original job
order: 135 nonstudent FP32 cells, 54 FP16, and 54 INT8. All 27 distilled-mobile
FP32 cells are excluded. The wrapper imports the exact original worker and calls
its `global_ready`, `panel_ready`, `ready`, `completed`, and `execute_cell`
functions directly. It preserves every original source/reference, panel, fit,
inference, reuse, publication, resource, and per-cell execution gate. There is no
replacement of the worker's ROOT, functions, hash bindings, or original plan.

The original idle worker owns the shared flock even while paused. Therefore this
amendment does not bypass that lock. Registration pins the original recovered
worker's boot ID, PID, start ticks, parent PID, command, launch, and process
receipt. A separately authorized `stop-original` operation validates that exact
identity, the absent global gate, no child process, and no evaluation output or
partial start receipt. It sends SIGTERM to the frozen worker's existing graceful
handler, resumes it only if stopped, and waits for the original recovery wrapper's
actual observed exit receipt. The stop proof requires exit zero, no evaluated
cells, no evaluation-start event, and no output. Unknown termination is never
represented as success. Frozen and existing partial files are never overwritten.

Only after root review and publication of both global freeze files may `run`
start. It requires the stop proof and original PID absence, then acquires the
same original worker flock before any evaluation. It requires CPU cores 18/19,
nice at least 10, hidden CUDA, two CPU threads, disabled bytecode, NAS temporary
storage, and the unchanged original resource floors. It requires all original
270 paths to remain output-free at startup. Each successful cell retains the
original worker's exact argv/namespace and original queue receipt; a separate
append-only ledger truthfully identifies this scheduling caller and binds the
original receipt. It is a fresh invocation, not an unregistered partial-resume
mechanism. Failure preserves any partial output for review.

Completion means only 243 cells, explicitly not all 270. The parent must observe
the scheduler's actual process exit, preserve its execution receipts and log,
and require the independent full student cold proof before restarting the
unchanged original evaluator. The original evaluator then validates and skips
the 243 existing publication/execution receipts and evaluates the remaining 27
student cells. Final indexing/reporting still requires all 270 cells and the
registered V3 finalization/student-cold publication checks. No outcome may be
used to change any frozen selection or scientific configuration.

Registration and commands (all output paths new, below the NAS study root):

```text
run-neural-nonstudent-evaluation.py register --output nonstudent-evaluation-v1/plan.json
run-neural-nonstudent-evaluation.py stop-original --plan .../plan.json --output .../original-stop.json --permit-stop
run-neural-nonstudent-evaluation.py run --plan .../plan.json --original-stop .../original-stop.json --output-root .../execution-v1 --permit-evaluation
```

Registration alone authorizes neither signaling the original worker nor opening
metrics. Both later actions require the root's concrete source/plan review.

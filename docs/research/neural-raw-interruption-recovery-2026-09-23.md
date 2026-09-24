# Raw inference after the September 23 interruption

The resumed session found no running WSL study processes. It did not observe
successful exits from the old processes. The interruption observation is saved
under `recovery-20260923T1758-v1` on the experiment NAS directory.

The FP32 queue had completed 111 of its 135 nonstudent logical tasks: 32 prior AV
tasks, 16 export AV tasks, 15 original tasks and 48 randomized nonstudent tasks.
The final 24 export nonstudent inference tasks had not started. The last completed
stage lacked its orchestration completion receipt despite all individual gates
and its final completed log entry being present.

`recover-neural-cached-raw.py` preserves every original queue file in a new NAS
recovery directory and verifies the exact interruption shape. Its independent
check binds all 111 existing fit, calibration, four-checkpoint, full-42-video raw
inference and reuse gates, including their evidence files. A new scheduling grant
is required before the unchanged final-stage command can run. The recovery checks
all 24 new tasks and rebinds the existing 111 after completion. Original queue
receipts are never backfilled and `priorObservedExitCode` remains null.

The precision raw workers had not launched. Their frozen worker and commands stay
unchanged. `audit-neural-precision-recovery-slot.py` replaces only the obsolete
process-provenance prerequisite and the old incomplete parent receipt. It requires
an actually observed zero exit from a new recovery process, its current boot and
process identity, and absence of those processes. All feature, fit, calibration,
numerical and reuse checks are identical to the original preflight. INT8 also
requires the new verified 135-task FP32 completion receipt. Unit tests reject
missing or failed exits, wrong boots or scopes, live processes and bad identities.

These are scheduling and evidence amendments. They change no training example,
model, checkpoint, decoder, threshold, precision recipe, inference population,
label revision or evaluation policy. Additional-video outcomes remain closed
until all 162 calibration selections and the global selection gate are complete.

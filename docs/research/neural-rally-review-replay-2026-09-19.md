# Replaying the rally-preserving review experiment

The source snapshot includes every original four-head score, timestamp, decoded
event, gold interval, registered numerical source and completed result needed to
replay this experiment. Videos, feature caches and detector checkpoints are
provenance references only; replay does not train detectors or read those files.

Extract `research-source.zip` into a new directory. The archive manifest records
the Python and NumPy versions used. Use that NumPy version for exact result-byte
comparison. No other third-party package is required for numerical replay.

From the extracted directory, verify every file hash, the registered sources,
the normalized arrays and all imports:

```sh
python -B repository/scripts/replay-neural-rally-review-snapshot.py --snapshot .
```

Replay one registered configuration and seed into a **new directory outside the
snapshot**:

```sh
python -B repository/scripts/replay-neural-rally-review-snapshot.py --snapshot . \
  --configuration production--compact_boost--local_events--evidence \
  --seed 3407 --output ../rally-review-replay
```

Use `--all --output ../rally-review-replay-all` instead of configuration and seed
to replay all 108 configuration/seed runs. The wrapper calls the unchanged
registered runner's `execute()` function, retains every independent audit, and
requires each result file to match its archived bytes exactly. It never changes
the snapshot or the original study. The output includes a verification receipt.

The experiment's first execution stopped before publishing any result because
an assertion passed interval dictionaries to an auditor expecting pairs. Its
closed failure receipt, original registration, protocol and 16 preserved source
files are included under `provenance/initial-attempt`. The completed second
execution adds an end-to-end synthetic runner test and fixes that call site; its
proposal rules and evaluation matrix are unchanged.

The immutable experiment contract is
`7386abe4befe96ba4c9607454f5f6936b312dd8459c5dae1a453d4cadc93440d`.

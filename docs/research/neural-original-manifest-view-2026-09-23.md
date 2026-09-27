# Original-manifest compatibility view

The first external evaluation stopped before scoring because the frozen evaluator
read `originalManifest.records`, while its registered legacy file stores the same
eighteen recordings as `exactRows` (8), `draftRows` (3), and `coverageRows` (7).
The failed start receipt and actual exit remain preserved. No training, calibration,
selection, prediction, accuracy result, or frozen source is changed by this repair.

An explicit, path- and hash-bound JSON read view adds only
`records = exactRows + draftRows + coverageRows`. Every existing key, full row,
value, type and ordering remains intact, and the original file bytes and identity
remain authoritative. All seven source groups must survive; using exact rows alone
would incorrectly mark draft-only or export training sources as unseen.

The independent correspondence audit binds all eighteen legacy rows to the
registered normalized input, original training tasks, and inventory. Qualification
compares the full projection and every declared panel membership for all162 tasks
and both production comparators without loading predictions or computing metrics.

Execution changes only `analysis.neural_recall_sweep.read` for that exact file.
The context restores the reader on success or failure and verifies that decoding,
interval/event metrics, hash functions, and caller-supplied evidence bindings remain
unchanged. A separately named companion binds the stage, original argv, output
identities, read sites, qualification, and restoration checks. Neural evaluation,
production evaluation, and the final independent audit must all carry companions.
Publication must reject missing or inconsistent companions. A report wrapper may
add provenance metadata, but the independently audited numerical content stays
unchanged.

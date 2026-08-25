# Side-switch feature loop handoff — 2026-08-25

## Handoff state

Branch: `t3code/analyze-side-switching-features`

Last implementation commit before this handoff document:
`46d850b Add mass-preserving mode transport extraction`.

The working tree was clean at that commit. T9 extraction was started and reached the
beginning of recording 6 of 11, then the session was intentionally interrupted. The
process is no longer running. The extractor writes atomically, and the expected T9
artifact does not exist, so there is no partial output to preserve or delete. Restart
T9 from the beginning.

No T9 label, model, precision, recall, F1, or coefficient run has occurred. Do not
open labels unless the frozen T9 engineering gates pass.

## Completed experimental results

The canonical plans and full ledgers are linked from
[the research index](./README.md). The most recent loop is:

| Iteration | Representation change | Engineering/model result | Decision |
| --- | --- | --- | --- |
| T4 | Same 224x224 detector on a smaller native far-court crop | Engineering pass; model P 50.00%, R 56.00%, F1 52.83% versus T0 P 50.00%, R 52.00%, F1 50.98% | Reject: +1.85 F1 missed +2.00 gate |
| T5 | Court-side temporal team pooling | Visibility 81.10%; reliable swap 16.19%. User-authorized diagnostic model P 49.12%, R 56.00%, F1 52.34%; reliable-swap coefficient negative in 11/11 folds | Reject |
| T6 | Hard dominant-jersey medoid consensus | Separation 0.5010 and positive margin 28.04%, but both-team visibility 61.57% and reliable swap 14.42% | Reject before labels |
| T7 | Soft robust medoid-distance weighting | Exact T5 visibility; separation 0.3387, positive margin 25.48%, reliable swap 19.39% | Reject before labels: separation gate |
| T8 | Explicit weighted two-mode set with Chamfer cost | Exact T5 visibility; separation 0.3707, positive margin 18.43%, reliable swap 14.58% | Reject before labels: separation and direction gates |
| T9 | Same T8 modes with exact support-mass optimal transport | Code and tests committed; full extraction interrupted, no artifact | In progress |

Important interpretation: T8 recovered most separation, but 90.69% of available
tracks had a substantive secondary mode and nearest-mode Chamfer matching allowed
several source modes to use one destination mode. T9 changes only that cost; it keeps
T8's modes, supports, availability, detector work, candidates, and decoder fixed.

## Immutable external artifacts

These artifacts are intentionally outside Git:

| Artifact | SHA-256 | Size |
| --- | --- | ---: |
| `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t7-soft-robust-jersey-consensus-features-v1.json` | `d592961f184122afe6e88c95e6405c2bdc57bd71986733480021d46037cba57a` | 15,774,791 bytes |
| `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t8-two-mode-jersey-transport-features-v1.json` | `4da68234930052d3669e7b40b003a6c6633068bd88f18e7629ae56536ff5d302` | 19,569,102 bytes |

Expected T9 output, currently absent:

`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t9-mass-preserving-mode-transport-features-v1.json`

The T9 source identity is the T8 SHA above. Its frozen contract is
[T9 mass-preserving mode transport](./side-switch-t9-mass-preserving-mode-transport-plan-2026-08-25.md).

## Resume T9

Use a Python environment with `analysis/requirements.txt`. The validated environment
on the source machine used Python 3.13, NumPy 2.2.4, and OpenCV 4.12.0.88. On a new
machine, from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r analysis/requirements.txt
PYTHONPATH=. .venv/bin/python -m unittest \
  analysis.tests.test_side_switch_t8_two_mode_transport \
  analysis.tests.test_side_switch_t9_mass_transport
test ! -e /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t9-mass-preserving-mode-transport-features-v1.json
PYTHONPATH=. .venv/bin/python scripts/extract-side-switch-t9-mass-transport-features.py
```

The complete run previously took about 26 minutes for T8 and is expected to make
exactly 25,400 detector tile calls over 635 endpoints and 3,175 frames. Do not run two
T9 extractors concurrently against the same output path.

After completion:

```bash
T9_ARTIFACT=/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t9-mass-preserving-mode-transport-features-v1.json
sha256sum "$T9_ARTIFACT"
jq '{engineeringDecision, parity, engineering: (.engineering | del(.correlations)), performance, sources}' "$T9_ARTIFACT"
```

If `engineeringDecision` is `fail`, do not train T9. Record the artifact hash,
measurements, failed checks, and next representation hypothesis in the T9 plan, the
main feature-development plan, feature-importance report, and research index; run
focused tests and commit the result.

If it is `pass`, first implement and commit the matched model profile/runner before
loading labels. The comparison is exact boundary T0 (`34` features) versus `34 + T9
core`, with immutable T4 as comparator. Follow the conditional gates in the T9 plan.
Use the T5 diagnostic runner and profile commits as structural references:

- `d0dfdbd Add diagnostic T5 model experiment`;
- `analysis/side_switch_feature_development.py`;
- `scripts/run-side-switch-feature-development.py`; and
- `analysis/tests/test_side_switch_feature_development.py`.

Then commit the model wiring, run the model once, write all event/calibration/fold/
coefficient results into the plans, and commit the result. Do not promote or port an
opened-scope success; new recording-held gold remains required.

## Move the repository to another machine

### Preferred: push and fetch the branch

On the source machine, after this handoff commit:

```bash
git status --short
git push -u origin t3code/analyze-side-switching-features
```

On the destination machine:

```bash
git clone ssh://git@internal.example:2222/vafrederico/volleycut.git
cd volleycut
git fetch origin t3code/analyze-side-switching-features
git switch --track origin/t3code/analyze-side-switching-features
git log -1 --oneline
```

If the repository already exists, omit `git clone` and run the remaining commands.

### Offline alternative: Git bundle

On the source machine:

```bash
git bundle create volleycut-side-switch-handoff.bundle \
  t3code/analyze-side-switching-features
git bundle verify volleycut-side-switch-handoff.bundle
```

Copy the bundle by the user's normal secure file-transfer method. On the destination:

```bash
git clone volleycut-side-switch-handoff.bundle volleycut
cd volleycut
git switch t3code/analyze-side-switching-features
```

## Move or mount the non-Git data

The Git branch is not sufficient by itself. The source machine currently has:

- 3.5 GiB under `/mnt/freenas/volleycut/labeling-v1-2026-08-09`;
- a 2.6 MiB pinned detector directory under that root; and
- 11 source videos totaling about 45 GiB under
  `/mnt/freenas/volleycut-raw-no-backup`.

Prefer mounting the same FreeNAS shares at the same absolute paths. The artifact
provenance and default commands deliberately use those paths. If copying instead,
copy the complete labeling root plus these exact videos while preserving names and
bytes:

```text
PXL_20260816_160023210.mp4
PXL_20260816_161923155.mp4
PXL_20260816_164327879.mp4
PXL_20260816_171720964.mp4
PXL_20260816_180646590.mp4
PXL_20260816_183701800.mp4
PXL_20260816_190429172.mp4
PXL_20260816_193307688.mp4
PXL_20260816_203801418.mp4
PXL_20260816_210449857.mp4
PXL_20260816_212717581.mp4
```

After mounting/copying, verify the T7/T8 hashes above and confirm that all video paths
listed by this command exist:

```bash
jq -r '.extractionAudit[] | .videoPath' \
  /mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-t8-two-mode-jersey-transport-features-v1.json \
  | sort -u | xargs -d '\n' -n1 test -f
```

## Continue in a new coding-agent session

The repository handoff is the canonical portable context. Start a new T3 Code/Codex
session on the destination machine in this branch and use:

```text
Read AGENTS.md and docs/research/side-switch-feature-loop-handoff-2026-08-25.md
completely. Continue the side-switch feature loop from T9. Preserve every frozen
engineering/model gate, commit between preregistration, implementation, extraction
result, and conditional model steps, and keep the plans and research index current.
```

Do not copy `~/.t3/userdata`, its SQLite database, or its `secrets/` directory to
transfer this work. Those files contain machine-local runtime state and signing/access
keys, and a raw cross-machine copy is neither a safe nor a documented session-transfer
mechanism. If the T3 Code UI exposes an account-backed handoff/share/sync action, it
may be used in addition to Git, but the new session should still read this document;
the repository plus immutable external artifacts is the reproducible source of truth.

Official OpenAI documentation searched during this handoff did not establish a
portable file-level export/import procedure for this T3 Code thread. Therefore this
document does not claim that copying a local session database will work.

## Commit sequence for the current tail

```text
08e3713 Preregister soft robust jersey consensus
6b56713 Add soft robust jersey consensus extraction
20d1d11 Record soft robust jersey consensus result
8822511 Preregister two-mode jersey transport
cea8b33 Add two-mode jersey transport extraction
540893c Record two-mode jersey transport result
61f2f6c Preregister mass-preserving mode transport
46d850b Add mass-preserving mode transport extraction
```

Earlier T4–T6 and T5 diagnostic commits are visible immediately below these in Git
history and are fully summarized in the linked plans.

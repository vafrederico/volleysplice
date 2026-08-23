# Side-switch parity feasibility — 2026-08-23

## Decision

Retain persistent side parity as the preferred **candidate architecture**, but reject
the existing V5 motion-component orientation coordinate as its state emission. Do not
replace the current `local-peak-soft-count` winner and do not port this diagnostic to
the browser or Android.

The important positive result is upstream: all 50 reviewed switches have stable rally
observations on both sides, and their truth-parity boundaries contain all 50 markers.
The important negative result is representation: the frozen score-zero palette sign
recognizes initial parity but usually fails to recognize swapped parity.

Proceed to the separately designed same-versus-swapped continuity verifier. A future
parity head should use stronger team identity, recording-relative calibration, and
abstention rather than treating zero as a universal state boundary.

## Experiment

Loop 1 from the
[rare-event improvement plan](./side-switch-rare-event-improvement-plan-2026-08-23.md)
was implemented as a new rally-level diagnostic. It reuses the detector-free V5
extractor but preserves the per-rally values that the earlier V5 artifact discarded.

For each of the 11 fully reviewed raw-phone recordings:

1. seven 256×144 frames are sampled across every production-detected rally;
2. motion-component player proposals produce near/far HSV palettes;
3. the first three score-zero rallies define a recording-local team-side prototype;
4. every later rally receives an orientation coordinate and quality;
5. manual marker order defines parity state zero/one, with rally observations touching
   ±4 seconds around a physical marker masked from state scoring; and
6. a causal state decoder starts at state zero and requires one, two, or three
   opposite-sign observations before confirming a flip.

No classifier, threshold, or feature weight was fit. The quality sensitivity uses a
label-independent floor equal to half of the recording's median quality. Event
proposals are the dead-time interval between the last accepted-state rally and the
first opposite-state rally. Markers are matched one-to-one by strict interval
containment and by the same interval padded four seconds on each side.

This is opened development evidence. All 11 recordings and all 50 markers had already
been reviewed and used by prior side-switch decisions.

## Results

Extraction produced 635 rally observations with zero frame failures. Thirty-three
observations overlap a transition mask, leaving 602 stable observations.

### State recognition

| Metric | Result |
| --- | ---: |
| Stable observations | 602 |
| Accuracy | 63.95% |
| Macro per-video accuracy | 63.66% |
| Balanced accuracy | 58.62% |
| Macro per-video balanced accuracy | 57.92% |
| State-zero recall | 95.93% (330/344) |
| State-one recall | 21.32% (55/258) |

The asymmetry is decisive. The first-three-rally prototype makes the initial assignment
easy by construction, but the same-versus-swapped cost rarely crosses zero after a
physical switch. State-one median coordinates remain positive in 10/11 recordings;
only `183701800` has a negative state-one median. `203801418` and `210449857` have zero
state-one recall. This is not a decoding or missing-frame failure.

The coordinates often move downward after a switch even when they do not become
negative. That makes recording-relative calibration worth revisiting with new data,
but selecting such a threshold on these same 11 opened videos would not establish
generalization.

### Candidate ceilings

| Candidate construction | Proposals | Strict TP / recall | ±4 TP / recall |
| --- | ---: | ---: | ---: |
| Every consecutive detected-rally boundary | 624 | 35 / 70% | 43 / 86% |
| Truth-parity boundary between stable rallies (oracle) | 50 | 50 / 100% | 50 / 100% |

The 624-boundary universe is larger than the earlier 352 reviewed modern gap rows and
raises the ±4 candidate ceiling from 33/50 to 43/50. The truth-parity oracle is not a
deployable result—it uses manual states—but proves that stable observations bracket all
50 switches. When a switch or rally mistake interrupts adjacent boundaries, comparing
the last stable state before it with the first stable state after it recovers the
remaining seven.

### Visual sign + persistence

Strict and ±4 results are identical for these state-derived intervals.

| Decoder | Proposals | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sign, persistence 1 | 90 | 9 | 81 | 41 | 10.00% | 18.00% | 12.86% |
| Sign, persistence 2 | 18 | 5 | 13 | 45 | 27.78% | 10.00% | 14.71% |
| Sign, persistence 3 | 6 | 2 | 4 | 48 | 33.33% | 4.00% | 7.14% |
| Persistence 2 + half-median quality | 18 | 4 | 14 | 46 | 22.22% | 8.00% | 11.76% |

Persistence suppresses flicker but cannot repair a biased emission. The primary
persistence-2 variant emits nothing in six videos and finds no true event in eight.
The quality filter removes one true match without reducing proposal count. None is
competitive with the 62-proposal current winner at 25 TP / 37 FP / 25 FN.

## What this changes

- Preserve the dense state-change formulation. It can expand candidate coverage and
  naturally tolerates redo/missing rally markers by comparing stable observations.
- Do not reuse `orientationCoordinate >= 0` as the parity classifier.
- Do not treat persistence as a precision solution until the swapped state is
  observable; longer persistence only destroys recall here.
- Build the next parity representation around direct same/swapped pair costs, multiple
  stable windows, better team isolation, recording-relative calibration, and an
  abstention/quality head.
- Continue immediately with the planned same-side continuity verifier because it asks
  the lower-risk local question and can veto false current-winner proposals without
  requiring a globally calibrated absolute state.

## Implementation and reproducibility

- Pure parity labels, state metrics, persistence decoding, and interval matching:
  [`side_switch_parity.py`](../../analysis/side_switch_parity.py)
- Immutable extractor/evaluator:
  [`evaluate-side-switch-parity-feasibility.py`](../../scripts/evaluate-side-switch-parity-feasibility.py)
- Focused tests:
  [`test_side_switch_parity.py`](../../analysis/tests/test_side_switch_parity.py)
- NAS artifact:
  `/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/side-switch-parity-feasibility-v1.json`
- Artifact SHA-256:
  `c2eda7a486433bf27aa30a529cb7cd4a908a0b29bcd8bd0f0d88c4e3dc317688`

The artifact binds these source hashes:

| Source | SHA-256 |
| --- | --- |
| `full-nas-video-corpus-v1.json` | `c177f2936dc1ea7f2953cf94a6b9720cd6d4bf4e652a4c7409499374c9fe1e24` |
| `side-switch-v4-multiframe-normalized-features.json` | `4d9ae424a48b81b630c72ad8650fbc2cf68be41b350b593339fe774564717069` |
| `full-video-side-switch-markers-full-nas-v1.json` | `50aabec1ecf5fb9d67edbdfa0a29a1997f4add5264c0b4a59a55b79ef6979e00` |

Run from the repository root:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate-side-switch-parity-feasibility.py
PYTHONPATH=. .venv/bin/python -m unittest analysis.tests.test_side_switch_parity
```

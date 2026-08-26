# Side-switch provisional recording-held gold plan — 2026-08-25

## Decision

Use the two newly reviewed Shoreline/KB recordings as a frozen, **provisional
recording-held gold revision** for the already-frozen side-switch candidates. Do not
fit a classifier, select a threshold, change a feature, or choose a decoder from these
two recordings.

This is recording-held rather than source-group-held evidence. Both source groups were
used by historical side-switch work, and the labeling tasks were initialized with
model proposals. The result must not be described as a pristine protected test.

## Frozen gold revision

| Recording | Source group | Video SHA-256 | Label SHA-256 | Human switches | Draft state |
| --- | --- | --- | --- | ---: | --- |
| `grass-source-11` | `shoreline-kob-20250616` | `2ca9bd9324a05807992f06893f53056c47a6c928b7a420d855d83726a9e496b0` | `ced51c22114541acec74b10d6992109e90425f6a53f0bc7e4581a6f4429fdaa5` | 4 | `in-progress`; `continuousVideoReviewed=false` |
| `grass-source-07` | `kb-kob-20250614` | `e377f7e7e1bd0c6290d76f346ed5f24858e917be91dc1bef5323f165476883ee` | `8bada37b00829755ef28e82fc2d549cba4c782b8e9fefbc58a31e6f087ac7a4e` | 3 | `in-progress`; `continuousVideoReviewed=true` |

The exact files are under
`/mnt/freenas/volleycut/intake-2026-08-25-shoreline-kb`. A later human edit creates a
new gold revision and must not overwrite this result.

## Evaluation contract

1. Reconstruct candidates from the frozen production inference and its cached F104
   inputs, not from the human-corrected rally intervals.
2. Extract features without loading `sideSwitches`; load the frozen marker snapshot
   only after candidate features and all classifier identities are fixed.
3. Apply the immutable full-development classifiers and thresholds trained on the 11
   opened raw-phone recordings. The two gold recordings remain evaluation-only and
   have `consent.train=false`.
4. Report the deployed full-union 34-input winner, then the matched boundary-only T0,
   T4, T14, T19, and T20 comparisons. These candidates were all specified and fitted
   before the new labels existed.
5. Use the existing ranked-candidate decoder unchanged. Report strict and ±4-second
   pooled event precision, recall, F1, TP/FP/FN, proposals, candidate coverage, and
   per-recording results.
6. Remove candidates whose transition anchor lies inside a label `ignoredInterval`.
7. Treat the result as provisional until both label files are marked complete and
   continuously reviewed. Never silently replace the bound label hashes.

## Interpretation gates

- This evaluation may compare already-frozen candidates, but it may not authorize a
  post-result feature edit on the same two recordings.
- Promotion still requires a completed label revision and preferably a new
  source-group-held set whose annotation was not seeded by the evaluated model.
- If draft completion changes either label hash, rerun all candidates against the new
  revision and retain both artifacts.

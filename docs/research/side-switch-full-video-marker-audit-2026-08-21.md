# Side-switch full-video marker audit — 2026-08-21

## Decision

Keep the side-switch specialists out of automatic production use. The completed
continuous review establishes 50 physical side-switch events across the 11 raw-phone
recordings, replacing the earlier candidate-conditioned positive inventory as the
canonical truth for this scope.

The most important new result is upstream: with each inter-rally proposal gap expanded
by four seconds on both sides, only 33/50 human events are covered by any of the 352
modern candidate gaps. The maximum achievable end-to-end recall from re-ranking those
same gaps is therefore 66%. Seventeen confirmed switches require better candidate
generation before any specialist or decoder can see them.

Among the comparable frozen outputs:

- independent no-cadence V5-state has the highest macro per-video recall, 56.36%, but
  emits 91 proposals and has only 30.77% pooled precision;
- independent soft count has the highest macro per-video precision, 55.91%, but matches
  only 20/50 events; and
- local peak plus soft count has the strongest pooled F1, 44.64%, with 25 matches from
  62 proposals.

V2 ranks slightly above no-cadence by macro recall, but that is not a fair held-out
comparison: seven of its eleven sequences are deterministic development-scope replay.
It is retained in the list only because the earlier 19-decoder inventory included it.

## Human truth policy

All 11 raw-phone recordings are marked fully reviewed. The marker artifact contains
4–5 manual physical-switch points per recording, 50 total. These direct full-video
markers are the exhaustive event inventory for this audit.

The 35 older `switch` decisions were produced by reviewing heuristic candidate gaps.
They are not unioned into the new truth because the completed pass placed markers for
all physical events, including events already visible in candidate rails. Unioning the
two stores would duplicate events and restore candidate-generation bias. Under the
primary matching contract, the older decisions match 30 manual events, contribute five
extra or duplicate proposals, and miss 20 manual events.

## Matching contract

A decoder output is an inter-rally interval with `gapStart`, `gapEnd`, and a
`transitionTime` anchor. A human label is one point known to lie inside a physical
switch. Primary evaluation matches a marker when it lies inside:

```text
[gapStart - 4 seconds, gapEnd + 4 seconds]
```

Assignment is monotonic and one-to-one. It first maximizes the number of matched events,
then minimizes total absolute distance to proposal anchors. One proposal cannot claim
two human events, and neighboring padded gaps cannot both claim one marker. The
four-second value is event-boundary allowance; it is separate from the earlier decoder
candidate margin measured in rally opportunities.

Strict unexpanded containment is also frozen as a sensitivity result. The primary
four-second allowance raises the candidate-coverage ceiling only from 31/50 to 33/50,
so the upstream conclusion is not caused by a broad tolerance.

Macro precision and recall are calculated per recording and then averaged across all
11 recordings. Pooled values use total TP/FP/FN. Every decoder emits at least one
proposal in every recording, so no undefined per-video precision is imputed.

## Sorted by average per-video recall

| Rank | Decoder | Macro R | Macro P | TP/FP/FN | Pooled P/R/F1 | Proposals/video |
| ---: | --- | ---: | ---: | --- | --- | ---: |
| 1 | V2 temporal no-op* | 57.27% | 24.32% | 28/110/22 | 20.29% / 56.00% / 29.79% | 12.5 |
| 2 | Independent control | 56.36% | 33.06% | 28/63/22 | 30.77% / 56.00% / 39.72% | 8.3 |
| 3 | V5-state no cadence | 56.36% | 33.06% | 28/63/22 | 30.77% / 56.00% / 39.72% | 8.3 |
| 4 | Local peak | 54.55% | 38.89% | 27/46/23 | 36.99% / 54.00% / 43.90% | 6.6 |
| 5 | Independent + soft count + context | 52.73% | 30.52% | 26/59/24 | 30.59% / 52.00% / 38.52% | 7.7 |
| 6 | Local peak + soft count | 50.91% | 41.19% | 25/37/25 | 40.32% / 50.00% / 44.64% | 5.6 |
| 7 | Independent + production context | 50.45% | 37.36% | 25/47/25 | 34.72% / 50.00% / 40.98% | 6.5 |
| 8 | Local peak + soft count + context | 46.82% | 40.15% | 23/33/27 | 41.07% / 46.00% / 43.40% | 5.1 |
| 9 | Local peak + context (selected) | 46.82% | 38.80% | 23/37/27 | 38.33% / 46.00% / 41.82% | 5.5 |
| 10 | Independent + soft count | 40.00% | 55.91% | 20/24/30 | 45.45% / 40.00% / 42.55% | 4.0 |
| 11 | V1 original | 29.09% | 15.75% | 14/99/36 | 12.39% / 28.00% / 17.18% | 10.3 |
| 12 | V1 no-blurry-beach | 25.00% | 13.69% | 12/94/38 | 11.32% / 24.00% / 15.38% | 9.6 |
| 13 | V5-state cadence | 23.18% | 31.52% | 11/27/39 | 28.95% / 22.00% / 25.00% | 3.5 |
| 14 | V5 | 23.18% | 23.94% | 11/33/39 | 25.00% / 22.00% / 23.40% | 4.0 |
| 15 | V6 | 18.18% | 19.09% | 9/39/41 | 18.75% / 18.00% / 18.37% | 4.4 |
| 16 | V6-state cadence | 13.18% | 19.24% | 6/26/44 | 18.75% / 12.00% / 14.63% | 2.9 |
| 17 | V4 | 12.27% | 20.00% | 6/31/44 | 16.22% / 12.00% / 13.79% | 3.4 |
| 18 | V3 original | 11.36% | 12.05% | 6/49/44 | 10.91% / 12.00% / 11.43% | 5.0 |
| 19 | V3 re-anchored/capped | 3.64% | 3.33% | 2/42/48 | 4.55% / 4.00% / 4.26% | 4.0 |

`Independent control` and `V5-state no cadence` are intentionally retained as separate
inventory rows even though their frozen proposal streams are identical.

## Sorted by average per-video precision

| Rank | Decoder | Macro P | Macro R | TP/FP/FN | Pooled P/R/F1 | Proposals/video |
| ---: | --- | ---: | ---: | --- | --- | ---: |
| 1 | Independent + soft count | 55.91% | 40.00% | 20/24/30 | 45.45% / 40.00% / 42.55% | 4.0 |
| 2 | Local peak + soft count | 41.19% | 50.91% | 25/37/25 | 40.32% / 50.00% / 44.64% | 5.6 |
| 3 | Local peak + soft count + context | 40.15% | 46.82% | 23/33/27 | 41.07% / 46.00% / 43.40% | 5.1 |
| 4 | Local peak | 38.89% | 54.55% | 27/46/23 | 36.99% / 54.00% / 43.90% | 6.6 |
| 5 | Local peak + context (selected) | 38.80% | 46.82% | 23/37/27 | 38.33% / 46.00% / 41.82% | 5.5 |
| 6 | Independent + production context | 37.36% | 50.45% | 25/47/25 | 34.72% / 50.00% / 40.98% | 6.5 |
| 7 | Independent control | 33.06% | 56.36% | 28/63/22 | 30.77% / 56.00% / 39.72% | 8.3 |
| 8 | V5-state no cadence | 33.06% | 56.36% | 28/63/22 | 30.77% / 56.00% / 39.72% | 8.3 |
| 9 | V5-state cadence | 31.52% | 23.18% | 11/27/39 | 28.95% / 22.00% / 25.00% | 3.5 |
| 10 | Independent + soft count + context | 30.52% | 52.73% | 26/59/24 | 30.59% / 52.00% / 38.52% | 7.7 |
| 11 | V2 temporal no-op* | 24.32% | 57.27% | 28/110/22 | 20.29% / 56.00% / 29.79% | 12.5 |
| 12 | V5 | 23.94% | 23.18% | 11/33/39 | 25.00% / 22.00% / 23.40% | 4.0 |
| 13 | V4 | 20.00% | 12.27% | 6/31/44 | 16.22% / 12.00% / 13.79% | 3.4 |
| 14 | V6-state cadence | 19.24% | 13.18% | 6/26/44 | 18.75% / 12.00% / 14.63% | 2.9 |
| 15 | V6 | 19.09% | 18.18% | 9/39/41 | 18.75% / 18.00% / 18.37% | 4.4 |
| 16 | V1 original | 15.75% | 29.09% | 14/99/36 | 12.39% / 28.00% / 17.18% | 10.3 |
| 17 | V1 no-blurry-beach | 13.69% | 25.00% | 12/94/38 | 11.32% / 24.00% / 15.38% | 9.6 |
| 18 | V3 original | 12.05% | 11.36% | 6/49/44 | 10.91% / 12.00% / 11.43% | 5.0 |
| 19 | V3 re-anchored/capped | 3.33% | 3.64% | 2/42/48 | 4.55% / 4.00% / 4.26% | 4.0 |

*V2 has mixed development and four-recording frozen-evaluation scope. Do not select it
from this ranking.

## Upstream misses by recording

| Recording suffix | Human | Candidate-covered | Upstream misses |
| --- | ---: | ---: | ---: |
| `160023210` | 4 | 4 | 0 |
| `161923155` | 5 | 3 | 2 |
| `164327879` | 4 | 3 | 1 |
| `171720964` | 5 | 5 | 0 |
| `180646590` | 5 | 3 | 2 |
| `183701800` | 4 | 4 | 0 |
| `190429172` | 5 | 3 | 2 |
| `193307688` | 4 | 2 | 2 |
| `203801418` | 5 | 1 | 4 |
| `210449857` | 4 | 4 | 0 |
| `212717581` | 5 | 1 | 4 |

The last two poorly covered recordings show why decoder-only cleanup is insufficient.
An end-to-end successor needs a side-switch candidate path that does not depend on the
production rally intervals already being correct.

## Strict-containment sensitivity

| Decoder | Strict TP/FP/FN | Strict P/R/F1 | Four-second TP/FP/FN | Four-second P/R/F1 |
| --- | --- | --- | --- | --- |
| V5-state no cadence | 26/65/24 | 28.57% / 52.00% / 36.88% | 28/63/22 | 30.77% / 56.00% / 39.72% |
| Local peak | 25/48/25 | 34.25% / 50.00% / 40.65% | 27/46/23 | 36.99% / 54.00% / 43.90% |
| Local peak + soft count | 23/39/27 | 37.10% / 46.00% / 41.07% | 25/37/25 | 40.32% / 50.00% / 44.64% |
| Independent + soft count | 18/26/32 | 40.91% / 36.00% / 38.30% | 20/24/30 | 45.45% / 40.00% / 42.55% |

The leading decoders gain only two matches from the boundary allowance, so their
relative interpretation is stable. Production context still does not transfer as a
useful cleanup signal, and cadence/re-anchoring remains strongly harmful.

## Artifacts and reproduction

All durable paths below are under
`/mnt/freenas/volleycut/labeling-v1-2026-08-09/reports/side-switch/`.

| Artifact | SHA-256 |
| --- | --- |
| `full-video-side-switch-markers-full-nas-v1.json` | `50aabec1ecf5fb9d67edbdfa0a29a1997f4add5264c0b4a59a55b79ef6979e00` |
| `side-switch-full-video-marker-evaluation-2026-08-21.json` | `b48058a517602c7cb1fd8c3ccc5a8a6d8807e0a7dc9eab5d0f9fe765b82795d7` |
| `side-switch-full-video-marker-evaluation-2026-08-21.md` | `8a9ecb57e4dc509e36811c68bacb779a80b54fa155a2fe72ae381b09a1338a6a` |
| `side-switch-full-video-marker-evaluation-2026-08-21.csv` | `c626b050e6fea812d8ed3b247ddae64f7024dabfd26e2f730d90b76cb0d6b920` |

The JSON contains both padding cases, every proposal interval, every one-to-one match,
all false-positive anchors, every missed human time, the upstream-versus-decoder miss
decomposition, per-video metrics, both rankings, source hashes, and exact implementation
file hashes. The Markdown artifact expands the complete 11-video × 19-decoder review;
the CSV is the long-form machine-readable table.

```bash
PYTHONPATH=. /home/developer/volleycut/.venv/bin/python \
  scripts/evaluate-side-switch-full-video-markers.py
```

The script refuses to overwrite the immutable outputs.

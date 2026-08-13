# Multistate follow-up execution — 2026-08-12

This ledger executes the follow-up order frozen after the serve-anchored four-state model improved
aggregate development metrics but failed its live-recall and ordinary-long preservation gates. All
reported measurements are development-only source-group out-of-fold (OOF) unless explicitly marked
as a label-only oracle. The protected retrospective split remains closed.

## Frozen order and hypotheses

The follow-up sequence was fixed before running new candidates:

1. diagnose per-state OOF classification/calibration, serve rejection, decoded durations, and
   label-only serve/end anchors;
2. evaluate a conservative binary-preserving hybrid under nested source-group selection and hard
   live-recall/ordinary-long guardrails;
3. ablate existing-label improvements separately: immediate-result supervision, calibrated OVR
   versus joint multiclass emissions, mixture duration priors, constrained selection, and safely
   cross-fitted serve/terminal evidence;
4. prepare label-dependent runner hooks for the 45-rally transition pilot and hard negatives without
   waiting for those labels.

The smallest frozen no-label hypotheses are:

- independently class-balanced one-vs-rest state heads are not calibrated joint posteriors;
- a single weak SERVE sample can veto an otherwise valid event;
- one geometric prior cannot represent the nearly fixed SETUP state and multimodal rally duration;
- preserving high-confidence binary ordinary-long proposals can recover the recall floor while
  multistate paths refine boundaries or add missing short events.

No oracle result can select or promote an inference candidate. Any selectable threshold, duration,
calibration, or edge weight must be selected from inner OOF predictions inside each outer fold; the
outer OOF diagnostic cache is assessment-only.

## 1. OOF state and anchor diagnostics

Implementation checkpoint: `1819139 Add multistate OOF diagnostic runner`.

The runner exactly reproduced all four frozen outer-fold binary and state-model fingerprints and
aggregate metrics, then stored both raw balanced-head sigmoid scores and the v1 normalized scores.
It evaluated 31,908 valid samples across eight development recordings. Test was neither prepared nor
read.

### State classification and calibration

| state | prevalence | mean normalized probability | argmax precision | argmax recall | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| DEAD | .5205 | .3342 | .7546 | .5731 | .1887 |
| SETUP | .1875 | .2603 | .4432 | .5634 | .0943 |
| SERVE | .0096 | .0944 | .1303 | .4804 | .0848 |
| LIVE | .2824 | .3111 | .6160 | .7220 | .0448 |

SERVE is especially distorted: a state with 0.96% prevalence receives 9.44% average probability
after the independently balanced heads are normalized. The domain failure is concentrated in the
held-out Yang beach source. Only 8/71 true Yang anchors are SERVE argmax and 15/71 have a decoded
SERVE within 0.5 seconds; SPU has 58/74 SERVE argmax and 43/74 within 0.5 seconds. Mean true-anchor
SERVE probability is .2256 on Yang versus .4726 on SPU.

### Durations

Truth has median duration 5.899 seconds and 63/306 events at most three seconds. Multistate output has
median 7.267 seconds and only 36/285 predictions at most three seconds; the binary control has median
7.858 seconds and only 7/308 predictions at most three seconds. Decoded SETUP runs range from 0.233
to 53.25 seconds with median 4.25 seconds, despite target SETUP runs being almost fixed around five
seconds. This supports explicit empirical SETUP and mixture LIVE priors rather than another search
over one geometric exit hazard.

### Development-label oracle results

| diagnostic | predictions / strict matches | event F1 | time IoU | live recall | objective |
| --- | ---: | ---: | ---: | ---: | ---: |
| unaltered multistate | 285 / 164 | .5550 | .5457 | .7826 | .5863 |
| force gold SERVE states | 350 / 210 | .6402 | .6311 | .8690 | .6718 |
| force gold end DEAD states | 349 / 219 | .6687 | .5917 | .7865 | .6633 |
| force both anchor states | 409 / 269 | .7524 | .6865 | .8665 | .7498 |
| force the complete gold state path | 306 / 306 | 1.0000 | .9762 | .9959 | .9923 |

Forcing SERVE alone restores 8.64 points of live recall and 8.55 points of objective. Forcing end
DEAD alone raises short-event strict recall from .1746 to .5556 and service-fault strict recall from
.2439 to .5854. Forcing both raises ordinary-long, short, and service-fault strict recall to .9022,
.7937, and .7805 respectively. The exact 306/306 gold-state result confirms the transition graph and
interval extraction are sound; the actionable bottlenecks are emissions, calibration, and boundary
evidence. Forced rows use development labels and are diagnostic ceilings, not candidate performance.

Artifacts:

- `reports/feature-order-2026-08-12/multistate-followup-diagnostics-v1.json`, SHA-256
  `3c0f2da46beaa656f43169fcf00f7cf38155ef134c0c8f8802c905c551d11b42`;
- `models/feature-order-2026-08-12/multistate-followup-oof-v1/oof-cache.json`, SHA-256
  `9e17d826390a195b96e0046edfa3892db28518c81c3ddee8cf1a36883cae9496`.

All eight per-recording cache hashes match their index. Provenance records clean Git commit
`181913999b771a971b0016b11a59a4f2ac29f98f`; wall time was 45.102 seconds.

## Protected-test status

The protected retrospective split is still unopened. It may be opened only for a candidate whose
full nested development report passes every predeclared aggregate, paired-source, live-recall,
ordinary-long, and short/fault gate. Diagnostic oracles are permanently ineligible.

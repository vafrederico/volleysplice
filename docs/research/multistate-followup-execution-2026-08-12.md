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

## 2. Conservative binary-preserving hybrid

Implementation checkpoint: `3af58f6 Add nested conservative multistate hybrid`.

The hybrid evaluated exactly four predeclared modes inside every outer fold: binary no-op, bounded
boundary snapping, strict isolated short rescue, and snapping plus rescue. Every mode structurally
retains each binary proposal. Snaps are limited to 0.75 seconds and long binary proposals must retain
at least 90% coverage and .85 IoU; rescues require independent high binary support, confident state
edges, isolated low-score flanks, and a two-per-recording cap. Each outer fold refit three true inner
OOF models that excluded the held source; every frozen binary and state-model fingerprint matched.

All four outer folds selected binary no-op. The no-op exactly reproduced the frozen binary control:

| output | F1 | time IoU | live recall | live precision | objective | ordinary strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen binary control | .52443 | .51310 | .83818 | .56952 | .56809 | 148 |
| nested conservative hybrid | .52443 | .51310 | .83818 | .56952 | .56809 | 148 |

The non-no-op inner evidence was too small or inconsistent:

- boundary snapping changed 58–102 boundaries per pooled inner validation set, but objective changes
  ranged only from -.0041 to +.0025 and never cleared the frozen +.01 macro-source gain gate; the SPU
  outer selection pool also lost ordinary-long strict matches;
- the strict rescue found no accepted addition in three pools and one addition in the SPU pool,
  gaining only .0020 objective and not clearing the macro or short-match gates;
- every selected no-op retained the live-recall and ordinary-long floor, preventing the original
  multistate candidate's source-specific regressions.

The final paired result is neutral: zero positive, zero negative, and four neutral source groups.
The hybrid is not promoted and the binary control remains selected. No protected labels were read.

Artifact:

- `reports/feature-order-2026-08-12/multistate-conservative-hybrid-v1-development.json`, SHA-256
  `2cb9fb2edcc59d0d04a549c9a6ee116e6a3e2548d682a65e3d574f541ab9e9a1`.

The report pins clean Git commit `3af58f6`; wall time was 190.334 seconds.

## 3. Existing-label emission and duration ablation

Implementation checkpoint: `4c98042 Add nested existing-label multistate ablations`.

The first existing-label study evaluated a fixed 2x2 family: frozen v1 balanced OVR emissions or a
fold-prior correction, crossed with frozen geometric duration priors or an empirical SETUP plus
two-geometric LIVE prior. Every prevalence and duration estimate used only its training fold, every
state-head fingerprint reproduced, and each outer holdout selected a candidate using three inner
source-group OOF folds under recall, precision, retained-dead-time, event-F1, and outcome-slice
guardrails. All four fixed configurations were also reported on outer OOF data for diagnostic
interpretation; those diagnostic rows did not select the nested candidate.

All four outer folds selected the exact v1 no-op. The fixed outer-OOF diagnostic rows were:

| emissions / durations | F1 | time IoU | live recall | live precision | objective | ordinary / short / fault strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 OVR / geometric | .55499 | .54566 | .78260 | .64316 | .58633 | 147 / 11 / 10 |
| v1 OVR / empirical + mixture | .54812 | .54063 | .80695 | .62094 | .58470 | 147 / 14 / 11 |
| prior-corrected OVR / geometric | .57439 | .54688 | .75843 | .66223 | .59375 | 146 / 13 / 12 |
| prior-corrected OVR / empirical + mixture | .57046 | .54675 | .77102 | .65274 | .59343 | 148 / 13 / 11 |

Prior correction improves precision and event F1, but it suppresses more live time and does not
preserve ordinary-long performance. The duration arm recovers some live recall and short matches,
but loses enough precision/F1 to lower objective. Crucially, every one of the 12 inner and four
outer LIVE mixture fits collapsed to effectively identical components (hazard gaps below
`2.7e-9`), so this corpus provides no evidence for a bimodal LIVE prior; the duration contrast is
interpreted as an empirical-SETUP ablation here.

The nested candidate exactly equals v1, paired evidence versus v1 is neutral, and it still fails the
operational binary live-recall and ordinary-long gates. Binary remains selected and protected test
remains closed.

Artifact:

- `reports/feature-order-2026-08-12/multistate-existing-label-ablation-v1-development.json`,
  SHA-256 `f9e2e0a053ebb0fec7fec28c179d7d126a2caadf4af662a55187564af22fe3ff`.

The report pins clean Git commit `4c98042`; wall time was 387.809 seconds.

## 4. Existing-tag immediate-result proxy

Implementation checkpoint: `8a0a32c Add nested immediate-result proxy study`.

This study used existing tags only: one gold-serve anchor per rally was positive exactly for `ace`
or `service-fault` and negative otherwise. Development support was 76 positives (35 aces and 41
service faults) and 230 negatives across all four source groups. The learned proxy used a fixed
17-column, at-most-two-second-lookahead bank; each fold trained it for 60 fixed epochs without using
held proxy labels for early stopping. Inference used no tag. Instead, the decoder privately split
LIVE into ordinary and result branches with fold-estimated duration priors; direct SERVE-to-DEAD was
removed and every event had at least one LIVE sample.

All four outer folds selected v1 no-op. Fixed outer-OOF diagnostics were:

| candidate | F1 | time IoU | live recall | live precision | objective | ordinary / short / ace / fault strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 no-op | .55499 | .54566 | .78260 | .64316 | .58633 | 147 / 11 / 7 / 10 |
| constant tag-duration branches | .55782 | .54708 | .78308 | .64480 | .58839 | 147 / 11 / 7 / 10 |
| learned proxy + tag-duration branches | .55254 | .54431 | .78071 | .64255 | .58430 | 146 / 12 / 6 / 11 |

The duration-only branch removed the one decoded direct SERVE-to-DEAD transition and gained only
.0021 objective, with no additional strict short, ace, or fault match. The learned proxy failed its
mechanism gate: outer-OOF log loss was .6419 versus .5658 for the training-fold prevalence baseline;
AUROC was .5054 and average precision .3013. Log loss was worse in all four source groups. It decoded
31 result branches but merely traded one ace and ordinary-long strict match for one fault/short
match. Existing coarse outcome tags therefore do not supply useful observable result evidence on
this feature bank, although the tag-conditioned duration distinction itself is real.

Binary remains operational. This study predeclared fresh-source validation—not the reused protected
test—as the only possible next assessment; the development gate failed, so neither was opened.

Artifact:

- `reports/feature-order-2026-08-12/multistate-immediate-result-proxy-v1-development.json`,
  SHA-256 `53edb324540feda8401359b4f151ad200184373c0c3d2b2c0a4e3bef16ac5182`.

The report pins clean Git commit `8a0a32c`; wall time was 186.485 seconds.

## 5. Joint multiclass emissions

Implementation checkpoint: `e9ab26f Add nested joint multistate emissions study`.

This study held the v1 graph, fold-geometric durations, and frozen transition bonus fixed while
changing only emission training. It compared the independently balanced/row-normalized v1 OVR
heads with a primary four-state softmax trained by ordinary categorical cross-entropy and a
separately named balanced softmax with fold-prior restoration. Softmax epoch caps came from the
three inner source folds inside each outer fold; state calibration and confusion were held-source
diagnostics and did not select the candidate.

Every outer fold selected the v1 OVR control. The fixed outer-OOF rows were:

| emissions | F1 | time IoU | live recall | live precision | objective | ordinary / short / fault strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 balanced OVR | .55499 | .54566 | .78260 | .64316 | .58633 | 147 / 11 / 10 |
| empirical-prior joint softmax | .46835 | .51706 | .71463 | .65160 | .51991 | 136 / 7 / 7 |
| balanced + prior-corrected joint softmax | .49840 | .50275 | .65554 | .68324 | .52328 | 132 / 15 / 12 |

The empirical model predicted 326 rallies but matched only 148, while the balanced/corrected model
predicted 320 and matched 156; v1 predicted 285 and matched 164. Both joint variants traded live
coverage and ordinary-long matches for precision. Their multiclass mechanism diagnostics also did
not support joint training:

| emissions | state accuracy | multiclass log loss | Brier | SERVE precision / recall / ECE |
| --- | ---: | ---: | ---: | ---: |
| v1 balanced OVR | .61245 | 1.30882 | .56838 | .1303 / .4804 / .0848 |
| empirical-prior joint softmax | .59631 | 2.31551 | .67552 | .0448 / .0752 / .0225 |
| balanced + prior-corrected joint softmax | .56375 | 3.11203 | .76521 | .1702 / .3399 / .0175 |

The primary joint model nearly eliminated rare SERVE recall; prior-corrected balanced training
recovered some SERVE detection and short/fault matches but further reduced overall live recall.
Normalized multiclass output alone therefore did not fix source-transfer calibration on this
linear feature substrate. Unanimous inner selection retained v1, and v1 still fails the frozen
operational-binary live-recall and ordinary-long gates. Binary remains selected and test stayed
closed.

Artifact:

- `reports/feature-order-2026-08-12/multistate-joint-emissions-v1-development.json`, SHA-256
  `74bdced6882a78e34940b77ba5e7665df57686a20b268286662ae29ccec9d862`.

Wall time was 417.929 seconds. The report pins clean Git commit `be40958` and content-hashes the
joint implementation, runner, tests, and all three prerequisite reports.

## 6. Cross-fitted serve and terminal edge evidence

Implementation checkpoints:

- `9d843f8 Add multistate transition-edge evidence decoder`;
- `243d69a Add nested multistate edge-evidence study`.

This last no-new-label study fixed the best diagnostic substrate from the 2x2 study—fold-prior-
corrected OVR emissions with geometric v1 durations—and evaluated the complete four-arm family:
no evidence, serve-edge only, terminal-edge only, and both. The serve and terminal specialists
were fit independently inside every source fold. Their balanced probabilities were mapped to
bounded `2p-1` evidence with coefficient one, affecting only `SETUP -> SERVE` or `LIVE -> DEAD` at
the transition destination sample. No evidence strength, threshold, or composition grid was
searched.

All four outer folds retained the no-edge control. Fixed outer-OOF diagnostics were:

| arm | F1 | time IoU | live recall | live precision | objective | start / end MAE | ordinary / short / fault strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no edge | .57439 | .54688 | .75843 | .66223 | .59375 | .5701 / 1.8699 | 146 / 13 / 12 |
| serve edge | .57686 | .54882 | .75851 | .66501 | .59569 | .5518 / 1.8625 | 146 / 14 / 13 |
| terminal edge | .56499 | .54703 | .75761 | .66308 | .58849 | .5841 / 1.8152 | 145 / 11 / 10 |
| both edges | .55903 | .54866 | .75749 | .66556 | .58569 | .5560 / 1.8086 | 143 / 11 / 10 |

The serve specialist is the only promising mechanism signal in this sequence. It gained one
matched rally, one strict short match, one strict fault match, 0.0183 seconds start MAE, and raised
within-0.5-second serve-anchor recall from .4575 to .4739. Its aggregate objective gain was only
.00195, however, versus the frozen +.01 source-macro requirement. Source objective deltas were
-.00084 KB, -.00470 Shoreline, +.00902 SPU, and +.00307 Yang: too small and inconsistent to justify
another selection on these reused sources.

Terminal evidence improved end MAE by .0547 seconds, satisfying its narrow boundary mechanism
criterion, but lowered objective in every source, lost one ordinary-long, two short, and two fault
strict matches, and hurt Yang most (-.02260 objective). Combining both mechanisms improved end MAE
by .0613 seconds but lost three ordinary-long matches and inherited the terminal regressions. The
combined arm was also ineligible because neither individual arm passed its full gate.

The no-edge control itself is not operational: compared with binary it loses two ordinary-long
strict matches and 7.97 points of live recall. Thus even the small serve-edge improvement cannot
open the protected test. Binary remains selected.

Artifact:

- `reports/feature-order-2026-08-12/multistate-edge-evidence-v1-development.json`, SHA-256
  `761cd3a5f61cda05a06817d666f764d93b87a44a81834ead9d9758e328bad2ff`.

Wall time was 244.885 seconds. The report pins clean Git commit `243d69a`, all prerequisite report
hashes, both specialist fingerprints and fold-local target counts, and confirms that protected
splits were not prepared.

## 7. Label-dependent experiment hooks

Implementation checkpoints:

- `9d843f8 Add multistate transition-edge evidence decoder`;
- `9e099fb Add transition-label experiment readiness gate`.

The transition-evidence primitive adds finite evidence only to an explicitly named legal,
non-self transition at its destination sample. Empty or zero evidence is exactly identical to the
legacy decoder. This is the frozen mechanism for future receiver-reaction evidence on
`SETUP -> SERVE` and collective-stand-down evidence on `LIVE -> DEAD`; it does not add a global
state feature or allow a forbidden edge.

The read-only gate scanned all nine current full-video drafts without mutation or validation
errors. It separates the eight train/validation recordings from the one protected test recording:

| scope | recordings | fully cued / pilot target | additional hard negatives to three/video |
| --- | ---: | ---: | ---: |
| development | 8 | 0 / 40 | 23 |
| protected test, sealed | 1 | 0 / 5 | 2 |

Consequently the user needs to provide 40 fully cued development rallies and 23 additional
development hard-negative intervals before those development experiments can run. The separate
five-rally and two-hard-negative protected debt remains sealed and is not a prerequisite for
development. Every registered runner is currently fail-closed:

- `reaction-supervised-serve-edge` uses `receiverReactionTime` and `startConfidence`;
- `stand-down-supervised-terminal-edge` uses stand-down time, terminal cue, end observability, and
  end confidence;
- `verified-immediate-result-branch` replaces the failed coarse ace/fault proxy with the verified
  field; and
- `hard-negative-dead-state` consumes categorized walking/retrieval, celebration/huddle,
  model-false-positive, and random-dead intervals.

Once labels arrive, freeze a new manifest from train/validation snapshots and rerun development
only. Mutable draft fields must not be attached to the existing report lineage, and the protected
recording must not be prepared before a newly frozen development candidate passes its gate.

Artifact:

- `reports/feature-order-2026-08-12/transition-label-experiment-gate-v1.json`, SHA-256
  `ccc04765e749070d7c645c91a2507bbc4bb96e330868748b165a18b5e2446d25`.

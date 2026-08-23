# Current side-switch winner production-port contract — 2026-08-23

## Status and identity

The selected side-switch model is
`side-switch-hard-negative-mining-v1/union34-top2-x2`. It is the current
**research-only** winner, not a shipped production model. The port specified here must
not be enabled in the editor until browser parity, device cost, and an independently
reviewed validation set pass.

The immutable source bindings are:

| Source | Identity |
| --- | --- |
| Current-winner pointer | `data/side-switch-current-research-winner-v1.json`, SHA-256 `163c7844267bc48410e89f86bd5cbf0a58b3145c1ec691932e4e3374dae7286c` |
| Model | `models/side-switch-hard-negative-mining-v1/model.json`, SHA-256 `c2570481c30dec62f56ac3284cd4028763ffd8e72a9ad07e9908fec10df387e3` |
| Feature artifact | `side-switch-full-union-v5-state-features-v1.json`, SHA-256 `9763cb3e5cd9baada64f4bf54f06140dcff5068bb8cd74a1d485c677d1e6c551` |
| Machine-readable port contract | [`data/side-switch-current-research-winner-production-port-v1.json`](../../data/side-switch-current-research-winner-production-port-v1.json), SHA-256 `48825e3e0b9ce11a6b67b8b7366d542d293e7b3e3eea4246e19f4dd9cd18613b` |

Hard-negative mining is training-only. The runtime remains one standardized 34-input
logistic head, threshold `0.39884973953581804`, and a cadence-free local-peak/soft-count
decoder.

## Browser feature-generation graph

The future web implementation needs this graph in addition to the current production
rally analysis:

```text
existing 4 Hz production traces and decoded range union
                         |
      every adjacent range boundary + dead-state internal peaks
                         |
        two visual comparison windows per candidate
                         |
     seven 256x144 court-ROI frames per comparison window
                         |
      V4 court-normalized palettes + V5 motion proposals
                         |
              22 visual comparison values
                         |
        10 reductions from both production model bundles
                         |
              candidate kind and generator score
                         |
                 ordered 34-value vector
                         |
       model imputation -> standardization -> logistic score
                         |
          cadence-free local suppression + soft count
                         |
              side-switch editor proposals
```

The required existing runtime inputs are the 4 Hz timestamps, rally and dead-state
probabilities, decoded ranges, and individual range/probability outputs from both
production bundles (`model-1ca43e38eefc` and `model-9c92b8e9333f`). Serve scores are not
inputs to the selected head. Suppression scores must remain quarantined.

## Candidate generation

Use the retained 704-candidate contract:

- emit every adjacent decoded-range boundary, with the transition at the midpoint
  between the preceding range end and following range start;
- inside each decoded range, scan the existing `initialInference` dead-state trace;
- exclude four seconds from each range edge;
- retain samples with `deadState >= 0.98`;
- apply score-ranked 14-second NMS inside that range; and
- emit a two-second interval centered on each internal peak.

Do not substitute the rejected expanded generator (`deadState >= 0.80`, ten-second
separation). Candidate metadata is `0/1` for boundary/internal kind and zero/dead-state
score for generator score.

## Visual extraction

Crop the same normalized court ROI already selected for production, then resize to
256×144. Estimate one recording-level net height from seven frames in each of the first
seven decoded ranges, and apply the frozen piecewise vertical warp that maps the net to
`y=0.5`.

For an adjacent boundary, compare the complete decoded range on each side. For an
internal candidate at `t`, compare `[t-4,t-1]` and `[t+1,t+4]` inside its containing
range. Sample seven evenly spaced frames from the frozen 8%-through-92% window contract.

Each sequence must reproduce the Python V4/V5 operations:

- upper-frame phase-correlation alignment;
- broad and tight motion-weighted HSV near/far palettes;
- motion-component proposal boxes, capped at six per frame;
- soft near/far assignment from proposal foot position;
- Hellinger assignment, palette-instability, coverage, proposal-count, and support
  reductions.

This is a new 256×144 sparse-candidate pass. The current 192×108, 4 Hz production base
features are not numerically interchangeable with these inputs.

## Exact ordered features

The browser must emit the model artifact's order, not alphabetical order.

| Group | Ordered inputs |
| --- | --- |
| V5 visual 22 | `v4BroadSameAssignmentCost`, `v4TightSameAssignmentCost`, `v4MeanSwapMargin`, `v4GlobalAppearanceChange`, `v4MaximumCameraShift`, `v4MinimumAlignmentResponse`, `playerSameAssignmentCost`, `playerSwappedAssignmentCost`, `playerSwapMargin`, `playerOrientationFlipEvidence`, `minimumPlayerSideSeparation`, `playerSideSeparationChange`, `beforePlayerPaletteInstability`, `afterPlayerPaletteInstability`, `playerGlobalAppearanceChange`, `minimumProposalCoverage`, `proposalCoverageChange`, `minimumProposalCount`, `proposalCountChange`, `minimumNearSupport`, `minimumFarSupport`, `sideSupportImbalanceChange` |
| Production state 10 | `productionBeforeSupportCount`, `productionAfterSupportCount`, `productionMinimumAdjacentSupportCount`, `productionMinimumAdjacentRallyPeak`, `productionGapLiveFraction`, `productionGapMeanRallyScore`, `productionGapPeakRallyScore`, `productionGapMeanDeadStateScore`, `productionGapPeakDeadStateScore`, `productionGapDurationSeconds` |
| Candidate metadata 2 | `candidateIsInternalDeadStatePeak`, `candidateGeneratorScore` |

The state reductions use both production bundles. Support counts come from the
overlap-connected range union; adjacent rally peak is the minimum supported peak;
gap-live fraction is the union duration of both bundles' ranges divided by gap duration;
and gap rally/dead statistics are the mean and maximum of the pointwise maximum across
the two bundle traces.

Missing or non-finite model inputs use the model artifact's 34 stored imputation values.
Then apply its stored mean and scale, dot with its weights, add bias, clip the logit to
`[-30,30]`, and apply the sigmoid.

## Decoder parity

Within a recording, sort candidates by descending probability, then transition time,
then event ID. Use threshold `0.39884973953581804`. Suppress a candidate when its
chronological candidate ordinal is less than two away from an already selected item.
There is no time-distance rule. The first six selected proposals are unpenalized; for
selection number seven onward subtract `0.5` logits for every output beyond six.

There is no score cadence, no re-anchoring, and no hard maximum output count.

## Explicit non-requirements

Do not port these as dependencies of the selected winner:

- the ten serve-anchor fields in `PRODUCTION-STATE20`;
- suppression-model scores or gates;
- V5 whole-set orientation parity;
- cadence or latent point-count priors;
- recording-reliability offsets;
- the expanded internal-candidate specialist; or
- the unpromoted boundary-only ablation.

## Acceptance gates

Before automatic editor use, require:

1. exact candidate inventory parity on all 11 audited recordings;
2. all 34 browser features compared with the frozen Python artifact, with explicit
   numeric tolerances and no reordered columns;
3. classifier-logit and final proposal parity, including deterministic ties;
4. peak memory and runtime measurements on representative phones; and
5. evaluation on a new exhaustively reviewed recording set.

The production editor should continue using its current rally model until all gates
pass. A review-only timeline can be added earlier if it is clearly labeled experimental.

Verify that the contract still matches the frozen pointer, model, feature artifact,
candidate generator, and decoder with:

```bash
python scripts/verify-side-switch-current-winner-production-port.py
```

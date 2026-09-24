**Optional follow-up preparation — 2026-09-19**

This preserves the engineering preparation made before the completed short-rally boost comparison. The user authorized up to two further hypotheses. After all preceding audits passed, both prepared interventions were selected and registered separately: shorter context for reviewed-export/short-boost, and baseline keep-head rescue across all three cohorts. Their prospective protocols and results are recorded separately; the following preparation did not use partial preceding-study outcomes.

The compact audiovisual TCN and DINO+TCN remain paired. Beach and the protected source group remain excluded. Product ranking stays at symmetric two-second padding, strictly less than three-second gap joining and ignored-time subtraction, using pooled `F1_padP_coreR`. All four padding cases remain required.

The first optional intervention changes temporal convolution dilations from `(1,2,4,8,16)` to `(1,1,2,2,2)`, reducing the network receptive field from 125 to 33 ticks without changing parameter counts or initialization. The original 128 central ticks plus 62-tick training halos remain fixed for paired sampling and dropout. Existing audiovisual feature windows and whole-record preprocessing also remain fixed; this is a temporal-network intervention. New adapter, fitter and runner files leave all 16 active-study sources untouched.

Six unordered pairs of excluded source groups can supply twelve restricted inner views. Each physical fit excludes both groups across every supervision tier; each logical view exposes only its designated inner group's probabilities. Four outer selections and refits remain separate. One cohort, two representations and three seeds would require 60 new physical fits representing 96 logical fits, with six immutable original-profile reference cells. No new cohort or loss arm has yet been selected.

The optional context pipeline has synthetic tests covering original-model tensor and prediction parity, initialization and dropout RNG, true temporal support, shared-owner validation isolation, resume corruption, full-grid selection ties and fallback, independent scaler/mask/exposure reconstruction, endpoint-sweep metrics and unchanged reference payloads. Its actual-data GPU preflight has not run. That preflight is designed to compare predetermined original-profile epoch-five fits with both historical exclusion orders, then check short-profile union/single/repeated validation fits and saved-checkpoint replay.

A separate CPU test issue was isolated below project code: in this installed CUDA-enabled PyTorch/WSL environment, setting `CUDA_VISIBLE_DEVICES=-1` reproduced a native abort even for a standalone scalar CPU backward operation. Leaving device visibility unset and explicitly using CPU tensors passed the minimal reproduction and all 25 combined model/fitter/runner tests with the usual thread limits. The precise native-library fault is unconfirmed. No active training environment or process was changed. Evidence is under `data/reports/neural-context-cpu-runtime-v1/`; diagnosis SHA-256 `04d72efb69efe132e2d82d6676bd0efa282a1ba02c9ac5100497394e43c9b571`.

The second optional intervention uses the auxiliary keep head to propose additional short rally cores, retaining the original checkpoint and live/boundary decoder. The prospective idea thresholds keep support, erodes two seconds from each end, keeps inferred positive core durations at most three seconds, and discards components clipped by valid-segment/video edges or touching exact ignored time. This is an approximate inverse of padded keep supervision; padding, clipping and gap joining are not generally invertible. It must not directly score keep intervals as core intervals or apply padding twice. Extra cores are unioned with baseline cores, so retained core time should be monotonic, while export precision and event F1 can worsen. An explicit no-op and any rescue threshold must be selected using inner predictions only. Helper, runner, preflight, registration and independent summary implementations are prepared without opening real candidate predictions. This inference variant would retain the fourth output head rather than use the three-head research derivative; its 65 extra output parameters add no feature-extraction requirement.

All 83 focused tests for both optional pipelines passed together in WSL in 34.1 seconds with the usual thread limits and CUDA visibility unset. The tests use synthetic fixtures; no real-data follow-up preflight, registration, fitting or candidate decoding has run. Independent reviews found no remaining implementation blocker. Actual-data qualification and completed-study audits are still required.

A separate follow-up source archiver passed 11 synthetic tests in both Windows and WSL and an independent read-only review. It requires completed study/audit artifacts and an explicitly supplied final report before creating an immutable archive. Its allowlist excludes videos, features, model weights and environments, and it binds each follow-up to the preceding study's verified source archive. No real follow-up archive has been created.

Preparation checklist (Tasks tool unavailable):

- [x] Isolate and test the context adapter, fitter and shared-owner runner.
- [x] Prepare context preflight and explicit registration helpers.
- [x] Prepare independent context tensor and full-grid/interval summary audits.
- [x] Finish optional keep-rescue helper, runner and independent summary audit.
- [x] Complete and audit the active short-boost transfer study.
- [x] Choose useful follow-up hypotheses, lock their populations and write prospective protocols.
- [x] Qualify, register, execute and audit each selected follow-up.
- [x] Write final comparisons and bind immutable source archives through their verification receipts.

Neither preparation nor a passing development screen authorizes production promotion. No protected-test evaluation or phone/browser inference benchmark is part of these optional preparations.

Selected follow-ups: [shorter-context protocol](./neural-short-context-protocol-2026-09-19.md), [shorter-context execution/results](./neural-short-context-results-2026-09-19.md), and [keep-head rescue protocol](./neural-keep-rescue-protocol-2026-09-19.md). Historical preparation statements above describe the state before registration.

Both selected follow-ups completed and passed independent engineering audits. [Shorter context](./neural-short-context-results-2026-09-19.md) was rejected because F1 and retention regressed in both architectures. [Keep-head rescue](./neural-keep-rescue-results-2026-09-19.md) passed the recovery screen in both architectures for reviewed-export supervision; standard F1 improvement screens still failed. The original WSL configuration was restored without restarting WSL.

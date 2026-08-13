# Model-training execution handoff — Windows Unsloth and WSL temporal model

## Purpose and decision

This is the execution specification for the two RTX 3080 12 GB experiments proposed in the
[small-model feasibility study](./small-multimodal-model-feasibility-rtx-3080-2026-08-12.md).
It is deliberately explicit enough for a coding model with no conversation history to implement,
run, audit, and stop the work safely.

The two tracks are independent:

1. **Track U — native-Windows Unsloth:** QLoRA-tune
   `unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit` on fixed 32-second video windows. This tests
   whether a compact VLM can learn the rally JSON contract and useful coarse volleyball semantics.
2. **Track T — WSL temporal model:** extract frozen DINOv2-S features at the 4 fps analysis grid,
   fuse them with the existing 90 audiovisual signals, and train a compact three-head temporal
   network. This is the recommended product-model experiment because it preserves dense timing and
   the existing decoder/evaluator.

Do not combine scores, checkpoints, or predictions from the two tracks unless a later, separately
specified fusion study does so without leakage. Track U is not a replacement for Track T: its 1 fps
input is useful for state and reset semantics but cannot establish quarter-second boundary accuracy.
Neither track is completed by this document; it defines the implementation and acceptance gates.

## Fixed storage map

| Item | Native Windows / Unsloth | WSL / non-Unsloth |
|---|---|---|
| VolleyCut data root | `Z:\volleycut` | `/mnt/z/volleycut` |
| Label workspace | `Z:\volleycut\labeling-v1-2026-08-09` | `/mnt/z/volleycut/labeling-v1-2026-08-09` |
| Frozen manifest | `Z:\volleycut\labeling-v1-2026-08-09\manifests\full-gold-v1.json` | `/mnt/z/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json` |
| Full Unsloth dataset | `Z:\volleycut\labeling-v1-2026-08-09\datasets\unsloth-rally-windows-v1-windows-z-2026-08-12` | not used by Track T |
| No-beach Unsloth dataset | `Z:\volleycut\labeling-v1-2026-08-09\datasets\unsloth-rally-windows-v1-no-beach-windows-z-2026-08-12` | not used by Track T |
| Existing audiovisual cache | not used directly | `/mnt/z/volleycut/labeling-v1-2026-08-09/features/audiovisual-v2` |

The JSONL files use forward-slash Windows paths such as
`Z:/volleycut/labeling-v1-2026-08-09/proxies/...`. Python and the Unsloth video reader accept that
form, and it avoids JSON escaping problems. The full artifact contains 254 train, 80 validation,
and 46 test windows. The no-beach artifact contains 176, 80, and 46 respectively. Use the full
artifact for the primary experiment and the no-beach artifact for exactly one frozen ablation; do
not concatenate them because the validation/test rows are identical and the training rows overlap.

If the Unsloth loader requires a path-layout change, regenerate a new checksummed artifact rather
than editing JSONL in place. From the Linux/NAS machine:

```bash
python3 scripts/export-unsloth-dataset.py \
  /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  /mnt/freenas/volleycut/labeling-v1-2026-08-09/datasets/NEW-WINDOWS-DATASET \
  --source-data-root /mnt/freenas/volleycut \
  --consumer-data-root Z:/volleycut
```

Add `--exclude-environment beach` for the filtered variant. For a WSL-consumed JSONL artifact, use
`--consumer-data-root /mnt/z/volleycut`. Always choose a new output directory: the exporter refuses
to overwrite an immutable dataset.

## Shared experimental rules

- Fit on `train` only. Use `validation` for early stopping, epoch selection, parsing/decoder choices,
  and thresholds. Keep `test` closed until all code, model settings, and decision rules are frozen.
- Preserve `sourceGroup` isolation. Never mix the overlapping nine-clip pilot with this corpus.
- Use unpadded gold `[start,end)` intervals for training and scoring. Export padding is a separate
  product setting and must not enter labels or headline metrics.
- Report chronological one-to-one event F1 at IoU 0.5, time IoU, live-time recall/precision,
  missed-live and retained-dead seconds, start/end MAE, strict recall for rallies at most 3 seconds,
  ace recall, and service-fault recall. Frame accuracy is not a primary metric.
- The development comparison uses four source groups: the three current training groups and the
  validation group. The indoor test group is a protected **retrospective regression**, not fresh
  generalization evidence. There is no held-out beach group.
- The gold is human-reviewed, but 290/345 rallies retain `ai-prelabel` provenance after correction.
  Disclose possible anchoring for a same-family VLM result. A clean generalization claim requires a
  newly recorded, blank-slate-labeled source group.
- Record the git commit, manifest SHA-256, dataset checksum file, model/repository commit, checkpoint
  SHA-256, complete Python package lock, GPU/driver/PyTorch/CUDA versions, random seed, commands,
  peak allocated and reserved VRAM, wall time, and all generated predictions.
- Writers must create a new run directory, write through a temporary file, and rename only after
  validation. Never mutate an earlier run or dataset.

## Track U — native-Windows Unsloth Qwen3-VL-2B QLoRA

### U1. Scope and model pin

Use `unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit`, not a thinking model and not an 8B model. At
the beginning of the run, resolve the Hugging Face revision to a full commit SHA and store it in the
run manifest. Do not silently follow `main` on a resumed run. The initial track is visual-only:
audio in the MP4 is not presented as model audio, and its existence must not be described as
audiovisual training.

### U2. Prepare the native-Windows environment

Run the training in PowerShell on Windows, not inside WSL. Keep the Python environment, model cache,
checkpoints, and active logs on a local SSD. Read the source videos from `Z:` and copy completed,
checksummed artifacts back to `Z:`. A network drive is appropriate for immutable media but is a poor
place for thousands of environment/checkpoint writes.

Follow the current official Unsloth Windows installation page at execution time. A clean `uv`
environment is the preferred path:

```powershell
winget install -e --id Python.Python.3.13
winget install -e --id astral-sh.uv
uv venv C:\volleycut-envs\unsloth-qwen3vl --python 3.13
C:\volleycut-envs\unsloth-qwen3vl\Scripts\Activate.ps1
uv pip install unsloth --torch-backend=auto
uv pip install datasets torchcodec pytest
```

If the current Unsloth compatibility table requires Python 3.12 instead, create a new 3.12
environment; do not downgrade an existing environment in place. Install or update the production
NVIDIA Windows driver first. Native extension failures may require Visual Studio Build Tools 2022
and the CUDA toolkit version documented by the then-current Unsloth installer. Do not add
FlashAttention to the first run: use the supported Windows attention backend selected by Unsloth,
record it, and treat any backend change as a new configuration.

Verify before downloading weights:

```powershell
nvidia-smi
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).total_memory)"
python -c "import unsloth, transformers, bitsandbytes, torchcodec; print(unsloth.__version__, transformers.__version__, bitsandbytes.__version__, torchcodec.__version__)"
```

Stop if CUDA is unavailable, the selected device is not the 12 GB RTX 3080, or inference falls back
to CPU. Do not use CPU/disk model offload to force a failed fit; reducing frames is the controlled
fallback.

### U3. Dataset preflight

Use these PowerShell variables so logs and commands agree:

```powershell
$VolleyDataRoot = "Z:\volleycut"
$VolleyWorkspace = "$VolleyDataRoot\labeling-v1-2026-08-09"
$VolleyDataset = "$VolleyWorkspace\datasets\unsloth-rally-windows-v1-windows-z-2026-08-12"
$VolleyRunRoot = "C:\volleycut-runs\qwen3vl2b-unsloth-v1"
```

Before any GPU work, the verifier described in U4 must:

1. validate `checksums.sha256` and `dataset.json`;
2. require exactly 254/80/46 rows and unique row IDs;
3. require the expected 3/1/1 source-group split and no cross-split source-group overlap;
4. resolve every video path, compare its recorded SHA-256, and validate segment bounds;
5. parse every assistant string with the strict target schema;
6. confirm each message video path equals `metadata.sourceVideo.path`;
7. decode one empty window, one short/ace-or-fault window, and one edge-censored window;
8. feed those three rows through the real processor and collator and prove that the requested
   segment—not the full source video—is sampled.

Prefer TorchCodec and record the chosen video-reader backend. A silent reader fallback is a failed
preflight. Measure random-seek latency for at least 30 windows. If median open/decode exceeds 2
seconds or failures are intermittent, stage the proxy tree on the local SSD and regenerate a new
path-rebased artifact pointing to that root; never hand-edit the checksummed dataset.

### U4. Required implementation deliverables

Implement and test these files before a real run:

```text
scripts/verify-unsloth-video-dataset.py
scripts/train-unsloth-qwen3vl.py
scripts/infer-unsloth-qwen3vl.py
scripts/evaluate-vlm-windows.py
analysis/vlm_intervals.py
analysis/tests/test_vlm_intervals.py
analysis/tests/test_unsloth_training_contract.py
```

`verify-unsloth-video-dataset.py` implements U3 and emits a machine-readable audit. The training
script accepts a dataset directory, resolved model revision, output directory, seed, and config;
refuses a nonempty output; logs peak memory; supports exact checkpoint resume; saves adapters plus
processor/tokenizer; and writes the effective config. The inference script performs deterministic
generation and retains raw text, parse state, latency, memory, and provenance for every row. The
evaluator strictly parses window JSON, stitches windows using U8, converts to the repository's
prediction shape, and calls the existing `analysis.evaluation` metrics rather than reimplementing
metric definitions.

Unit tests must use synthetic JSON/messages and no downloaded model. Put the real three-row
collator, one-step GPU, and save/reload tests behind an explicit integration-test flag.

### U5. Frozen initial configuration

Use this as the first real configuration after compatibility gates pass:

```yaml
model_id: unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit
load_in_4bit: true
max_seq_length: 2048
gradient_checkpointing: unsloth
dataset: unsloth-rally-windows-v1-windows-z-2026-08-12
video_fps: 1.0
max_frames: 32
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
num_train_epochs: 3
learning_rate: 0.0001
optimizer: adamw_8bit
weight_decay: 0.001
warmup_ratio: 0.05
lr_scheduler_type: linear
lora_rank: 8
lora_alpha: 8
lora_dropout: 0.0
finetune_vision_layers: false
finetune_language_layers: true
finetune_attention_modules: true
finetune_mlp_modules: false
seeds: [3407, 1729, 20260812]
generation_do_sample: false
generation_max_new_tokens: 192
```

This is intentionally smaller than a generic demo recipe. The corpus has only 254 training
windows, and the frozen vision tower limits memory and overfitting. With accumulation 16, one epoch
contains about 16 optimizer updates; logs must distinguish microbatches from optimizer steps.

Use `FastVisionModel.from_pretrained`, `FastVisionModel.get_peft_model`,
`UnslothVisionDataCollator`, `SFTTrainer`, and `SFTConfig` from the current official Qwen3-VL
notebook. The following trainer settings are mandatory unless an upstream breaking change is
documented in the run manifest:

```python
remove_unused_columns = False
dataset_text_field = ""
dataset_kwargs = {"skip_prepare_dataset": True}
max_length = 2048
```

Loss must apply only to assistant-answer tokens. Do not hardcode chat-marker token IDs: derive them
from the pinned processor/chat template, inspect one tokenized example, and test that user, video,
padding, and prompt tokens are masked while every target JSON token is supervised.

### U6. Compatibility and memory gates

Run the gates in order and retain their logs even when they fail:

1. **U0 — load/collate/infer:** process the three preflight rows at 32 frames. If measured free
   headroom is below the larger of 1 GB or 10% of usable VRAM, retry at 16 and then 8 frames. Reduce
   temporal frames before spatial resolution. Do not proceed below 8 frames per 32-second window.
2. **U1 — one optimizer step:** require finite loss and gradients in LoRA parameters, no gradients
   in the vision tower, memory headroom, adapter save, fresh-process reload, and byte-for-byte stable
   deterministic output for the same input/config.
3. **U2 — overfit smoke:** run 20 optimizer steps on a fixed 16-row subset containing empty,
   ordinary, short, ace/fault, and censored examples. Loss must decline and JSON validity must not
   worsen. Discard this adapter.
4. **U3 — 200-step compatibility:** repeat the 16 rows as needed, save at step 100, terminate, resume
   in a new process, and finish step 200. Require no NaNs, memory growth, reader failures, or resume
   mismatch. Discard this adapter.

After U3, freeze the package lock, resolved model revision, reader backend, frame budget, LoRA
targets, and trainer configuration. Any change restarts the relevant gate.

### U7. Real training and selection

Train the full 254-window training split for three seeds. Save at every epoch. After each epoch,
compute validation loss and generate all 80 validation rows deterministically. Select the epoch per
seed by the validation score

```text
0.55 * event_F1_at_IoU_0.5 + 0.30 * time_IoU + 0.15 * live_time_recall
```

not by training loss and not by test performance. The validation targets may be opened only inside
the evaluator; never include reference answers in an inference prompt or repair loop. Report all
three seeds and their median; do not select the best seed as the headline result.

### U8. Strict parsing and window stitching

The only accepted assistant payload is:

```json
{"liveAtStart":false,"liveAtEnd":false,"rallies":[{"start":1.25,"end":7.5,"outcome":"ordinary"}]}
```

Reject unknown keys, non-finite/non-numeric times, booleans in numeric fields, intervals with
`start >= end`, values outside the window, unsorted/overlapping rallies within an answer, invalid
outcomes, and wrong field types. The only automatic text cleanup permitted before JSON parsing is
trimming whitespace and removing one surrounding Markdown code fence. Never use another LLM to
repair output. Report strict-valid and cleanup-valid rates separately, and score invalid output as
an empty prediction rather than dropping it.

Convert each valid window-relative interval back to source time, then stitch overlapping windows
per recording with this frozen deterministic algorithm:

1. sort candidates by source start, source end, and window ID;
2. connect candidates from **different** windows when their intersection is at least 50% of the
   shorter duration, or when both starts and both ends are within 1.0 second;
3. emit one interval for each connected component;
4. use the median of uncensored starts, falling back to the minimum start if all are left-censored;
5. use the median of uncensored ends, falling back to the maximum end if all are right-censored;
6. choose majority outcome, breaking ties `service-fault`, then `ace`, then `ordinary` because the
   immediate-result classes are the critical slice;
7. merge intervals that truly overlap after aggregation; merely touching half-open intervals remain
   distinct; and
8. bound the final intervals to the probed source duration and validate repository output schema.

Test single-window predictions, duplicated overlap predictions, a rally censored in both adjacent
windows, empty/invalid rows, outcome ties, transitive three-window components, and touching
intervals. The overlap and 1-second constants are validation choices: once the first real run
begins, tune them only on validation and record any departure as a new decoder version.

### U9. Acceptance, no-beach ablation, and protected test

The Windows path is operationally qualified only when the three-seed validation median has:

- at least 99.5% strict JSON validity and 100% cleanup-validity;
- repeatable deterministic generation and successful adapter save/reload;
- real-time factor at most 1.0 on the RTX 3080 for the complete windowed pipeline;
- the required VRAM headroom;
- no regression against the pinned model's zero-shot validation event F1 or time IoU; and
- at least 5 percentage points better strict short/fault recall without more than a 5-point increase
  in retained-video fraction.

These gates qualify the experiment, not production. Compare the tuned model to a deterministic
zero-shot prompt on the same validation rows and model revision. Also report the existing temporal
baseline; do not imply that a VLM has replaced it merely because its own zero-shot score improved.

Only after the full run is frozen, repeat the exact three-seed configuration with
`unsloth-rally-windows-v1-no-beach-windows-z-2026-08-12`. The only changes may be dataset path and
the derived number of optimizer steps. This removes 78 train windows, 71 unique rallies, and one
source group, leaving a substantially weaker two-group training set. Interpret the comparison as
an environment-sensitivity ablation, not evidence of unseen-beach generalization.

If the full configuration passes its validation gates, select the median-seed configuration by the
predeclared rule, freeze its checkpoint/decoder hashes, and run the 46-window test once. Do not
return to validation or retrain in response to the test result.

### U10. Windows artifacts

Train locally, then copy verified outputs to:

```text
Z:\volleycut\labeling-v1-2026-08-09\models\qwen3vl2b-unsloth-v1\SEED\
Z:\volleycut\labeling-v1-2026-08-09\reports\qwen3vl2b-unsloth-v1\
```

Each seed directory contains adapter weights, adapter config, processor/tokenizer files, resolved
base-model revision, effective config, environment lock, checkpoints, training/validation logs,
raw generations, parsed/stitched predictions, memory/timing telemetry, and checksums. The report
directory contains the dataset audit, per-seed and aggregate metrics, slice tables, test-open ledger
if applicable, and a promotion/non-promotion decision. Do not export GGUF or merge the adapter into
base weights until the validation decision is recorded; conversion adds no evidence and makes
provenance harder to audit.

## Track T — WSL frozen DINOv2-S plus temporal model

### T1. Scope

Run this track in WSL with data under `/mnt/z/volleycut`. Use frozen DINOv2 ViT-S/14 as a semantic
frame encoder and train only a compact temporal network. This is a distinct experiment from the
completed frozen-MobileNetV2 high-resolution study, which decreased development event F1 from
0.5244 to 0.3498 and was correctly not promoted. Reuse that study's provenance, cache validation,
fold, and immutable-artifact patterns; do not use its negative result as evidence that DINOv2 will
work, and do not present DINOv2 as already validated.

### T2. WSL environment and storage

Ubuntu 24.04 WSL with the Windows NVIDIA driver is sufficient; bare-metal Linux is not required.
The required and conditional `sudo` package blocks are already recorded in the
[feasibility report](./small-multimodal-model-feasibility-rtx-3080-2026-08-12.md#wsl2-packages-to-preinstall).
For a fresh distribution, install the required application/media packages and the reasonable
model-extension build preparation:

```bash
sudo apt-get update
sudo apt-get install --yes \
  ca-certificates curl git \
  python3 python3-venv \
  ffmpeg

sudo apt-get install --yes \
  build-essential python3-dev \
  cmake ninja-build pkg-config
```

Optional workstation utilities are:

```bash
sudo apt-get install --yes \
  git-lfs jq ripgrep rsync tmux unzip
git lfs install
```

Only if a video package cannot use a wheel and must compile against the system FFmpeg, add:

```bash
sudo apt-get install --yes \
  libavcodec-dev libavdevice-dev libavfilter-dev \
  libavformat-dev libavutil-dev \
  libswresample-dev libswscale-dev
```

Do not preinstall a Linux NVIDIA driver, cuDNN, `nvidia-cuda-toolkit`, GUI/X11 packages, or FFmpeg
development headers for the normal wheel path. Do not use `sudo pip`. Ubuntu 24.04's FFmpeg 6.1 is
preferred because the repository uses `-fps_mode`; stock Ubuntu 22.04's FFmpeg 4.4 is too old for
that call unless separately upgraded. Verify the prepared system:

```bash
/usr/lib/wsl/lib/nvidia-smi
python3 --version
ffmpeg -version
ffprobe -version
ffmpeg -hide_banner -h full | grep -q fps_mode
gcc --version
cmake --version
ninja --version
```

Create a separate model environment:

```bash
export VOLLEYCUT_DATA_ROOT=/mnt/z/volleycut
export VOLLEYCUT_LABELING_WORKSPACE=/mnt/z/volleycut/labeling-v1-2026-08-09

python3 -m venv .venv-temporal
.venv-temporal/bin/python -m pip install --upgrade pip
.venv-temporal/bin/python -m pip install -r analysis/requirements.txt
```

Then use the current [PyTorch selector](https://pytorch.org/get-started/locally/) to install a
compatible CUDA `torch` and `torchvision` pair; record the exact command and freeze the resolved
versions. Do not copy a historical wheel URL into automation. Install only the additional packages
the implementation actually imports, expected initially to be `timm` and `einops`. Do not install
a Linux NVIDIA driver. A CUDA toolkit is unnecessary for wheel-only PyTorch and should be added
only if a pinned extension truly needs compilation.

The canonical manifest, proxies, labels, and final artifacts remain on `/mnt/z`. Because repeated
small reads and cache writes across a Windows mount can be slow, a run may build its DINO cache on
the WSL ext4 filesystem, then checksum-copy the completed directory to `/mnt/z` and validate it
there. Never maintain two mutable cache copies.

### T3. Required implementation deliverables

Implement:

```text
analysis/dinov2_embeddings.py
analysis/semantic_temporal_model.py
analysis/semantic_temporal_study.py
scripts/extract-dinov2-embeddings.py
scripts/evaluate-semantic-temporal-model.py
analysis/tests/test_dinov2_embeddings.py
analysis/tests/test_semantic_temporal_model.py
analysis/tests/test_semantic_temporal_study.py
```

The extractor owns pinned-backbone loading, preprocessing, timestamp selection, ROI/full-frame
views, batched inference, content-addressed cache writing, resume, and cache validation. The model
file contains only tensor/model/loss code. The study file owns source-group folds, train-only
normalization, deterministic sampling/training, decoder fitting, metrics, ablations, and immutable
reports. The runner defaults to development groups and must require an explicit
`--open-retrospective-test` flag to touch the protected test source.

Unit tests use synthetic videos/tensors and stub backbones. Add one explicit GPU integration test
for deterministic extraction, frozen gradients, cache resume, and peak memory.

### T4. Pin and qualify DINOv2

Pin a full commit of the official `facebookresearch/dinov2` repository and the official
`dinov2_vits14` checkpoint SHA-256. Load from the pinned local repository with `torch.hub` source
set to local; do not execute mutable remote code during a recorded run. Record the Apache-2.0
license and all preprocessing values.

Before corpus extraction, test one fixed frame twice in fresh processes. Require the expected
384-dimensional class/patch representation, finite values, identical output within the declared
floating-point tolerance, no parameter gradients, and safe memory. The backbone stays frozen in
all experiments; fine-tuning it is out of scope for this corpus.

### T5. Frame representation and cache contract

Decode from each existing 960x540 proxy at the recording's exact 4 fps analysis timestamps. Use
the annotated court ROI when available; otherwise use the full frame. Preserve aspect ratio with
letterboxing to a multiple of the 14-pixel patch size and apply the official DINOv2 normalization.
Run the frame through DINOv2 once and retain:

- the 384-dimensional class token; and
- a 3x3 spatial average pool over the patch-token grid, yielding nine 384-dimensional region tokens.

The cached tensor per recording is therefore `[T, 10, 384]`, stored as FP16 after checking the
FP16-versus-FP32 maximum/mean error on at least 100 development frames. Do not upscale a 192x108
analysis image, and do not describe these features as ball tracking.

Before any label-informed training, benchmark input sizes 224 and 336 on at least one long proxy.
Choose the largest size that sustains extraction with the larger of 1 GB or 10% usable-VRAM
headroom and practical throughput. This is a resource choice only; do not select resolution by
validation accuracy unless it is declared as a formal ablation.

Each cache must include and validate:

```text
schema version; recording ID; source/proxy path and SHA-256; duration;
exact timestamp vector and cadence; ROI/fallback rule; resize/letterbox/normalization;
DINO repository commit; checkpoint SHA-256; torch/vision/CUDA versions;
input size; tensor shape/dtype; batch size; extractor git commit; created time;
per-file checksum and completion marker
```

The cache key is content-addressed from those fields. Resume only missing validated chunks, reject
metadata mismatch, and publish only after all development recordings pass a second checksum/shape
audit. Extract the eight development recordings first; do not read the test video during feature
or architecture development.

### T6. Sequence examples and targets

Align DINO ticks to the existing audiovisual cache by timestamp, never by assuming array indices.
Require the nearest timestamp to be within half an analysis tick. A recording example contains:

```text
timestamps:       [T]
valid_mask:       [T]
dino_tokens:      [T, 10, 384]
audiovisual:      [T, 90]
live_target:      [T]
serve_target:     [T]
end_target:       [T]
hard_negative:    [T]  # when annotated; otherwise false
```

Construct `live_target` from exact half-open intervals. Construct serve/end targets as Gaussian
pulses centered on annotated boundaries, sigma 0.35 seconds and truncated at plus/minus 1 second.
Ignored ticks are masked from every loss. Fit audiovisual scalers and class weights on the current
training fold only.

Train on 256-tick (64-second) chunks with stride 128 ticks. Keep every all-dead chunk; do not
gold-center sampling or undersample negatives. At inference, cover the recording exhaustively and
average logits in overlapping chunk regions before applying sigmoid and decoding.

### T7. Frozen initial architecture and optimizer

Implement this initial network exactly before exploring alternatives:

```text
DINO [T,10,384]
  -> shared LayerNorm(384) + Linear(384,32) + GELU per token
  -> flatten tokens to [T,320]

AV [T,90] -> fold-trained standardization

concat [T,410] -> Linear(410,128)
  -> five residual Conv1d blocks
       hidden=128, kernel=3, dilations=[1,2,4,8,16]
       GroupNorm(groups=8), GELU, dropout=0.15
  -> independent Linear(128,1) heads: live, serve, end
```

Use same-length centered padding. The temporal receptive field is 63 ticks, or 15.75 seconds at 4
fps. Assert fewer than 2 million trainable parameters and log the exact count. Do not add a
transformer, bidirectional recurrence, CRF, or pretrained text model in the first experiment.

Use

```text
loss = BCE_live + 0.5 * BCE_serve + 0.5 * BCE_end
```

with positive weights derived only from the current training fold and clipped to `[1,20]`. Optimize
with AdamW, learning rate `3e-4`, weight decay `1e-4`, batch size 8 chunks, gradient-norm clipping at
1.0, at most 40 epochs, and early-stopping patience 6. Use seeds `[3407,1729,20260812]`. If batch 8
does not meet the memory gate, reduce physical batch and use gradient accumulation to keep effective
batch 8. Early stopping is based on inner-validation weighted objective, not training loss.

### T8. Decoder contract

Produce three probabilities on the original recording timestamps. Start with the repository's
existing hysteresis interval decoder as the live-state proposal generator, fitting its inner
thresholds on inner validation only. Add two bounded refinements:

- choose a serve-head-supported start within `[-2,+1]` seconds of each proposal start; and
- choose an end-head-supported end within `[-1,+3]` seconds of each proposal end.

Add a short-result path that pairs a serve peak with the next end peak 0.25–3.0 seconds later, using
live probability only as supporting evidence. It must be a no-op when confidence is below a tuned
threshold and it must pass the same chronological overlap resolution as the current decoder.

All thresholds, peak separation, and proposal arbitration are selected only inside inner
development folds. The decoder must have a configured no-op mode that reproduces live-only output;
test this parity. Do not tune boundary windows or minimum durations on an outer group or the test
source.

### T9. Nested source-group development study

Use the four development groups (three current train groups plus the current validation group) in
nested leave-one-source-group-out evaluation:

1. hold one group out as the outer evaluation group;
2. use leave-one-group-out splits among the remaining three groups to choose epoch, scaler,
   thresholds, and decoder settings;
3. choose a single configuration by the median inner objective, not the most favorable group;
4. retrain on all three outer-training groups; and
5. evaluate the untouched outer group once for each seed.

Run these predeclared candidates in order:

1. existing audiovisual features with the new same-capacity TCN (capacity control);
2. DINO class token only;
3. DINO class plus 3x3 region tokens;
4. DINO regions plus audiovisual features;
5. candidate 4 plus serve/end heads and boundary refinement; and
6. candidate 5 plus the short-result path.

Also report the historical logistic OOF result as context, but use candidate 1 as the architectural
control. Do not proceed to candidate 5 if candidate 4 cannot improve representation metrics; do not
proceed to candidate 6 if serve/end peaks have no inner-fold boundary signal.

The promotion gate relative to the matched control is:

- at least `+0.02` absolute event F1 at IoU 0.5;
- at least `+0.02` absolute time IoU;
- no more than 1 percentage point lower live-time recall;
- positive objective delta for every seed and improvement on at least three of four outer groups;
  and
- no decrease in strict recall for rallies at most 3 seconds or service faults.

Include per-group/per-seed distributions, confidence intervals from source-group bootstrap only when
statistically meaningful, exact failure slices, peak-memory/runtime, and retained-video fraction.
Nine videos are not nine independent samples; do not bootstrap frames or overlapping windows as if
they were independent.

### T10. No-beach temporal ablation

The JSONL Unsloth dataset is not Track T's input. For a corresponding no-beach temporal study,
filter recordings with `environment == "beach"` in the manifest-driven fold builder and emit the
exact included recording/source-group IDs in the report. This leaves only three development source
groups, so use three-group outer leave-one-group-out with two groups for each outer training set.
Freeze architecture, resolution, optimizer, decoder search space, and seeds from the full study;
the only intentional change is the environment filter. This is a sensitivity check with low power,
not a claim about generalizing to beach footage.

### T11. V-JEPA comparison and optional boundary refiner

Only after the complete DINO pipeline and controls run reproducibly, add the frozen official V-JEPA
2.1 base checkpoint `vjepa2_1_vit_base_384`. Pin the official repository commit and checkpoint
checksum, then repeat shape, deterministic-output, frozen-gradient, throughput, and memory gates.
Do not substitute a larger model if base is unavailable.

Use the smallest valid clip/frame count supported by the pinned model and a clip stride no greater
than 1 second. Interpolate or repeat each clip representation onto the 4 fps sequence without
claiming new timing resolution. Compare, under the same outer folds and same-capacity head:

1. audiovisual plus V-JEPA;
2. audiovisual plus DINO; and
3. audiovisual plus DINO plus V-JEPA.

Prefer the smallest representation that passes the same promotion gate. If semantics improve event
coverage but end MAE remains high, the next experiment may add a short high-frame-rate boundary
refiner (for example X3D-S) around predicted starts/ends. Training crops may be centered on gold
boundaries only within training groups; validation/test crops must be centered on model proposals.
The refiner must be a no-op below confidence and must improve boundary MAE without reducing event
F1, time IoU, or short/fault recall. It is not part of the initial deliverable.

### T12. Protected test and WSL artifacts

Open the protected test only after a signed/immutable development report passes the promotion gate.
Then extract its DINO cache with the already pinned extractor, fit once on the frozen development
pool, apply the frozen decoder, and evaluate with the explicit `--open-retrospective-test` flag.
Record who/what opened it, timestamp, config/report hashes, and command. Never use the result to
change the model.

Publish to new directories:

```text
/mnt/z/volleycut/labeling-v1-2026-08-09/features/dinov2-vits14-v1/
/mnt/z/volleycut/labeling-v1-2026-08-09/models/dinov2-temporal-v1/
/mnt/z/volleycut/labeling-v1-2026-08-09/reports/dinov2-temporal-v1/
```

The feature directory holds validated per-recording tensors/metadata/checksums. The model directory
holds one fold/seed checkpoint, train-only scalers, effective configs, environment lock, and hashes
per subdirectory. The report directory holds fold assignments, metrics, raw/decoded predictions,
ablation tables, telemetry, protected-test ledger if opened, and the promotion decision.

## Required execution order for another coding model

The implementing model must follow this order and stop at each failed gate:

1. Read the analysis contract, labeling guide, evaluation implementation, prior feasibility report,
   this handoff, and the completed MobileNet high-resolution report.
2. Record repo status and preserve unrelated/user changes. Create no dataset or model artifact inside
   the git repository.
3. Independently verify the manifest, source groups, media paths, environment tags, and checksums.
4. Implement shared strict schemas, provenance records, atomic artifact writing, and unit tests.
5. Implement and pass the Windows dataset verifier and Track U integration preflight.
6. Pass U0–U3 before launching three real Unsloth seeds; stop and document failure rather than
   changing several memory/LoRA variables at once.
7. Run full Track U validation, then its exactly matched no-beach ablation. Open test only if U9 says
   to do so.
8. Build/pin the DINO environment and backbone, implement the extractor, and qualify a one-video
   resource sweep without labels.
9. Extract/validate development-only caches, then implement the temporal model, decoder, folds, and
   synthetic tests.
10. Run Track T candidates in T9 order for all outer folds and seeds. Freeze and write the development
    decision before any protected test access.
11. Run the no-beach sensitivity analysis only after the full configuration is frozen.
12. Add V-JEPA or a boundary refiner only as a separately versioned follow-up; do not expand the
    initial study mid-run.

When blocked by an upstream API change, consult the pinned upstream documentation/source, add a
minimal compatibility test, document the exact departure here or in the run report, and resume from
the earliest affected gate. Do not guess around a video collator, chat template, timestamp transform,
or evaluation mismatch.

## Definition of done

The request is implemented—not merely started—when both tracks have:

- reproducible setup instructions and locked environments;
- tested loaders with correct source-time segments and timestamps;
- deterministic, resumable training/inference runners;
- immutable, checksummed features/checkpoints/predictions/reports on `Z:` or `/mnt/z` as assigned;
- all declared seeds and source-group folds, or a recorded stop-gate failure with evidence;
- one aggregate report with metrics and failure slices, not just loss curves;
- explicit validation-only and protected-test status;
- a promotion/non-promotion decision against the declared baseline; and
- no change to the product `analysis.json` contract or export-padding behavior.

## Primary upstream references

- [Unsloth native Windows installation](https://unsloth.ai/docs/get-started/install-and-update/windows-installation)
- [Unsloth system requirements](https://unsloth.ai/docs/get-started/beginner-start-here/unsloth-requirements)
- [Unsloth vision fine-tuning](https://unsloth.ai/docs/basics/vision-fine-tuning)
- [Unsloth Qwen3-VL guide](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune)
- [Official Unsloth Qwen3-VL vision notebook](https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision.ipynb)
- [Qwen3-VL official fine-tuning implementation](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune)
- [DINOv2 official repository and ViT-S/14](https://github.com/facebookresearch/dinov2)
- [V-JEPA 2 official repository](https://github.com/facebookresearch/vjepa2)
- [PyTorch installation selector](https://pytorch.org/get-started/locally/)

# Small multimodal model feasibility and RTX 3080 plan — 2026-08-12

## Decision

Using a small multimodal model is feasible on one RTX 3080, but replacing VolleyCut with a
generative LLM that watches a full match and emits exact timestamps is not the recommended first
move. The best risk-adjusted design is a **hybrid temporal model**:

1. cache frozen visual embeddings from a small image/video encoder;
2. fuse them with the existing 90 audiovisual signals;
3. train a lightweight temporal network with separate live-state, serve-contact, and rally-end
   outputs;
4. keep the existing interval decoder, evaluation contract, and export format;
5. benchmark a 2B temporal VLM as an optional coarse proposer or teacher, not as the sole final
   boundary detector.

The first implementation should use **DINOv2-S plus a small multi-scale temporal head** as the
lowest-cost framewise baseline, then compare it with **V-JEPA 2.1 ViT-B** under the same folds before
selecting a backbone. The first direct-VLM benchmark should be **TimeLens2-2B zero-shot**. The first
generative fine-tuning candidate should be **Qwen3-VL-2B with LoRA**, or QLoRA only after its
unofficial quantized training path passes a compatibility test. **SmolVLM2-500M** is the cheapest
training-pipeline check. **Gemma 4 E2B** is a secondary audiovisual experiment whose value for
non-speech sports audio is unproven. **Muse Glimmer-30B is rejected** for this hardware and task.

The remote card is reported to be the **12 GB RTX 3080** variant; confirm that with `nvidia-smi`
during Phase 0. That extra 2 GB materially improves the 2B-model path, but does not make static
weight fit equivalent to video-training fit. All fine-tuning memory ranges below are planning
estimates, not measured results. The first execution gate remains a short peak-VRAM benchmark on
the actual machine.

This work can plausibly improve review-assist cuts. The current corpus is sufficient for an
engineering feasibility study, but not for a clean claim that a tuned VLM generalizes: it contains
only five independent source groups, and the validation and test sources have already been reused.

## What the model actually has to solve

VolleyCut predicts dense binary rally state at 4 fps. A valid interval starts at serve-ball contact
and ends at the first instant play is dead. Aces and service faults count; celebration, retrieval,
and setup do not. The immutable product boundary remains ordered, bounded `[start, end)` intervals
in `analysis.json`, with padding applied separately. See the [analysis contract](../../analysis/README.md),
[labeling guide](../labeling-guide.md), and [analysis format](../analysis-format.md).

The promoted rally-only review-assist artifact, `full-audiovisual-v2-final`, is a class-weighted
logistic model over 90 audiovisual base signals at five centered temporal offsets, or 450 inputs.
Its corresponding frozen feature-study report supplies the most defensible reference: the nested
source-group out-of-fold development score below. The final fixed-split artifact and the OOF
reference are related, but not the same fitted model.

| Metric | Current development result |
|---|---:|
| Event F1 at IoU 0.5 | 0.4655 |
| Time IoU | 0.4862 |
| Live-time recall | 86.26% |
| Matched rallies | 155 / 306 |
| End-boundary MAE | 2.419 s |

The one-video indoor regression result is much higher—0.6905 event F1 and 0.6715 time IoU—but it
is repeatedly inspected, so it is not fresh generalization evidence. The full evidence is in the
[audiovisual feature study](./audiovisual-feature-ablation-2026-08-11.md).

The critical failure slice is not ordinary long rallies. Development strict recall is 63.56% for
ordinary long rallies, but only 20.00% for aces, 12.20% for service faults, and 9.52% for rallies at
most three seconds. The regression source matches all 28 ordinary long rallies and none of its 10
short rallies or 7 service faults at IoU 0.5. A serve specialist finds 32 of 39 contacts, including
6 of 7 faults, yet interval composition still matches no faults and lowers event F1. That result
shows that **end-of-play state and precise interval closure**, not just serve recognition, are the
highest-value representation problem. See the
[serve-specialist study](./serve-specialist-audiovisual-experiment-2026-08-11.md) and
[end/dead-state study](./end-and-dead-state-audio-experiment-2026-08-12.md).

A text-only LLM is structurally mismatched. Turning motion and audio features into prose would
discard timing information and add a stochastic text interface. The relevant model classes are a
video-language model (VLM), a frozen visual/video encoder, or a small temporal classifier.

## RTX 3080 feasibility

The official [RTX 3080 specifications](https://www.nvidia.com/en-gb/geforce/graphics-cards/30-series/rtx-3080-3080ti/)
list 10 GB and 12 GB GDDR6X variants. This workstation is expected to have 12 GB, but Phase 0 must
record the actual value before selecting a tuning configuration. The following judgments reserve
memory for visual activations, attention state, CUDA kernels, and the training stack; static weight
size alone is not enough.

| Candidate | Published/static evidence | 10/12 GB judgment | Role and decision |
|---|---|---|---|
| **DINOv2-S + temporal head** | About 21M backbone parameters; standard ViT-S/14 code and weights are Apache 2.0 | Comfortable for frozen extraction and head training | **Lowest-cost framewise baseline.** Low engineering and overfitting risk, but no native temporal pretraining. |
| **V-JEPA 2.1 ViT-B + temporal head** | 80M parameters at 384 px; repository is primarily MIT with identified Apache-2.0 files | Frozen extraction/head training should fit; activation peak still requires a clip sweep | **First temporal-representation comparison.** Designed for temporally consistent video features. |
| **TimeLens2-2B** | 2.439B BF16 parameters, about 4.88 GB of static BF16 parameters by calculation; Apache 2.0 | Modest-window inference should fit; use 8/4-bit if activations exceed the measured budget | **Recommended first VLM benchmark.** Specialized for multi-span video temporal grounding, but not validated on volleyball or exact contact frames. |
| **Qwen3-VL-2B-Instruct** | 2.128B BF16 parameters, about 4.26 GB static; Apache 2.0 | BF16/FP16 bounded-clip inference fits; short-window QLoRA is estimated at 7–11 GB but is not an official recipe | **Best practical VLM tuning base.** Mature video/timestamp and BF16 LoRA tooling; quantized training needs a stop-gated proof. |
| **Qwen3.5-2B** | About 2.27B parameters/4.56 GB BF16 by calculation; Apache 2.0; video input defaults to 2 fps | Bounded-clip BF16 inference should fit and must be measured | **Current generalist zero-shot comparator.** Keep Qwen3-VL as the tuning base because its video-LoRA path is more mature. |
| **SmolVLM2-500M / 2.2B** | The official 2.2B model card states 5.2 GB for video inference; official work demonstrates 500M tuning in Colab; Apache 2.0 | Both infer; 500M full tuning and 2.2B QLoRA are plausible but unproven on a 10/12 GB 3080 | **Pipeline baseline.** Cheap and useful, but expected to trail a temporal-grounding model on boundary precision. |
| **Gemma 4 E2B** | 5.1B total parameters, 2.3B effective; official load estimates are 11.4 GB BF16, 5.7 GB SFP8, and 2.9 GB Q4 | Q4 inference fits; constrained QLoRA is plausible on 12 GB and risky on 10 GB | **Secondary only.** Google documents up to 60 s assuming 1 fps, not a hard 1-fps cap; exact timing and non-speech audio remain unverified. |
| **Gemma 4 E4B / 12B** | Official E4B load estimates: 17.9 GB BF16, 8.9 GB SFP8, 4.5 GB Q4; 12B Q4: 6.7 GB | Quantized inference only; tuning leaves too little activation headroom | Do not fine-tune on this card. |
| **Muse Glimmer-30B** | About 29.6B parameters; official smallest quantized package targets a 24 GB class system; image/text only and not optimized for video | Does not fit 10 or 12 GB without heavy CPU offload | **Reject.** Too large, no native audio, and poorly aligned to video boundary work. |

Google's figures are model-load estimates including a stated 20% overhead; they are not full
runtime or training peaks. The documentation does not separately quantify KV cache,
visual/audio activations, framework workspace, or training state. Similarly, the Qwen and TimeLens
figures above are parameter-size calculations, not measured end-to-end peaks. This project's
memory gate requires measured headroom of at least the larger of 1 GB or 10% of usable VRAM.

### Windows, WSL2, or bare-metal Linux

Bare-metal Linux is **not required**. The recommended workstation environment is **Windows 11
with WSL2 and an Ubuntu LTS distribution**. This provides the Linux user space assumed by most
model repositories while using the Windows NVIDIA driver to expose the RTX 3080 to CUDA. NVIDIA
describes WSL2 GPU workloads as near-native and supports CUDA applications and containers there.

Native Windows is technically viable for part of the plan:

- official PyTorch CUDA builds support Windows;
- current `bitsandbytes` wheels support Windows x86-64, CUDA 11.8–13.0, and the RTX 3080's `sm86`;
- DINOv2 extraction, the small temporal head, FFmpeg preprocessing, and many zero-shot Transformers
  inference paths should run natively.

It is not the recommended primary environment because FlashAttention remains Linux-first: its
official requirements list Linux and describe Windows compilation as insufficiently tested. Model
training repositories also commonly assume Bash, GNU build tools, Linux paths, and Linux wheels.
Native Windows would therefore create a second support path and may require slower PyTorch SDPA or
eager-attention fallbacks, which can erase the 12 GB card's limited memory margin.

WSL2 is a suitable middle ground for this study. Use it directly first; Docker is optional. Docker
Desktop GPU support on Windows itself uses the WSL2 backend, so adding a container improves
reproducibility but does not avoid WSL. Separate Python environments inside WSL are simpler for the
initial memory probe and for debugging Qwen, Gemma, and SmolVLM dependency differences.

Recommended layout and setup constraints:

1. use current Windows 11, the latest production NVIDIA **Windows** driver, and current WSL2;
2. install **Ubuntu 24.04 LTS**. Its packaged FFmpeg 6.1 supports the `-fps_mode` option used by
   VolleyCut; stock Ubuntu 22.04 ships FFmpeg 4.4 and would require a separate FFmpeg upgrade. Use
   Ubuntu's Python 3.12 for the current application and separate 3.11/3.12 model-family environments
   when their pinned stacks require them;
3. do **not** install a Linux NVIDIA display driver inside WSL; NVIDIA says the Windows driver is
   the only display driver required;
4. install the ordinary PyTorch CUDA wheel first. Install a WSL-Ubuntu CUDA **toolkit** only when a
   package such as FlashAttention must compile; do not install `cuda`, `cuda-12-x`, or
   `cuda-drivers` meta-packages that attempt to add a Linux driver;
5. keep the repository, virtual environments, decoded proxies used during training, and feature
   caches in the WSL ext4 filesystem such as `/home/<user>/volleycut`, not `/mnt/c/...`. Microsoft
   documents substantial cross-filesystem overhead for Linux tools operating on Windows-mounted
   files. Large archival sources may remain on Windows/NAS storage, but stage hot clips and caches
   into WSL for training;
6. limit FlashAttention compilation parallelism, for example `MAX_JOBS=2` or `4`, to avoid exhausting
   system RAM;
7. begin data loading with conservative workers and pinned memory disabled. NVIDIA documents
   limited pinned-system-memory availability in WSL; increase workers and enable pinning only after
   a stable measured run;
8. use PyTorch's allocated/reserved peak counters as the primary memory measurement because WSL's
   `nvidia-smi`/NVML feature set remains incomplete. Record NVML peak use only when the required
   query is available.

The 12 GB update changes the practical model judgment as follows:

- DINOv2-S, V-JEPA 2.1 ViT-B, X3D-S, and SmolVLM2 remain comfortable candidates;
- TimeLens2-2B and Qwen3-VL-2B bounded-clip inference have useful additional headroom;
- Qwen3-VL-2B short-window QLoRA becomes a credible compatibility experiment, but its quantized
  training path is still not an official Qwen recipe and must pass the 200-step stop gate;
- Gemma 4 E2B Q4 inference and a very constrained QLoRA experiment become more plausible, but its
  11.4 GB official BF16 load estimate still leaves no runtime/training margin;
- Gemma E4B/12B tuning and Muse Glimmer remain out of scope.

The final fallback order is therefore **WSL2 Ubuntu → bare-metal Linux only if a reproducible WSL
kernel/driver limitation appears → native Windows for inference/lightweight-model work only**. A
dual boot is not justified for the initial experiments.

### WSL2 packages to preinstall

The following is the recommended `sudo` preparation for a fresh Ubuntu 24.04 WSL distribution.
The base block covers VolleyCut itself. The build block is worth installing now because the planned
model work may compile Python/C++ extensions.

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

Useful but nonessential workstation tools:

```bash
sudo apt-get install --yes \
  git-lfs ccache jq ripgrep rsync tmux unzip
git lfs install
```

Do not install Ubuntu's `nodejs` or `npm` packages for this checkout: Ubuntu 24.04 supplies Node
18, while Next.js 16 requires Node 20.9 or newer. Install the current Node LTS and its bundled npm
without `sudo` through nvm, Volta, or another user-scoped version manager.

No additional packages are required for the normal headless path. In particular, do not preinstall
`libopencv-dev`, `libgl1`, GTK/X11 libraries, cuDNN, or FFmpeg development headers. VolleyCut pins
`opencv-python-headless`, the Ubuntu `ffmpeg` package also installs `ffprobe`, and ordinary PyTorch,
`bitsandbytes`, and Transformers wheels carry or depend on the runtime libraries they need.

Install the following only if a particular package cannot use a wheel and must compile against
FFmpeg locally:

```bash
sudo apt-get install --yes \
  libavcodec-dev libavdevice-dev libavfilter-dev \
  libavformat-dev libavutil-dev \
  libswresample-dev libswscale-dev
```

Other conditional packages are:

- `libgl1` only if an upstream model environment insists on GUI `opencv-python` rather than a
  headless build;
- `nfs-common` only for an NFS mount, or `cifs-utils` only for a direct SMB mount;
- Docker Desktop or Docker Engine only if a pinned container becomes useful. It is not needed for
  the first experiments.

#### Optional CUDA toolkit for compiling FlashAttention

Do **not** install a Linux CUDA toolkit merely to run PyTorch or `bitsandbytes`; their prebuilt
wheels use the Windows NVIDIA driver exposed through WSL. Add a toolkit only if FlashAttention or
another CUDA extension must compile. First pin the PyTorch CUDA build, then install the matching
version from NVIDIA's WSL-Ubuntu repository. For a PyTorch `cu128` environment, the preparation is:

```bash
curl --fail --location --output /tmp/cuda-keyring.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update
sudo apt-get install --yes cuda-toolkit-12-8
```

Configure that build in the shell or model environment, not globally:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=4  # use 2 on a system with limited RAM
```

Never install `cuda`, `cuda-12-x`, `cuda-13-x`, `cuda-drivers`, `nvidia-driver-*`,
`nvidia-dkms-*`, `nvidia-utils-*`, `cuda-compat-*`, Ubuntu's `nvidia-cuda-toolkit`, or an NVIDIA
Linux `.run` installer inside WSL. Those either attempt to install a Linux driver or risk mixing
CUDA installation methods. The Windows NVIDIA driver is the only GPU driver WSL needs. Also avoid
`sudo pip`, `--break-system-packages`, and global ML packages; keep Python dependencies in virtual
environments.

#### Preparation verification

Before copying model weights, run:

```bash
/usr/lib/wsl/lib/nvidia-smi
python3 --version
ffmpeg -version
ffprobe -version
ffmpeg -hide_banner -h full | grep -q fps_mode
ffmpeg -hide_banner -encoders | grep -E 'libx264|[[:space:]]aac[[:space:]]'
gcc --version
cmake --version
ninja --version
node --version  # install current Node LTS separately; must be >=20.9
```

After cloning into the WSL filesystem and installing repository dependencies:

```bash
npm install
npm run analysis:setup
.venv/bin/python -m analysis doctor
```

If the optional CUDA toolkit was installed, `nvcc --version` reports its version. The CUDA version
shown by `nvidia-smi` is the driver's maximum supported API, not proof that the corresponding
toolkit is installed.

### Why the VLM should not own the final timestamp

Small VLM video paths commonly default to 1–2 fps. TimeLens2's official example uses 2 fps, Qwen's
video tools default around 2 fps, and Google documents Gemma's 60-second maximum assuming 1 fps;
this is not a stated Gemma architectural cap. These documented/default rates can recognize setup,
active play, celebration, and reset, but do not establish reliable discrimination at every sample
on the nominal 4 fps analysis grid. Truth remains continuous-time `[start, end)`, and evaluation
uses each recording's actual sampled timestamps/cadence. Low-rate VLMs also make a full-match prompt
expensive and require explicit stitching across context windows.

The ball itself is not a safe shortcut. The existing extractor's 192×108 view often makes it
subpixel. Even the separate 960×540 generic detector pilot reached only 0.522 out-of-fold presence
F1 and 0.157 box F1 at IoU 0.5; tiling hurt transfer and held-out beach recall was zero. A small VLM
should therefore learn **team motion, receiving formation, loss of coordinated play, stand-down,
retrieval, and reset semantics**. Any future ball-dependent branch needs aligned high-resolution
crops, volleyball-specific labels, and a separately qualified detector. See the
[ball-presence pilot](./minimum-ball-presence-pilot-2026-08-11.md).

## Candidate details

### 1. Frozen encoder plus temporal head — recommended first experiment

Use [DINOv2-S](https://github.com/facebookresearch/dinov2) to cache one full-court embedding and a
small grid of regional patch pools at each 4 fps tick. Read frames from the existing 960-wide proxy
rather than enlarging the current 192×108 feature view. Concatenate the cached visual representation
with:

- the existing 90 audiovisual signals;
- quality and availability masks;
- normalized court/region geometry where available;
- explicit valid/ignored masks.

Project the combined vector to 128–256 dimensions and pass it through a compact centered temporal
convolution or transformer with two receptive fields:

- **short branch, 2–8 seconds:** serve motion, contact evidence, abrupt dives, faults, whistles,
  immediate stand-down;
- **long branch, 32–64 seconds:** setup versus live continuity, interruptions, celebration,
  retrieval, and reset.

Predict three dense sequences: `p_live`, `p_serve_contact`, and `p_rally_end`. Train the state head
with balanced BCE or focal loss and the boundary heads against soft pulses around the annotated
start/end ticks. Preserve complete dead time, ignored masks, and hard negatives. Decode the three
signals chronologically into the existing interval contract; include a separate path for short
outcomes instead of relying on a global minimum-duration rule.

This design keeps most parameters frozen, trains very little capacity against only three training
source groups, and retains exact 4 fps outputs. It also lets the study isolate whether semantic
visual representation adds value beyond the current low-level features. DINOv2 is an image model,
so it is the cheapest baseline rather than a preselected production representation.

Before selecting a backbone, compare at least one genuinely temporal representation under the same
head and folds:

- frozen [V-JEPA 2.1](https://github.com/facebookresearch/vjepa2) ViT-B/16 clip and dense features;
  this is the 80M, 384-pixel member intended to produce temporally consistent features; or
- [X3D-S](https://github.com/facebookresearch/SlowFast) on short 16–30 fps boundary crops.

X3D-S is the better precise-refinement candidate if the frozen per-frame model finds rallies but
still places serve or end boundaries too coarsely. [VideoMAE](https://github.com/MCG-NJU/VideoMAE)
is technically relevant but its main repository describes most of the project as CC BY-NC 4.0;
do not make it a product dependency without clearing the exact code and checkpoint license.

### 2. TimeLens2-2B — recommended coarse grounding benchmark

[TimeLens2-2B](https://huggingface.co/MCG-NJU/TimeLens2-2B) is based on Qwen3-VL-2B and trained for
video temporal grounding: a video and query produce all matching `[start, end]` spans. Its model
card reports a 44.5 mean-IoU average over seven grounding benchmarks, while the
[paper](https://arxiv.org/abs/2607.17423) reports a 14.2-point improvement over its Qwen3-VL-2B
base. That is promising transfer evidence, not a VolleyCut result.

Run it zero-shot first with one frozen task query and deterministic decoding. Use fixed,
wall-clock-aligned 32-second windows at 2 fps, an 8-second overlap, and no gold-centered windows at
prediction time. Some rallies can cross a window boundary, so the schema must expose
`liveAtStart`/`liveAtEnd` (or equivalent censored-span flags) and use a deterministic stitching rule.
The prompt should reproduce the label contract exactly: every rally from serve-ball contact to the
first dead-ball instant, including aces and faults and excluding setup, celebration, and retrieval.

Before evaluation, use one frozen adapter to convert model-native spans or bins into seconds and
then validate bounded, ordered, non-overlapping half-open intervals. Report malformed, dropped, and
repaired outputs separately; do not silently interpret an inclusive or unknown native endpoint as
VolleyCut's `[start, end)` convention. Score the converted spans directly and also test them as:

- proposals sent to the precise temporal refiner;
- a feature indicating coarse semantic state;
- a review queue, never as labels.

Do not reproduce TimeLens2's large SFT/RL training recipe on the 3080. A later short-window LoRA is
plausible through the Qwen stack, but only after zero-shot usefulness and actual VRAM are measured.

### 3. Qwen3-VL-2B — recommended VLM fine-tuning candidate

[Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) has explicit video-time
alignment, configurable frame budgets, and an
[official video/LoRA training path](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/README.md).
It is the most practical model for a controlled generative experiment on this card.

Use hierarchical examples rather than full matches:

- coarse examples: 16–32 seconds at 1–2 fps, returning all rally spans in the window;
- boundary examples: very short clips sampled at 8–16 fps, returning temporal-patch/bin indices or
  timestamps as coarse evidence for a separate high-frame-rate refiner;
- hard negatives: warmups, between-point setup, adjacent-court motion, celebrations, retrieval,
  camera motion, aborted tosses, preparation bounces, and unrelated non-serve impacts.

Return a tiny constrained schema whose bins are defined by the processor's temporal patches, for
example:

```json
{"liveAtStart":false,"liveAtEnd":false,"events":[{"startBin":7,"endBin":29}]}
```

Qwen's official guide documents BF16 video LoRA, not a 4-bit QLoRA loader. Treat QLoRA as a custom
compatibility branch: first prove 4-bit multimodal loading, one forward/backward pass, 200 stable
steps, gradient flow into adapters, and an adapter-only save/reload. Stop on unsupported kernels or
OOM rather than hiding the problem with CPU offload.

If that gate passes, start with LoRA rank 8, batch size 1, gradient accumulation to an effective
batch of 16–32, gradient checkpointing, frozen vision tower, short output targets, and 8-bit
optimizer states. Sweep only frame count and resolution during the memory probe. Increase rank to
16 or unfreeze the projector only if the rank-8 run is stable and useful. Qwen uses temporal
patching, so sampled input frames must not be treated as independently resolvable output ticks.
Run three seeds before making a model decision.

### 4. SmolVLM2 and Gemma 4 — controls, not the main bet

[SmolVLM2](https://huggingface.co/blog/smolvlm2) is ideal for proving that clip construction,
collation, constrained output, and evaluation work. The official evidence establishes 5.2 GB
video inference for the 2.2B model and demonstrates 500M tuning in Colab, not specifically on this
3080. Treat a 500M full tune or 2.2B QLoRA as a Phase 0 measurement candidate and then, if it fits,
as a low-cost control. A smaller model winning would be operationally valuable, but it should not
be assumed to have TimeLens/Qwen boundary ability.

[Gemma 4](https://ai.google.dev/gemma/docs/core) is interesting because E2B accepts image/video
frames and audio. Its [model card](https://ai.google.dev/gemma/docs/core/model_card_4) documents a
60-second video maximum assuming 1 fps and 30 seconds of audio; it does not state a hard 1-fps cap.
The documented audio capabilities are speech recognition and speech translation, not whistle,
impact, crowd, or synchronized audiovisual event detection. Treat synchronized video frames and
audio as separate inputs until a processor smoke test proves muxed-track ingestion and alignment.
Benchmark E2B Q4 on a small, frozen development sample only after the primary runs. Proceed to
QLoRA only on a 12 GB card with a measured configuration that retains headroom and only if the
screen first establishes useful non-speech event evidence.

[Muse Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) is not a small-model option
for this machine. Its model card describes image-plus-text input, frame-by-frame video handling,
and 24/32 GB deployment targets. CPU offload would add substantial latency without solving the
temporal-alignment mismatch.

## Data and leakage constraints

The frozen full corpus contains 9 recordings, 151.37 minutes, 345 rallies, and 2,614.739 live
seconds. It has only five independent source groups: three train, one grass validation, and one
indoor regression test; beach appears only in training. The overlapping 90-second pilot excerpts
must never be added to their full recordings as extra examples.

The current data is enough to answer:

- can the model load and train on the 3080;
- do frozen semantic embeddings add development OOF signal;
- can a VLM follow the interval schema;
- is a representation promising enough to justify more labels?

It is not enough to establish:

- generalization across indoor, grass, and beach capture styles;
- reliable per-outcome performance for short rallies, aces, and faults;
- benefit from tuning a high-capacity generative model rather than memorizing venue/camera cues;
- unattended export quality.

Human gold is usable, but 290 of 345 final rows retain `ai-prelabel` provenance: 185/230 train,
70/76 validation, and 35/39 test. These are continuously human-reviewed assisted labels, and the
tag persists after correction, so it does not mean the boundaries are uncorrected. Every candidate
must nevertheless disclose possible annotation anchoring; this is annotation dependence, not
manifest source-split leakage. No tracked report establishes independent boundary agreement or
adjudication; an [earlier comparison](./analysis-vs-pilot-gold-2026-08-09.md) also requested that
pass. The next sealed test groups should therefore be labeled from blank video without model
suggestions and receive independent boundary review/adjudication before supporting a subsecond-gain
claim.

Checkpoint pretraining exposure is a separate unknown. Source-group separation cannot prove that a
web-pretrained encoder or VLM never saw these public/source videos. Audit disclosed source-video
IDs where possible and record unresolved exposure as unknown rather than claiming a fully clean
pretraining split.

Create a sequence artifact before training any new architecture:

```text
recording_id, source_group, split, video/proxy hashes, ROI
times[T], valid_mask[T], ignored_mask[T]
visual_embeddings[T,D], audiovisual_features[T,90]
live_target[T], serve_pulse[T], end_pulse[T]
event table: start, end, outcome tags
hard-negative spans and reason tags
```

Split whole source groups before generating frames, clips, embeddings, crops, augmentations, or
model suggestions. A source group includes sister recordings, excerpts, proxies, re-encodes, and
device-contiguous sessions. Label-free frozen embeddings may be generated for any split, but
scalers, PCA, codebooks, calibration, thresholds, and supervised feature selection must fit only on
the active fitting groups. Preserve the existing within-recording percentile transform for the 90
audiovisual signals; only learned dataset-level normalization is fold-fit. Label documents and
manifests already carry `hardNegatives`, but the typed `Recording` and binary training target do not
expose them directly. Carry those spans into the sequence artifact so sampling can target them
without changing evaluation truth. The frozen manifest contains only two explicit hard negatives,
both timeouts, so deterministic ordinary-dead-time sampling remains necessary and reason-specific
hard-negative results are underpowered until more spans are tagged.

For an immediate decision-grade follow-up, collect at least 12 new independent source groups: six
for development and six sealed for final evaluation, with two indoor, two grass, and two beach
groups in each pool. Assign them using capture metadata before annotation. Hash and freeze the
sealed blank-human truth before generating any model suggestions, then require independent boundary
review/adjudication. The longer-term target is about 30 total groups—roughly ten per environment—so
model comparisons are not determined by one venue. These counts are engineering targets rather
than a formal power calculation.

## Experiment plan

### Phase 0 — hardware and environment probe, half a day

On the remote machine:

1. record GPU name, exact memory, driver, CUDA, OS, PyTorch, Transformers, and model revisions;
2. verify FP16/BF16 support and quantization kernels;
3. if using WSL, record the Windows build, WSL version/kernel, distribution, and whether the project
   and caches are in the Linux filesystem;
4. measure `max_memory_allocated` and `max_memory_reserved`, plus NVML peak use when WSL exposes the
   required query, rather than relying on occasional `nvidia-smi` snapshots;
5. benchmark batch 1 at 8/16/32/64 frames and 224/336/384 px where supported;
6. record throughput and fail any configuration with less than `max(1 GB, 10% of usable VRAM)`
   measured headroom;
7. when activations cause OOM, reduce frames, resolution, or context before changing weight
   quantization;
8. pin separate model-family environments for Qwen, Gemma, and SmolVLM, recording PyTorch,
   Transformers, FlashAttention, bitsandbytes, CUDA, and driver versions.

This phase converts every VLM memory estimate in this report into a measured envelope. Pin the
working environment before producing comparable scores.

### Phase 1 — frozen direct-model screen, one to two days

Freeze wall-clock windowing, prompt, decoding, and stitching before inspecting outputs. Run fixed,
exhaustive, continuous windows covering complete development recordings; prediction windows must
not be selected or centered using gold boundaries. Run on development groups only:

- TimeLens2-2B zero-shot;
- Qwen3-VL-2B-Instruct zero-shot;
- Qwen3.5-2B zero-shot as the current generalist comparator;
- SmolVLM2-500M or 2.2B zero-shot;
- Gemma 4 E2B Q4 on a small stratified diagnostic subset if the processor is stable.

Score only the exhaustive windows for headline precision/recall and the advance decision.
Separately inspect gold-stratified ordinary, short, ace/fault, ordinary-dead-time, and the two
explicit timeout hard-negative clips as oracle failure diagnostics only. Carry
`liveAtStart`/`liveAtEnd` across censored windows. Record event metrics, JSON validity,
repeatability, peak VRAM, and real-time factor. This is a screen, not model selection from the
current test source.

**Advance criterion:** compare with frozen current-v2 predictions and a duration-matched heuristic.
On a threshold sweep matched for retained-video fraction, require higher any-overlap recall. At the
chosen operating point, require at least a five-point short/fault recall gain while retained-video
fraction rises by no more than five points. Raw schema validity must be at least 99.5%; score
repaired JSON separately. Require real-time factor at most 1.0 for offline processing and
effectively identical repeated deterministic inference. Otherwise stop generative work and
continue only with frozen encoders.

### Phase 2 — recommended frozen-feature temporal model, three to five days

Implement DINOv2-S caching and the three-head temporal model. Under identical source-group folds,
run these preregistered ablations:

1. current 90 audiovisual signals only;
2. frozen global visual embedding only;
3. global plus regional embeddings;
4. global/regional embeddings plus the 90 audiovisual signals;
5. candidate 4 plus separate serve and end heads;
6. candidate 5 plus the short-outcome decoding path.

Cache embeddings once. Train the small head with three seeds. Keep the test source closed; use
nested leave-one-source-group-out development evaluation and the existing decoder/evaluator.
Replay a same-fold, same-seed baseline control through exactly the same evaluator and recordings;
the historical frozen OOF reference used one seed, so it is contextual rather than a complete
paired control. Aggregate the three candidate/control seeds by median. The development objective
remains `0.55 × event F1 + 0.30 × time IoU + 0.15 × live recall`.

**Advance criterion:** relative to the frozen full-audiovisual development OOF reference and the
paired control, improve event F1 by at least 0.02 and time IoU by at least 0.02, lose no more than
one percentage point of live recall, improve the weighted objective in at least three of four
groups, and do not reduce strict short-event recall. The objective delta versus the paired control
must be positive for every seed.

**Refiner-entry criterion:** a model may enter Phase 3 without the 0.02 time-IoU gain if event F1
improves by at least 0.02, time IoU is no worse than baseline, live-recall loss is at most one point,
and the objective delta is positive. This separates a promising coverage model from one already
precise enough to advance directly.

### Phase 3 — precise video refinement, three to five days

If Phase 2 improves rally coverage but not boundaries, compare one temporal encoder—V-JEPA 2.1
ViT-B frozen features or X3D-S—on high-frame-rate crops. Because current end MAE is 2.419 seconds,
begin with an eight-second `[-4 s, +4 s]` crop around each predicted boundary at the highest frame
rate that passes Phase 0, using full-court and court-ROI views. A later micro-refiner may narrow the
crop. Predict a per-frame contact or dead transition rather than text.

Evaluate the refiner as an exact no-op when confidence is insufficient. Require unchanged proposal
count, non-decreasing F1 at IoU 0.5, and at least a 0.02 gain in F1 at IoU 0.7 or time IoU. Matched
end MAE must improve by at least 0.2 seconds while boundary recall over all gold events and
short/long event recall do not fall.

### Phase 4 — one parameter-efficient VLM run, after a positive screen

Choose exactly one:

- Qwen3-VL-2B LoRA, or QLoRA only after the explicit compatibility gate, if it has the best
  temporal/schema screen;
- SmolVLM2-500M full tuning if the goal is a low-cost end-to-end feasibility check;
- TimeLens2-2B short-window LoRA if its zero-shot spans are clearly strongest and its Qwen-based
  training path works within the measured envelope.

Do not sweep multiple model families on the current three training groups. First run a fixed-split
three-seed tuning feasibility test, approximately 9–60 GPU hours at the estimates below. A full
four-fold OOF × three-seed study would require at least 36–240 GPU hours before inner tuning, so
defer it until the representation passes the fixed split and more development groups exist. Treat
the result as representation research until new groups exist.

### Phase 5 — freeze and evaluate once on new groups

Collect and validate new source groups in parallel with model work. Freeze model weights, prompts,
thresholds, decoder, dependency revisions, and artifact hashes before opening the six new sealed
groups. Report per-environment and per-source deltas. The existing indoor test remains a regression
check, not the headline test.

Promote only to the same **review-assist** status if the candidate passes the development gates and,
on the fresh test, pooled event F1 and time-IoU deltas are nonnegative, live-recall loss is at most
one point, short/fault recall does not fall, the objective improves in at least four of six groups,
mean delta is nonnegative in every environment, and no group objective delta is below -0.05. With
only two groups per environment, paired group deltas are primary and bootstrap intervals are
descriptive. Unattended exports need a separate product gate with much stronger event
precision/recall and boundary-tail evidence; this study should not grant that status.

## Evaluation and reporting contract

Use unpadded predictions for model selection and report symmetric export padding separately. For
every generative candidate, apply the same frozen native-range-to-seconds adapter and interval
validator described above before scoring. For every candidate record:

- event precision, recall, and F1 at IoU 0.3, 0.5, and 0.7;
- time IoU and live-time precision/recall;
- matched-event start/end MAE with its denominator, plus boundary recall over every gold event
  within 0.25, 0.5, 1, and 2 seconds;
- missed live seconds, retained dead seconds, and video-retained fraction;
- short (at most 3 s), ace, service-fault, ordinary-long, environment, format, and quality slices;
- paired result for every source group and source-group bootstrap intervals where meaningful;
- peak allocated/reserved VRAM, wall time, real-time factor, and cache size;
- for generative models, raw schema validity, invalid-output repair rate, and three-run stability.

Do not use frame accuracy as the headline metric. Do not select on the repeatedly inspected indoor
test. Do not convert VLM suggestions into truth without continuous human review of the whole video.

## Expected effort and artifacts

Planning estimates, to be replaced by Phase 0 measurements:

| Work item | Expected 3080 time | Main artifact |
|---|---:|---|
| Hardware/model memory sweep | 0.5 day | Pinned environment and VRAM/throughput matrix |
| Direct VLM screen | 1–2 days | Frozen prompts, raw outputs, development-only report |
| DINOv2 pooled-feature cache | 0.5–2 GPU hours | Versioned embedding cache with video hashes |
| Temporal-head implementation and grouped ablations | 3–5 days; roughly 1–3 GPU hours after caching | Models, OOF predictions, ablation report |
| One temporal-backbone/refiner comparison | 3–5 days; roughly 12–36 GPU hours | Frozen/partial-tune comparison |
| One 500M full tune or 2B LoRA/QLoRA fixed-split study | roughly 9–60 GPU hours for three seeds | Adapter/full weights, logs, raw generations |
| Four-fold × three-seed generative OOF study | at least 36–240 GPU hours before inner tuning | Deferred until the representation and data justify it |

The ranges depend heavily on resolution, frame count, storage throughput, and the exact 3080. A
200-step smoke run must precede any full tuning job. Store external caches and weights outside Git;
commit configuration, hashes, prompts, evaluation summaries, and the promotion decision to this
research directory. Cache only pooled DINOv2 outputs: a class token plus nine regional vectors is
roughly 0.28 GB in FP16 or 0.56 GB in FP32 for the current corpus, while full patch-token caching is
roughly 7 GB and is unnecessary. Reserve approximately 75–100 GB on the remote machine for isolated
environments, one copy of each base model, run logs, and adapter-only checkpoints.

## Final recommendation

Proceed, but frame the work as **small video-model representation learning**, not “fine-tune an LLM
to cut matches.” Start with the DINOv2-S temporal model because it cheaply preserves the 4 fps task,
reuses the strongest existing audiovisual evidence, fits comfortably, and has limited capacity to
memorize five source groups. Compare it with V-JEPA 2.1 ViT-B before choosing the backbone. In
parallel, run TimeLens2-2B zero-shot as the direct temporal-VLM benchmark. Spend the larger tuning
budget on Qwen3-VL-2B only if that screen demonstrates useful semantic or boundary signal and its
LoRA/QLoRA path passes the hardware gate.

Do not spend time on Muse Glimmer for this GPU. Do not make Gemma 4 the primary path; its documented
audio evidence is speech-oriented, and its video/runtime profile is a poor first match for exact
volleyball boundaries. Most importantly, add untouched indoor, grass, and beach groups while the
modeling work proceeds. Data diversity, not the 3080, is the limiting factor for a credible
promotion decision.

## External source access log

Availability and model-card claims were checked on 2026-08-12. No external weights were downloaded
or executed for this feasibility report.

- [NVIDIA GeForce RTX 3080/3080 Ti specifications](https://www.nvidia.com/en-gb/geforce/graphics-cards/30-series/rtx-3080-3080ti/)
- [NVIDIA CUDA on WSL2 guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [Microsoft GPU-accelerated ML in WSL](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)
- [Microsoft WSL filesystem guidance](https://learn.microsoft.com/en-us/windows/wsl/filesystems)
- [PyTorch Windows and CUDA installation](https://pytorch.org/get-started/locally/)
- [bitsandbytes platform/CUDA installation matrix](https://huggingface.co/docs/bitsandbytes/en/installation)
- [FlashAttention requirements](https://github.com/Dao-AILab/flash-attention)
- [Docker Desktop GPU support](https://docs.docker.com/desktop/features/gpu/)
- [Ubuntu 24.04 FFmpeg 6.1 package](https://packages.ubuntu.com/noble/ffmpeg)
- [Ubuntu 22.04 FFmpeg 4.4 package](https://packages.ubuntu.com/jammy/ffmpeg)
- [Next.js system requirements](https://nextjs.org/docs/pages/getting-started/installation)
- [TorchCodec FFmpeg requirements](https://github.com/meta-pytorch/torchcodec)
- [PyAV source-build requirements](https://pyav.org/docs/stable/overview/installation.html)
- [Gemma 4 overview and memory estimates](https://ai.google.dev/gemma/docs/core)
- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Gemma 4 video guide](https://ai.google.dev/gemma/docs/capabilities/vision/video)
- [Muse Glimmer-30B model card](https://huggingface.co/meta-models/Muse-Glimmer-30B)
- [TimeLens2 repository](https://github.com/MCG-NJU/TimeLens2),
  [model card](https://huggingface.co/MCG-NJU/TimeLens2-2B), and
  [paper](https://arxiv.org/abs/2607.17423)
- [Qwen3-VL-2B model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) and
  [fine-tuning guide](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/README.md)
- [Qwen3-VL repository and video processor](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- [SmolVLM2 release and fine-tuning guidance](https://huggingface.co/blog/smolvlm2)
- [SmolVLM2-2.2B model card](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)
- [DINOv2 repository](https://github.com/facebookresearch/dinov2)
- [V-JEPA 2 repository](https://github.com/facebookresearch/vjepa2)
- [PySlowFast/X3D repository](https://github.com/facebookresearch/SlowFast)
- [VideoMAE repository and license](https://github.com/MCG-NJU/VideoMAE)

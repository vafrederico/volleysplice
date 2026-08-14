# WSL DINOv2 temporal execution — 2026-08-13

## Status

Track T implementation, no-label hardware/backbone qualification, development extraction, and the
label-informed nested study completed. Canonical `/mnt/z` access was restored during the run after
an earlier WSL permission failure; CUDA and proxy decoding were healthy throughout.

The frozen development manifest contains 8 train/validation recordings, 4 source groups, and 306
gold rallies. DINOv2 extraction produced 28,908 analysis ticks at 4 fps. The 72-trial development
study selected `dino_regions_plus_audiovisual_boundary_heads` and passed the preregistered promotion
gate. The required 54-trial no-beach sensitivity study also completed; with 6 recordings and 3
source groups, it selected `dino_class_only` under the filtered data contract. This is an environment
sensitivity result, not beach generalization evidence. No model checkpoint was published, and the
protected test split remains unopened.

Immutable development artifacts:

- DINO extraction index: `/mnt/z/volleycut/labeling-v1-2026-08-09/reports/dinov2-vits14-development-extraction-2026-08-13.json`
  (SHA-256 `52ce3e1e73cdd27971daf0210fb44981f59540a7ac18cf27cc7eb322a06a7470`);
- temporal study report: `/mnt/z/volleycut/labeling-v1-2026-08-09/reports/dinov2-temporal-development-study-2026-08-13.json`
  (SHA-256 `0f8238f3133f7bfa5af152c0de3a5ee351409a15a4098519529b7267209715cd`).
- no-beach sensitivity report: `/mnt/z/volleycut/labeling-v1-2026-08-09/reports/dinov2-temporal-development-study-no-beach-2026-08-13.json`
  (SHA-256 `6c19a3e07a41dfefbd46fd78342b2001621542c02c6860a75e1c524371526cc0`).

## Implementation

The repository now contains:

- `analysis/dinov2_embeddings.py` — local-checkout-only DINOv2 loading, official preprocessing,
  4-fps timestamping, `[T,10,384]` pooled-token caches, resumable chunks, metadata validation,
  and atomic publication;
- `analysis/semantic_temporal_model.py` — the frozen-DINO plus 90-signal five-block temporal head,
  fold-local scaling/class weights, masked three-head loss, deterministic seeding, and parameter
  count guard;
- `analysis/semantic_temporal_study.py` — target construction, chunk coverage, live/boundary/short
  decoding, and nested source-group development study contracts;
- `scripts/extract-dinov2-embeddings.py` — development-only extraction plus a manifest-independent
  resource sweep; and
- `scripts/evaluate-semantic-temporal-model.py` — development study runner with a fail-closed
  protected-test path.

Synthetic contract coverage was added in the three corresponding test modules. The full repository
suite passes **477 tests**.

## Qualified environment

The documented WSL GPU path is available through `/usr/lib/wsl/lib/nvidia-smi` (that directory is
not on the Codex shell's default `PATH`). The measured device is:

| Item | Measured value |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 |
| VRAM | 10,240 MiB total; 8,593 MiB free at probe |
| Windows driver | 576.80 |
| PyTorch | 2.11.0+cu128 |
| TorchVision | 0.26.0+cu128 |
| CUDA visible to PyTorch | yes; one device |
| DINO repository | `facebookresearch/dinov2`, commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8` |
| DINO checkpoint | official `dinov2_vits14_pretrain.pth` |
| Checkpoint SHA-256 | `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` |
| DINO license | Apache-2.0 |

The resolved Python package set is captured in
[`dinov2-temporal-environment-2026-08-13.lock`](./dinov2-temporal-environment-2026-08-13.lock).

The 10-GB measurement is recorded as a hardware deviation from the handoff's 12-GB assumption;
the run used measured headroom rather than the assumption.

## No-label qualification

The one-frame 224/336 sweep and a 32-frame proxy throughput probe both passed. The larger 336-pixel
input was selected because it stayed comfortably inside the memory gate and retained practical
throughput.

| Input | One-frame shape | Peak allocated VRAM | 32-frame throughput |
|---:|---|---:|---:|
| 224 | `[1,10,384]` | 102,738,944 bytes | 130.946 frames/s |
| 336 | `[1,10,384]` | 109,050,880 bytes | 93.990 frames/s |

Both outputs were finite. Two fresh processes produced the same fixed-frame output hash:
`ab71e6efe6b4b9aae93512d23ea027abb8a90b8bfc0884d7bac4c00ca95b0749`.

The machine-readable sweep was written temporarily to
`/tmp/volleycut-dinov2-resource-2026-08-13-v2.json` with SHA-256
`6ee5b086d74f93ca3c76c467c08dae3ebf39602a413f1b1a8a1e104ca6017693`.

The official DINO code emitted its expected “xFormers is not available” warnings and used its
regular PyTorch attention/FFN fallback. This is recorded as the qualified backend; adding xFormers
would be a new environment/resource configuration.

## Earlier WSL access failure and recovery

The following canonical inputs initially returned `PermissionError` for the WSL process:

```text
/mnt/z/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json
/mnt/z/volleycut/labeling-v1-2026-08-09/features/audiovisual-v2/*.npz
```

They were readable again later in the same run. The restored access was verified by loading the
manifest, reading all 18 existing audiovisual NPZ files, regenerating four current-key audiovisual
caches, extracting all eight development DINO caches, and joining every sequence at the 4-fps grid.
The proxy used for the no-label probe was readable and hashed as
`236b5acf162441b6b81d2f83a0f85b42ed5706acf3629027da80e2e74188ceca`.

The original failure was therefore an access-state issue, not a missing path: the affected paths
were the manifest above and the `features/audiovisual-v2/*.npz` cache directory under the same
labeling workspace. No test recording was accessed to resolve it.

## Development decision

The primary development aggregate (24 held-out recording/seed rows) was:

| Candidate | Event F1 | Time IoU | Live-time recall | Objective |
|---|---:|---:|---:|---:|
| Audiovisual TCN control | 0.4640 | 0.4543 | 0.6724 | 0.4923 |
| DINO regions + audiovisual + boundary heads | **0.5544** | **0.5179** | **0.7003** | **0.5653** |

Relative to the control, the selected candidate improved event F1 by 0.0904 and time IoU by
0.0636, while improving live-time recall by 0.0280. Its strict short-rally and service-fault
recalls were non-decreasing, and the seed/group paired gate passed.

The no-beach sensitivity report selected DINO class-only (event F1 0.5987, time IoU 0.5968,
live-time recall 0.8029) against its filtered audiovisual control (0.3631 / 0.3303 / 0.4704).
Because the filter removes one environment and weakens the source-group set, this result is kept as
sensitivity evidence only. The protected indoor test was not loaded by either report.

# Unsloth rally-window dataset — 2026-08-12

## Result

The frozen, continuously human-reviewed rally corpus has been exported as a local
Unsloth-compatible video conversation dataset:

```text
/mnt/freenas/volleycut/labeling-v1-2026-08-09/datasets/
  unsloth-rally-windows-v1-2026-08-12/
```

The artifact is intentionally small (less than 1 MB of JSON and documentation) because it
references the existing verified 960x540 proxy videos. It does not duplicate or transcode the
2.5-hour media corpus. Current `unsloth-zoo` accepts OpenAI-style `messages` with a `video` content
item and supports `video_start`, `video_end`, `fps`, `min_frames`, and `max_frames`; its video
readers sample only the requested temporal segment. The export therefore works directly with
`UnslothVisionDataCollator` while retaining the canonical source-video hashes.

A second filtered artifact excludes both beach recordings:

```text
/mnt/freenas/volleycut/labeling-v1-2026-08-09/datasets/
  unsloth-rally-windows-v1-no-beach-2026-08-12/
```

It has 176 train windows from four recordings/two source groups, 80 validation windows, and 46
test windows. The validation and test sets are unchanged because they contain no beach recording.
The filtered train set contains 159 unique gold rallies (272 window appearances) and 9 empty
windows. Its metadata records `selection.excludedEnvironments: ["beach"]`.

For the native-Windows Unsloth workstation, use the corresponding path-rebased artifacts:

```text
Z:\volleycut\labeling-v1-2026-08-09\datasets\
  unsloth-rally-windows-v1-windows-z-2026-08-12\
  unsloth-rally-windows-v1-no-beach-windows-z-2026-08-12\
```

They live on the same NAS under `/mnt/freenas/volleycut/.../datasets/`. Their row counts, labels,
windows, source hashes, and split assignments match the two original artifacts; only the embedded
consumer video paths differ. The JSONL uses `Z:/volleycut/...` so Python can read the strings
without backslash escaping. These are the variants to use for native-Windows training. The full
execution instructions are in the
[Windows/WSL model-training handoff](./model-training-execution-handoff-2026-08-12.md).

This is a **visual video** SFT dataset. Unsloth's video reader does not turn the MP4 audio track
into model audio input. A later Gemma audiovisual experiment needs a separate, explicitly aligned
audio-content export and a processor smoke test; it should not silently claim that this dataset
trained on audio.

## Dataset construction

The exporter uses the fixed coarse-window contract from the RTX 3080 feasibility plan:

- exhaustive 32-second wall-clock windows;
- 24-second stride, producing an 8-second overlap;
- no gold-centered or event-selected training windows;
- a conservative request of 1 sampled frame per second, capped at 32 frames for a full window;
- exact train, validation, and test assignments from `full-gold-v1.json`;
- absolute paths to verified proxy videos plus each proxy's SHA-256;
- labels expressed relative to the window using half-open `[start,end)` spans;
- `liveAtStart` and `liveAtEnd` for rallies censored by a window edge;
- `ordinary`, `ace`, and `service-fault` outcomes;
- empty windows retained as ordinary dead-time negatives;
- explicit hard-negative overlaps retained as metadata;
- any window touching an ignored interval excluded. The frozen corpus currently has no ignored
  intervals, so no row was removed by that rule.

One assistant answer is compact JSON:

```json
{"liveAtStart":false,"liveAtEnd":false,"rallies":[{"start":15.9,"end":20.8,"outcome":"ordinary"}]}
```

The user instruction restates the full labeling contract: serve-ball contact through the first
dead-ball instant; aces and faults included; setup, celebration, retrieval, timeouts, and warmups
excluded. The video-content row carries absolute source times, while the answer carries
window-relative times.

## Counts and split safety

| Split | Examples | Recordings | Source groups | Unique gold rallies | Rally appearances | Empty examples |
|---|---:|---:|---:|---:|---:|---:|
| Train | 254 | 6 | 3 | 230 | 380 | 14 |
| Validation | 80 | 2 | 1 | 76 | 121 | 4 |
| Test | 46 | 1 | 1 | 39 | 65 | 5 |
| **Total** | **380** | **9** | **5** | **345** | **566** | **23** |

Rally appearances exceed unique rally counts because the fixed overlap intentionally shows some
events in two adjacent windows. That duplication never crosses a source-group split. The audit
confirmed the exact five-group assignment:

```text
train      yang-showmatch-20260703, kb-kob-20250614, spu-semi-20260521
validation shoreline-kob-20250616
test       spu-match1-20260526
```

Only `train.jsonl` is fitting data. `validation.jsonl` is for model/decoder selection and
`test.jsonl` must remain a regression-only evaluation file. The test source was repeatedly
inspected in previous development, so it is not fresh generalization evidence.

## Files and use

The artifact contains:

```text
README.md
dataset.json
checksums.sha256
train.jsonl
validation.jsonl
test.jsonl
```

Load it from its directory with Hugging Face Datasets:

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files={
        "train": "train.jsonl",
        "validation": "validation.jsonl",
        "test": "test.jsonl",
    },
)
```

For Unsloth SFT, pass `dataset["train"]` to `SFTTrainer` with
`UnslothVisionDataCollator(model, tokenizer)`, `remove_unused_columns=False`,
`dataset_text_field=""`, and `dataset_kwargs={"skip_prepare_dataset": True}`. Start with batch
size 1 on the 12 GB RTX 3080. The 1 fps setting is a compatibility baseline, not a claim of
boundary precision; test 2 fps only after the Phase 0 VRAM probe passes.

The original artifacts embed the Linux NAS mount. The `windows-z` artifacts embed the native
Windows `Z:` mapping. A WSL consumer at `/mnt/z/volleycut` needs another path-rebased artifact or a
loader-level translation that is explicitly tested and recorded. For the current Unsloth stack,
prefer TorchCodec for video loading; if more than one reader backend is installed,
`FORCE_UNSLOTH_VIDEO_READER=torchcodec` makes that choice explicit. Install Unsloth, Hugging Face
Datasets, and the TorchCodec version compatible with the selected PyTorch build inside the model
virtual environment, not with `sudo`.

Regenerate without overwriting an existing artifact:

```bash
python3 scripts/export-unsloth-dataset.py \
  /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  /mnt/freenas/volleycut/labeling-v1-2026-08-09/datasets/unsloth-rally-windows-v1-2026-08-12
```

Regenerate the no-beach variant with:

```bash
python3 scripts/export-unsloth-dataset.py \
  /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  /mnt/freenas/volleycut/labeling-v1-2026-08-09/datasets/unsloth-rally-windows-v1-no-beach-2026-08-12 \
  --exclude-environment beach
```

Generate immutable native-Windows variants from the NAS source with:

```bash
python3 scripts/export-unsloth-dataset.py \
  /mnt/freenas/volleycut/labeling-v1-2026-08-09/manifests/full-gold-v1.json \
  /mnt/freenas/volleycut/labeling-v1-2026-08-09/datasets/NEW-WINDOWS-DATASET \
  --source-data-root /mnt/freenas/volleycut \
  --consumer-data-root Z:/volleycut
```

Add `--exclude-environment beach` for its filtered counterpart. For a WSL consumer, replace the
last value with `/mnt/z/volleycut`. Supply both root options together; the exporter rejects relative
consumer roots and source videos outside the declared source root.

Use a new output name for a different frame-rate/window profile. For example, add
`--sample-fps 2` only after measuring that 64-frame windows fit with safe VRAM headroom.

## Verification

The generated `checksums.sha256` validates the README, dataset metadata, and all three JSONL files.
The completed audit also verified:

- every JSONL row has exactly one video request and one assistant answer;
- every referenced source video exists;
- every video segment is positive and bounded;
- every assistant answer parses as the constrained JSON schema;
- row identifiers are unique within each split;
- all 230/76/39 source rallies appear in train/validation/test respectively;
- no source group occurs in more than one split;
- all 380/302 row-level video references in the Windows-rebased full/no-beach artifacts resolve
  back to existing NAS media and contain no leaked `/mnt/freenas` paths; and
- the repository's full 466-test Python suite passes after the exporter/path-rebasing additions.

## Limitations

This corpus has only five independent source groups, including just three training groups and no
held-out beach group. The training split's 230 rallies are too small for an unconstrained end-to-end
VLM study, which is why the planned run is a narrow adapter feasibility test rather than a model
sweep.

The gold is human-reviewed, but 290 of 345 final rally rows retain `ai-prelabel` provenance. The
labeling policy deliberately preserves provenance after correction, so this does not show that the
boundaries are uncorrected. It does require disclosure, and a same-family student/teacher result
needs a newly blank-slate-labeled source group for an independent claim.

Finally, the proxy resolution supports team-state, formation, stand-down, and reset semantics. It
does not make this a ball-tracking dataset: the existing generic ball pilot failed its grouped
transfer gates even at 960x540.

## Primary format references

- [Unsloth vision fine-tuning](https://unsloth.ai/docs/basics/vision-fine-tuning)
- [Unsloth Qwen3-VL fine-tuning guide](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune/qwen3-vl-how-to-run-and-fine-tune)
- [Current Unsloth video collation and segment-reader implementation](https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/vision_utils.py)
- [Qwen3-VL video message format and processor](https://github.com/QwenLM/Qwen3-VL)

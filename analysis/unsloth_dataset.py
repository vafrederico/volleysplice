from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable

from .ffmpeg import probe
from .schema import ANNOTATION_POLICY_ID, DatasetManifest, Interval, Recording


UNSLOTH_DATASET_SCHEMA_VERSION = 1
UNSLOTH_ROW_SCHEMA = "volleycut-unsloth-rally-window-v1"
DEFAULT_WINDOW_SECONDS = 32.0
DEFAULT_STRIDE_SECONDS = 24.0
DEFAULT_SAMPLE_FPS = 1.0
VALID_ENVIRONMENTS = {"indoor", "beach", "grass", "broadcast", "unknown"}
DEFAULT_INSTRUCTION = (
    "Find every live volleyball rally in this video window. A rally starts at serve-ball "
    "contact and ends at the first instant play is dead. Include aces and service faults. "
    "Exclude setup, celebration, ball retrieval, timeouts, and warmups. Return only compact "
    "JSON with keys liveAtStart, liveAtEnd, and rallies. Each rally must have start, end, and "
    "outcome (ordinary, ace, or service-fault). Times are seconds relative to this window, "
    "rounded to three decimals, using half-open [start,end) intervals. liveAtStart/liveAtEnd "
    "mean a rally crosses that window boundary. Return an empty rallies array when none occur."
)


class UnslothDatasetError(RuntimeError):
    """Raised when an Unsloth export would be invalid or unsafe."""


def _normalize_excluded_environments(values: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(str(value).strip().lower() for value in values))
    if any(not value for value in normalized):
        raise UnslothDatasetError("excluded environments must be non-empty strings")
    unknown = sorted(set(normalized) - VALID_ENVIRONMENTS)
    if unknown:
        raise UnslothDatasetError(
            f"unknown excluded environment(s): {', '.join(unknown)}; "
            f"expected one of {sorted(VALID_ENVIRONMENTS)}"
        )
    return normalized


def _rebase_video_path(
    video: Path,
    *,
    source_data_root: Path | None,
    consumer_data_root: str | None,
) -> str:
    if source_data_root is None and consumer_data_root is None:
        return str(video)
    if source_data_root is None or consumer_data_root is None:
        raise UnslothDatasetError(
            "source and consumer data roots must be supplied together"
        )
    source_root = source_data_root.expanduser().resolve()
    try:
        relative = video.resolve().relative_to(source_root)
    except ValueError as error:
        raise UnslothDatasetError(
            f"video {video} is outside source data root {source_root}"
        ) from error
    consumer = consumer_data_root.strip()
    if re.fullmatch(r"[A-Za-z]:[\\/].*", consumer):
        return PureWindowsPath(consumer, *relative.parts).as_posix()
    if consumer.startswith("/"):
        return str(PurePosixPath(consumer, *relative.parts))
    raise UnslothDatasetError(
        "consumer data root must be an absolute POSIX path or Windows drive path"
    )


@dataclass(frozen=True)
class Window:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def _round_seconds(value: float) -> float:
    rounded = round(float(value) + 0.0, 3)
    return 0.0 if rounded == 0 else rounded


def aligned_windows(
    duration: float,
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    stride_seconds: float = DEFAULT_STRIDE_SECONDS,
) -> tuple[Window, ...]:
    """Return deterministic wall-clock windows that exhaustively cover a recording."""
    values = (duration, window_seconds, stride_seconds)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise UnslothDatasetError("duration, window seconds, and stride seconds must be positive")
    if stride_seconds > window_seconds:
        raise UnslothDatasetError("stride seconds cannot exceed window seconds")

    windows: list[Window] = []
    start = 0.0
    while start < duration:
        end = min(duration, start + window_seconds)
        windows.append(Window(start=_round_seconds(start), end=_round_seconds(end)))
        if end >= duration - 1e-9:
            break
        start += stride_seconds
    return tuple(windows)


def intervals_overlap_window(intervals: Iterable[Interval], window: Window) -> bool:
    return any(interval.start < window.end and window.start < interval.end for interval in intervals)


def _outcome(tags: Iterable[str]) -> str:
    tag_set = set(tags)
    if "ace" in tag_set and "service-fault" in tag_set:
        raise UnslothDatasetError("a rally cannot be tagged as both ace and service-fault")
    if "ace" in tag_set:
        return "ace"
    if "service-fault" in tag_set:
        return "service-fault"
    return "ordinary"


def labels_for_window(recording: Recording, window: Window) -> dict[str, Any]:
    rallies: list[dict[str, Any]] = []
    live_at_start = False
    live_at_end = False
    for interval in recording.rallies:
        if interval.start >= window.end or interval.end <= window.start:
            continue
        left_censored = interval.start < window.start
        right_censored = interval.end > window.end
        live_at_start = live_at_start or left_censored
        live_at_end = live_at_end or right_censored
        rallies.append(
            {
                "start": _round_seconds(max(interval.start, window.start) - window.start),
                "end": _round_seconds(min(interval.end, window.end) - window.start),
                "outcome": _outcome(interval.tags),
            }
        )
    return {
        "liveAtStart": live_at_start,
        "liveAtEnd": live_at_end,
        "rallies": rallies,
    }


def _raw_intervals_for_window(rows: Any, window: Window) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        start = row.get("start")
        end = row.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        if float(start) >= window.end or float(end) <= window.start:
            continue
        selected.append(
            {
                **{key: value for key, value in row.items() if key not in {"start", "end"}},
                "start": _round_seconds(max(float(start), window.start) - window.start),
                "end": _round_seconds(min(float(end), window.end) - window.start),
            }
        )
    return selected


def _rally_metadata(recording: Recording, window: Window) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for interval in recording.rallies:
        if interval.start >= window.end or interval.end <= window.start:
            continue
        selected.append(
            {
                "sourceStart": _round_seconds(interval.start),
                "sourceEnd": _round_seconds(interval.end),
                "tags": list(interval.tags),
                "leftCensored": interval.start < window.start,
                "rightCensored": interval.end > window.end,
            }
        )
    return selected


def _window_id(recording_id: str, window: Window) -> str:
    start_ms = round(window.start * 1000)
    end_ms = round(window.end * 1000)
    return f"{recording_id}__{start_ms:010d}_{end_ms:010d}"


def build_row(
    recording: Recording,
    window: Window,
    *,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    instruction: str = DEFAULT_INSTRUCTION,
    video_path: str | None = None,
) -> dict[str, Any]:
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise UnslothDatasetError("sample FPS must be positive")
    if not instruction.strip():
        raise UnslothDatasetError("instruction must not be empty")
    labels = labels_for_window(recording, window)
    max_frames = max(4, math.ceil(window.duration * sample_fps))
    assistant = json.dumps(labels, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    resolved_video_path = video_path or str(recording.video)
    video_part = {
        "type": "video",
        "video": resolved_video_path,
        "video_start": _round_seconds(window.start),
        "video_end": _round_seconds(window.end),
        "fps": float(sample_fps),
        "min_frames": 4,
        "max_frames": max_frames,
    }
    return {
        "id": _window_id(recording.id, window),
        "messages": [
            {
                "role": "user",
                "content": [
                    video_part,
                    {"type": "text", "text": instruction},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant}],
            },
        ],
        "metadata": {
            "schema": UNSLOTH_ROW_SCHEMA,
            "recordingId": recording.id,
            "split": recording.split,
            "sourceGroup": recording.source_group,
            "environment": recording.environment,
            "window": {
                "sourceStart": _round_seconds(window.start),
                "sourceEnd": _round_seconds(window.end),
                "duration": _round_seconds(window.duration),
            },
            "sourceVideo": {
                "path": resolved_video_path,
                "sha256": recording.content_sha256,
            },
            "sourceRallies": _rally_metadata(recording, window),
            "hardNegatives": _raw_intervals_for_window(
                recording.raw.get("hardNegatives", []), window
            ),
        },
    }


def build_rows(
    manifest: DatasetManifest,
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    stride_seconds: float = DEFAULT_STRIDE_SECONDS,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    instruction: str = DEFAULT_INSTRUCTION,
    excluded_environments: Iterable[str] = (),
    source_data_root: Path | None = None,
    consumer_data_root: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    excluded = set(_normalize_excluded_environments(excluded_environments))
    rows: dict[str, list[dict[str, Any]]] = {}
    for recording in manifest.recordings:
        if recording.environment in excluded:
            continue
        consumer_video = _rebase_video_path(
            recording.video,
            source_data_root=source_data_root,
            consumer_data_root=consumer_data_root,
        )
        media = probe(recording.video)
        for window in aligned_windows(
            float(media["duration"]),
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
        ):
            if intervals_overlap_window(recording.ignored_intervals, window):
                continue
            rows.setdefault(recording.split, []).append(
                build_row(
                    recording,
                    window,
                    sample_fps=sample_fps,
                    instruction=instruction,
                    video_path=consumer_video,
                )
            )
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recording_ids = {row["metadata"]["recordingId"] for row in rows}
    source_groups = {row["metadata"]["sourceGroup"] for row in rows}
    examples_with_rallies = 0
    rally_appearances = 0
    hard_negative_examples = 0
    for row in rows:
        target = json.loads(row["messages"][1]["content"][0]["text"])
        count = len(target["rallies"])
        examples_with_rallies += int(count > 0)
        rally_appearances += count
        hard_negative_examples += int(bool(row["metadata"]["hardNegatives"]))
    return {
        "examples": len(rows),
        "recordings": len(recording_ids),
        "sourceGroups": len(source_groups),
        "examplesWithRallies": examples_with_rallies,
        "emptyExamples": len(rows) - examples_with_rallies,
        "rallyAppearances": rally_appearances,
        "hardNegativeExamples": hard_negative_examples,
    }


def _readme(metadata: dict[str, Any]) -> str:
    split_lines = [
        f"| {name} | {summary['examples']} | {summary['recordings']} | "
        f"{summary['sourceGroups']} | {summary['examplesWithRallies']} | {summary['emptyExamples']} |"
        for name, summary in metadata["splits"].items()
    ]
    excluded_environments = metadata["selection"]["excludedEnvironments"]
    selection_note = (
        "This export excludes the following environments: "
        + ", ".join(f"`{value}`" for value in excluded_environments)
        + "."
        if excluded_environments
        else "This export includes every environment in the frozen manifest."
    )
    path_mapping = metadata["selection"]["videoPathMapping"]
    path_note = (
        f"Video paths were rebased from `{path_mapping['sourceDataRoot']}` to "
        f"`{path_mapping['consumerDataRoot']}` for the consumer machine."
        if path_mapping
        else "Video paths use the paths resolved on the export machine."
    )
    return "\n".join(
        [
            "# VolleyCut Unsloth rally-window dataset",
            "",
            "This is a local, video-referenced OpenAI/ChatML JSONL dataset for Unsloth's "
            "`UnslothVisionDataCollator`. Each example uses one `video` content item with "
            "`video_start`/`video_end` plus one text instruction, followed by a compact JSON answer.",
            "",
            "The source videos are not copied. The `video` fields are absolute paths to the verified "
            "960x540 proxy files. On another machine, either mount/copy the label workspace at the "
            "same path or regenerate the JSONL files there with the exporter.",
            "",
            selection_note,
            "",
            path_note,
            "",
            "This is a visual-video dataset. The video reader samples frames but does not turn the "
            "MP4 audio track into model audio input. Any audiovisual model study needs a separate, "
            "explicitly aligned audio export and processor check.",
            "",
            "## Splits",
            "",
            "| Split | Examples | Recordings | Source groups | With rallies | Empty |",
            "|---|---:|---:|---:|---:|---:|",
            *split_lines,
            "",
            "Use `train.jsonl` for fitting and `validation.jsonl` for model/decoder selection. "
            "Never concatenate `test.jsonl` into training; the test source has also been repeatedly "
            "inspected and is a regression set rather than fresh generalization evidence.",
            "",
            "## Loading",
            "",
            "```python",
            "from datasets import load_dataset",
            "",
            "dataset = load_dataset(",
            "    \"json\",",
            "    data_files={",
            "        \"train\": \"train.jsonl\",",
            "        \"validation\": \"validation.jsonl\",",
            "        \"test\": \"test.jsonl\",",
            "    },",
            ")",
            "```",
            "",
            "Pass `dataset[\"train\"]` to `SFTTrainer` with `remove_unused_columns=False`, "
            "`dataset_text_field=\"\"`, `dataset_kwargs={\"skip_prepare_dataset\": True}`, and "
            "`UnslothVisionDataCollator(model, tokenizer)`. Begin with batch size 1 on a 12 GB "
            "RTX 3080. The rows request the conservative sample rate recorded in `dataset.json`. "
            "Prefer a PyTorch-compatible TorchCodec wheel with system FFmpeg; when multiple video "
            "backends are present, set `FORCE_UNSLOTH_VIDEO_READER=torchcodec` explicitly.",
            "",
            "## Label contract and caveats",
            "",
            f"The annotation policy is `{ANNOTATION_POLICY_ID}`. Timestamps in answers are relative "
            "to each window and use `[start,end)`. `liveAtStart` and `liveAtEnd` identify censored "
            "rallies crossing a window boundary. Empty windows are intentional dead-time negatives.",
            "",
            "Gold labels were continuously human-reviewed, but AI-prelabel provenance remains on "
            "many corrected rally rows. This does not mean those boundaries are uncorrected; it does "
            "mean same-family teacher/student claims need a newly blank-slate-labeled source group.",
            "",
            "`checksums.sha256` covers the dataset metadata and split JSONL files. Source video "
            "hashes are recorded per example in `metadata.sourceVideo.sha256`.",
            "",
        ]
    )


def export_dataset(
    manifest: DatasetManifest,
    output: str | Path,
    *,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    stride_seconds: float = DEFAULT_STRIDE_SECONDS,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    instruction: str = DEFAULT_INSTRUCTION,
    excluded_environments: Iterable[str] = (),
    source_data_root: Path | None = None,
    consumer_data_root: str | None = None,
) -> dict[str, Any]:
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise UnslothDatasetError(f"refusing to overwrite existing dataset: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f".{output_path.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise UnslothDatasetError(f"temporary export path already exists: {temporary}")
    temporary.mkdir()
    try:
        normalized_excluded_environments = _normalize_excluded_environments(
            excluded_environments
        )
        rows_by_split = build_rows(
            manifest,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            sample_fps=sample_fps,
            instruction=instruction,
            excluded_environments=normalized_excluded_environments,
            source_data_root=source_data_root,
            consumer_data_root=consumer_data_root,
        )
        if "train" not in rows_by_split or "validation" not in rows_by_split:
            raise UnslothDatasetError("dataset must contain train and validation examples")
        source_groups: dict[str, str] = {}
        for split, rows in rows_by_split.items():
            for row in rows:
                group = row["metadata"]["sourceGroup"]
                previous = source_groups.setdefault(group, split)
                if previous != split:
                    raise UnslothDatasetError(
                        f"source group {group!r} crosses {previous!r} and {split!r}"
                    )
            split_path = temporary / f"{split}.jsonl"
            with split_path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

        manifest_digest = _sha256_file(manifest.path)
        metadata = {
            "schemaVersion": UNSLOTH_DATASET_SCHEMA_VERSION,
            "name": output_path.name,
            "format": "unsloth-openai-video-messages",
            "rowSchema": UNSLOTH_ROW_SCHEMA,
            "annotationPolicy": ANNOTATION_POLICY_ID,
            "selection": {
                "excludedEnvironments": list(normalized_excluded_environments),
                "videoPathMapping": (
                    {
                        "sourceDataRoot": str(source_data_root.expanduser().resolve()),
                        "consumerDataRoot": consumer_data_root,
                    }
                    if source_data_root is not None and consumer_data_root is not None
                    else None
                ),
            },
            "source": {
                "manifest": str(manifest.path),
                "manifestSha256": manifest_digest,
                "manifestName": manifest.name,
            },
            "windowing": {
                "kind": "exhaustive-wall-clock",
                "windowSeconds": float(window_seconds),
                "strideSeconds": float(stride_seconds),
                "overlapSeconds": float(window_seconds - stride_seconds),
                "sampleFps": float(sample_fps),
                "goldCentered": False,
                "excludedIgnoredWindows": True,
            },
            "instruction": instruction,
            "splits": {
                split: _split_summary(rows)
                for split, rows in sorted(rows_by_split.items())
            },
            "warnings": [
                "Do not train on validation or test rows.",
                "The current test source is a repeatedly inspected regression set, not fresh generalization evidence.",
                "Many human-reviewed rally rows retain AI-prelabel provenance; use a new blank-slate source group for an independent same-family claim.",
                "The proxy is 960x540 and this task should not be represented as ball tracking.",
            ],
        }
        metadata_path = temporary / "dataset.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(_readme(metadata), encoding="utf-8")

        checksum_targets = sorted(
            path for path in temporary.iterdir() if path.name != "checksums.sha256"
        )
        checksum_text = "".join(
            f"{_sha256_file(path)}  {path.name}\n" for path in checksum_targets
        )
        (temporary / "checksums.sha256").write_text(checksum_text, encoding="utf-8")
        temporary.replace(output_path)
    except BaseException:
        if temporary.exists():
            for path in temporary.iterdir():
                path.unlink()
            temporary.rmdir()
        raise
    return metadata

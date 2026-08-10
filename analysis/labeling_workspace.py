from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .annotations import create_label_draft
from .media import normalize_video
from .schema import ENVIRONMENTS, SPLITS, ManifestError, _read_game, _read_roi


Progress = Callable[[str], None]


def _resolve_source(value: Any, plan_file: Path, where: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{where}.video must be a non-empty path string")
    source = Path(value).expanduser()
    if not source.is_absolute():
        source = (plan_file.parent / source).resolve()
    else:
        source = source.resolve()
    if not source.is_file():
        raise ManifestError(f"{where}.video does not exist: {source}")
    return source


def prepare_labeling_workspace(
    plan_path: str | Path,
    workspace_path: str | Path,
    *,
    fps: float = 30.0,
    max_width: int = 960,
    crf: int = 24,
    preset: str = "ultrafast",
    threads: int = 1,
    progress: Progress | None = None,
) -> dict[str, Any]:
    plan_file = Path(plan_path).expanduser().resolve()
    workspace = Path(workspace_path).expanduser().resolve()
    try:
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read labeling plan {plan_file}: {error}") from error
    if not isinstance(plan, dict) or plan.get("schemaVersion") != 1:
        raise ManifestError("labeling plan schemaVersion must be 1")
    rows = plan.get("recordings")
    if not isinstance(rows, list) or not rows:
        raise ManifestError("labeling plan recordings must be a non-empty array")
    # Resolve every source before the first expensive transcode. A bad later row
    # must not waste hours of preparation before it is discovered.
    sources = []
    for index, row in enumerate(rows):
        where = f"recordings[{index}]"
        if not isinstance(row, dict):
            raise ManifestError(f"{where} must be an object")
        sources.append(_resolve_source(row.get("video"), plan_file, where))

    report: dict[str, Any] = {
        "workspace": str(workspace),
        "recordings": len(rows),
        "proxiesCreated": 0,
        "proxiesReused": 0,
        "tasksCreated": 0,
        "tasksReused": 0,
        "items": [],
    }
    seen_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    for index, row in enumerate(rows):
        where = f"recordings[{index}]"
        if not isinstance(row, dict):
            raise ManifestError(f"{where} must be an object")
        recording_id = row.get("id")
        source_group = row.get("sourceGroup")
        split = row.get("split")
        environment = row.get("environment")
        if not isinstance(recording_id, str) or not recording_id.strip():
            raise ManifestError(f"{where}.id must be a non-empty string")
        if recording_id in seen_ids:
            raise ManifestError(f"duplicate recording id: {recording_id}")
        seen_ids.add(recording_id)
        if not isinstance(source_group, str) or not source_group.strip():
            raise ManifestError(f"{where}.sourceGroup must be a non-empty string")
        if split not in SPLITS:
            raise ManifestError(f"{where}.split must be one of {sorted(SPLITS)}")
        old_split = group_splits.setdefault(source_group, split)
        if old_split != split:
            raise ManifestError(
                f"sourceGroup {source_group!r} crosses {old_split!r} and {split!r} splits"
            )
        if environment not in ENVIRONMENTS:
            raise ManifestError(f"{where}.environment must be one of {sorted(ENVIRONMENTS)}")
        source = sources[index]
        game = _read_game(row.get("game"), where)
        roi = _read_roi(row.get("roi"), where)
        capture = row.get("capture", {})
        if not isinstance(capture, dict):
            raise ManifestError(f"{where}.capture must be an object")

        proxy_name = (
            f"{recording_id}.mp4"
            if recording_id.endswith("-full")
            else f"{recording_id}-full.mp4"
        )
        proxy = workspace / "proxies" / environment / proxy_name
        provenance = proxy.with_suffix(proxy.suffix + ".provenance.json")
        task = workspace / "tasks" / "full" / f"{recording_id}.labels.json"
        if proxy.exists() != provenance.exists():
            raise ManifestError(
                f"incomplete proxy state for {recording_id}: video and provenance must both exist or neither"
            )
        if proxy.is_file() and provenance.is_file():
            report["proxiesReused"] += 1
            if progress:
                progress(f"[{index + 1}/{len(rows)}] reuse proxy {proxy.name}")
        else:
            if progress:
                progress(f"[{index + 1}/{len(rows)}] normalize {source.name} -> {proxy.name}")
            normalize_video(
                source,
                proxy,
                fps=fps,
                max_width=max_width,
                crf=crf,
                preset=preset,
                threads=threads,
            )
            report["proxiesCreated"] += 1

        if task.is_file():
            report["tasksReused"] += 1
        else:
            create_label_draft(
                proxy,
                task,
                recording_id=recording_id,
                source_group=source_group,
                split=split,
                environment=environment,
                players_per_team=game.get("playersPerTeam"),
                target_points=game.get("targetPoints"),
                format_name=game.get("format"),
                roi=roi,
                capture=capture,
            )
            report["tasksCreated"] += 1
        report["items"].append(
            {
                "id": recording_id,
                "proxy": str(proxy),
                "task": str(task),
            }
        )
    return report

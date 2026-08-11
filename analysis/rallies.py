from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionSettings:
    high_threshold: float = 0.46
    low_threshold: float = 0.25
    max_gap_seconds: float = 0.75
    min_rally_seconds: float = 3.0
    onset_lead_seconds: float = 0.75
    ending_tail_seconds: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "highThreshold": self.high_threshold,
            "lowThreshold": self.low_threshold,
            "maxGapSeconds": self.max_gap_seconds,
            "minRallySeconds": self.min_rally_seconds,
            "onsetLeadSeconds": self.onset_lead_seconds,
            "endingTailSeconds": self.ending_tail_seconds,
        }


def detect_rallies(
    signals: list[dict[str, float]],
    duration: float,
    camera_stability: float,
    settings: DetectionSettings = DetectionSettings(),
) -> list[dict[str, object]]:
    if not signals or duration <= 0:
        return []
    times = np.asarray([sample["time"] for sample in signals], dtype=np.float64)
    activity = np.asarray([sample["activity"] for sample in signals], dtype=np.float64)
    motion = np.asarray([sample["motion"] for sample in signals], dtype=np.float64)
    audio = np.asarray([sample["audio"] for sample in signals], dtype=np.float64)
    if times.size < 2:
        return []
    period = max(0.05, float(np.median(np.diff(times))))
    active = activity >= settings.low_threshold
    seeded = activity >= settings.high_threshold
    gap_samples = max(0, round(settings.max_gap_seconds / period))

    if gap_samples:
        false_indexes = np.flatnonzero(~active)
        for index in false_indexes:
            left = max(0, index - gap_samples)
            right = min(active.size, index + gap_samples + 1)
            if np.any(active[left:index]) and np.any(active[index + 1:right]):
                before = index - np.flatnonzero(active[:index])[-1] if np.any(active[:index]) else gap_samples + 1
                after_matches = np.flatnonzero(active[index + 1:])
                after = int(after_matches[0] + 1) if after_matches.size else gap_samples + 1
                if before + after - 1 <= gap_samples:
                    active[index] = True

    boundaries = np.diff(np.pad(active.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(boundaries == 1)
    ends = np.flatnonzero(boundaries == -1)
    rallies: list[dict[str, object]] = []
    for start_index, end_index in zip(starts, ends, strict=True):
        if end_index <= start_index or not np.any(seeded[start_index:end_index]):
            continue
        start = max(0.0, float(times[start_index]) - settings.onset_lead_seconds)
        end_sample = min(times.size - 1, end_index - 1)
        end = min(duration, float(times[end_sample]) + period + settings.ending_tail_seconds)
        if end - start < settings.min_rally_seconds:
            continue
        segment = slice(start_index, end_index)
        peak_activity = float(np.max(activity[segment]))
        density = float(np.mean(activity[segment] >= settings.high_threshold))
        duration_score = 1.0 - min(1.0, abs((end - start) - 14) / 45)
        confidence = np.clip(
            0.28 + peak_activity * 0.28 + density * 0.18 + duration_score * 0.12 + camera_stability * 0.14,
            0.05,
            0.9,
        )
        rallies.append({
            "id": f"R{len(rallies) + 1:02d}",
            "start": round(start, 3),
            "end": round(end, 3),
            "confidence": round(float(confidence), 3),
            "included": True,
            "evidence": {
                "motionPeak": round(float(np.max(motion[segment])), 3),
                "audioPeak": round(float(np.max(audio[segment])), 3),
            },
        })
    return rallies

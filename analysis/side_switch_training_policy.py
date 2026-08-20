"""Frozen dataset policy for future on-device side-switch iterations."""

from __future__ import annotations


SIDE_SWITCH_CADENCE_POINTS = 7
SIDE_SWITCH_RECORDING_START_POINT = 0
SIDE_SWITCH_ONE_SET_PER_RECORDING = True
SIDE_SWITCH_DEFAULT_RALLY_MARGIN = 2
SIDE_SWITCH_FIT_EXCLUSIONS = {
    "beach-source-02": (
        "image becomes blurry roughly halfway through the recording; exclude from "
        "all new side-switch fitting"
    ),
}


class SideSwitchTrainingPolicyError(ValueError):
    pass


def expected_switch_point_totals(completed_points: int) -> tuple[int, ...]:
    """Return deterministic switch opportunities for one set starting at point zero."""

    if isinstance(completed_points, bool) or not isinstance(completed_points, int):
        raise SideSwitchTrainingPolicyError("completed points must be an integer")
    if completed_points < SIDE_SWITCH_RECORDING_START_POINT:
        raise SideSwitchTrainingPolicyError("completed points cannot be negative")
    return tuple(
        range(
            SIDE_SWITCH_CADENCE_POINTS,
            completed_points + 1,
            SIDE_SWITCH_CADENCE_POINTS,
        )
    )


def expected_switch_gap_windows(
    completed_rallies: int,
    *,
    rally_margin: int = SIDE_SWITCH_DEFAULT_RALLY_MARGIN,
) -> tuple[tuple[int, int], ...]:
    """Project score cadence to inclusive rally-gap candidate windows.

    Rally count is only a proxy for point count: re-dos do not score, rally markers can
    be missed, and players can switch early or late. These windows generate candidates;
    they do not assert that a switch occurred.
    """

    if isinstance(completed_rallies, bool) or not isinstance(completed_rallies, int):
        raise SideSwitchTrainingPolicyError("completed rallies must be an integer")
    if completed_rallies < 0:
        raise SideSwitchTrainingPolicyError("completed rallies cannot be negative")
    if isinstance(rally_margin, bool) or not isinstance(rally_margin, int):
        raise SideSwitchTrainingPolicyError("rally margin must be an integer")
    if rally_margin < 0:
        raise SideSwitchTrainingPolicyError("rally margin cannot be negative")

    return tuple(
        (
            max(1, point_total - rally_margin),
            min(completed_rallies, point_total + rally_margin),
        )
        for point_total in expected_switch_point_totals(
            completed_rallies + rally_margin
        )
    )


def validate_side_switch_fit_recordings(recording_ids: list[str] | tuple[str, ...]) -> None:
    """Reject quality-excluded recordings before any new side-switch fit."""

    blocked = sorted(set(recording_ids) & set(SIDE_SWITCH_FIT_EXCLUSIONS))
    if blocked:
        reasons = "; ".join(
            f"{recording_id}: {SIDE_SWITCH_FIT_EXCLUSIONS[recording_id]}"
            for recording_id in blocked
        )
        raise SideSwitchTrainingPolicyError(
            f"side-switch fit contains excluded recordings: {reasons}"
        )

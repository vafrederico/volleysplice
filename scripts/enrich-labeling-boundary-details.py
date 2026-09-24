#!/usr/bin/env python3
"""Attach traceable review explanations without altering predictions or selection."""
from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import json
from pathlib import Path


def close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) < 1e-8


def unique_by_id(rows: list[dict]) -> dict[str, dict]:
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate lineage ID")
    return result


def nearest_index(times: list[float], time: float) -> int:
    if not times:
        raise ValueError("Native signal samples are required")
    position = bisect.bisect_left(times, time)
    if position == 0:
        return 0
    if position == len(times):
        return position - 1
    return position - 1 if time - times[position - 1] <= times[position] - time else position


def enrich(manifest: dict, adviser: dict) -> dict:
    result = copy.deepcopy(manifest)
    record = adviser["record"]
    recordings = [row for row in result["recordings"] if row["recordingId"] == record["id"]]
    if len(recordings) != 1 or not close(recordings[0]["durationSeconds"], record["durationSeconds"]):
        raise ValueError("Adviser recording does not match reference manifest")
    parents = unique_by_id(record["productionEvents"])
    candidates = unique_by_id(adviser["plan"]["eventCandidates"])
    proposals = unique_by_id(adviser["plan"]["proposals"])
    jobs = {job["parentId"]: job for job in adviser["jobs"]}
    if len(jobs) != len(adviser["jobs"]):
        raise ValueError("Duplicate review parent")
    selected = set(adviser["queue"]["selectedParentIds"])
    enriched_count = 0
    for reference in recordings[0]["references"]:
        if "research" not in reference:
            continue
        research = reference["research"]
        flags = unique_by_id(research["boundaryFlags"])
        if set(flags) != set(proposals):
            raise ValueError("Boundary flag identities changed from adviser proposals")
        regions = {region["parentId"]: region for region in research["reviewRegions"]}
        if len(regions) != len(research["reviewRegions"]) or set(regions) != set(jobs):
            raise ValueError("Review regions changed from adviser jobs")
        if {key for key, region in regions.items() if region["recommended"]} != selected:
            raise ValueError("Recommended parent selection changed")
        if research["queue"]["selectedParentCount"] != len(selected):
            raise ValueError("Selected parent count changed")
        signals = research["signals"]
        for region in regions.values():
            parent = parents[region["parentId"]]
            if not close(region["start"], parent["start"]) or not close(region["end"], parent["end"]):
                raise ValueError("Review region is not the original production parent")
        for flag in research["boundaryFlags"]:
            proposal = proposals[flag["id"]]
            candidate = candidates[proposal["candidateId"]]
            parent = parents[flag["parentId"]]
            if proposal["parentId"] != flag["parentId"] or candidate["parentId"] != flag["parentId"] or proposal["kind"] != flag["kind"] or not close(proposal["time"], flag["time"]):
                raise ValueError("Boundary flag lineage does not match adviser")
            if flag["id"] not in jobs[flag["parentId"]]["splitIds"]:
                raise ValueError("Boundary flag is not in its original review job")
            siblings = sorted((row for row in candidates.values() if row["parentId"] == flag["parentId"] and row["componentId"] == candidate["componentId"]), key=lambda row: (row["start"], row["end"], row["id"]))
            position = next(index for index, row in enumerate(siblings) if row["id"] == candidate["id"])
            is_end = flag["kind"] == "end"
            previous_time = None
            comparison = "new_boundary"
            if flag["kind"] == "initial_start" and position == 0 and close(candidate["componentStart"], parent["start"]):
                previous_time = parent["start"]
                comparison = "original_parent_start"
            elif is_end and position == len(siblings) - 1 and close(candidate["componentEnd"], parent["end"]):
                previous_time = parent["end"]
                comparison = "original_parent_end"
            native_index = nearest_index(signals["times"], flag["time"])
            native_sample = {"index": native_index, "time": signals["times"][native_index]}
            native_sample.update({head: signals[head][native_index] for head in ("live", "serve", "end", "keep")})
            original_time = candidate["originalNeuralEnd" if is_end else "originalNeuralStart"]
            instruction = (
                "Check the serve contact for the existing first rally. Confirm this is a correction of its start, not a missed later rally."
                if flag["kind"] == "initial_start" else
                "Check that a distinct new serve and rally occur here. Confirm the previous rally ended and preserve the gap between the two rallies."
                if flag["kind"] == "additional_start" else
                "Check when this proposed rally becomes dead. Keep the following dead-time gap separate from any later rally; do not treat the new start as this rally's end."
                if comparison == "new_boundary" else
                "Check when the final rally becomes dead and whether the original production end includes extra time or misses play."
            )
            details = {"candidateId": candidate["id"], "componentId": candidate["componentId"],
                "candidateStart": candidate["start"], "candidateEnd": candidate["end"],
                "source": proposal["source"], "observed": proposal["observed"],
                "comparison": comparison, "previousTime": previous_time,
                "shiftSeconds": None if previous_time is None else flag["time"] - previous_time,
                "originalNeuralTime": original_time, "headShiftSeconds": flag["time"] - original_time,
                "sample": native_sample, "reviewInstruction": instruction}
            if position > 0:
                details["previousProposedEnd"] = siblings[position - 1]["end"]
            if position + 1 < len(siblings):
                details["nextProposedStart"] = siblings[position + 1]["start"]
            flag["details"] = details
            enriched_count += 1
    if not enriched_count:
        raise ValueError("No research flags were enriched")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--adviser", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    references_bytes = args.references.read_bytes()
    adviser_bytes = args.adviser.read_bytes()
    result = enrich(json.loads(references_bytes), json.loads(adviser_bytes))
    result["boundaryDetailsProvenance"] = {
        "schemaVersion": 1, "generator": Path(__file__).name,
        "referencesSha256": hashlib.sha256(references_bytes).hexdigest(),
        "adviserSha256": hashlib.sha256(adviser_bytes).hexdigest(),
        "note": "Explanations only: source model predictions, export cores, signals, flags and selected review parents preserved.",
    }
    # Refuse to overwrite any existing source or derived artifact.
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()

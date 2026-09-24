"""Approximate rally supervision from individual reviewed export cores.

This is an explicit experimental label contract, not promotion to semantic gold.
Ignored spans mask targets; they never create a new serve/end boundary.
"""
from __future__ import annotations
from analysis.private_ledger import private_value

from .source_exposure_inventory import subtract_ranges

ALLOWED_GROUPS = frozenset({private_value('source-group-006'), private_value('source-group-004')})


def derive_proxy_record(record: dict, feedback: dict, minimum_seconds: float = .25) -> dict:
    if record["sourceGroup"] not in ALLOWED_GROUPS or record.get("protected"):
        raise ValueError("proxy scope must exclude protected and reserved September17 sources")
    if not record["coverageReview"]["exhaustiveKeptDiscardedReviewConfirmed"]:
        raise ValueError("coverage review not confirmed")
    active = {rid for item in feedback["finalExportIntervals"] for rid in item.get("cutIds", [])}
    if not active:
        raise ValueError("individual final-export cut identity is required; joined export ranges cannot create rallies")
    ignored = record["ignoredIntervals"]
    window = record.get("gameWindow") or {"start": 0., "end": record["durationSeconds"]}
    ignored = list(ignored) + [{"start": 0., "end": window["start"], "reason": "outside-game-window"},
                               {"start": window["end"], "end": record["durationSeconds"], "reason": "outside-game-window"}]
    ignored = [r for r in ignored if r["end"] > r["start"]]
    rallies, excluded, audit = [], [], []
    seen_ids, seen_ranges = {}, {}
    for raw in feedback["corrections"]["correctedRanges"]:
        cut_id, start, end = raw["id"], float(raw["coreStart"]), float(raw["coreEnd"])
        identity = {"id": cut_id, "start": start, "end": end, "durationSeconds": end-start,
                    "sourceOrigin": raw.get("origin"), "sourceFields": "corrections.correctedRanges coreStart/coreEnd"}
        if not 0 <= start < end <= record["durationSeconds"] + .01:
            raise ValueError("invalid saved individual core")
        if cut_id in seen_ids and seen_ids[cut_id] != (start, end):
            raise ValueError("one core identity has conflicting endpoints")
        duplicate = cut_id in seen_ids or (start, end) in seen_ranges
        seen_ids[cut_id] = (start, end)
        suppression = [row for row in feedback.get("finalExportProvenance", [])
                       if str(row.get("kind", "")).startswith("suppression-") and cut_id in row.get("cutIds", [])]
        masks = ignored + suppression
        visible = subtract_ranges(start, end, masks)
        reason = ("not-included" if not raw.get("included") else "absent-from-final-export-membership" if cut_id not in active else
                  "duplicate-core-identity-or-range" if duplicate else "micro-range-under-0.25s" if end-start < minimum_seconds else
                  "fully-outside-valid-review-scope" if not visible else None)
        if reason:
            excluded.append({**identity, "reason": reason,
                             "duplicateOf": seen_ranges.get((start, end)) if duplicate else None})
            continue
        seen_ranges[(start, end)] = cut_id
        if suppression:
            # Suppression may remove only part of a core. Treat that removed part
            # as an explicit unknown area for approximate boundary supervision.
            ignored.extend(suppression)
        rallies.append({"id": cut_id, "start": start, "end": end,
                        "tags": ["reviewed-export-rally-proxy", "approximate-endpoints"],
                        "sourceOrigin": raw.get("origin"), "sourceCutId": cut_id})
        audit.append({**identity, "validCoveragePieces": [{"start": a, "end": b} for a, b in visible],
                      "validSeconds": sum(b-a for a, b in visible),
                      "endpointsChanged": False,
                      "startInsideIgnored": any(r["start"] <= start < r["end"] for r in masks),
                      "endInsideIgnored": any(r["start"] < end <= r["end"] for r in masks)})
    rallies.sort(key=lambda r: (r["start"], r["end"], r["id"]))
    if any(a["end"] > b["start"] for a, b in zip(rallies, rallies[1:])):
        raise ValueError("overlapping saved individual cores require explicit resolution")
    return {"id": record["id"], "sourceGroup": record["sourceGroup"], "environment": record["environment"],
            "video": record["video"], "contentSha256": record.get("contentSha256"), "roi": record.get("roi"),
            "durationSeconds": record["durationSeconds"], "protected": False,
            "labelTier": "reviewed-export-rally-proxy", "independentSemanticGold": False,
            "rallies": rallies, "ignoredIntervals": ignored, "gameWindow": window,
            "originalKeepTargets": record["keepTargets"],
            "annotation": {"continuousVideoReviewed": True, "reviewScope": "all kept/discarded export coverage manually reviewed",
                           "independentEndpointGold": False, "approximateRallySupervisionAuthorized": True},
            "sourceFeedback": {k: record["feedback"][k] for k in ("path", "sha256", "sizeBytes")},
            "derivation": {"minimumIndividualCoreSeconds": minimum_seconds, "joinedExportBoundariesUsed": False,
                           "endpointsChanged": False, "sourceCoreCount": len(feedback["corrections"]["correctedRanges"]),
                           "retained": audit, "excluded": excluded},
            "experimentalUse": "source-group-held train/calibration arms only; final accuracy remains evaluated by original independent tier"}

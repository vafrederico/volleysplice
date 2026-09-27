#!/usr/bin/env python3
"""Independent read-only bindings, original-label reconciliation and proxy checks."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.private_ledger import private_value
from analysis.source_exposure_inventory import identity, read, sha256

parser = argparse.ArgumentParser()
parser.add_argument("--inventory", type=Path, required=True)
parser.add_argument("--proxy", type=Path, required=True)
parser.add_argument("--original", type=Path, default=Path(private_value('private-reference-0095')))
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
inventory, proxies, original = read(args.inventory), read(args.proxy), read(args.original)
assert proxies["inventory"]["sha256"] == sha256(args.inventory)
checked = {}


def check_bindings(value):
    if isinstance(value, dict):
        if value.get("path") and value.get("sha256"):
            path = Path(value["path"])
            if str(path) not in checked:
                checked[str(path)] = sha256(path)
            assert checked[str(path)] == value["sha256"], f"changed dependency: {path}"
        for name, child in value.items():
            # Catalog entries are historical pointers, sometimes superseded by
            # a completed label. The pinned catalog itself binds those facts;
            # labelSource and labelRevisions bind the current authoritative files.
            if name != "catalogLabel":
                check_bindings(child)
    elif isinstance(value, list):
        for child in value:
            check_bindings(child)


check_bindings(inventory)
check_bindings(proxies)
records = {r["id"]: r for r in inventory["records"]}
assert len(records) == 44 and sum(r["environment"] != "beach" for r in records.values()) == 42
reconciliation = []


def pairs(rows):
    return [(r["start"], r["end"]) for r in rows]


for tier, key in (("exact", "exactRows"), ("draft", "draftRows"), ("coverage", "coverageRows")):
    for index, old in enumerate(original[key]):
        new = records[old["id"]]
        shared = {k: old.get(k) == new.get(k) for k in ("video", "sourceGroup", "contentSha256", "roi", "durationSeconds")}
        assert all(shared.values()), (old["id"], shared)
        result = {"id": old["id"], "tier": tier, "originalRow": {"field": key, "index": index}, "mediaIdentityUnchanged": shared}
        if tier in {"exact", "draft"}:
            result.update(rallyEndpointsUnchanged=pairs(old["rallies"]) == pairs(new["rallies"]),
                          ignoredEndpointsUnchanged=pairs(old.get("ignoredIntervals", [])) == pairs(new["ignoredIntervals"]),
                          labelSourceShaUnchanged=old["labelSource"]["sha256"] == new["labelSource"]["sha256"])
            assert result["rallyEndpointsUnchanged"] and result["ignoredEndpointsUnchanged"] and result["labelSourceShaUnchanged"]
        else:
            result.update(literalKeepTargetsUnchanged=pairs(old["keepTargets"]) == pairs(new["keepTargets"]),
                          oldRetainedCoreCount=len(old["retainedCoverage"]), currentCoverageCoreCount=len(new["rallies"]),
                          ignoredEndpointsUnchanged=pairs(old["ignoredIntervals"]) == pairs(new["ignoredIntervals"]))
            assert result["literalKeepTargetsUnchanged"]
            if not result["ignoredEndpointsUnchanged"]:
                assert old["id"] == private_value('recording-037')
                assert old["ignoredIntervals"] == []
                assert pairs(new["ignoredIntervals"]) == [(943.735, 943.7352)]
                result["difference"] = "inventory derives source.gameWindow tail0.0002s; frozen training row stays unchanged"
        reconciliation.append(result)

proxy_audit = []
for row in proxies["records"]:
    assert not row["protected"] and row["sourceGroup"] in {private_value('source-group-004'), private_value('source-group-006')}
    feedback = read(Path(row["sourceFeedback"]["path"]))
    cuts = {cut["id"]: cut for cut in feedback["corrections"]["correctedRanges"]}
    active = {rid for interval in feedback["finalExportIntervals"] for rid in interval["cutIds"]}
    for rally in row["rallies"]:
        source = cuts[rally["id"]]
        assert source["included"] and rally["id"] in active
        assert (rally["start"], rally["end"]) == (source["coreStart"], source["coreEnd"])
        assert rally["end"] - rally["start"] >= .25
    assert len({(r["start"], r["end"]) for r in row["rallies"]}) == len(row["rallies"])
    proxy_audit.append({"id": row["id"], "individualSavedEndpointsVerified": len(row["rallies"]),
                        "excludedMicroIds": [r["id"] for r in row["derivation"]["excluded"] if r["reason"] == "micro-range-under-0.25s"]})
assert sum(r["individualSavedEndpointsVerified"] for r in proxy_audit) == 674
for row in records.values():
    if row["protected"]:
        assert row["consent"]["train"] is False and row["inferenceOnly"] is True
    if row["id"].startswith(private_value('private-reference-0094')):
        assert row["sourceGroup"] == private_value('source-group-006')
        assert row["environment"] == "grass"
        assert row["productionExposure"]["rallyPipeline"]["trainingOrRelated"] is True
assert records[private_value('indoor-source-05')]["productionExposure"]["rallyPipeline"]["sameGroupCalibrationHeads"] == ["production-suppression"]

# Alias supplement binds original short excerpts and pilot90 clips to the full
# source. These files remain metadata only, not additional training examples.
data_root = args.inventory.parents[3]
alias_supplement = []
for path in (data_root / "v0-2026-08-09/manifests/real-v0.json", data_root / "labeling-v1-2026-08-09/manifests/pilot-gold-v1.json"):
    manifest = read(path)
    for row in manifest["recordings"]:
        youtube_id = row.get("sourceExcerpt", {}).get("rawVideoId")
        canonical = next((r for r in records.values() if (youtube_id and f"-{youtube_id}-full" in r["id"]) or r["id"] == row["id"].replace("-pilot90", "-full")), None)
        assert canonical is not None and canonical["sourceGroup"] == row["sourceGroup"]
        alias_supplement.append({"alias": row["id"], "sourceId": canonical["id"], "sourceGroup": canonical["sourceGroup"],
                                 "protected": canonical["protected"], "sourceExcerpt": row.get("sourceExcerpt"), "manifest": identity(path)})
result = {"kind": "source-exposure-inventory-audit-v1", "createdAt": datetime.now(timezone.utc).isoformat(), "passed": True,
          "inventory": identity(args.inventory), "proxyManifest": identity(args.proxy), "originalManifest": identity(args.original),
          "auditCode": identity(Path(__file__).resolve()), "dependencyCount": len(checked), "dependencyHashes": checked,
          "original18Reconciliation": reconciliation, "allOriginalExactDraftLabelHashesUnchanged": True,
          "allOriginalCoverageKeepTargetsUnchanged": True, "preserveOriginal18TrainingRowsByteForByte": True,
          "sourceAliasSupplement": alias_supplement, "proxyEndpointAudit": proxy_audit,
          "noNewPredictionsOrMetrics": True, "limitations": ["Full video bytes are not rehashed by this audit; feature preparation must bind actual source bytes.",
              "Artifact/source identity checks and saved-coordinate equality do not establish independent endpoint accuracy.",
              "Exposure classification is supported by pinned production metadata; historical diagnostic evaluation is not claimed absent."]}
with args.output.open("x", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2, allow_nan=False); handle.write("\n")
print(json.dumps({"audit": identity(args.output), "passed": True, "dependencyCount": len(checked), "originalRows": len(reconciliation)}, indent=2))

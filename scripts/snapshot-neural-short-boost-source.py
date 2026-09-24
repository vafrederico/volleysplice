#!/usr/bin/env python3
"""Archive completed transfer research sources and small immutable provenance."""
from __future__ import annotations
from analysis.private_ledger import private_value

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
ROOT = Path(private_value('private-reference-0084'))
spec = importlib.util.spec_from_file_location("old_source_archive", REPO/"scripts/snapshot-neural-event-balanced-source.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
prior = old.prior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()
    root, study = args.root, args.root/"study"
    registration = old.load(study/"preregistration.json")
    contract = registration["contract"]
    old.require(old.contract_hash(contract) == registration["sha256"]
                and contract["experiment"] == "bounded-short-boost-dino-transfer-v1"
                and len(contract["code"]) == 16, "invalid transfer contract")
    report = old.load(study/"report.json")
    expected = {(c, k, a, s) for c in contract["cohorts"] for k in contract["kinds"]
                for a in contract["lossArms"] for s in contract["seeds"]}
    actual = {(r["cohort"], r["kind"], r["lossArm"], r["seed"]) for r in report["results"]}
    old.require(actual == expected and len(report["results"]) == 54
                and report["contractSha256"] == registration["sha256"]
                and report["status"] == "completed-short-boost-transfer-development"
                and not report["protectedTestOpened"] and not report["productionPromotionAllowed"], "incomplete transfer report")
    historical = old.verify_prior_archive(contract)
    preflight = old.load(Path(contract["preflight"]["path"]))
    old.file_identity(Path(contract["preflight"]["path"]), contract["preflight"]["sha256"])
    old.require(preflight["passed"] and preflight["code"] == contract["code"], "preflight changed")
    old.file_identity(Path(contract["dinoManifest"]["path"]), contract["dinoManifest"]["sha256"])
    prefix = prior.git_prefix()
    git_head = prior.git(prefix, "rev-parse", "HEAD").decode().strip()
    tracked = set(prior.git(prefix, "ls-files", "-z").decode().split("\x00")) - {""}
    untracked = set(prior.git(prefix, "ls-files", "--others", "--exclude-standard", "-z").decode().split("\x00")) - {""}
    changed = set(prior.git(prefix, "diff", "--name-only", "-z", "HEAD", "--").decode().split("\x00")) - {""}
    registered = {"analysis/"+name: sha for name, sha in contract["code"].items()}
    candidates = {p for p in tracked | untracked if prior.research_allowed(p)} | set(registered)
    candidates.add("analysis/tests/test_transfer_temporal_model.py")
    candidates |= {p for p in tracked if p.startswith("analysis/") and p.endswith(".py")}
    sources, origins, rows = {}, {}, []

    def add(path, allowed_root, member, expected_sha=None):
        old.require(member not in sources, "duplicate archive path")
        data = prior.safe_read(path, allowed_root)
        digest = old.digest(data)
        old.require(expected_sha is None or digest == expected_sha, f"changed source: {path}")
        sources[member], origins[member] = data, path
        rows.append({"archivePath": member, "sourcePath": str(path), "sha256": digest, "sizeBytes": len(data)})

    for name in sorted(candidates):
        add(REPO/name, REPO, "repository/"+name, registered.get(name))
    add(study/"preregistration.json", root, "provenance/preregistration.json")
    for name in ("dino-manifest.json", "extraction-plan.json", "dataset-audit-pts-v1.json",
                 "manifest-pts-v1.json", "pts-repair-plan-v1.json", "protocol-initial.md",
                 "protocol-initial.json", "protocol-amendment-pts-v1.md", "protocol-amendment-pts-v1.json",
                 "pts-preparation-completion-v1.json"):
        add(root/name, root, "provenance/"+name)
    for path in sorted(root.glob("*supplement*.json")):
        add(path, root, "provenance/"+path.name)
    for leaf in sorted(root.glob("cache-precompute-*")):
        old.require(leaf.is_dir() and not leaf.is_symlink(), "unexpected precompute provenance leaf")
        for path in sorted(leaf.rglob("*")):
            if path.is_file():
                old.require(path.suffix in (".json", ".py", ".log", ".txt"), "unexpected precompute artifact")
                add(path, root, "provenance/"+path.relative_to(root).as_posix())
    for directory in ("extraction-sources", "dino-metadata", "record-audits",
                      "pts-extraction-sources", "pts-source-freeze-v1"):
        for path in sorted((root/directory).iterdir()):
            old.require(path.suffix in (".json", ".py"), "unexpected extraction provenance file")
            add(path, root, "provenance/"+directory+"/"+path.name)
    for directory in ("pts-records-v1", "decoder-audit"):
        for path in sorted((root/directory).rglob("*")):
            if path.is_file() and path.suffix in (".json", ".py"):
                add(path, root, "provenance/"+path.relative_to(root).as_posix())
    for path in sorted((root/"preflight").rglob("*.json")):
        add(path, root, "preflight/"+path.relative_to(root/"preflight").as_posix())
    for path in sorted((root/"preflight/sources").iterdir()):
        add(path, root, "preflight/sources/"+path.name, contract["code"].get(path.name))
    for monitor in sorted(study.glob("resource-monitor-v*")):
        old.require(monitor.is_dir() and not monitor.is_symlink(), "unexpected resource-monitor directory")
        for path in sorted(monitor.iterdir()):
            old.require(path.suffix in (".py", ".ps1", ".json", ".jsonl"), "unexpected resource-monitor artifact")
            add(path, root, "provenance/study/"+monitor.name+"/"+path.name)
    for recovery in sorted(root.glob("runtime-recovery-v*")):
        old.require(recovery.is_dir() and not recovery.is_symlink(), "unexpected runtime-recovery directory")
        for path in sorted(recovery.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".py", ".ps1"):
                add(path, root, "provenance/"+path.relative_to(root).as_posix())
    external = {"studyReport": old.file_identity(study/"report.json"),
                "summary": old.file_identity(study/"summary.json"),
                "summaryMarkdown": old.file_identity(study/"summary.md"),
                "lossIdentities": old.file_identity(study/"loss-identities.json"),
                "historicalArchive": historical}
    for path in sorted(root.glob("*audit*.json")):
        external[path.name] = old.file_identity(path)
    previous = root.parent/"2026-09-19-event-balanced/source-snapshot"
    external["previousHypothesisArchive"] = old.file_identity(previous/"research-source.zip",
        "2acf81a969af9a904474ceb8a1e5f75e4f917b48e20d55a6a4eeab6787c117ab")
    total = sum(map(len, sources.values()))
    old.require(total <= prior.MAX_TOTAL_BYTES, "source snapshot exceeds narrow size budget")
    manifest = {"schemaVersion": 1, "kind": "short-boost-transfer-research-source-snapshot",
        "createdAt": datetime.now(timezone.utc).isoformat(), "contractSha256": registration["sha256"],
        "gitHead": git_head, "repositoryPath": str(REPO), "registeredSourcesVerified": 16,
        "files": rows, "sourceFileCount": len(sources), "sourceBytes": total,
        "externalBindings": external, "allowlistPatterns": list(prior.PATTERNS),
        "additionalExplicitResearchFiles": ["analysis/tests/test_transfer_temporal_model.py"],
        "changedOutsideAllowlistExcluded": sorted(changed-candidates),
        "untrackedOutsideAllowlistExcluded": sorted(untracked-candidates),
        "restore": "Checkout gitHead and overlay repository/. Keep preflight and extraction source snapshots separate. Prior archive preserves earlier worker versions. Weights, raw videos and feature arrays remain at bound NAS paths.",
        "scope": "Research sources/docs and small provenance only; no videos, feature arrays, weights, secrets or environment directories."}
    manifest_bytes = (json.dumps(manifest, indent=2, allow_nan=False)+"\n").encode()
    zip_bytes = prior.zip_roundtrip(sources, manifest_bytes)
    for member, path in origins.items():
        old.require(path.read_bytes() == sources[member], f"source changed during snapshot: {path}")
    old.require(prior.git(prefix, "rev-parse", "HEAD").decode().strip() == git_head, "Git HEAD changed")
    verification = {"mode": "created" if args.create else "validation-only-no-writes", "gitHead": git_head,
        "contractSha256": registration["sha256"], "registeredSourcesVerified": 16,
        "sourceFileCount": len(sources), "sourceBytes": total, "zipBytes": len(zip_bytes),
        "zipSha256": old.digest(zip_bytes), "manifestSha256": old.digest(manifest_bytes),
        "allSourceBytesStable": True, "zipRoundtripPassed": True, "fullStudyComplete": True}
    if args.create:
        output = root/"source-snapshot"
        output.mkdir(parents=True, exist_ok=False)
        for name, data in (("research-source.zip", zip_bytes), ("snapshot-manifest.json", manifest_bytes),
                           ("verification.json", (json.dumps(verification, indent=2)+"\n").encode())):
            with (output/name).open("xb") as handle:
                handle.write(data)
        old.file_identity(output/"research-source.zip", verification["zipSha256"])
        old.file_identity(output/"snapshot-manifest.json", verification["manifestSha256"])
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()

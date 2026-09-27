#!/usr/bin/env python3
"""Audit matched inputs, reproduce compact baselines and smoke-test DINO fitting."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value
spec = importlib.util.spec_from_file_location("old_preflight", REPO/"scripts/audit-neural-event-balanced-preflight.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
identity, read, require = helper.identity, helper.read, helper.require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(private_value('private-reference-0057'))
    parser.add_argument("--manifest", type=Path, default=root/"2026-09-19-short-boost-transfer/manifest-pts-v1.json")
    parser.add_argument("--reference-study", type=Path, default=root/"2026-09-19-expanded/study")
    parser.add_argument("--dino-manifest", type=Path, default=root/"2026-09-19-short-boost-transfer/dino-manifest.json")
    parser.add_argument("--feature-content-audit", type=Path,
                        default=root/"2026-09-19-short-boost-transfer/decoder-audit/repair-content-v1/content-audit.json")
    parser.add_argument("--output", type=Path, default=root/"2026-09-19-short-boost-transfer/preflight")
    args = parser.parse_args()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    from analysis import neural_expanded_development as expanded
    from analysis import neural_short_boost_weighting as weighting
    from analysis import neural_short_boost_transfer as runner
    from analysis.transfer_temporal_model import model_metadata

    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"kind": "bounded-short-boost-transfer-preflight-v1", "passed": False,
              "createdAt": datetime.now(timezone.utc).isoformat(), "script": identity(Path(__file__)),
              "helperScript": identity(REPO/"scripts/audit-neural-event-balanced-preflight.py"),
              "comparisons": [], "dinoSmokeTests": [], "correctedCompactSmokeTests": [], "chunkChecks": [],
              "limits": ["No decoder selection or candidate outcomes evaluated.",
                         "DINO smoke checks test fit/replay mechanics, not predictive quality.",
                         "Positive-mass matching is per recording, not per sampled minibatch."]}
    try:
        registration = read(args.reference_study/"preregistration.json")
        old = registration["contract"]
        require(runner.canonical_hash(old) == registration["sha256"], "historical registration changed")
        manifest = runner.verify_input_revision(args.manifest, old["manifestSha256"])
        original_manifest = Path(manifest["originalManifest"]["path"])
        content_audit = read(args.feature_content_audit)
        require(content_audit["kind"] == "repaired-feature-content-parity-v1"
                and content_audit["passed"] is True and content_audit["ticksPerRecord"] == 8
                and [r["recordingId"] for r in content_audit["records"]] == manifest["featureRevision"]["recordingIds"],
                "repaired feature-content parity is incomplete")
        for field in ("repairPlan", "script"):
            item = content_audit[field]
            require(helper.sha256(Path(item["path"])) == item["sha256"], "feature-content audit source binding changed")
        require(content_audit["repairPlan"]["sha256"] == read(args.dino_manifest)["repairPlan"]["sha256"],
                "feature-content parity used a different extraction plan")
        code = {name: helper.sha256(REPO/"analysis"/name) for name in (*old["code"],
            "neural_event_balanced_development.py", "neural_event_weighting.py",
            "neural_short_boost_transfer.py", "neural_short_boost_weighting.py", "transfer_temporal_model.py")}
        require(all(code[name] == digest for name, digest in old["code"].items()), "historical sources changed")
        environment = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                       "device": "cuda", "gpu": torch.cuda.get_device_name()}
        require(environment == old["environment"], "preflight environment differs from historical baseline")
        report.update({"code": code, "historicalRegistration": identity(args.reference_study/"preregistration.json"),
                       "manifest": identity(args.manifest), "dinoManifest": identity(args.dino_manifest),
                       "featureContentAudit": identity(args.feature_content_audit),
                       "environment": environment, "torchThreads": torch.get_num_threads(),
                       "cublasWorkspaceConfig": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                       "originalManifest": identity(original_manifest),
                       "protocolAmendment": identity(Path(manifest["protocolAmendment"]["path"]))})
        sources = args.output/"sources"
        sources.mkdir()
        for name in code:
            (sources/name).write_bytes((REPO/"analysis"/name).read_bytes())
        (sources/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        helper_path = Path(report["helperScript"]["path"])
        (sources/helper_path.name).write_bytes(helper_path.read_bytes())
        report["sourceSnapshots"] = [identity(path) for path in sorted(sources.iterdir())]
        report["verifiedInputReferences"] = []
        for path, digest in helper.declared_references(read(args.manifest)).items():
            actual = identity(Path(path))
            require(actual["sha256"] == digest, f"declared input changed: {path}")
            report["verifiedInputReferences"].append(actual)
        data = runner.load_data(args.manifest, args.dino_manifest)
        historical_data = expanded.load_data(original_manifest)
        require([len(data[t]) for t in ("exact", "draft", "coverage")] == [8, 3, 7], "population changed")
        groups = sorted({r.example.group for r in data["exact"]})
        require(groups == old["groups"], "source groups changed")
        outer, inner = groups[:2]
        train = [r for r in data["exact"] if r.example.group not in {outer, inner}]
        validation = [r.example for r in data["exact"] if r.example.group == inner]
        mean, scale = expanded.base.fit_scaler([r.example for r in train])
        report["weightDiagnostics"] = {arm: runner.weighting_metadata(data, arm) for arm in runner.ARMS}
        for tier, rows in data.items():
            for kind in runner.KINDS:
                original = expanded.make_chunks(rows, mean, scale, kind)
                for arm in runner.ARMS:
                    candidate = weighting.make_weighted_chunks(rows, mean, scale, kind, arm)
                    require(len(original) == len(candidate), "chunk count changed")
                    for a, b in zip(original, candidate):
                        require(all(a[j].dtype == b[j].dtype and a[j].shape == b[j].shape
                                    and a[j].tobytes() == b[j].tobytes() for j in range(3)) and a[3] == b[3],
                                "frozen chunk fields changed")
                    for seed in expanded.base.SEEDS:
                        require(expanded.epoch_batches(original, np.random.default_rng(seed))
                                == expanded.epoch_batches(candidate, np.random.default_rng(seed)), "sampling changed")
                    report["chunkChecks"].append({"tier": tier, "kind": kind, "arm": arm,
                        "chunks": len(original), "firstFourFieldsBitExact": True, "allThreeSeedsPaired": True})
                    del candidate
                del original
        for rows in data.values():
            for row in rows:
                e = row.example
                eligible = e.valid & (row.mask[:, 0] > 0) & (e.targets[:, 0] > 0)
                short = np.zeros(len(e.times), dtype=bool)
                for event in e.truth:
                    if event.end-event.start <= 3:
                        short |= (e.times >= event.start) & (e.times < event.end)
                short &= eligible
                P, S = int(eligible.sum()), int(short.sum())
                for arm in runner.ARMS:
                    expected = np.ones(len(e.times), dtype=np.float32)
                    if arm == "short_boost":
                        expected[short] = 2
                    elif arm == "global_control":
                        expected[eligible] = 1+S/P if P else 1
                    actual, _ = weighting.live_event_weights(row, arm)
                    require(actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes(),
                            f"independent multiplier mismatch: {e.id}/{arm}")
        replay = {"purpose": "predetermined baseline parity and DINO mechanics only", "code": code,
                  "manifestSha256": report["manifest"]["sha256"],
                  "dinoManifestSha256": report["dinoManifest"]["sha256"],
                  "outerIndex": 0, "innerIndex": 0, "seed": 3407}
        replay_hash = runner.canonical_hash(replay)
        helper.write_new(args.output/"control-contract.json", {"sha256": replay_hash, "contract": replay})
        for cohort in expanded.COHORTS:
            auxiliary = expanded.auxiliary_for_fold(data, cohort, {outer, inner})
            historical_auxiliary = expanded.auxiliary_for_fold(historical_data, cohort, {outer, inner})
            reference = args.reference_study/"fits"/cohort/"tcn/3407/outer-0/inner-0"
            meta = read(reference/"completed.json")
            for name in ("weights-5.npz", "predictions-5.npz"):
                require(helper.sha256(reference/name) == meta["artifacts"][name], "old checkpoint changed")
            destination = args.output/"compact-baseline-replay"/cohort
            print(f"COMPACT BASELINE REPLAY {cohort} epoch5", flush=True)
            runner.fit_model(train, historical_auxiliary, validation, "tcn", 3407, (5,), destination,
                             "cuda", replay_hash, "baseline")
            current = read(destination/"completed.json")
            comparison = {"cohort": cohort, "weights": helper.compare_npz(reference/"weights-5.npz", destination/"weights-5.npz"),
                "predictions": helper.compare_npz(reference/"predictions-5.npz", destination/"predictions-5.npz"),
                "historyBitExact": current["history"] == meta["history"][:5],
                "identityBitExact": all(current[k] == meta[k] for k in ("trainIds", "auxiliaryIds", "validationIds",
                    "trainGroups", "auxiliaryGroups", "validationGroups", "positiveWeight", "supervisedCounts", "scalerTrainIds"))}
            report["comparisons"].append(comparison)
            require(comparison["weights"]["bitExact"] and comparison["predictions"]["bitExact"]
                    and comparison["historyBitExact"] and comparison["identityBitExact"], "compact replay mismatch")
            current_reference = current
            if cohort == "reviewed_export":
                print("CORRECTED COMPACT BASELINE DETERMINISM reviewed_export epoch1 twice", flush=True)
                current_locations = [args.output/"corrected-compact-smoke"/f"repeat-{i}" for i in range(2)]
                for location in current_locations:
                    runner.fit_model(train, auxiliary, validation, "tcn", 3407, (1,), location,
                                     "cuda", replay_hash, "baseline")
                comparison = {"cohort": cohort,
                    "weights": helper.compare_npz(current_locations[0]/"weights-1.npz", current_locations[1]/"weights-1.npz"),
                    "predictions": helper.compare_npz(current_locations[0]/"predictions-1.npz", current_locations[1]/"predictions-1.npz")}
                report["correctedCompactSmokeTests"].append(comparison)
                require(comparison["weights"]["bitExact"] and comparison["predictions"]["bitExact"],
                        "corrected compact nondeterministic smoke")
                current_reference = read(current_locations[0]/"completed.json")
            print(f"DINO BASELINE DETERMINISM {cohort} epoch1 twice", flush=True)
            locations = [args.output/"dino-baseline-smoke"/cohort/f"repeat-{i}" for i in range(2)]
            for location in locations:
                outputs = runner.fit_model(train, auxiliary, validation, "dino_tcn", 3407, (1,), location,
                                           "cuda", replay_hash, "baseline")
                resumed = runner.fit_model(train, auxiliary, validation, "dino_tcn", 3407, (1,), location,
                                           "cuda", replay_hash, "baseline")
                for key in outputs[1]:
                    require(np.array_equal(outputs[1][key], resumed[1][key]), "DINO resume mismatch")
            test = {"cohort": cohort, "weights": helper.compare_npz(locations[0]/"weights-1.npz", locations[1]/"weights-1.npz"),
                    "predictions": helper.compare_npz(locations[0]/"predictions-1.npz", locations[1]/"predictions-1.npz")}
            report["dinoSmokeTests"].append(test)
            require(test["weights"]["bitExact"] and test["predictions"]["bitExact"], "DINO nondeterministic smoke")
            dino = read(locations[0]/"completed.json")
            require(dino["model"] == model_metadata("dino_tcn") and dino["parameters"] == 46868, "DINO architecture changed")
            require(dino["history"][0]["exposureSha256"] == current_reference["history"][0]["exposureSha256"]
                    and dino["history"][0]["optimizerSteps"] == current_reference["history"][0]["optimizerSteps"],
                    "DINO sampling differs from compact")
        require(all(helper.sha256(REPO/"analysis"/name) == digest for name, digest in code.items()),
                "implementation changed during preflight")
        require(identity(Path(__file__)) == report["script"] and identity(helper_path) == report["helperScript"],
                "preflight source changed during execution")
        require(all(identity(Path(entry["path"])) == entry for entry in report["sourceSnapshots"]),
                "preflight source snapshots changed")
        require(all(helper.sha256(sources/name) == digest for name, digest in code.items()),
                "analysis source snapshots differ from registered sources")
        require(identity(args.manifest) == report["manifest"] and identity(args.dino_manifest) == report["dinoManifest"],
                "manifest changed during preflight")
        require(identity(args.feature_content_audit) == report["featureContentAudit"],
                "feature-content audit changed during preflight")
        report["passed"] = True
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["completedAt"] = datetime.now(timezone.utc).isoformat()
        helper.write_new(args.output/"preflight-report.json", report)
        print(json.dumps({"passed": report["passed"], "report": identity(args.output/"preflight-report.json")}))


if __name__ == "__main__":
    main()

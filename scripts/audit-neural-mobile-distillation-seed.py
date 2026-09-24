#!/usr/bin/env python3
"""Audit one finished student seed while other registered seeds are still fitting.

Reuses a hash-bound, already executed independent input/teacher audit; rehashes
its entire evidence inventory. This is a scoped seed pass, never a full-study
pass or permission to rank an incomplete seed population.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import importlib.util
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.private_ledger import private_value


def audit_seed(study, seed, input_audit):
    import cv2
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    auditor_path = REPO / "scripts/audit-neural-mobile-distillation.py"
    spec = importlib.util.spec_from_file_location("independent_distillation_auditor", auditor_path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    evidence = audit.Evidence()
    source_ref = evidence.capture(auditor_path)
    old, tensor, interval = audit.legacy_helpers(evidence)
    registration, engineering = audit.verify_registration(study, evidence, old)
    contract = registration["contract"]
    audit.require(seed in contract["seeds"], "Seed is not registered")
    # A finished result is a prerequisite; partial fit folders are not audited.
    result_ref = evidence.capture(study / f"result-{seed}.json")
    result = audit.read(result_ref["path"])
    audit.require(result["seed"] == seed and result["contractSha256"] == registration["sha256"],
                  "Finished seed result identity differs")
    input_ref = evidence.capture(input_audit)
    prior = audit.read(input_ref["path"])
    audit.require(prior.get("kind") == "independent-distilled-mobile-input-teacher-audit-v1"
        and prior.get("passed") is True and prior["auditor"] == source_ref
        and prior["registration"] == audit.identity(study / "preregistration.json")
        and prior["imageIndex"] == contract["images"] and prior["teacherIndex"] == contract["teacherTargets"]
        and prior["gpuUsed"] is False and prior["trainingPerformed"] is False
        and len(prior["records"]) == 18, "Prior independent input/teacher gate changed or incomplete")
    for reference in prior["evidence"]:
        evidence.bind(reference)
    data, config = audit.load_inputs(contract, evidence)
    images = {r["id"]: r for r in audit.read(contract["images"]["path"])["records"]}
    teachers = {r["id"]: r for r in audit.read(contract["teacherTargets"]["path"])["records"]}
    checked = audit.audit_seed(study, seed, registration, data, images, teachers, config,
                               evidence, old, tensor, interval)
    return {"kind": "independent-mobile-distillation-seed-audit-v1", "passed": True,
        "fullStudyPassed": False, "seed": seed, "registration": audit.identity(study / "preregistration.json"),
        "contractSha256": registration["sha256"], "result": checked,
        "inputTeacherAudit": input_ref, "inputTeacherEvidenceRehashed": True,
        "auditor": source_ref, "runner": audit.identity(Path(__file__)),
        "scope": "One completed registered seed, including every feasible outer and all inner fits, strict99 reselection, and complete/infeasible handling. Other seeds are not claimed audited.",
        "replayScope": "Three encoder feature frames per fold/record and first/middle/last real-context head chunks per held record/checkpoint. Input/teacher replay is bound through the earlier 54-frame independent gate. Not full training, full-frame, or full-tick replay.",
        "inputScope": "Every prepared image/teacher/feature and fit artifact in the evidence inventory is hash-verified. Raw videos are not fully rehashed.",
        "tolerance": {"teacher": {"rtol": audit.TEACHER_RTOL, "atol": audit.TEACHER_ATOL},
            "encoder": {"rtol": audit.FEATURE_RTOL, "atol": audit.FEATURE_ATOL,
                "storedFloat16Rounding": "plus half local FP16 ULP"},
            "head": {"rtol": old.REPLAY_RTOL, "atol": old.REPLAY_ATOL}},
        "evidence": evidence.finish(), "gpuUsed": False, "trainingPerformed": False,
        "protectedTestOpened": False, "finishedAtUtc": datetime.now(UTC).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(3407, 1729, 20260918))
    parser.add_argument("--input-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not str(args.output.resolve()).startswith(private_value('private-reference-0060')) or args.output.exists():
        raise ValueError("Scoped audit requires a new direct-NAS output")
    for name in ("TMPDIR", "TMP", "TEMP", "TORCH_HOME", "HF_HOME", "XDG_CACHE_HOME", "CUDA_CACHE_PATH"):
        path = args.study.parent / "audit-runtime" / name.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    sys.dont_write_bytecode = True
    result = audit_seed(args.study.resolve(), args.seed, args.input_audit.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": True, "fullStudyPassed": False, "seed": args.seed,
        "completeEvaluation": result["result"]["completeEvaluation"], "output": str(args.output)}))


if __name__ == "__main__":
    main()

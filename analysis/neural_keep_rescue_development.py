"""Optional registered CPU-only keep-head rescue over immutable baseline scores.

No model fitting, feature extraction, epoch search or live-decoder search occurs.
For each outer fold, only the five declared rescue options are ranked on that
fold's original three inner prediction files at its already selected baseline
epoch/decoder. The outer group's probabilities are opened after selection.
No experiment is selected or registered by importing or executing this module.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from . import neural_short_boost_transfer as source
from . import neural_keep_rescue as rescue
from .crop_evaluation import RecordingIntervals, evaluate_f1_pad_p_core_r
from .neural_evaluation import evaluate_predictions
from .schema import Interval, mask_for_times

REPO = Path(__file__).resolve().parents[1]
KINDS = ("tcn", "dino_tcn")
COHORTS = ("exact", "draft", "reviewed_export")
VARIANTS = ("reference", "selected")
EXPERIMENT = "keep-head-short-rescue-development-v1"
NEW_SOURCES = {"neural_keep_rescue.py", "neural_keep_rescue_development.py"}
AUDIT_KINDS = {"neural-short-boost-independent-tensor-audit-v1", "independent-short-boost-transfer-interval-audit-v1"}
RESCUE_SELECTION = {
    "recallEligibilityFloor": .95,
    "ranking": "F1_padP_coreR descending among eligible; first option wins exact ties",
    "fallback": "highest F1 flagged infeasible if no option meets recall floor",
    "scope": "pooled original three inner source-held prediction sets only",
    "baselineEpochAndDecoderFixed": True,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def digest(path):
    return source.base.file_sha256(Path(path))


def identity(path):
    return {"path": str(path), "sha256": digest(path)}


def verified(bound):
    require(digest(bound["path"]) == bound["sha256"], "Changed bound input: " + bound["path"])
    return read(bound["path"])


def write_immutable(path, value):
    path = Path(path)
    if path.exists():
        require(read(path) == value, "Immutable result changed: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def contains_binding(value, bound):
    if isinstance(value, dict):
        return (value.get("path") == bound["path"] and value.get("sha256") == bound["sha256"]) or any(
            contains_binding(v, bound) for v in value.values())
    return isinstance(value, list) and any(contains_binding(v, bound) for v in value)


def completed_reference(reference):
    """Require the complete preceding factorial and both report-bound audits."""
    root = Path(reference["path"])
    reg = verified({"path": str(root/"preregistration.json"), "sha256": reference["preregistrationFileSha256"]})
    require(canonical_hash(reg["contract"]) == reg["sha256"] == reference["contractSha256"], "Reference registration differs")
    report_binding = {"path": str(root/"report.json"), "sha256": reference["reportSha256"]}
    report = verified(report_binding)
    summary = verified({"path": str(root/"summary.json"), "sha256": reference["summarySha256"]})
    require(report["status"] == "completed-short-boost-transfer-development"
            and summary["status"] == "completed-short-boost-transfer-audit"
            and report["contractSha256"] == summary["contractSha256"] == reg["sha256"]
            and contains_binding(summary, report_binding)
            and report["protectedTestOpened"] is False and report["productionPromotionAllowed"] is False,
            "Reference study not complete and audited")
    expected = {(c, k, a, s) for c in COHORTS for k in KINDS for a in source.ARMS for s in reg["contract"]["seeds"]}
    require(len(report["results"]) == len(expected) == 54
            and {(r["cohort"], r["kind"], r["lossArm"], r["seed"]) for r in report["results"]} == expected
            and all(r["contractSha256"] == reg["sha256"] for r in report["results"]), "Incomplete/duplicate reference grid")
    audits = [verified(item) for item in reference["independentAudits"]]
    require(len(audits) == 2 and len({r["path"] for r in reference["independentAudits"]}) == 2
            and {r["kind"] for r in audits} == AUDIT_KINDS, "Distinct tensor and interval reference audits required")
    for audit in audits:
        require(audit["passed"] is True and audit["contractSha256"] == reg["sha256"]
                and contains_binding(audit, report_binding), "Reference audit failed or unrelated")
    tensor = next(a for a in audits if a["kind"] == "neural-short-boost-independent-tensor-audit-v1")
    fits = {Path(r["completed"]["path"]).resolve(): r["completed"]["sha256"] for r in tensor["fits"]}
    require(len(tensor["fits"]) == len(fits) == 864, "Reference audited fit inventory differs")
    return reg, report, fits


def validate_registration(path):
    registration = read(path)
    contract = registration["contract"]
    require(canonical_hash(contract) == registration["sha256"] and contract["experiment"] == EXPERIMENT, "Bad rescue registration")
    require(bool(contract["cohorts"]) and contract["cohorts"] == [c for c in COHORTS if c in contract["cohorts"]]
            and contract["kinds"] == list(KINDS) and contract["variants"] == list(VARIANTS)
            and contract["baselineLossArm"] == "baseline", "Explicit supported rescue scope required")
    require(contract["rescueOptions"] == list(rescue.KEEP_RESCUE_OPTIONS)
            and contract["rescueRecipe"] == rescue.rescue_metadata()
            and contract["rescueSelection"] == RESCUE_SELECTION, "Rescue recipe or selection changed")
    old_reg, report, fits = completed_reference(contract["referenceStudy"])
    old = old_reg["contract"]
    for field in ("seeds", "groups", "checkpointEpochs", "decoderCandidates", "selection", "primaryMetric",
                  "targetPaddingSeconds", "joinGapSeconds", "paddingSweep", "manifestSha256", "dinoManifest",
                  "screen", "retentionRecoveryScreen", "evaluationPopulation"):
        require(contract[field] == old[field], "Baseline comparison changes " + field)
    require(len(contract["groups"]) == 4 and len(contract["seeds"]) == 3
            and contract["primaryMetric"] == "F1_padP_coreR" and contract["targetPaddingSeconds"] == 2
            and contract["joinGapSeconds"] == 3 and contract["paddingSweep"] == [0, 1, 2, 3], "Metric or fold scope differs")
    require(len(old["code"]) == 16 and set(contract["code"]) == set(old["code"]) | NEW_SOURCES
            and all(contract["code"][k] == v for k, v in old["code"].items()), "Frozen source inheritance differs")
    for name, sha in contract["code"].items():
        require(Path(name).name == name and digest(REPO/"analysis"/name) == sha, "Registered source changed")
    require(digest(contract["protocolSnapshot"]["path"]) == contract["protocolSnapshot"]["sha256"], "Prospective protocol changed")
    manifest = verified(contract["manifest"])
    require(contract["manifest"]["sha256"] == contract["manifestSha256"], "Manifest identity differs")
    require(len(manifest["exactRows"]) == 8 and sorted({r["sourceGroup"] for r in manifest["exactRows"]}) == contract["groups"],
            "Exact evaluation population differs")
    exact = verified(manifest["exactManifest"])
    require(exact["recordings"] == manifest["exactRows"], "Exact label revision differs")
    preflight = verified(contract["preflight"])
    report_bound = {"path": str(Path(contract["referenceStudy"]["path"])/"report.json"), "sha256": contract["referenceStudy"]["reportSha256"]}
    require(preflight["kind"] == "keep-rescue-engineering-preflight-v1"
            and preflight["passed"] is True and preflight["noOpReplayQualified"] is True
            and preflight["selectionIsolationQualified"] is True
            and preflight["code"] == contract["code"] and preflight["manifest"] == contract["manifest"]
            and preflight["cohorts"] == contract["cohorts"]
            and preflight["executionEnvironment"] == contract["executionEnvironment"]
            and preflight["referenceContractSha256"] == old_reg["sha256"]
            and preflight["referenceReport"] == report_bound
            and preflight["referenceAudits"] == contract["referenceStudy"]["independentAudits"], "Rescue preflight binding differs")
    require(contract["executionEnvironment"] == {"python": platform.python_version(), "numpy": np.__version__, "device": "cpu"},
            "CPU decoder execution environment differs")
    return registration, manifest, report, fits


def load_evaluation_examples(manifest):
    """Read exact timestamps and duration only; never AV values/DINO tokens."""
    result = []
    for row in manifest["exactRows"]:
        require(row["environment"] in ("grass", "indoor") and row["consent"]["train"] is True
                and row["sourceGroup"] not in manifest["protectedSourceGroups"], "Forbidden exact evaluation input")
        cache = row["featureCaches"]["audiovisual"]
        require(digest(cache["path"]) == cache["sha256"], "Exact timestamp cache changed")
        with np.load(cache["path"], allow_pickle=False) as archive:
            times = archive["times"].astype(np.float64)
            duration = float(json.loads(str(archive["metadata_json"].item()))["duration"])
        require(times.ndim == 1 and len(times) and np.isfinite(times).all()
                and np.all(np.diff(times) > 0) and np.all(np.abs(np.diff(times)-.25) <= .1)
                and np.isfinite(duration) and duration > 0, "Invalid exact evaluation timeline")
        intervals = lambda key: tuple(Interval(float(r["start"]), float(r["end"]), tuple(r.get("tags", []))) for r in row.get(key, []))
        truth, ignored = intervals("rallies"), intervals("ignoredIntervals")
        empty = np.empty((len(times), 0), dtype=np.float32)
        result.append(source.base.Example(row["id"], row["sourceGroup"], duration, times, empty, empty,
                                         mask_for_times(times, ignored), truth, ignored, row["environment"]))
    return result


def source_fit_root(baseline, reference_contract, reference_path):
    cohort, kind, seed = (baseline[k] for k in ("cohort", "kind", "seed"))
    require(baseline["lossArm"] == "baseline", "Rescue must reuse baseline loss checkpoints")
    origin = baseline["origin"]
    historical = source.reuses_reference(cohort, kind, "baseline")
    if historical:
        reference = reference_contract["referenceStudy"]
        root = Path(reference["path"])/"fits"/cohort/kind/str(seed)
        require(origin["type"] == "reused-reference" and origin["studyPath"] == reference["path"]
                and origin["referenceContractSha256"] == reference["contractSha256"]
                and origin["reportSha256"] == reference["reportSha256"], "Historical baseline origin changed")
        contract_hash = reference["contractSha256"]
    else:
        root = Path(reference_path)/"fits"/cohort/kind/"baseline"/str(seed)
        require(origin["type"] == "trained", "Fresh baseline origin changed")
        contract_hash = baseline["contractSha256"]
    require(Path(origin["fitRoot"]).resolve() == root.resolve(), "Baseline fit path changed")
    return root, contract_hash


def expected_membership(manifest, cohort, excluded, held):
    active = () if cohort == "exact" else ("draft",) if cohort == "draft" else ("draft", "coverage")
    require(cohort in COHORTS, "Unsupported cohort")
    return {"trainIds": [r["id"] for r in manifest["exactRows"] if r["sourceGroup"] not in excluded],
            "auxiliaryIds": {tier: [r["id"] for r in manifest[tier+"Rows"] if r["sourceGroup"] not in excluded] for tier in active},
            "validationIds": [r["id"] for r in manifest["exactRows"] if r["sourceGroup"] == held]}


def load_bound_predictions(folder, epoch, examples, *, expected, kind, seed, contract_hash, audited_fits):
    """Load one selected epoch and the exact held scope; refuse outer leakage."""
    completion = folder/"completed.json"
    require(audited_fits.get(completion.resolve()) == digest(completion), "Baseline completion differs from prior tensor audit")
    meta = read(completion)
    require(meta["contractSha256"] == contract_hash and meta["kind"] == kind and meta["seed"] == seed
            and all(meta[key] == value for key, value in expected.items()) and epoch in meta["epochs"]
            and meta.get("lossArm", "baseline") == "baseline"
            and meta["scalerTrainIds"] == meta["trainIds"], "Baseline fit identity/epoch/membership differs")
    require([e.id for e in examples] == expected["validationIds"], "Requested prediction scope differs")
    fitting = set(meta["trainIds"]) | {rid for ids in meta["auxiliaryIds"].values() for rid in ids}
    training_groups = set(meta["trainGroups"]) | {g for groups in meta["auxiliaryGroups"].values() for g in groups}
    require(not fitting & {e.id for e in examples} and not training_groups & {e.group for e in examples}, "Baseline source-group leakage")
    inventory = {f"{stem}-{ep}.npz" for ep in meta["epochs"] for stem in ("weights", "predictions")}
    require(set(meta["artifacts"]) == inventory, "Baseline artifact inventory differs")
    paths = {stem: folder/f"{stem}-{epoch}.npz" for stem in ("weights", "predictions")}
    for path in paths.values():
        require(digest(path) == meta["artifacts"][path.name], "Baseline selected artifact changed")
    with np.load(paths["predictions"], allow_pickle=False) as archive:
        require(len(archive.files) == len(examples) and set(archive.files) == {e.id for e in examples}, "Baseline NPZ includes wrong/outer scope")
        probabilities = {}
        for e in examples:
            values = archive[e.id]
            require(values.shape == (len(e.times), 4) and values.dtype == np.float32 and np.isfinite(values).all()
                    and np.all((values >= 0) & (values <= 1)) and np.all(values[~e.valid] == 0), "Invalid baseline probability trace")
            probabilities[e.id] = values.copy()
    return probabilities, {"completed": identity(completion), **{k: identity(v) for k, v in paths.items()},
                           "epoch": epoch, "membership": deepcopy(expected)}


def select_rescue(examples, probabilities, decoder, options=None):
    options = tuple(rescue.KEEP_RESCUE_OPTIONS if options is None else options)
    require(options == tuple(rescue.KEEP_RESCUE_OPTIONS), "Rescue option grid/order changed")
    require(len({e.id for e in examples}) == len(examples) and set(probabilities) == {e.id for e in examples}, "Inner prediction population differs")
    candidates = []
    for index, threshold in enumerate(options):
        rows = [RecordingIntervals(e.id, "development", e.duration, e.truth,
                tuple(rescue.decode_with_keep_rescue(e, probabilities[e.id], decoder, keep_threshold=threshold)), e.ignored) for e in examples]
        score = evaluate_f1_pad_p_core_r(rows, [2.], 3.)[0]
        candidates.append({"keepThreshold": threshold, "optionIndex": index,
                           "innerF1_padP_coreR": score["F1_padP_coreR"], "innerR_core": score["R_core"]})
    eligible = [r for r in candidates if r["innerR_core"] >= .95]
    selected = max(eligible or candidates, key=lambda r: r["innerF1_padP_coreR"])
    return {**selected, "recallEligibilityPassed": bool(eligible), "recallEligibilityFloor": .95, "candidates": candidates}


def serialized_row(example, predictions):
    row = example.row(predictions)
    return {**row, **{key: [r.to_dict() for r in row[key]] for key in ("rallies", "ignoredIntervals", "predictions")}}


def reference_result(baseline, reference, contract_hash):
    return {**deepcopy(baseline), "rescueVariant": "reference", "contractSha256": contract_hash,
            "origin": {"type": "reused-keep-rescue-reference", "studyPath": reference["path"],
                       "reportSha256": reference["reportSha256"], "referenceContractSha256": reference["contractSha256"],
                       "sourceOrigin": deepcopy(baseline["origin"]),
                       "sourceResult": {k: baseline[k] for k in ("cohort", "kind", "lossArm", "seed")}}}


def run_cell(examples, manifest, contract, contract_hash, baseline, reference_contract, audited_fits, *, load_fn=None, select_fn=None):
    """Decode one cell; dependencies injectable for synthetic isolation tests."""
    loader, selector = load_fn or load_bound_predictions, select_fn or select_rescue
    cohort, kind, seed = (baseline[k] for k in ("cohort", "kind", "seed"))
    require(cohort in contract["cohorts"] and kind in contract["kinds"] and seed in contract["seeds"], "Unregistered rescue cell")
    root, source_contract_hash = source_fit_root(baseline, reference_contract, contract["referenceStudy"]["path"])
    require([e.id for e in examples] == [r["id"] for r in manifest["exactRows"]], "Exact input order changed")
    selections = {r["heldSourceGroup"]: r for r in baseline["selections"]}
    require(len(selections) == len(baseline["selections"]) == len(contract["groups"]) and set(selections) == set(contract["groups"]), "Baseline outer selections differ")
    output, selected, no_op_rows = [], [], []
    for outer_index, outer in enumerate(contract["groups"]):
        fixed = selections[outer]
        epoch, decoder = fixed["epoch"], fixed["decoder"]
        require(epoch in contract["checkpointEpochs"] and decoder in contract["decoderCandidates"], "Baseline selection outside grid")
        inner_probabilities, bindings = {}, []
        fitting = [e for e in examples if e.group != outer]
        for inner_index, inner in enumerate(g for g in contract["groups"] if g != outer):
            held = [e for e in examples if e.group == inner]
            members = expected_membership(manifest, cohort, {outer, inner}, inner)
            values, bound = loader(root/f"outer-{outer_index}"/f"inner-{inner_index}", epoch, held,
                expected=members, kind=kind, seed=seed, contract_hash=source_contract_hash, audited_fits=audited_fits)
            require(set(values) == {e.id for e in held} and not set(inner_probabilities) & set(values), "Inner view includes duplicate/outer rows")
            inner_probabilities.update(values)
            bindings.append({"innerValidationGroup": inner, **bound})
        require(set(inner_probabilities) == {e.id for e in fitting}, "Inner selection exposes missing/outer group")
        choice = selector(fitting, inner_probabilities, decoder, contract["rescueOptions"])
        require(choice["keepThreshold"] in contract["rescueOptions"]
                and choice["optionIndex"] == contract["rescueOptions"].index(choice["keepThreshold"])
                and choice["recallEligibilityFloor"] == .95
                and choice["recallEligibilityPassed"] == (choice["innerR_core"] >= .95), "Invalid rescue selection")
        # Outer probabilities are loaded only after rescue selection finishes.
        held = [e for e in examples if e.group == outer]
        values, bound = loader(root/f"outer-{outer_index}"/"refit", epoch, held,
            expected=expected_membership(manifest, cohort, {outer}, outer), kind=kind, seed=seed,
            contract_hash=source_contract_hash, audited_fits=audited_fits)
        require(set(values) == {e.id for e in held}, "Outer prediction scope differs")
        for e in held:
            no_op_rows.append(serialized_row(e, rescue.decode_with_keep_rescue(e, values[e.id], decoder, keep_threshold=None)))
            output.append(serialized_row(e, rescue.decode_with_keep_rescue(e, values[e.id], decoder, keep_threshold=choice["keepThreshold"])))
        selected.append({"heldSourceGroup": outer, "epoch": epoch, "decoder": deepcopy(decoder),
                         "baselineSelection": deepcopy(fixed), **choice, "innerPredictionBindings": bindings,
                         "outerPredictionBinding": bound})
    require(no_op_rows == baseline["predictions"] and evaluate_predictions(no_op_rows) == baseline["evaluation"],
            "No-op baseline replay differs from immutable reference")
    return {"cohort": cohort, "kind": kind, "architecture": kind, "lossArm": "baseline", "rescueVariant": "selected", "seed": seed,
            "contractSha256": contract_hash, "origin": {"type": "rescued-reference-checkpoints", "fitRoot": str(root),
                "sourceContractSha256": source_contract_hash, "sourceOrigin": deepcopy(baseline["origin"]),
                "referenceStudy": deepcopy(contract["referenceStudy"])},
            "selections": selected, "evaluation": evaluate_predictions(output), "predictions": output,
            "baselineNoOpReplayPassed": True, "newFits": 0}


def result_path(output, row):
    return output/f"result-{row['cohort']}-{row['kind']}-baseline-{row['rescueVariant']}-{row['seed']}.json"


def run_study(registration_path):
    registration, manifest, report, audited = validate_registration(registration_path)
    c, output = registration["contract"], registration_path.parent
    require(not (output/"report.json").exists(), "Completed rescue study exists")
    old_reg = verified({"path": str(Path(c["referenceStudy"]["path"])/"preregistration.json"),
                        "sha256": c["referenceStudy"]["preregistrationFileSha256"]})
    selected = [r for r in report["results"] if r["cohort"] in c["cohorts"] and r["lossArm"] == "baseline"]
    expected = {(cohort, kind, seed) for cohort in c["cohorts"] for kind in KINDS for seed in c["seeds"]}
    require(len(selected) == len(expected) and {(r["cohort"], r["kind"], r["seed"]) for r in selected} == expected, "Missing/duplicate baseline cells")
    started = time.perf_counter()
    execution_path = output/f"execution-{time.time_ns()}.json"
    execution = {"status": "running", "contractSha256": registration["sha256"], "device": "cpu", "newFits": 0,
                 "startedAt": datetime.now(timezone.utc).isoformat(), "candidateCells": len(selected), "referenceCells": len(selected)}
    write_immutable(execution_path, execution)
    try:
        examples = load_evaluation_examples(manifest)
        results = []
        for baseline in selected:
            reference = reference_result(baseline, c["referenceStudy"], registration["sha256"])
            write_immutable(result_path(output, reference), reference)
            candidate = run_cell(examples, manifest, c, registration["sha256"], baseline, old_reg["contract"], audited)
            write_immutable(result_path(output, candidate), candidate)
            results.extend((reference, candidate))
        final_registration, _, _, _ = validate_registration(registration_path)
        require(final_registration == registration, "Registration changed during rescue execution")
        final = {"schemaVersion": 1, "status": "completed-keep-rescue-development", "contractSha256": registration["sha256"],
                 "manifestSha256": c["manifestSha256"], "records": len(examples), "sourceGroups": c["groups"],
                 "protectedTestOpened": False, "productionPromotionAllowed": False,
                 "execution": {"started": identity(execution_path), "device": "cpu", "newFits": 0,
                               "candidateCells": len(selected), "referenceCells": len(selected), "wallSeconds": time.perf_counter()-started},
                 "results": results}
        write_immutable(output/"report.json", final)
    except BaseException as error:
        write_immutable(execution_path.with_name(execution_path.stem+"-failed.json"),
                        {**execution, "status": "failed", "started": identity(execution_path),
                         "errorType": type(error).__name__, "error": str(error)})
        raise
    write_immutable(execution_path.with_name(execution_path.stem+"-completed.json"),
                    {**execution, "status": "completed", "started": identity(execution_path), "report": identity(output/"report.json")})
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        registration, _, _, _ = validate_registration(args.registration)
        print(json.dumps({"validated": True, "contractSha256": registration["sha256"], "newFits": 0}))
    else:
        run_study(args.registration)


if __name__ == "__main__":
    main()

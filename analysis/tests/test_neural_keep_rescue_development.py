"""Synthetic prediction files only: no neural fitting or actual outcomes."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analysis import neural_keep_rescue_development as runner
from analysis.schema import Interval, mask_for_times


def fixture(root, *, cohort="reviewed_export", kind="dino_tcn", seed=3407):
    """Reusable synthetic four-group reference cell with16 tiny checkpoint pairs."""
    root = Path(root)
    groups = ["A", "B", "C", "D"]
    times = np.arange(120, dtype=np.float64)/4+.125
    ignored = (Interval(24, 25),)
    examples = [runner.source.base.Example(f"exact-{g}-{i}", g, 30., times.copy(), np.empty((120, 0), np.float32),
                  np.empty((120, 0), np.float32), mask_for_times(times, ignored), (Interval(12, 14),), ignored, "grass")
                for g in groups for i in range(2)]
    manifest = {"exactRows": [{"id": e.id, "sourceGroup": e.group, "environment": "grass", "consent": {"train": True},
                              "rallies": [r.to_dict() for r in e.truth], "ignoredIntervals": [r.to_dict() for r in e.ignored]}
                             for e in examples],
                "draftRows": [{"id": f"draft-{g}", "sourceGroup": g} for g in ("A", "C", "aux-only")],
                "coverageRows": [{"id": f"coverage-{i}", "sourceGroup": g} for i, g in enumerate(("A", "A", "B", "C", "D", "aux-only", "aux-only"))],
                "protectedSourceGroups": ["protected"]}
    decoder = {"smoothing": .5, "enter": .5, "minimum": .25, "boundary": False}
    reference = {"path": str(root/"reference"), "contractSha256": "current-contract", "reportSha256": "current-report"}
    contract = {"cohorts": [cohort], "kinds": list(runner.KINDS), "seeds": [seed], "groups": groups,
                "checkpointEpochs": [5, 15, 30, 60], "decoderCandidates": [decoder],
                "rescueOptions": list(runner.rescue.KEEP_RESCUE_OPTIONS), "referenceStudy": reference}
    previous = {"referenceStudy": {"path": str(root/"historical"), "contractSha256": "historical-contract", "reportSha256": "historical-report"}}
    historical = runner.source.reuses_reference(cohort, kind, "baseline")
    if historical:
        origin = {"type": "reused-reference", "studyPath": previous["referenceStudy"]["path"],
                  "reportSha256": "historical-report", "referenceContractSha256": "historical-contract",
                  "fitRoot": str(root/"historical"/"fits"/cohort/kind/str(seed))}
    else:
        origin = {"type": "trained", "fitRoot": str(root/"reference"/"fits"/cohort/kind/"baseline"/str(seed))}
    probabilities = {}
    for e in examples:
        score = np.zeros((120, 4), np.float32)
        score[(e.times >= 10) & (e.times < 16), 3] = 1
        score[~e.valid] = 0
        probabilities[e.id] = score
    baseline_rows = [runner.serialized_row(e, runner.rescue.decode_with_keep_rescue(e, probabilities[e.id], decoder)) for e in examples]
    selections = [{"heldSourceGroup": g, "epoch": 5, "decoder": deepcopy(decoder), "innerF1_padP_coreR": 0.,
                   "innerR_core": 0., "recallEligibilityPassed": False, "recallEligibilityFloor": .95} for g in groups]
    baseline = {"cohort": cohort, "kind": kind, "architecture": kind, "lossArm": "baseline", "seed": seed,
                "contractSha256": "current-contract", "origin": origin, "selections": selections,
                "predictions": baseline_rows, "evaluation": runner.evaluate_predictions(baseline_rows)}
    fit_root, source_contract = runner.source_fit_root(baseline, previous, reference["path"])
    audit = {}
    by_id = {r["id"]: r for key in ("exactRows", "draftRows", "coverageRows") for r in manifest[key]}
    for outer_index, outer in enumerate(groups):
        views = [(f"inner-{i}", {outer, held}, held) for i, held in enumerate(g for g in groups if g != outer)]
        views.append(("refit", {outer}, outer))
        for name, excluded, held in views:
            folder = fit_root/f"outer-{outer_index}"/name
            folder.mkdir(parents=True)
            members = runner.expected_membership(manifest, cohort, excluded, held)
            np.savez_compressed(folder/"predictions-5.npz", **{rid: probabilities[rid] for rid in members["validationIds"]})
            np.savez_compressed(folder/"weights-5.npz", synthetic_marker=np.array([1], np.float32))
            meta = {"contractSha256": source_contract, "kind": kind, "seed": seed, "epochs": [5], **members,
                    "scalerTrainIds": members["trainIds"],
                    "validationGroups": [held],
                    "trainGroups": sorted({by_id[rid]["sourceGroup"] for rid in members["trainIds"]}),
                    "auxiliaryGroups": {tier: sorted({by_id[rid]["sourceGroup"] for rid in ids}) for tier, ids in members["auxiliaryIds"].items()},
                    "artifacts": {p.name: runner.digest(p) for p in folder.glob("*.npz")}}
            if not historical:
                meta["lossArm"] = "baseline"
            runner.write_immutable(folder/"completed.json", meta)
            audit[(folder/"completed.json").resolve()] = runner.digest(folder/"completed.json")
    return examples, manifest, contract, baseline, previous, audit


def gate_fixture(root):
    """A complete synthetic provenance envelope, with no real study access."""
    root = Path(root)
    repo, reference = root/"repo", root/"reference"
    (repo/"analysis").mkdir(parents=True)
    code = {}
    for name in [f"frozen-{i}.py" for i in range(16)] + sorted(runner.NEW_SOURCES):
        path = repo/"analysis"/name
        path.write_text("# synthetic source " + name)
        code[name] = runner.digest(path)
    def save(path, value):
        runner.write_immutable(path, value)
        return runner.identity(path)
    groups = ["A", "B", "C", "D"]
    rows = [{"id": f"r-{g}-{i}", "sourceGroup": g} for g in groups for i in range(2)]
    exact = save(root/"exact.json", {"recordings": rows})
    manifest = save(root/"manifest.json", {"exactRows": rows, "exactManifest": exact})
    dino = save(root/"dino.json", {"synthetic": True})
    old = {"code": {k: v for k, v in code.items() if k not in runner.NEW_SOURCES},
           "seeds": [3407, 1729, 20260918], "groups": groups, "checkpointEpochs": [5, 15, 30, 60],
           "decoderCandidates": [], "selection": {}, "primaryMetric": "F1_padP_coreR", "targetPaddingSeconds": 2,
           "joinGapSeconds": 3, "paddingSweep": [0, 1, 2, 3], "manifestSha256": manifest["sha256"],
           "dinoManifest": dino, "screen": {}, "retentionRecoveryScreen": {}, "evaluationPopulation": {"records": 8}}
    old_hash = runner.canonical_hash(old)
    old_registration = save(reference/"preregistration.json", {"contract": old, "sha256": old_hash})
    cells = [{"cohort": c, "kind": k, "lossArm": a, "seed": s, "contractSha256": old_hash}
             for c in runner.COHORTS for k in runner.KINDS for a in runner.source.ARMS for s in old["seeds"]]
    report = save(reference/"report.json", {"status": "completed-short-boost-transfer-development", "contractSha256": old_hash,
                  "protectedTestOpened": False, "productionPromotionAllowed": False, "results": cells})
    summary = save(reference/"summary.json", {"status": "completed-short-boost-transfer-audit", "contractSha256": old_hash,
                   "inputs": {"report": report}})
    audits = []
    for name, kind in (("tensor", "neural-short-boost-independent-tensor-audit-v1"),
                       ("interval", "independent-short-boost-transfer-interval-audit-v1")):
        value = {"kind": kind, "passed": True, "contractSha256": old_hash, "report": report}
        if name == "tensor":
            value["fits"] = [{"completed": {"path": str(reference/f"fit-{i}/completed.json"), "sha256": f"synthetic-{i}"}} for i in range(864)]
        audits.append(save(reference/f"{name}.json", value))
    reference_binding = {"path": str(reference), "contractSha256": old_hash,
                         "preregistrationFileSha256": old_registration["sha256"], "reportSha256": report["sha256"],
                         "summarySha256": summary["sha256"], "independentAudits": audits}
    protocol_path = root/"protocol.md"
    protocol_path.write_text("# Prospective synthetic protocol\n")
    preflight = save(root/"preflight.json", {"kind": "keep-rescue-engineering-preflight-v1", "passed": True, "noOpReplayQualified": True, "selectionIsolationQualified": True,
                    "code": code, "manifest": manifest, "cohorts": ["exact"], "referenceContractSha256": old_hash,
                    "executionEnvironment": {"python": runner.platform.python_version(), "numpy": np.__version__, "device": "cpu"},
                    "referenceReport": report, "referenceAudits": audits})
    contract = {**deepcopy(old), "experiment": runner.EXPERIMENT, "cohorts": ["exact"], "kinds": list(runner.KINDS),
                "variants": list(runner.VARIANTS), "baselineLossArm": "baseline", "rescueOptions": list(runner.rescue.KEEP_RESCUE_OPTIONS),
                "rescueRecipe": runner.rescue.rescue_metadata(), "rescueSelection": runner.RESCUE_SELECTION,
                "code": code, "manifest": manifest, "referenceStudy": reference_binding, "preflight": preflight,
                "protocolSnapshot": runner.identity(protocol_path),
                "executionEnvironment": {"python": runner.platform.python_version(), "numpy": np.__version__, "device": "cpu"}}
    registration_path = root/"rescue"/"preregistration.json"
    save(registration_path, {"contract": contract, "sha256": runner.canonical_hash(contract)})
    return registration_path, repo


class KeepRescueRunnerTests(unittest.TestCase):
    def test_cell_uses_only_original_inner_groups_before_opening_outer_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = fixture(temporary)
            examples, manifest, contract, baseline, previous, audit = args
            calls = []
            def load(folder, epoch, held, **kwargs):
                calls.append((folder.name, tuple(e.group for e in held), epoch, deepcopy(kwargs["expected"])))
                return runner.load_bound_predictions(folder, epoch, held, **kwargs)
            def choose(rows, scores, settings, options):
                outer_index = len([c for c in calls if c[0] == "SELECT"])
                outer = contract["groups"][outer_index]
                self.assertEqual(len(calls), outer_index*5+3)
                self.assertNotIn(outer, {e.group for e in rows})
                self.assertEqual(set(scores), {e.id for e in rows})
                self.assertEqual(settings, baseline["selections"][outer_index]["decoder"])
                calls.append(("SELECT", outer))
                return runner.select_rescue(rows, scores, settings, options)
            result = runner.run_cell(examples, manifest, contract, "new-contract", baseline, previous, audit,
                                     load_fn=load, select_fn=choose)
            self.assertEqual([c[0] for c in calls], ["inner-0", "inner-1", "inner-2", "SELECT", "refit"]*4)
            self.assertEqual(result["newFits"], 0)
            self.assertTrue(result["baselineNoOpReplayPassed"])
            self.assertEqual(len(result["selections"]), 4)
            self.assertTrue(all(r["keepThreshold"] is not None and r["recallEligibilityPassed"] for r in result["selections"]))
            self.assertEqual([r["paddingSecondsBeforeAndAfter"] for r in result["evaluation"]["padding"]], [0, 1, 2, 3])
            self.assertGreater(result["evaluation"]["primary"]["R_core"], baseline["evaluation"]["primary"]["R_core"])
            for c in calls:
                if c[0] != "SELECT":
                    self.assertEqual(c[2], 5)
                    forbidden = set(c[1])
                    train_ids = c[3]["trainIds"] + [rid for rows in c[3]["auxiliaryIds"].values() for rid in rows]
                    lookup = {r["id"]: r for key in ("exactRows", "draftRows", "coverageRows") for r in manifest[key]}
                    self.assertFalse(forbidden & {lookup[rid]["sourceGroup"] for rid in train_ids})

    def test_historical_compact_baseline_and_current_dino_paths_are_explicit(self):
        for cohort, kind, historical in (("exact", "tcn", True), ("draft", "tcn", True),
                                         ("reviewed_export", "tcn", False), ("exact", "dino_tcn", False)):
            with tempfile.TemporaryDirectory() as temporary:
                examples, manifest, contract, baseline, previous, audit = fixture(temporary, cohort=cohort, kind=kind)
                root, contract_hash = runner.source_fit_root(baseline, previous, contract["referenceStudy"]["path"])
                self.assertEqual("historical" in root.parts, historical)
                self.assertEqual(contract_hash, "historical-contract" if historical else "current-contract")
                result = runner.run_cell(examples, manifest, contract, "new", baseline, previous, audit)
                self.assertEqual(result["origin"]["sourceOrigin"], baseline["origin"])
                changed = deepcopy(baseline)
                changed["origin"]["fitRoot"] = str(Path(temporary)/"wrong")
                with self.assertRaisesRegex(ValueError, "path changed"):
                    runner.source_fit_root(changed, previous, contract["referenceStudy"]["path"])

    def test_selection_eligibility_ties_noop_and_no_eligible_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            examples, _, contract, _, _, _ = fixture(temporary)
            probabilities = {e.id: np.zeros((len(e.times), 4), np.float32) for e in examples}
            decoder = contract["decoderCandidates"][0]
            # All equal and infeasible: explicit no-op remains first.
            actual = runner.select_rescue(examples, probabilities, decoder)
            self.assertIsNone(actual["keepThreshold"])
            self.assertFalse(actual["recallEligibilityPassed"])
            def scores(pairs):
                return [[{"F1_padP_coreR": f1, "R_core": recall}] for f1, recall in pairs]
            with patch.object(runner, "evaluate_f1_pad_p_core_r", side_effect=scores([(1, .94), (.8, .95), (.8, .99), (.7, 1), (.5, 1)])):
                actual = runner.select_rescue(examples, probabilities, decoder)
            self.assertEqual(actual["optionIndex"], 1)
            self.assertTrue(actual["recallEligibilityPassed"])
            with patch.object(runner, "evaluate_f1_pad_p_core_r", side_effect=scores([(.5, .8), (.9, .94), (.9, .9), (.6, .7), (.4, .8)])):
                actual = runner.select_rescue(examples, probabilities, decoder)
            self.assertEqual(actual["optionIndex"], 1)
            self.assertFalse(actual["recallEligibilityPassed"])
            with self.assertRaisesRegex(ValueError, "grid/order"):
                runner.select_rescue(examples, probabilities, decoder, list(reversed(contract["rescueOptions"])))

    def test_wrong_or_outer_inner_scores_never_reach_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            examples, manifest, contract, baseline, previous, audit = fixture(temporary)
            def leaking(folder, epoch, held, **kwargs):
                scores, evidence = runner.load_bound_predictions(folder, epoch, held, **kwargs)
                scores[examples[0].id] = np.zeros((120, 4), np.float32)
                return scores, evidence
            with patch.object(runner, "select_rescue", side_effect=AssertionError("selection saw outer data")):
                with self.assertRaisesRegex(ValueError, "Inner view"):
                    runner.run_cell(examples, manifest, contract, "new", baseline, previous, audit, load_fn=leaking)

    def test_bound_npz_rejects_changed_bytes_metadata_shape_and_ignored_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            examples, manifest, contract, baseline, previous, audit = fixture(temporary)
            root, sha = runner.source_fit_root(baseline, previous, contract["referenceStudy"]["path"])
            folder = root/"outer-0/inner-0"
            held = [e for e in examples if e.group == "B"]
            kwargs = dict(expected=runner.expected_membership(manifest, "reviewed_export", {"A", "B"}, "B"),
                          kind="dino_tcn", seed=3407, contract_hash=sha, audited_fits=audit)
            scores, _ = runner.load_bound_predictions(folder, 5, held, **kwargs)
            pp, completed = folder/"predictions-5.npz", folder/"completed.json"
            original_meta = runner.read(completed)
            pp.write_bytes(pp.read_bytes()+b"tamper")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                runner.load_bound_predictions(folder, 5, held, **kwargs)
            for malformed in (np.zeros((119, 4), np.float32), np.ones((120, 4), np.float32)):
                changed = {**scores, held[0].id: malformed}
                np.savez_compressed(pp, **changed)
                meta = deepcopy(original_meta)
                meta["artifacts"][pp.name] = runner.digest(pp)
                completed.write_text(json.dumps(meta))
                audit[completed.resolve()] = runner.digest(completed)
                with self.assertRaisesRegex(ValueError, "Invalid baseline"):
                    runner.load_bound_predictions(folder, 5, held, **kwargs)
            meta = deepcopy(original_meta)
            meta["seed"] = 7
            completed.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "prior tensor audit"):
                runner.load_bound_predictions(folder, 5, held, **kwargs)

    def test_reference_wrapper_preserves_full_payload_and_nested_historical_origin(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, contract, baseline, _, _ = fixture(temporary, cohort="exact", kind="tcn")
            saved = deepcopy(baseline)
            wrapper = runner.reference_result(baseline, contract["referenceStudy"], "new")
            self.assertEqual(baseline, saved)
            for key, value in baseline.items():
                if key not in ("origin", "contractSha256"):
                    self.assertEqual(wrapper[key], value)
            self.assertEqual(wrapper["origin"]["sourceOrigin"], baseline["origin"])
            wrapper["origin"]["sourceOrigin"]["fitRoot"] = "changed"
            self.assertEqual(baseline, saved)

    def test_exact_loader_does_not_need_feature_values_or_dino_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            examples, manifest, _, _, _, _ = fixture(root)
            for row, e in zip(manifest["exactRows"], examples):
                cache = root/f"{e.id}.npz"
                np.savez(cache, times=e.times, metadata_json=np.asarray(json.dumps({"duration": e.duration})))
                row["featureCaches"] = {"audiovisual": runner.identity(cache)}
            loaded = runner.load_evaluation_examples(manifest)
            self.assertEqual([e.id for e in loaded], [e.id for e in examples])
            for a, b in zip(loaded, examples):
                np.testing.assert_array_equal(a.times, b.times)
                np.testing.assert_array_equal(a.valid, b.valid)
                self.assertEqual(a.values.shape, (120, 0))
                self.assertEqual(a.truth, b.truth)

    def test_baseline_noop_replay_refuses_altered_reference_intervals(self):
        with tempfile.TemporaryDirectory() as temporary:
            examples, manifest, contract, baseline, previous, audit = fixture(temporary)
            baseline["predictions"][0]["predictions"] = [{"start": 1., "end": 2.}]
            with self.assertRaisesRegex(ValueError, "No-op baseline replay"):
                runner.run_cell(examples, manifest, contract, "new", baseline, previous, audit)

    def test_registration_requires_complete_reference_both_audits_and_bound_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, repo = gate_fixture(temporary)
            original = runner.read(path)
            def replace_registration(contract):
                path.write_text(json.dumps({"contract": contract, "sha256": runner.canonical_hash(contract)}))
            with patch.object(runner, "REPO", repo):
                registration, manifest, report, fits = runner.validate_registration(path)
                self.assertEqual(registration, original)
                self.assertEqual((len(manifest["exactRows"]), len(report["results"]), len(fits)), (8, 54, 864))
                # Even self-consistent rehashing cannot reinterpret scope/recipe.
                for key, value, message in (("rescueOptions", [.35, None, .5, .65, .8], "recipe"),
                                            ("cohorts", ["exact", "exact"], "scope"),
                                            ("targetPaddingSeconds", 1, "changes")):
                    changed = deepcopy(original["contract"])
                    changed[key] = value
                    replace_registration(changed)
                    with self.assertRaisesRegex(ValueError, message):
                        runner.validate_registration(path)
                replace_registration(original["contract"])
                preflight_path = Path(original["contract"]["preflight"]["path"])
                preflight = runner.read(preflight_path)
                for key, value in (("kind", "unrelated-preflight"), ("selectionIsolationQualified", False), ("referenceReport", {"path": "wrong", "sha256": "wrong"}),
                                   ("referenceAudits", []), ("executionEnvironment", {"device": "other"})):
                    preflight_path.write_text(json.dumps({**preflight, key: value}))
                    changed = deepcopy(original["contract"])
                    changed["preflight"] = runner.identity(preflight_path)
                    replace_registration(changed)
                    with self.assertRaisesRegex(ValueError, "preflight binding"):
                        runner.validate_registration(path)
                # A changed source is detected independently of unchanged registration bytes.
                preflight_path.write_text(json.dumps(preflight, indent=2)+"\n")
                replace_registration(original["contract"])
                (repo/"analysis"/"frozen-0.py").write_text("# altered")
                with self.assertRaisesRegex(ValueError, "source changed"):
                    runner.validate_registration(path)

    def test_reference_gate_rejects_duplicate_missing_or_unrelated_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = gate_fixture(temporary)
            reference = runner.read(path)["contract"]["referenceStudy"]
            for mutation in ("duplicate-audit", "wrong-kind", "unrelated-report", "duplicate-fit", "missing-cell"):
                changed = deepcopy(reference)
                if mutation == "duplicate-audit":
                    changed["independentAudits"][1] = changed["independentAudits"][0]
                elif mutation == "missing-cell":
                    target = Path(changed["path"])/"report.json"
                    value = runner.read(target)
                    value["results"].pop()
                    other = target.with_name("incomplete-report.json")
                    runner.write_immutable(other, value)
                    # Keep the normal path and restore afterwards; hash alone is insufficient.
                    raw = target.read_bytes()
                    target.write_bytes(other.read_bytes())
                    changed["reportSha256"] = runner.digest(target)
                else:
                    old = changed["independentAudits"][0]
                    value = runner.read(old["path"])
                    if mutation == "wrong-kind":
                        value["kind"] = "unknown-audit"
                    elif mutation == "unrelated-report":
                        value["report"]["sha256"] = "wrong"
                    else:
                        value["fits"][-1] = value["fits"][0]
                    new = Path(temporary)/f"{mutation}.json"
                    runner.write_immutable(new, value)
                    changed["independentAudits"][0] = runner.identity(new)
                try:
                    with self.assertRaises(ValueError, msg=mutation):
                        runner.completed_reference(changed)
                finally:
                    if mutation == "missing-cell":
                        target.write_bytes(raw)

    def test_full_cpu_orchestration_creates_six_candidates_six_references_without_fits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report, audit = {"results": []}, {}
            for kind in runner.KINDS:
                for seed in (3407, 1729, 20260918):
                    examples, manifest, contract, baseline, previous, bound = fixture(root, cohort="exact", kind=kind, seed=seed)
                    report["results"].append(baseline)
                    audit.update(bound)
            contract["seeds"] = [3407, 1729, 20260918]
            contract["manifestSha256"] = "synthetic-manifest"
            previous_path = root/"reference/preregistration.json"
            runner.write_immutable(previous_path, {"contract": previous})
            contract["referenceStudy"]["preregistrationFileSha256"] = runner.digest(previous_path)
            registration = {"contract": contract, "sha256": "synthetic-new-contract"}
            registration_path = root/"new/preregistration.json"
            runner.write_immutable(registration_path, registration)
            with patch.object(runner, "validate_registration", return_value=(registration, manifest, report, audit)), \
                    patch.object(runner, "load_evaluation_examples", return_value=examples), \
                    patch.object(runner.source, "fit_model", side_effect=AssertionError("decoder experiment tried fitting")):
                result = runner.run_study(registration_path)
                self.assertEqual(result["status"], "completed-keep-rescue-development")
                self.assertEqual(result["execution"]["newFits"], 0)
                self.assertEqual(result["execution"]["candidateCells"], 6)
                self.assertEqual(result["execution"]["referenceCells"], 6)
                self.assertEqual(len(result["results"]), 12)
                self.assertEqual(len(list(registration_path.parent.glob("result-*.json"))), 12)
                self.assertEqual(len(list(registration_path.parent.glob("execution-*-completed.json"))), 1)
                with self.assertRaisesRegex(ValueError, "Completed rescue"):
                    runner.run_study(registration_path)


if __name__ == "__main__":
    unittest.main()

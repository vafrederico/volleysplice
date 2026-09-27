"""Synthetic manual accounting and metamorphic tests; no model quality reads."""
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/audit-neural-combination-accounting.py"
SPEC = importlib.util.spec_from_file_location("combination_accounting_audit_test", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def intervals(*pairs):
    return [{"start": start, "end": end} for start, end in pairs]


def record(identifier="manual", group="A", duration=20, rallies=((5, 10),),
           predictions=((8, 13),), ignored=((0, 2),)):
    return {"id": identifier, "sourceGroup": group, "durationSeconds": duration,
            "rallies": intervals(*rallies), "predictions": intervals(*predictions),
            "ignoredIntervals": intervals(*ignored)}


def expected(pad, *, model, human, valid, tn, fn, fp, core, retained):
    intersection = model - fp
    precision, recall = intersection / model if model else 0, retained / core if core else 0
    return {"paddingSecondsBeforeAndAfter": pad, "joinGapSeconds": 3,
            "P_pad": precision, "R_core": recall,
            "F1_padP_coreR": 2 * precision * recall / (precision + recall) if precision + recall else 0,
            "paddedModelExportSeconds": model, "paddedHumanExportSeconds": human,
            "exportDurationDifferenceSeconds": model - human, "evaluableVideoSeconds": valid,
            "correctlyRemovedSeconds": tn, "incorrectlyRemovedSeconds": fn,
            "incorrectExportSeconds": fp, "missedCoreSeconds": core - retained,
            "coreHumanSeconds": core, "paddedIntersectionSeconds": intersection,
            "coreIntersectionSeconds": retained}


def manual_rows():
    return [expected(p, model=5 + 2*p, human=5 + 2*p, valid=18, tn=10 - 2*p,
                     fn=3, fp=3, core=5, retained=2 + p) for p in range(4)]


class CombinationAccountingAuditTests(unittest.TestCase):
    def test_manual_tn_fn_fp_and_core_accounting_at_all_padding_cases(self):
        records, rows = [record()], manual_rows()
        before = deepcopy((records, rows))
        result = audit.audit_duration_records(records, rows)
        self.assertTrue(result["passed"])
        self.assertEqual(result["metricValuesCompared"], 56)
        self.assertEqual(result["partitionIdentitiesChecked"], 56)
        self.assertEqual((records, rows), before)

    def test_gap_exactly_three_stays_cut_but_next_float_below_joins(self):
        self.assertEqual(audit._canonical_export([(1, 2), (5, 6)], [], 20, 0), [(1, 2), (5, 6)])
        near = math.nextafter(5.0, -math.inf)
        self.assertEqual(audit._canonical_export([(1, 2), (near, 6)], [], 20, 0), [(1, 6)])
        self.assertEqual(audit._canonical_export([(1, 2), (2, 3)], [], 20, 0), [(1, 3)])

    def test_ignored_spans_are_subtracted_after_join_and_never_rejoined(self):
        result = audit._canonical_export([(1, 4), (6, 10)], [(4, 6)], 20, 0)
        self.assertEqual(result, [(1, 4), (6, 10)])
        self.assertEqual(sum(end-start for start, end in result), 7)

    def test_clip_before_padding_and_clip_padded_bounds(self):
        normalized = audit._intervals(intervals((-5, 1), (9, 15), (-4, -2), (11, 14)), 10, "test")
        self.assertEqual(normalized, [(0, 1), (9, 10)])
        self.assertEqual(audit._canonical_export(normalized, [], 10, 2), [(0, 3), (7, 10)])
        self.assertEqual(audit._canonical_export(normalized, [], 10, 3), [(0, 10)])

    def test_micro_pooling_not_mean_per_video_recall_or_f1(self):
        records = [record("large", "A", 100, ((0, 90),), ((0, 45),), ()),
                   record("small", "B", 100, ((0, 10),), ((0, 10),), ())]
        rows = [expected(p, model=55+2*p, human=100+2*p, valid=200, tn=100-2*p,
                         fn=45, fp=0, core=100, retained=55+p) for p in range(4)]
        self.assertEqual(audit.audit_duration_records(records, rows)["sourceGroupCount"], 2)
        rows[0]["R_core"] = .75
        with self.assertRaisesRegex(ValueError, "R_core"):
            audit.audit_duration_records(records, rows)

    def test_final_export_overrides_are_not_padded_again(self):
        overrides = {"manual": {str(p): intervals((8, 13)) for p in range(4)}}
        rows = [expected(p, model=5, human=5+2*p, valid=18, tn=10-p,
                         fn=3+p, fp=3-p, core=5, retained=2) for p in range(4)]
        result = audit.audit_duration_records([record()], rows, overrides)
        self.assertEqual(result["overrideExportRowsAudited"], 4)
        with self.assertRaises(ValueError):
            audit.audit_duration_records([record()], rows)

    def test_final_export_overrides_preserve_short_suppression_gaps(self):
        data = record(duration=20, rallies=((0, 20),), predictions=((0, 20),), ignored=((8, 9),))
        overrides = {"manual": {str(p): intervals((1, 2), (3, 4), (7, 10)) for p in range(4)}}
        rows = [expected(p, model=4, human=19, valid=19, tn=0, fn=15, fp=0, core=19, retained=4)
                for p in range(4)]
        self.assertTrue(audit.audit_duration_records([data], rows, overrides)["passed"])

    def test_numeric_pairs_work_in_records_and_final_export_overrides(self):
        data = record()
        for key in ("rallies", "predictions", "ignoredIntervals"):
            data[key] = [[row["start"], row["end"]] for row in data[key]]
        self.assertTrue(audit.audit_duration_records([data], manual_rows())["passed"])
        overrides = {"manual": {str(p): [(8, 13)] for p in range(4)}}
        rows = [expected(p, model=5, human=5+2*p, valid=18, tn=10-p,
                         fn=3+p, fp=3-p, core=5, retained=2) for p in range(4)]
        self.assertTrue(audit.audit_duration_records([data], rows, overrides)["passed"])

    def test_invalid_numeric_pairs_are_rejected_for_records_and_overrides(self):
        for bad in ([], [1], [1, 2, 3], "12", [1, math.nan], [math.inf, 2],
                    [4, 3], [2, 2], [True, 2], [1, "2"]):
            with self.subTest(pair=bad):
                data = record()
                data["predictions"] = [bad]
                with self.assertRaises(ValueError):
                    audit.audit_duration_records([data], manual_rows())
                overrides = {"manual": {str(p): [bad] for p in range(4)}}
                with self.assertRaises(ValueError):
                    audit.audit_duration_records([record()], manual_rows(), overrides)

    def test_order_overlap_and_duplicate_interval_invariance(self):
        data = record()
        data["predictions"] = intervals((11, 13), (8, 12), (8, 12))
        data["rallies"] = intervals((8, 10), (5, 8), (5, 8))
        data["ignoredIntervals"] = intervals((1, 2), (0, 1.5), (0, 1.5))
        rows = list(reversed(manual_rows()))
        rows[0]["unrelatedDiagnostic"] = "ignored"
        self.assertTrue(audit.audit_duration_records([data], rows)["passed"])

    def test_empty_prediction_export_has_zero_precision_recall_and_f1(self):
        data = record(predictions=())
        rows = [expected(p, model=0, human=5+2*p, valid=18, tn=13-2*p,
                         fn=5+2*p, fp=0, core=5, retained=0) for p in range(4)]
        self.assertTrue(audit.audit_duration_records([data], rows)["passed"])

    def test_each_required_metric_tamper_is_rejected(self):
        for field in audit.METRIC_FIELDS:
            with self.subTest(field=field):
                rows = manual_rows()
                rows[2][field] += .001
                with self.assertRaisesRegex(ValueError, field):
                    audit.audit_duration_records([record()], rows)

    def test_missing_duplicate_padding_or_wrong_join_threshold_is_rejected(self):
        candidates = [manual_rows()[:-1], manual_rows(), manual_rows(), manual_rows()]
        candidates[1][3]["paddingSecondsBeforeAndAfter"] = 2
        del candidates[2][0]["missedCoreSeconds"]
        candidates[3][2]["joinGapSeconds"] = .5
        for rows in candidates:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                audit.audit_duration_records([record()], rows)

    def test_override_scope_cannot_silently_fall_back_to_canonical_exports(self):
        for overrides in ({}, {"unknown": {}}, {"manual": {"2": []}},
                          {"manual": {p: [] for p in range(4)}}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                audit.audit_duration_records([record()], manual_rows(), overrides)

    def test_duplicate_recordings_nonfinite_and_reversed_boundaries_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            audit.audit_duration_records([record(), record()], manual_rows())
        for start, end in ((math.nan, 2), (1, math.inf), (4, 3), (2, 2), (True, 2)):
            data = record()
            data["predictions"] = intervals((start, end))
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                audit.audit_duration_records([data], manual_rows())


if __name__ == "__main__":
    unittest.main()

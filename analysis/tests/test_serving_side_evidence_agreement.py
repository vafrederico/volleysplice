"""Agreement boundaries and source isolation for the feature-family control."""
from copy import deepcopy
import unittest

import numpy as np

from analysis import serving_side_evidence_agreement as study
from analysis.serving_side_v2 import LogisticModel


class EvidenceAgreementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [{"rowIndex": i, "sourceGroup": f"synthetic-{i // 121}", "label": i % 2}
                    for i in range(968)]
        cls.features = np.random.default_rng(17).normal(size=(968, 237))
        cls.first = study.fit_predict(cls.rows, cls.features, ("synthetic-0",))

    def test_confidence_is_weakest_agreeing_family_and_ties_route_near(self):
        probabilities = np.array([
            [.9, .8, .7], [.1, .2, .3], [.9, .4, .8], [.1, .2, .5],
            [.5, .5, .7], [.5, .49, .7], [1., 1., 1.], [0., 0., 0.],
        ])
        scores = study.agreement_scores(probabilities)
        np.testing.assert_array_equal(scores[:, 0], probabilities[:, 0])
        confidence = scores[:, 1] * np.maximum(scores[:, 0], 1 - scores[:, 0])
        np.testing.assert_allclose(confidence, [.7, .7, 0., 0., .5, 0., 1., 1.], rtol=0, atol=1e-15)
        self.assertTrue(np.all((scores[:, 1] >= 0) & (scores[:, 1] <= 1)))
        self.assertEqual(study.agreement_scores(np.empty((0, 3))).shape, (0, 2))

    def test_invalid_probability_arrays_fail_closed(self):
        for values in (np.zeros(3), np.zeros((2, 2)), np.array([[.5, np.nan, .5]]),
                       np.array([[.5, np.inf, .5]]), np.array([[1.1, .5, .5]]),
                       np.array([[-.1, .5, .5]])):
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                study.agreement_scores(values)

    def test_models_fit_only_their_training_columns_and_reproduce_scores(self):
        self.assertEqual(self.first["trainingRows"], list(range(121, 968)))
        self.assertEqual(self.first["heldRows"], list(range(121)))
        for column, (name, feature_slice) in enumerate(zip(study.FAMILY_NAMES,
                                                        (slice(None), slice(82), slice(82, None)), strict=True)):
            model = LogisticModel.from_dict(self.first["models"][name])
            self.assertEqual(model.l2, .1)
            np.testing.assert_allclose(model.mean, self.features[121:, feature_slice].mean(axis=0), rtol=0, atol=1e-15)
            np.testing.assert_array_equal(model.predict_proba(self.features[:121, feature_slice]),
                                          self.first["familyProbabilities"][:, column])
        np.testing.assert_array_equal(self.first["scores"], study.agreement_scores(self.first["familyProbabilities"]))

    def test_changed_held_labels_and_features_cannot_change_any_fitted_model(self):
        rows, features = deepcopy(self.rows), self.features.copy()
        features[:121] = 10000.
        for row in rows[:121]:
            row["label"] = 1 - row["label"]
        changed = study.fit_predict(rows, features, ("synthetic-0",))
        self.assertEqual(changed["models"], self.first["models"])
        self.assertEqual(changed["trainingRows"], self.first["trainingRows"])

    def test_empty_exclusion_delivers_all_development_predictions(self):
        final = study.fit_predict(self.rows, self.features, ())
        self.assertEqual(final["trainingRows"], list(range(968)))
        self.assertEqual(final["heldRows"], list(range(968)))
        self.assertEqual(final["scores"].shape, (968, 2))
        model = LogisticModel.from_dict(final["models"]["full"])
        np.testing.assert_array_equal(final["scores"][:, 0], model.predict_proba(self.features))

    def test_invalid_inputs_and_partitions_fail_closed(self):
        invalid_features = [self.features.astype(np.float32), self.features[:967], self.features[:, :236]]
        nonfinite = self.features.copy(); nonfinite[0, 0] = np.nan
        invalid_features.append(nonfinite)
        for values in invalid_features:
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                study.fit_predict(self.rows, values, ("synthetic-0",))
        for excluded in (("missing",), ("synthetic-0", "synthetic-0"), "synthetic-0",
                         tuple(f"synthetic-{i}" for i in range(8))):
            with self.subTest(excluded=excluded), self.assertRaises(ValueError):
                study.fit_predict(self.rows, self.features, excluded)
        for field, value in (("label", "unclear"), ("sourceGroup", ""), ("rowIndex", 1), ("sourceSplit", "test")):
            rows = deepcopy(self.rows); rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                study.fit_predict(rows, self.features, ("synthetic-0",))
        one_class = deepcopy(self.rows)
        for row in one_class[121:]:
            row["label"] = 1
        with self.assertRaisesRegex(ValueError, "both serving sides"):
            study.fit_predict(one_class, self.features, ("synthetic-0",))


if __name__ == "__main__":
    unittest.main()

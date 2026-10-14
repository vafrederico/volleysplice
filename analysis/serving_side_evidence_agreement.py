"""Source-isolated agreement control for the frozen serving-side population.

The three independently fitted feature-family models vote on the full model's
side. The exported score columns are near probability ``q`` and acceptance
factor ``a``; ``a * max(q, 1-q)`` is the weakest agreeing family probability,
or zero when a family disagrees. This is a deterministic confidence score,
not an independently calibrated probability of correctness.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .serving_side_v2 import fit_logistic


ROW_COUNT = 968
FEATURE_COUNT = 237
COURT_FEATURE_COUNT = 82
L2 = 0.1
FAMILY_NAMES = ("full", "court", "flight")


def agreement_scores(family_probabilities: np.ndarray) -> np.ndarray:
    """Return ``[q, a]`` from near probabilities ordered full/court/flight.

    Every family routes exact 0.5 ties to near. Therefore a tied family
    disagrees with a far full-model prediction, and agrees with a near one.
    """
    probabilities = np.asarray(family_probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(FAMILY_NAMES):
        raise ValueError("Family probabilities must have shape [N, 3]")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Family probabilities must be finite values in [0, 1]")
    near = probabilities[:, 0] >= 0.5
    agree = np.all((probabilities >= 0.5) == near[:, None], axis=1)
    class_probabilities = np.where(near[:, None], probabilities, 1.0 - probabilities)
    confidence = np.where(agree, class_probabilities.min(axis=1), 0.0)
    # The full-model class probability is always at least 0.5, including ties.
    acceptance = confidence / class_probabilities[:, 0]
    return np.column_stack((probabilities[:, 0], acceptance))


def fit_predict(
    rows: Sequence[Mapping[str, Any]],
    features: np.ndarray,
    excluded: Sequence[str],
) -> dict[str, Any]:
    """Fit fixed-L2 family models without any excluded-source supervision.

    An empty exclusion sequence fits all development rows and returns their
    predictions solely for final-model delivery; those are in-sample outputs.
    Feature ordering must match the frozen specialist contract: 82 court-flow
    columns followed by 155 flight columns. This function performs no I/O.
    """
    if not isinstance(features, np.ndarray) or features.dtype != np.float64:
        raise ValueError("Features must be a float64 NumPy matrix")
    if len(rows) != ROW_COUNT or features.shape != (ROW_COUNT, FEATURE_COUNT):
        raise ValueError("Frozen population requires exactly 968 rows and 237 features")
    if not np.isfinite(features).all():
        raise ValueError("Features must be finite")
    if isinstance(excluded, (str, bytes)):
        raise ValueError("Excluded groups must be a sequence of source-group names")
    excluded = tuple(excluded)
    if any(not isinstance(group, str) or not group for group in excluded):
        raise ValueError("Excluded source-group names must be nonempty strings")
    if len(set(excluded)) != len(excluded):
        raise ValueError("Duplicate excluded source group")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("Rows must contain metadata mappings")
        if not isinstance(row.get("sourceGroup"), str) or not row["sourceGroup"]:
            raise ValueError("Every row needs a source group")
        if row.get("label") not in (0, 1):
            raise ValueError("Every row needs a known binary near/far label")
        if "rowIndex" in row and row["rowIndex"] != index:
            raise ValueError("Frozen row order differs")
        if row.get("sourceSplit") == "test":
            raise ValueError("Protected test rows cannot enter development fitting")
    groups = {row["sourceGroup"] for row in rows}
    if not set(excluded).issubset(groups):
        raise ValueError("Unknown excluded source group")
    train = np.asarray([i for i, row in enumerate(rows) if row["sourceGroup"] not in excluded], dtype=np.int64)
    held = (np.asarray([i for i, row in enumerate(rows) if row["sourceGroup"] in excluded], dtype=np.int64)
            if excluded else np.arange(ROW_COUNT, dtype=np.int64))
    if not len(train) or not len(held):
        raise ValueError("Training and prediction partitions must be nonempty")
    if excluded and {rows[i]["sourceGroup"] for i in train}.intersection(excluded):
        raise ValueError("Held source entered the training partition")
    truth = np.asarray([rows[i]["label"] for i in train], dtype=np.int64)
    if set(truth.tolist()) != {0, 1}:
        raise ValueError("Training partition needs both serving sides")

    slices = (slice(None), slice(0, COURT_FEATURE_COUNT), slice(COURT_FEATURE_COUNT, None))
    probabilities, models = [], {}
    for name, columns in zip(FAMILY_NAMES, slices, strict=True):
        model = fit_logistic(features[train, columns], truth, l2=L2)
        probabilities.append(model.predict_proba(features[held, columns]))
        models[name] = model.to_dict()
    family_probabilities = np.column_stack(probabilities)
    return {
        "trainingRows": train.tolist(),
        "heldRows": held.tolist(),
        "scores": agreement_scores(family_probabilities),
        "familyProbabilities": family_probabilities,
        "models": models,
    }

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from .ball_annotation import BALL_ANNOTATION_TASK_TYPE, BallAnnotationError, validate_ball_annotation_task


class BallReviewRepairError(BallAnnotationError):
    """Raised when a review cannot be safely rebuilt from its pristine task."""


def _sha256(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _reject_constant(value: str) -> None:
    raise BallReviewRepairError(f"non-finite JSON number {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BallReviewRepairError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _load_json(source: str, *, decimal_numbers: bool = False) -> Any:
    options: dict[str, Any] = {
        "object_pairs_hook": _unique_object,
        "parse_constant": _reject_constant,
    }
    if decimal_numbers:
        options.update(parse_int=Decimal, parse_float=Decimal)
    try:
        return json.loads(source, **options)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise BallReviewRepairError(f"invalid JSON: {error}") from error


def _top_level_member_spans(source: str) -> dict[str, tuple[int, int]]:
    decoder = json.JSONDecoder(parse_constant=_reject_constant)
    length = len(source)

    def skip(offset: int) -> int:
        while offset < length and source[offset] in " \t\r\n":
            offset += 1
        return offset

    offset = skip(0)
    if offset >= length or source[offset] != "{":
        raise BallReviewRepairError("JSON root must be an object")
    offset = skip(offset + 1)
    spans: dict[str, tuple[int, int]] = {}
    if offset < length and source[offset] == "}":
        offset = skip(offset + 1)
        if offset != length:
            raise BallReviewRepairError("JSON has trailing data")
        return spans
    while True:
        try:
            key, key_end = decoder.raw_decode(source, offset)
        except json.JSONDecodeError as error:
            raise BallReviewRepairError(f"invalid JSON object key: {error}") from error
        if not isinstance(key, str):
            raise BallReviewRepairError("JSON object keys must be strings")
        if key in spans:
            raise BallReviewRepairError(f"duplicate top-level JSON key {key!r}")
        offset = skip(key_end)
        if offset >= length or source[offset] != ":":
            raise BallReviewRepairError("JSON object member is missing ':'")
        value_start = skip(offset + 1)
        try:
            _, value_end = decoder.raw_decode(source, value_start)
        except json.JSONDecodeError as error:
            raise BallReviewRepairError(f"invalid JSON value for {key!r}: {error}") from error
        spans[key] = (value_start, value_end)
        offset = skip(value_end)
        if offset < length and source[offset] == ",":
            offset = skip(offset + 1)
            continue
        if offset < length and source[offset] == "}":
            offset = skip(offset + 1)
            if offset != length:
                raise BallReviewRepairError("JSON has trailing data")
            return spans
        raise BallReviewRepairError("JSON root object is incomplete")


def _same_json_value(expected: Any, actual: Any) -> bool:
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(expected) == set(actual)
            and all(_same_json_value(expected[key], actual[key]) for key in expected)
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(_same_json_value(left, right) for left, right in zip(expected, actual))
        )
    if isinstance(expected, Decimal):
        return isinstance(actual, Decimal) and expected == actual
    return type(expected) is type(actual) and expected == actual


def _require_exact_keys(value: Any, expected: set[str], where: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise BallReviewRepairError(f"{where} contains missing or unsupported fields")


def _validate_mutable_shape(review: Any) -> None:
    _require_exact_keys(
        review,
        {"schemaVersion", "taskType", "immutable", "suggestions", "annotations"},
        "review",
    )
    _require_exact_keys(review["suggestions"], {"status", "model", "frames"}, "suggestions")
    _require_exact_keys(review["annotations"], {"review", "frames"}, "annotations")
    _require_exact_keys(
        review["annotations"]["review"],
        {"status", "annotator", "reviewedAt", "notes"},
        "annotations.review",
    )


def _safe_child(parent: Path, candidate: Path, where: str) -> Path:
    resolved = candidate.resolve()
    try:
        resolved.relative_to(parent.resolve())
    except ValueError as error:
        raise BallReviewRepairError(f"{where} escapes its expected directory") from error
    return resolved


def canonicalize_ball_review_immutables(
    pilot_root: str | Path,
    output_directory: str | Path,
    *,
    reviews_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild review files with pristine immutable JSON, without touching their source.

    Only the top-level ``immutable`` value is replaced. Integral-float lexical drift is
    accepted, while any mathematically different value, extra immutable key, duplicate
    key, unsupported mutable field, or invalid annotation aborts the entire operation.
    """

    root = Path(pilot_root).expanduser().resolve()
    reviews = (
        Path(reviews_directory).expanduser().resolve()
        if reviews_directory is not None
        else root / "reviews"
    )
    output = Path(output_directory).expanduser().resolve()
    if output == reviews or output.parent != reviews.parent:
        raise BallReviewRepairError("output must be a new sibling of the source reviews directory")
    if output.exists():
        raise BallReviewRepairError(f"output already exists: {output}")
    if not reviews.is_dir() or reviews.is_symlink():
        raise BallReviewRepairError(f"source reviews directory is unavailable or unsafe: {reviews}")

    index_path = root / "index.json"
    try:
        index_source = index_path.read_text(encoding="utf-8")
    except OSError as error:
        raise BallReviewRepairError(f"cannot read pilot index {index_path}: {error}") from error
    index = _load_json(index_source)
    if (
        not isinstance(index, dict)
        or index.get("artifactType") != "volleycut-ball-presence-pilot-index"
        or not isinstance(index.get("tasks"), list)
    ):
        raise BallReviewRepairError("pilot index is invalid")

    expected_reviews: dict[str, Mapping[str, Any]] = {}
    for position, row in enumerate(index["tasks"]):
        if not isinstance(row, dict):
            raise BallReviewRepairError(f"pilot index task {position + 1} is invalid")
        recording_id = row.get("recordingId")
        if not isinstance(recording_id, str) or not recording_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in recording_id
        ):
            raise BallReviewRepairError(f"pilot index task {position + 1} has an unsafe recording id")
        filename = f"{recording_id}.ball-presence.json"
        if filename in expected_reviews:
            raise BallReviewRepairError(f"duplicate pilot recording {recording_id!r}")
        expected_reviews[filename] = row

    supplied_reviews = {item.name for item in reviews.glob("*.ball-presence.json") if item.is_file()}
    unexpected = supplied_reviews - set(expected_reviews)
    if unexpected:
        raise BallReviewRepairError(f"reviews directory contains tasks absent from the pilot index: {sorted(unexpected)}")

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    artifacts: list[dict[str, Any]] = []
    try:
        for filename in sorted(supplied_reviews):
            row = expected_reviews[filename]
            review_path = reviews / filename
            if review_path.is_symlink():
                raise BallReviewRepairError(f"review artifact is a symlink: {review_path}")
            task_value = row.get("task")
            if not isinstance(task_value, str):
                raise BallReviewRepairError(f"pilot index task path is invalid for {filename}")
            pristine_path = _safe_child(root / "tasks", root / task_value, "pristine task path")
            if pristine_path.name != filename or pristine_path.is_symlink():
                raise BallReviewRepairError(f"pristine task path is unsafe for {filename}")
            try:
                pristine_bytes = pristine_path.read_bytes()
                review_bytes = review_path.read_bytes()
                pristine_source = pristine_bytes.decode("utf-8")
                review_source = review_bytes.decode("utf-8")
            except (OSError, UnicodeDecodeError) as error:
                raise BallReviewRepairError(f"cannot read {filename}: {error}") from error
            if _sha256(pristine_bytes) != row.get("initialTaskSha256"):
                raise BallReviewRepairError(f"SHA-pinned pristine task changed for {filename}")

            pristine = _load_json(pristine_source)
            review = _load_json(review_source)
            pristine_decimal = _load_json(pristine_source, decimal_numbers=True)
            review_decimal = _load_json(review_source, decimal_numbers=True)
            validate_ball_annotation_task(pristine, task_path=pristine_path, verify_images=False)
            _validate_mutable_shape(review)
            if review.get("taskType") != BALL_ANNOTATION_TASK_TYPE:
                raise BallReviewRepairError(f"review task type is invalid for {filename}")
            if not _same_json_value(pristine_decimal["immutable"], review_decimal["immutable"]):
                raise BallReviewRepairError(f"review immutable provenance changed for {filename}")

            candidate = copy.deepcopy(review)
            candidate["immutable"] = copy.deepcopy(pristine["immutable"])
            validate_ball_annotation_task(candidate, task_path=review_path, verify_images=False)

            pristine_spans = _top_level_member_spans(pristine_source)
            review_spans = _top_level_member_spans(review_source)
            if "immutable" not in pristine_spans or not {"immutable", "annotations"} <= set(review_spans):
                raise BallReviewRepairError(f"required top-level fields are missing in {filename}")
            pristine_start, pristine_end = pristine_spans["immutable"]
            review_start, review_end = review_spans["immutable"]
            pristine_immutable_source = pristine_source[pristine_start:pristine_end]
            repaired_source = (
                review_source[:review_start]
                + pristine_immutable_source
                + review_source[review_end:]
            )
            repaired = _load_json(repaired_source)
            validate_ball_annotation_task(repaired, task_path=review_path, verify_images=False)
            repaired_spans = _top_level_member_spans(repaired_source)
            annotation_start, annotation_end = review_spans["annotations"]
            repaired_annotation_start, repaired_annotation_end = repaired_spans["annotations"]
            annotation_source = review_source[annotation_start:annotation_end]
            if annotation_source != repaired_source[repaired_annotation_start:repaired_annotation_end]:
                raise BallReviewRepairError(f"annotation bytes changed while repairing {filename}")

            destination = temporary / filename
            destination.write_text(repaired_source, encoding="utf-8")
            artifacts.append(
                {
                    "recordingId": row["recordingId"],
                    "taskId": pristine["immutable"]["taskId"],
                    "immutableDigestSha256": pristine["immutable"]["digestSha256"],
                    "pristineTaskSha256": _sha256(pristine_bytes),
                    "sourceReviewSha256": _sha256(review_bytes),
                    "annotationsJsonSha256": _sha256(annotation_source),
                    "outputReviewSha256": _sha256(repaired_source),
                    "filename": filename,
                }
            )

        receipt = {
            "schemaVersion": 1,
            "artifactType": "volleycut-ball-review-immutable-canonicalization",
            "sourcePilotIndexSha256": _sha256(index_source),
            "sourceReviewsPathHint": str(reviews),
            "outputPathHint": str(output),
            "artifactCount": len(artifacts),
            "artifacts": artifacts,
        }
        (temporary / "canonicalization-receipt.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        return receipt
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

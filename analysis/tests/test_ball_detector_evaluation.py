from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from analysis.ball_detector_evaluation import (
    Box,
    Detection,
    EvaluationFrame,
    ProtocolRequirements,
    TruthObject,
    BallDetectorEvaluationError,
    bbox_iou,
    evaluate_ball_detector_tasks,
    evaluate_frames_at_threshold,
    load_completed_reviewed_tasks,
    load_completed_sol_reviews,
    merge_blind_review_with_detector_suggestions,
    prepare_detector_blind_sol_review,
    _threshold_freeze_gate,
    _validate_pilot_index_binding,
)
from analysis.ball_annotation import _immutable_digest, validate_ball_annotation_task
from analysis.tests.test_ball_annotation import task_fixture


def _object(
    bbox: dict[str, float],
    *,
    role: str,
    visibility: str = "clear",
    object_id: str = "ball-1",
) -> dict[str, object]:
    return {
        "id": object_id,
        "category": "volleyball",
        "role": role,
        "bbox": bbox,
        "visibility": visibility,
        "truncated": False,
    }


def completed_task() -> dict[str, object]:
    task = task_fixture()
    frame_ids = list(task["annotations"]["frames"])
    for annotation in task["annotations"]["frames"].values():
        annotation.update(
            {
                "status": "reviewed",
                "primaryBallState": "out_of_frame",
                "objects": [],
                "proposalExposure": "not_shown",
            }
        )
    small = {"x": 0.10, "y": 0.20, "width": 0.01, "height": 0.015}
    medium = {"x": 0.20, "y": 0.30, "width": 0.03, "height": 0.04}
    other = {"x": 0.60, "y": 0.40, "width": 0.02, "height": 0.025}
    task["annotations"]["frames"][frame_ids[0]].update(
        {
            "primaryBallState": "localizable",
            "objects": [_object(small, role="primary-court")],
        }
    )
    task["annotations"]["frames"][frame_ids[1]].update(
        {
            "primaryBallState": "localizable",
            "objects": [
                _object(
                    medium,
                    role="primary-court",
                    visibility="motion-blurred",
                )
            ],
        }
    )
    task["annotations"]["frames"][frame_ids[2]]["objects"] = [
        _object(other, role="other-court")
    ]
    task["annotations"]["review"] = {
        "status": "complete",
        "annotator": "fixture-reviewer",
        "reviewedAt": "2026-08-11T12:00:00+00:00",
        "notes": "fixture",
    }

    suggestions = {
        frame_id: {"ballPresenceProbability": 0.0, "detections": []}
        for frame_id in frame_ids
    }
    for index, (score, bbox) in enumerate(
        ((0.9, small), (0.1, medium), (0.8, other))
    ):
        suggestions[frame_ids[index]] = {
            "ballPresenceProbability": score,
            "detections": [{"confidence": score, "bbox": bbox}],
        }
    false_box = {"x": 0.8, "y": 0.7, "width": 0.02, "height": 0.02}
    suggestions[frame_ids[3]] = {
        "ballPresenceProbability": 0.7,
        "detections": [{"confidence": 0.7, "bbox": false_box}],
    }
    task["suggestions"] = {
        "status": "complete",
        "model": {
            "modelId": "fixture-sports-ball-detector",
            "modelSha256": "c" * 64,
            "modelPath": "/machine-specific/model.onnx",
            "sourceTask": {
                "pathHint": "/machine-specific/task.json",
                "sha256": "d" * 64,
            },
            "settings": {"scoreFloor": 0.01, "nmsThreshold": 0.5},
        },
        "frames": suggestions,
    }
    return task


def _resign(task: dict[str, object]) -> None:
    digest = _immutable_digest(task["immutable"])
    task["immutable"]["digestSha256"] = digest
    task["immutable"]["taskId"] = f"ball-presence-{digest[:24]}"


def blind_merge_fixture(
    root: Path,
) -> tuple[Path, Path, Path, dict[str, object], dict[str, object]]:
    base = completed_task()
    image_path = root / "images" / "frame.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"immutable fixture image")
    image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    for frame in base["immutable"]["frames"]:
        frame["image"] = {
            "path": "../images/frame.png",
            "sha256": image_sha,
            "width": 960,
            "height": 540,
            "format": "png",
        }
    _resign(base)

    reviewed = copy.deepcopy(base)
    reviewed["suggestions"] = {"status": "empty", "model": None, "frames": {}}
    proposals = copy.deepcopy(base)
    proposals["annotations"] = {
        "review": {
            "status": "unreviewed",
            "annotator": None,
            "reviewedAt": None,
            "notes": "",
        },
        "frames": {
            frame_id: {
                "status": "unreviewed",
                "primaryBallState": None,
                "objects": [],
                "notes": "",
            }
            for frame_id in base["annotations"]["frames"]
        },
    }
    reviewed_path = root / "reviewed" / "task.json"
    proposal_path = root / "proposals" / "task.json"
    output_path = root / "merged" / "task.json"
    reviewed_path.parent.mkdir()
    proposal_path.parent.mkdir()
    reviewed_path.write_text(json.dumps(reviewed), encoding="utf-8")
    proposal_path.write_text(json.dumps(proposals), encoding="utf-8")
    return reviewed_path, proposal_path, output_path, reviewed, proposals


def write_merged_evaluation_task(
    root: Path,
    task: dict[str, object],
    name: str = "fixture",
) -> tuple[Path, dict[str, object]]:
    artifact_root = root / name
    payload = copy.deepcopy(task)
    image_path = artifact_root / "images" / "frame.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(f"immutable image {name}".encode("utf-8"))
    image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    for frame in payload["immutable"]["frames"]:
        frame["image"] = {
            "path": "../images/frame.png",
            "sha256": image_sha,
            "width": 960,
            "height": 540,
            "format": "png",
        }
    _resign(payload)
    initial = copy.deepcopy(payload)
    initial["suggestions"] = {"status": "empty", "model": None, "frames": {}}
    initial["annotations"] = {
        "review": {
            "status": "unreviewed",
            "annotator": None,
            "reviewedAt": None,
            "notes": "",
        },
        "frames": {
            frame_id: {
                "status": "unreviewed",
                "primaryBallState": None,
                "objects": [],
                "notes": "",
            }
            for frame_id in payload["annotations"]["frames"]
        },
    }
    initial_path = artifact_root / "initial" / "task.json"
    initial_path.parent.mkdir()
    initial_path.write_text(json.dumps(initial), encoding="utf-8")
    payload["suggestions"]["model"]["sourceTask"] = {
        "pathHint": str(initial_path.resolve()),
        "sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest(),
    }
    reviewed = copy.deepcopy(payload)
    reviewed["suggestions"] = {"status": "empty", "model": None, "frames": {}}
    proposals = copy.deepcopy(payload)
    proposals["annotations"] = {
        "review": {
            "status": "unreviewed",
            "annotator": None,
            "reviewedAt": None,
            "notes": "",
        },
        "frames": {
            frame_id: {
                "status": "unreviewed",
                "primaryBallState": None,
                "objects": [],
                "notes": "",
            }
            for frame_id in payload["annotations"]["frames"]
        },
    }
    reviewed_path = artifact_root / "reviewed" / "task.json"
    proposal_path = artifact_root / "proposals" / "task.json"
    merged_path = artifact_root / "merged" / "task.json"
    reviewed_path.parent.mkdir()
    proposal_path.parent.mkdir()
    reviewed_path.write_text(json.dumps(reviewed), encoding="utf-8")
    proposal_path.write_text(json.dumps(proposals), encoding="utf-8")
    merge_blind_review_with_detector_suggestions(
        reviewed_path,
        proposal_path,
        merged_path,
    )
    return merged_path, payload


def write_completed_sol_review(
    root: Path,
    human_task: dict[str, object],
    *,
    name: str = "fixture",
    mutate: object | None = None,
) -> Path:
    source_path = Path(
        human_task["suggestions"]["model"]["sourceTask"]["pathHint"]
    )
    index_path = root / name / "index.json"
    if not index_path.exists():
        index_path.write_text(
            json.dumps(pilot_index_for_tasks([human_task])),
            encoding="utf-8",
        )
    output_path = root / name / "sol" / "task.json"
    prepare_detector_blind_sol_review(
        source_path,
        output_path,
        pilot_index_path=index_path,
        agent_id="sol-labeler-agent",
        model_id="gpt-5.6-sol",
        run_id="fixture-run-001",
    )
    sol = json.loads(output_path.read_text(encoding="utf-8"))
    sol_frames = {}
    for frame_id, annotation in human_task["annotations"]["frames"].items():
        copied = copy.deepcopy(annotation)
        copied.pop("proposalExposure", None)
        sol_frames[frame_id] = copied
    sol["annotations"] = {
        "review": {
            "status": "complete",
            "annotator": "sol-labeler-agent",
            "reviewedAt": sol["solReviewProvenance"]["preparedAt"],
            "notes": "detector-blind Sol fixture",
        },
        "frames": sol_frames,
    }
    if mutate is not None:
        mutate(sol)
    output_path.write_text(json.dumps(sol), encoding="utf-8")
    return output_path


def pilot_index_for_tasks(tasks: list[dict[str, object]]) -> dict[str, object]:
    first = tasks[0]
    return {
        "schemaVersion": 1,
        "artifactType": "volleycut-ball-presence-pilot-index",
        "manifest": {"sha256": first["immutable"]["manifest"]["sha256"]},
        "samplingPolicyId": first["immutable"]["sampling"]["policyId"],
        "round": first["immutable"]["sampling"]["round"],
        "developmentOnly": True,
        "recordingCount": len(tasks),
        "windowCount": sum(len(task["immutable"]["windows"]) for task in tasks),
        "frameCount": sum(len(task["immutable"]["frames"]) for task in tasks),
        "tasks": [
            {
                "recordingId": task["immutable"]["recording"]["id"],
                "split": task["immutable"]["recording"]["split"],
                "taskId": task["immutable"]["taskId"],
                "frameCount": len(task["immutable"]["frames"]),
                "initialTaskSha256": task["suggestions"]["model"]["sourceTask"][
                    "sha256"
                ],
            }
            for task in tasks
        ],
    }


def minimal_requirements() -> ProtocolRequirements:
    return ProtocolRequirements(
        min_recordings=1,
        min_source_groups=1,
        min_windows=1,
        min_reviewed_frames=1,
        min_primary_positive_frames=1,
        min_any_ball_positive_frames=1,
        min_ball_free_frames=1,
        min_primary_positive_windows=1,
        min_any_ball_positive_windows=1,
        min_ball_free_windows=1,
        min_primary_positive_source_groups=1,
        min_ball_free_source_groups=1,
    )


class BallDetectorMetricTests(unittest.TestCase):
    def test_iou_and_role_aware_point_metrics(self) -> None:
        self.assertEqual(bbox_iou(Box(0, 0, 1, 1), Box(0, 0, 1, 1)), 1.0)
        self.assertEqual(bbox_iou(Box(0, 0, 0.1, 0.1), Box(0.2, 0.2, 0.1, 0.1)), 0.0)
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = write_merged_evaluation_task(
                Path(temporary), completed_task()
            )
            loaded = load_completed_reviewed_tasks([path])
            metrics = evaluate_frames_at_threshold(loaded.frames, 0.5)

        primary = metrics["framePresence"]["primaryLocalizable"]
        self.assertEqual(
            (
                primary["truePositive"],
                primary["falsePositive"],
                primary["falseNegative"],
            ),
            (1, 2, 1),
        )
        self.assertAlmostEqual(primary["precision"], 1 / 3)
        self.assertAlmostEqual(primary["recall"], 1 / 2)
        self.assertAlmostEqual(primary["f1"], 0.4)
        any_ball = metrics["framePresence"]["anyAnnotatedRealBall"]
        self.assertEqual(
            (
                any_ball["truePositive"],
                any_ball["falsePositive"],
                any_ball["falseNegative"],
            ),
            (2, 1, 1),
        )
        self.assertAlmostEqual(any_ball["f1"], 2 / 3)
        self.assertAlmostEqual(
            metrics["calibration"]["primaryLocalizableBrier"], 1.95 / 270
        )
        self.assertAlmostEqual(
            metrics["calibration"]["anyAnnotatedRealBallBrier"], 1.35 / 270
        )

        primary_iou = metrics["localization"]["primaryLocalizable"]["atIou050"]
        self.assertEqual(
            (
                primary_iou["matchedBoxes"],
                primary_iou["predictedBoxes"],
                primary_iou["truthBoxes"],
            ),
            (1, 3, 2),
        )
        any_iou = metrics["localization"]["anyAnnotatedRealBall"]["atIou050"]
        self.assertEqual(
            (
                any_iou["matchedBoxes"],
                any_iou["predictedBoxes"],
                any_iou["truthBoxes"],
            ),
            (2, 3, 3),
        )
        false_rate = metrics["falseDetectionsOnTrulyBallFreeFrames"]
        self.assertEqual(false_rate["falseDetections"], 1)
        self.assertEqual(false_rate["ballFreeFrames"], 267)
        self.assertAlmostEqual(false_rate["falseDetectionsPer1000BallFreeFrames"], 1000 / 267)

    def test_ranked_ap_is_true_confidence_ranked_ap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = write_merged_evaluation_task(
                Path(temporary), completed_task()
            )
            frames = load_completed_reviewed_tasks([path]).frames
        metrics = evaluate_frames_at_threshold(frames, 0.5)
        ranked = metrics["localization"]["primaryLocalizable"]["rankedAveragePrecision"]["atIou050"]
        self.assertAlmostEqual(ranked["averagePrecision"], 0.75)
        self.assertIn("global confidence ranking", ranked["method"])
        self.assertIn("inference score floor", ranked["scopeCaveat"])

    def test_indeterminate_absence_is_excluded_instead_of_scored_negative(self) -> None:
        task = completed_task()
        frame_ids = list(task["annotations"]["frames"])
        task["annotations"]["frames"][frame_ids[3]]["primaryBallState"] = "indeterminate"
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = write_merged_evaluation_task(Path(temporary), task)
            frames = load_completed_reviewed_tasks([path]).frames
        metrics = evaluate_frames_at_threshold(frames, 0.5)
        primary = metrics["framePresence"]["primaryLocalizable"]
        any_ball = metrics["framePresence"]["anyAnnotatedRealBall"]
        self.assertEqual(primary["excludedIndeterminateFrames"], 1)
        self.assertEqual(any_ball["excludedIndeterminateFrames"], 1)
        self.assertEqual(primary["falsePositive"], 1)
        self.assertEqual(any_ball["falsePositive"], 0)
        self.assertEqual(
            metrics["falseDetectionsOnTrulyBallFreeFrames"]["falseDetections"], 0
        )

    def test_fully_occluded_primary_ball_is_not_called_truly_ball_free(self) -> None:
        task = completed_task()
        frame_ids = list(task["annotations"]["frames"])
        task["annotations"]["frames"][frame_ids[3]]["primaryBallState"] = (
            "fully_occluded"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = write_merged_evaluation_task(Path(temporary), task)
            frames = load_completed_reviewed_tasks([path]).frames
        false_metrics = evaluate_frames_at_threshold(frames, 0.5)[
            "falseDetectionsOnTrulyBallFreeFrames"
        ]
        self.assertEqual(false_metrics["ballFreeFrames"], 266)
        self.assertEqual(false_metrics["falseDetections"], 0)

    def test_center_match_uses_four_pixels_when_ball_is_smaller(self) -> None:
        truth = TruthObject(Box(0.10, 0.10, 0.02, 0.02), "primary-court", "clear", False)
        prediction = Detection(Box(0.14, 0.10, 0.02, 0.02), 0.9)
        frame = EvaluationFrame(
            key="task/frame",
            task_id="task",
            recording_id="recording",
            source_group="group",
            environment="indoor",
            split="train",
            window_key="task/window",
            window_id="window",
            stratum="mid_live_a",
            width=100,
            height=100,
            primary_ball_state="localizable",
            truth_objects=(truth,),
            ball_presence_probability=0.9,
            detections=(prediction,),
        )
        center = evaluate_frames_at_threshold((frame,), 0.5)["localization"][
            "primaryLocalizable"
        ]["centerMatch"]
        self.assertEqual(center["matchedTruthBoxes"], 1)
        self.assertEqual(center["recall"], 1.0)


class BlindReviewMergeTests(unittest.TestCase):
    def test_merges_only_after_blind_review_and_keeps_both_source_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reviewed_path, proposal_path, output_path, reviewed, proposals = (
                blind_merge_fixture(root)
            )
            reviewed_sha = hashlib.sha256(reviewed_path.read_bytes()).hexdigest()
            proposal_sha = hashlib.sha256(proposal_path.read_bytes()).hexdigest()
            written = merge_blind_review_with_detector_suggestions(
                reviewed_path,
                proposal_path,
                output_path,
            )
            merged = validate_ball_annotation_task(written, verify_images=True)

        self.assertEqual(merged["annotations"], reviewed["annotations"])
        self.assertEqual(merged["suggestions"], proposals["suggestions"])
        provenance = merged["blindMergeProvenance"]
        self.assertEqual(provenance["reviewedLabels"]["sha256"], reviewed_sha)
        self.assertEqual(provenance["detectorProposals"]["sha256"], proposal_sha)
        self.assertEqual(
            provenance["immutableDigestSha256"],
            merged["immutable"]["digestSha256"],
        )

    def test_rejects_hidden_channels_in_completed_human_review(self) -> None:
        def add_root_channel(task: dict[str, object]) -> None:
            task["detectorProposals"] = {"seen": True}

        def add_frame_channel(task: dict[str, object]) -> None:
            frame = next(iter(task["annotations"]["frames"].values()))
            frame["modelBoxSeen"] = True

        def add_object_channel(task: dict[str, object]) -> None:
            frame = next(iter(task["annotations"]["frames"].values()))
            frame["objects"][0]["detectorConfidence"] = 0.99

        def remove_exposure_audit(task: dict[str, object]) -> None:
            frame = next(iter(task["annotations"]["frames"].values()))
            frame.pop("proposalExposure")

        for name, mutate in (
            ("root", add_root_channel),
            ("frame", add_frame_channel),
            ("object", add_object_channel),
            ("missing-exposure", remove_exposure_audit),
        ):
            with self.subTest(channel=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                reviewed_path, proposal_path, output_path, reviewed, _ = (
                    blind_merge_fixture(root)
                )
                mutate(reviewed)
                reviewed_path.write_text(json.dumps(reviewed), encoding="utf-8")
                with self.assertRaisesRegex(
                    BallDetectorEvaluationError, "exactly|unsupported fields"
                ):
                    merge_blind_review_with_detector_suggestions(
                        reviewed_path, proposal_path, output_path
                    )

    def test_rejects_nonblind_inputs_and_changed_immutable_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reviewed_path, proposal_path, output_path, reviewed, proposals = (
                blind_merge_fixture(root)
            )
            reviewed["suggestions"] = copy.deepcopy(proposals["suggestions"])
            reviewed_path.write_text(json.dumps(reviewed), encoding="utf-8")
            with self.assertRaisesRegex(
                BallDetectorEvaluationError, "empty suggestions"
            ):
                merge_blind_review_with_detector_suggestions(
                    reviewed_path, proposal_path, output_path
                )

            reviewed["suggestions"] = {
                "status": "empty",
                "model": None,
                "frames": {},
            }
            reviewed_path.write_text(json.dumps(reviewed), encoding="utf-8")
            unreviewed_annotations = copy.deepcopy(proposals["annotations"])
            proposals["annotations"] = copy.deepcopy(reviewed["annotations"])
            proposal_path.write_text(json.dumps(proposals), encoding="utf-8")
            with self.assertRaisesRegex(
                BallDetectorEvaluationError, "must remain unreviewed"
            ):
                merge_blind_review_with_detector_suggestions(
                    reviewed_path, proposal_path, output_path
                )

            proposals["annotations"] = unreviewed_annotations
            proposals["immutable"]["recording"]["environment"] = "beach"
            _resign(proposals)
            proposal_path.write_text(json.dumps(proposals), encoding="utf-8")
            with self.assertRaisesRegex(
                BallDetectorEvaluationError, "different immutable provenance"
            ):
                merge_blind_review_with_detector_suggestions(
                    reviewed_path, proposal_path, output_path
                )

    def test_refuses_wrong_depth_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reviewed_path, proposal_path, output_path, _, _ = blind_merge_fixture(root)
            wrong_depth = root / "nested" / "merged" / "task.json"
            with self.assertRaisesRegex(
                BallDetectorEvaluationError, "sibling-depth"
            ):
                merge_blind_review_with_detector_suggestions(
                    reviewed_path, proposal_path, wrong_depth
                )
            merge_blind_review_with_detector_suggestions(
                reviewed_path, proposal_path, output_path
            )
            with self.assertRaises(FileExistsError):
                merge_blind_review_with_detector_suggestions(
                    reviewed_path, proposal_path, output_path
                )


class SolReviewEvaluationTests(unittest.TestCase):
    def test_prepares_detector_empty_provenance_bound_copy_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            merged_path, human = write_merged_evaluation_task(root, completed_task())
            source_path = Path(
                human["suggestions"]["model"]["sourceTask"]["pathHint"]
            )
            output_path = root / "fixture" / "sol" / "task.json"
            index_path = root / "fixture" / "index.json"
            index_path.write_text(
                json.dumps(pilot_index_for_tasks([human])),
                encoding="utf-8",
            )
            written = prepare_detector_blind_sol_review(
                source_path,
                output_path,
                pilot_index_path=index_path,
                agent_id="sol-agent",
                model_id="gpt-5.6-sol",
                run_id="run-123",
            )
            prepared = validate_ball_annotation_task(written, verify_images=True)
            provenance = prepared["solReviewProvenance"]
            self.assertEqual(prepared["suggestions"]["status"], "empty")
            self.assertEqual(prepared["annotations"]["review"]["status"], "unreviewed")
            self.assertTrue(provenance["detectorSuggestionsAbsent"])
            self.assertEqual(provenance["reviewer"]["agentId"], "sol-agent")
            self.assertEqual(provenance["reviewer"]["modelId"], "gpt-5.6-sol")
            self.assertEqual(
                provenance["sourceTask"]["sha256"],
                hashlib.sha256(source_path.read_bytes()).hexdigest(),
            )
            with self.assertRaises(FileExistsError):
                prepare_detector_blind_sol_review(
                    source_path,
                    output_path,
                    pilot_index_path=index_path,
                    agent_id="sol-agent",
                    model_id="gpt-5.6-sol",
                    run_id="run-123",
                )

            merged = json.loads(merged_path.read_text(encoding="utf-8"))
            proposal_path = Path(
                merged["blindMergeProvenance"]["detectorProposals"]["pathHint"]
            )
            with self.assertRaisesRegex(
                BallDetectorEvaluationError, "empty detector suggestions"
            ):
                prepare_detector_blind_sol_review(
                    proposal_path,
                    root / "fixture" / "bad-sol" / "task.json",
                    pilot_index_path=index_path,
                    agent_id="sol-agent",
                    model_id="gpt-5.6-sol",
                    run_id="run-124",
                )

    def test_sol_preparation_rejects_non_development_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, human = write_merged_evaluation_task(root, completed_task())
            source_path = Path(
                human["suggestions"]["model"]["sourceTask"]["pathHint"]
            )
            source = json.loads(source_path.read_text(encoding="utf-8"))
            index_path = root / "fixture" / "index.json"
            index_path.write_text(
                json.dumps(pilot_index_for_tasks([human])),
                encoding="utf-8",
            )
            source["immutable"]["recording"]["split"] = "test"
            _resign(source)
            source_path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(
                BallDetectorEvaluationError, "development pilot"
            ):
                prepare_detector_blind_sol_review(
                    source_path,
                    root / "fixture" / "sol-test" / "task.json",
                    pilot_index_path=index_path,
                    agent_id="sol-agent",
                    model_id="gpt-5.6-sol",
                    run_id="run-test",
                )

    def test_sol_preparation_rejects_hidden_detector_channels(self) -> None:
        mutations = {
            "root-extra": lambda source: source.__setitem__(
                "detectorLeak", {"boxes": [1, 2, 3]}
            ),
            "immutable-extra": lambda source: source["immutable"].__setitem__(
                "detectorLeak", {"boxes": [1, 2, 3]}
            ),
            "frame-note": lambda source: next(
                iter(source["annotations"]["frames"].values())
            ).__setitem__("notes", "DETECTOR BOX AT 10,20"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                _, human = write_merged_evaluation_task(root, completed_task())
                source_path = Path(
                    human["suggestions"]["model"]["sourceTask"]["pathHint"]
                )
                source = json.loads(source_path.read_text(encoding="utf-8"))
                mutate(source)
                if name == "immutable-extra":
                    _resign(source)
                source_path.write_text(json.dumps(source), encoding="utf-8")
                index = pilot_index_for_tasks([human])
                index["tasks"][0]["taskId"] = source["immutable"]["taskId"]
                index["tasks"][0]["initialTaskSha256"] = hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest()
                index_path = root / "fixture" / "index.json"
                index_path.write_text(json.dumps(index), encoding="utf-8")
                output_path = root / "fixture" / "sol-hidden" / "task.json"
                with self.assertRaisesRegex(
                    BallDetectorEvaluationError, "canonical|proposal channels|blank"
                ):
                    prepare_detector_blind_sol_review(
                        source_path,
                        output_path,
                        pilot_index_path=index_path,
                        agent_id="sol-agent",
                        model_id="gpt-5.6-sol",
                        run_id=f"run-{name}",
                    )
                self.assertFalse(output_path.exists())
                self.assertFalse(
                    output_path.with_name(
                        f"{output_path.name}.preparation-receipt.json"
                    ).exists()
                )

    def test_sol_preparation_rejects_internally_inconsistent_pilot_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, human = write_merged_evaluation_task(root, completed_task())
            source_path = Path(
                human["suggestions"]["model"]["sourceTask"]["pathHint"]
            )
            index = pilot_index_for_tasks([human])
            index["recordingCount"] += 1
            index_path = root / "fixture" / "index.json"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaisesRegex(
                BallDetectorEvaluationError, "aggregate counts"
            ):
                prepare_detector_blind_sol_review(
                    source_path,
                    root / "fixture" / "sol-bad-index" / "task.json",
                    pilot_index_path=index_path,
                    agent_id="sol-agent",
                    model_id="gpt-5.6-sol",
                    run_id="run-bad-index",
                )

    def test_sol_fixed_point_metrics_match_human_truth_without_fake_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            merged_path, human = write_merged_evaluation_task(root, completed_task())
            sol_path = write_completed_sol_review(root, human)
            report = evaluate_ball_detector_tasks(
                [merged_path],
                sol_task_inputs=[sol_path],
                thresholds=[0.5],
                requirements=minimal_requirements(),
            )

        comparison = report["solComparison"]
        self.assertTrue(comparison["available"])
        self.assertFalse(comparison["confidence"]["available"])
        self.assertFalse(comparison["confidence"]["brierReported"])
        primary = comparison["overallMicro"]["framePresence"][
            "primaryLocalizable"
        ]
        any_ball = comparison["overallMicro"]["framePresence"][
            "anyAnnotatedRealBall"
        ]
        self.assertEqual(primary["precision"], 1.0)
        self.assertEqual(primary["recall"], 1.0)
        self.assertEqual(any_ball["f1"], 1.0)
        primary_boxes = comparison["overallMicro"]["localization"][
            "primaryLocalizable"
        ]
        self.assertEqual(primary_boxes["atIou050"]["f1"], 1.0)
        self.assertFalse(primary_boxes["rankedAveragePrecision"]["reported"])
        self.assertEqual(
            comparison["provenance"]["reviewerRuns"][0]["modelId"],
            "gpt-5.6-sol",
        )
        json.dumps(report, allow_nan=False)

    def test_unknown_sol_role_counts_for_any_ball_but_not_primary(self) -> None:
        def make_role_unknown(sol: dict[str, object]) -> None:
            frame_id = next(iter(sol["annotations"]["frames"]))
            annotation = sol["annotations"]["frames"][frame_id]
            annotation["primaryBallState"] = "out_of_frame"
            annotation["objects"][0]["role"] = "unknown"

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            merged_path, human = write_merged_evaluation_task(root, completed_task())
            sol_path = write_completed_sol_review(
                root,
                human,
                mutate=make_role_unknown,
            )
            report = evaluate_ball_detector_tasks(
                [merged_path],
                sol_task_inputs=[sol_path],
                thresholds=[0.5],
                requirements=minimal_requirements(),
            )

        overall = report["solComparison"]["overallMicro"]
        self.assertEqual(
            overall["framePresence"]["primaryLocalizable"]["recall"], 0.5
        )
        self.assertEqual(
            overall["framePresence"]["anyAnnotatedRealBall"]["recall"], 1.0
        )
        self.assertEqual(overall["annotationCoverage"]["roleCounts"]["unknown"], 1)

    def test_rejects_tampered_sol_source_binding_or_wrong_annotator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            merged_path, human = write_merged_evaluation_task(root, completed_task())
            sol_path = write_completed_sol_review(root, human)
            loaded_human = load_completed_reviewed_tasks([merged_path])
            sol = json.loads(sol_path.read_text(encoding="utf-8"))
            sol["solReviewProvenance"]["sourceTask"]["sha256"] = "0" * 64
            sol_path.write_text(json.dumps(sol), encoding="utf-8")
            with self.assertRaisesRegex(
                BallDetectorEvaluationError, "source task SHA-256"
            ):
                load_completed_sol_reviews([sol_path], loaded_human)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            merged_path, human = write_merged_evaluation_task(root, completed_task())
            sol_path = write_completed_sol_review(root, human)
            loaded_human = load_completed_reviewed_tasks([merged_path])
            sol = json.loads(sol_path.read_text(encoding="utf-8"))
            sol["annotations"]["review"]["annotator"] = "different-agent"
            sol_path.write_text(json.dumps(sol), encoding="utf-8")
            with self.assertRaisesRegex(BallDetectorEvaluationError, "agentId"):
                load_completed_sol_reviews([sol_path], loaded_human)

    def test_rejects_tampered_receipt_impossible_chronology_and_zero_code_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            merged_path, human = write_merged_evaluation_task(root, completed_task())
            sol_path = write_completed_sol_review(root, human)
            loaded_human = load_completed_reviewed_tasks([merged_path])
            sol = json.loads(sol_path.read_text(encoding="utf-8"))
            receipt_path = Path(
                sol["solReviewProvenance"]["preparationReceipt"]["pathHint"]
            )
            receipt_path.write_text(
                receipt_path.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                BallDetectorEvaluationError, "receipt SHA-256"
            ):
                load_completed_sol_reviews([sol_path], loaded_human)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            merged_path, human = write_merged_evaluation_task(root, completed_task())
            sol_path = write_completed_sol_review(root, human)
            loaded_human = load_completed_reviewed_tasks([merged_path])
            sol = json.loads(sol_path.read_text(encoding="utf-8"))
            sol["annotations"]["review"]["reviewedAt"] = "2000-01-01T00:00:00+00:00"
            sol_path.write_text(json.dumps(sol), encoding="utf-8")
            with self.assertRaisesRegex(BallDetectorEvaluationError, "not precede"):
                load_completed_sol_reviews([sol_path], loaded_human)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            merged_path, human = write_merged_evaluation_task(root, completed_task())
            sol_path = write_completed_sol_review(root, human)
            loaded_human = load_completed_reviewed_tasks([merged_path])
            sol = json.loads(sol_path.read_text(encoding="utf-8"))
            sol["solReviewProvenance"]["implementationSha256"] = "0" * 64
            sol_path.write_text(json.dumps(sol), encoding="utf-8")
            with self.assertRaisesRegex(
                BallDetectorEvaluationError, "implementationSha256"
            ):
                load_completed_sol_reviews([sol_path], loaded_human)

    def test_sol_inputs_must_exactly_cover_the_human_task_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_path, first = write_merged_evaluation_task(
                root,
                completed_task(),
                name="first",
            )
            second_task = completed_task()
            second_task["immutable"]["recording"].update(
                {"id": "development-recording-2", "sourceGroup": "source-b"}
            )
            second_path, second = write_merged_evaluation_task(
                root,
                second_task,
                name="second",
            )
            first_sol = write_completed_sol_review(root, first, name="first")
            second_sol = write_completed_sol_review(root, second, name="second")
            both_human = load_completed_reviewed_tasks([first_path, second_path])
            with self.assertRaisesRegex(BallDetectorEvaluationError, "missing task ids"):
                load_completed_sol_reviews([first_sol], both_human)
            first_human = load_completed_reviewed_tasks([first_path])
            with self.assertRaisesRegex(
                BallDetectorEvaluationError, "no matching human-review task"
            ):
                load_completed_sol_reviews([second_sol], first_human)

    def test_assisted_human_frames_are_excluded_for_both_methods(self) -> None:
        task = completed_task()
        frame_ids = list(task["annotations"]["frames"])
        task["annotations"]["frames"][frame_ids[0]]["proposalExposure"] = (
            "shown_before_label_finalized"
        )

        def poison_excluded_sol_frames(sol: dict[str, object]) -> None:
            assisted = sol["annotations"]["frames"][frame_ids[0]]
            assisted["primaryBallState"] = "out_of_frame"
            assisted["objects"] = []

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            merged_path, human = write_merged_evaluation_task(root, task)
            sol_path = write_completed_sol_review(
                root,
                human,
                mutate=poison_excluded_sol_frames,
            )
            report = evaluate_ball_detector_tasks(
                [merged_path],
                sol_task_inputs=[sol_path],
                thresholds=[0.5],
                requirements=minimal_requirements(),
            )
            merged = json.loads(merged_path.read_text(encoding="utf-8"))

        self.assertEqual(report["coverage"]["frames"], 269)
        self.assertEqual(report["coverage"]["excludedAssistedHumanFrames"], 1)
        self.assertEqual(report["coverage"]["excludedHumanFramesWithoutExposureAudit"], 0)
        self.assertEqual(
            report["solComparison"]["provenance"][
                "humanFramesExcludedForProposalExposure"
            ],
            1,
        )
        exposure = merged["blindMergeProvenance"]["humanReviewExposure"]
        self.assertFalse(exposure["blanketBlindnessClaim"])
        self.assertEqual(exposure["qualityEligibleFrameCount"], 269)
        detector_primary = report["operatingPoint"]["overallMicro"]["framePresence"][
            "primaryLocalizable"
        ]
        self.assertEqual(detector_primary["positiveFrames"], 1)
        self.assertEqual(detector_primary["truePositive"], 0)
        sol_overall = report["solComparison"]["overallMicro"]["framePresence"]
        self.assertEqual(sol_overall["primaryLocalizable"]["positiveFrames"], 1)
        self.assertEqual(sol_overall["primaryLocalizable"]["recall"], 1.0)
        self.assertEqual(sol_overall["anyAnnotatedRealBall"]["falsePositive"], 0)


class BallDetectorEvaluationProtocolTests(unittest.TestCase):
    def test_report_groups_metrics_but_coverage_alone_does_not_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_path, task = write_merged_evaluation_task(root, completed_task())
            index = pilot_index_for_tasks([task])
            index_path = root / "index.json"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            report = evaluate_ball_detector_tasks(
                [task_path],
                thresholds=[0.5],
                requirements=minimal_requirements(),
                pilot_index_path=index_path,
            )

        self.assertTrue(report["protocolGate"]["coverageSufficient"])
        json.dumps(report, allow_nan=False)
        self.assertFalse(report["sourceGroupLeaveOneOut"]["available"])
        self.assertFalse(report["outOfFoldQualityGate"]["passes"])
        self.assertIsNone(
            report["thresholdSelection"]["frozenAllDevelopmentThreshold"]
        )
        self.assertIsNone(report["thresholdSelection"]["promotedThreshold"])
        self.assertFalse(report["downstreamEligibility"]["eligible"])
        self.assertIn("source-a", report["operatingPoint"]["grouped"]["sourceGroup"])
        self.assertIn("indoor", report["operatingPoint"]["grouped"]["environment"])
        self.assertEqual(set(report["operatingPoint"]["grouped"]["stratum"]), {
            "serve_window", "mid_live_a", "mid_live_b", "end_transition", "ordinary_dead", "high_motion_dead"
        })
        self.assertIn("clear", report["operatingPoint"]["grouped"]["visibility"]["primaryLocalizable"])
        self.assertEqual(report["operatingPoint"]["macro"]["byWindow"]["units"], 6)
        self.assertIn("temporally correlated", report["operatingPoint"]["macro"]["correlationWarning"])

    def test_without_index_best_threshold_remains_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = write_merged_evaluation_task(
                Path(temporary), completed_task()
            )
            report = evaluate_ball_detector_tasks(
                [path], thresholds=[0.5], requirements=minimal_requirements()
            )
        self.assertFalse(report["protocolGate"]["coverageSufficient"])
        self.assertIn("pilotIndexCoverage", report["protocolGate"]["failures"])
        self.assertIsNone(report["thresholdSelection"]["promotedThreshold"])
        self.assertEqual(report["thresholdSelection"]["diagnosticBestThreshold"], 0.5)

    def test_group_held_out_quality_can_freeze_but_not_promote_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_paths = []
            indexed_payloads = []
            for index, (source_group, environment) in enumerate(
                (("source-a", "indoor"), ("source-b", "indoor"), ("source-c", "beach"))
            ):
                task = completed_task()
                frame_ids = list(task["suggestions"]["frames"])
                task["suggestions"]["frames"][frame_ids[1]] = {
                    "ballPresenceProbability": 0.85,
                    "detections": [
                        {
                            "confidence": 0.85,
                            "bbox": task["annotations"]["frames"][frame_ids[1]][
                                "objects"
                            ][0]["bbox"],
                        }
                    ],
                }
                recording_id = f"development-recording-{index}"
                task["immutable"]["recording"].update(
                    {
                        "id": recording_id,
                        "sourceGroup": source_group,
                        "environment": environment,
                    }
                )
                task_path, task = write_merged_evaluation_task(
                    root,
                    task,
                    name=recording_id,
                )
                task_paths.append(task_path)
                indexed_payloads.append(task)
            index = pilot_index_for_tasks(indexed_payloads)
            index_path = root / "index.json"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            report = evaluate_ball_detector_tasks(
                task_paths,
                requirements=minimal_requirements(),
                pilot_index_path=index_path,
            )

        self.assertTrue(report["protocolGate"]["coverageSufficient"])
        json.dumps(report, allow_nan=False)
        self.assertTrue(report["sourceGroupLeaveOneOut"]["available"])
        self.assertTrue(report["outOfFoldQualityGate"]["passes"])
        self.assertTrue(report["thresholdFreezeGate"]["passes"])
        self.assertTrue(
            all(
                fold["selectionStatus"] == "primary-precision-constrained"
                for fold in report["sourceGroupLeaveOneOut"]["folds"]
            )
        )
        self.assertAlmostEqual(
            report["sourceGroupLeaveOneOut"]["pooledOutOfFold"]["framePresence"]
            ["primaryLocalizable"]["precision"],
            1.0,
        )
        self.assertEqual(
            report["thresholdSelection"]["frozenAllDevelopmentThreshold"], 0.85
        )
        self.assertIsNone(report["thresholdSelection"]["promotedThreshold"])
        self.assertFalse(report["downstreamEligibility"]["complete"])
        self.assertEqual(
            report["downstreamEligibility"]["missingRequiredGates"],
            ["falseTrackGate", "throughputGate"],
        )

    def test_index_binding_rejects_substituted_task_round_policy_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, task = write_merged_evaluation_task(root, completed_task())
            loaded = load_completed_reviewed_tasks([path])
            index = pilot_index_for_tasks([task])
            binding = _validate_pilot_index_binding(loaded, index)
            self.assertTrue(binding["exact"])
            self.assertEqual(
                binding["taskBindings"][0]["initialTaskSha256"],
                task["suggestions"]["model"]["sourceTask"]["sha256"],
            )

            mutations = {
                "round": lambda value: value.__setitem__("round", value["round"] + 1),
                "sampling-policy": lambda value: value.__setitem__(
                    "samplingPolicyId", "substituted-policy"
                ),
                "recording-count": lambda value: value.__setitem__(
                    "recordingCount", value["recordingCount"] + 1
                ),
                "window-count": lambda value: value.__setitem__(
                    "windowCount", value["windowCount"] + 1
                ),
                "frame-count": lambda value: value.__setitem__(
                    "frameCount", value["frameCount"] + 1
                ),
                "task-id": lambda value: value["tasks"][0].__setitem__(
                    "taskId", "ball-presence-substituted"
                ),
                "task-frame-count": lambda value: value["tasks"][0].__setitem__(
                    "frameCount", value["tasks"][0]["frameCount"] - 1
                ),
                "initial-task-hash": lambda value: value["tasks"][0].__setitem__(
                    "initialTaskSha256", "e" * 64
                ),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    changed = copy.deepcopy(index)
                    mutate(changed)
                    with self.assertRaises(BallDetectorEvaluationError):
                        _validate_pilot_index_binding(loaded, changed)

    def test_freeze_gate_rejects_any_diagnostic_selection_fallback(self) -> None:
        coverage = {"coverageSufficient": True}
        quality = {"passes": True}
        constrained_folds = {
            "folds": [
                {
                    "heldOutSourceGroup": "source-a",
                    "selectionStatus": "primary-precision-constrained",
                },
                {
                    "heldOutSourceGroup": "source-b",
                    "selectionStatus": "primary-precision-constrained",
                },
            ]
        }
        passing = _threshold_freeze_gate(
            coverage,
            quality,
            "primary-precision-constrained",
            constrained_folds,
        )
        self.assertTrue(passing["passes"])

        all_development_fallback = _threshold_freeze_gate(
            coverage,
            quality,
            "diagnostic-fallback",
            constrained_folds,
        )
        self.assertFalse(all_development_fallback["passes"])
        fallback_folds = copy.deepcopy(constrained_folds)
        fallback_folds["folds"][1]["selectionStatus"] = "diagnostic-fallback"
        fold_fallback = _threshold_freeze_gate(
            coverage,
            quality,
            "primary-precision-constrained",
            fallback_folds,
        )
        self.assertFalse(fold_fallback["passes"])
        self.assertIn("everyFoldSelectionConstrained", fold_fallback["failures"])

    def test_rejects_unfinished_review(self) -> None:
        task = completed_task()
        task["annotations"]["review"]["status"] = "in_progress"
        task["annotations"]["review"]["reviewedAt"] = None
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task.json"
            path.write_text(json.dumps(task), encoding="utf-8")
            with self.assertRaises(BallDetectorEvaluationError):
                load_completed_reviewed_tasks([path])

    def test_rejects_presence_probability_that_disagrees_with_boxes(self) -> None:
        task = completed_task()
        frame_id = next(iter(task["suggestions"]["frames"]))
        task["suggestions"]["frames"][frame_id]["ballPresenceProbability"] = 0.5
        with tempfile.TemporaryDirectory() as temporary:
            path, _ = write_merged_evaluation_task(Path(temporary), task)
            with self.assertRaisesRegex(
                BallDetectorEvaluationError, "maximum stored detection confidence"
            ):
                load_completed_reviewed_tasks([path])

    def test_requires_valid_sha_pinned_blind_merge_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            absent = root / "absent.json"
            absent.write_text(json.dumps(completed_task()), encoding="utf-8")
            with self.assertRaisesRegex(
                BallDetectorEvaluationError, "exact blindMergeProvenance"
            ):
                load_completed_reviewed_tasks([absent])

            merged_path, _ = write_merged_evaluation_task(
                root,
                completed_task(),
                name="merged-fixture",
            )
            merged = json.loads(merged_path.read_text(encoding="utf-8"))
            merged["blindMergeProvenance"]["immutableDigestSha256"] = "0" * 64
            merged_path.write_text(json.dumps(merged), encoding="utf-8")
            with self.assertRaisesRegex(
                BallDetectorEvaluationError, "immutable digest"
            ):
                load_completed_reviewed_tasks([merged_path])

    def test_rejects_changed_blind_merge_source_or_merged_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            merged_path, _ = write_merged_evaluation_task(root, completed_task())
            merged = json.loads(merged_path.read_text(encoding="utf-8"))
            reviewed_path = Path(
                merged["blindMergeProvenance"]["reviewedLabels"]["pathHint"]
            )
            reviewed_path.write_text(
                reviewed_path.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                BallDetectorEvaluationError, "source SHA-256 does not match"
            ):
                load_completed_reviewed_tasks([merged_path])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            merged_path, _ = write_merged_evaluation_task(root, completed_task())
            merged = json.loads(merged_path.read_text(encoding="utf-8"))
            merged["annotations"]["review"]["notes"] = "changed after blind merge"
            merged_path.write_text(json.dumps(merged), encoding="utf-8")
            with self.assertRaisesRegex(
                BallDetectorEvaluationError, "do not exactly match"
            ):
                load_completed_reviewed_tasks([merged_path])

    def test_rejects_test_or_challenge_tasks_even_if_resigned(self) -> None:
        task = completed_task()
        task["immutable"]["recording"]["split"] = "test"
        _resign(task)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task.json"
            path.write_text(json.dumps(task), encoding="utf-8")
            with self.assertRaisesRegex(BallDetectorEvaluationError, "development pilot"):
                load_completed_reviewed_tasks([path])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from analysis.nanodet_person_detector import (MODEL_ID, PINS, UPSTREAM_COMMIT,
    NanoDetPersonDetector, box_to_player, nanodet_identity)


class NanoDetAdapterTests(unittest.TestCase):
    def test_box_geometry_is_explicit_fixed_proxy_not_invented_observed_pose(self):
        person = box_to_player(np.array([10,20,110,220]),.7)
        self.assertEqual((person.x,person.y,person.width,person.height),(30,60,60,90))
        self.assertEqual((person.hip_x,person.hip_y),(60,140))
        self.assertEqual((person.shoulder_x,person.shoulder_y),(60,70))
        self.assertEqual(person.score,.7)
        for box in ([10,10,0,20],[0,0,0,20],[0,np.nan,10,20]):
            with self.assertRaises(ValueError):
                box_to_player(np.asarray(box),.7)

    def test_only_people_survive_with_cross_tile_duplicate_nms(self):
        detector = NanoDetPersonDetector.__new__(NanoDetPersonDetector)
        detector.maximum_detections = 24
        detector.tile_mode = "four-tiles"
        detector.demo = SimpleNamespace(letterbox=lambda image: (image,None),
            unletterbox=lambda box,shape,scale: np.asarray(box))
        detector.model = SimpleNamespace(infer=lambda image: np.array([
            [40,5,80,45,.9,0], [41,5,81,45,.8,0], [90,0,120,40,.95,1]]))
        result=detector.detect(np.zeros((100,200,3),np.uint8))
        self.assertEqual(len(result.detections),4)
        self.assertEqual(result.raw_candidates,8)
        self.assertEqual(len(detector.last_body_boxes),4)
        self.assertTrue(all(p.score==.9 for p in result.detections))

    def test_empty_output_is_empty_evidence(self):
        detector = NanoDetPersonDetector.__new__(NanoDetPersonDetector)
        detector.maximum_detections=24
        detector.tile_mode = "four-tiles"
        detector.demo=SimpleNamespace(letterbox=lambda image: (image,None))
        detector.model=SimpleNamespace(infer=lambda image: np.array([]))
        result=detector.detect(np.zeros((100,200,3),np.uint8))
        self.assertEqual(result.detections,())
        self.assertEqual(detector.last_body_boxes,[])

    def test_single_roi_has_no_quadrant_ownership_filter(self):
        detector = NanoDetPersonDetector.__new__(NanoDetPersonDetector)
        detector.maximum_detections, detector.tile_mode = 24, "single-roi"
        detector.demo = SimpleNamespace(letterbox=lambda image: (image,None),
            unletterbox=lambda box,shape,scale: np.asarray(box))
        calls=[]
        def infer(image):
            calls.append(image.shape)
            return np.array([[150,60,190,95,.9,0]])
        detector.model=SimpleNamespace(infer=infer)
        result=detector.detect(np.zeros((100,200,3),np.uint8))
        self.assertEqual(len(calls),1)
        self.assertEqual(len(result.detections),1)
        self.assertEqual(result.detections[0].hip_x,170)

    def test_artifact_verification_checks_code_model_and_license(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in PINS:
                (root/name).write_bytes(b"placeholder")
            (root/"model.json").write_text(json.dumps({"modelId":MODEL_ID,"upstreamCommit":UPSTREAM_COMMIT,
                "architectureQualification":{"initializerElements":941863}}))
            def pinned_hash(path):
                return PINS.get(Path(path).name,"metadata-hash")
            with patch("analysis.nanodet_person_detector.sha256_file",side_effect=pinned_hash):
                identity=nanodet_identity(root)
                self.assertEqual(identity["modelSha256"],PINS["model.onnx"])
                self.assertEqual(set(identity["artifacts"]),set(PINS))
            with patch("analysis.nanodet_person_detector.sha256_file",return_value="wrong"):
                with self.assertRaisesRegex(ValueError,"artifact mismatch"):
                    nanodet_identity(root)


if __name__ == "__main__":
    unittest.main()

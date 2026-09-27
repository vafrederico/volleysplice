from __future__ import annotations
from analysis.private_ledger import private_value

import importlib.util
import json
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analysis.player_motion_features import (FEATURE_NAMES, PlayerMotionConfig, PlayerMotionExtractor,
    Track, crop_roi, duplicate_detection, match_tracks, nearest_frame_indexes,
    propagate_tracks, select_detections, side_statistics)
from analysis.side_switch_player_detector import PlayerDetection, PersonDetectionResult

SPEC = importlib.util.spec_from_file_location("player_precompute", Path(__file__).resolve().parents[2]/"scripts/precompute-neural-player-features.py")
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def person(x=25., y=70., score=.9):
    return PlayerDetection(x-3, y-12, 6, 12, x, y, x, y-10, score)


class Detector:
    def __init__(self, detections=()):
        self.detections, self.calls = detections, 0
    def detect(self, frame):
        self.calls += 1
        return PersonDetectionResult(tuple(self.detections), len(self.detections), 1.)


class Capture:
    def __init__(self, times, offset=0):
        self.times, self.offset, self.ordinal = times, offset, -1
    def grab(self):
        self.ordinal += 1
        return self.ordinal < len(self.times)
    def retrieve(self):
        return True, np.full((16, 16, 3), self.ordinal, np.uint8)
    def get(self, key):
        return (self.times[self.ordinal]+self.offset)*1000
    def set(self, key, value):
        raise AssertionError("random frame seeking is prohibited")


class PlayerMotionTests(unittest.TestCase):
    def test_nearby_distinct_people_are_not_fixed_frame_radius_duplicates(self):
        self.assertFalse(duplicate_detection(person(25), person(30)))
        self.assertTrue(duplicate_detection(person(25), person(25.1)))

    def test_twelve_players_survive_without_side_cap(self):
        people = [person(10+i*7, 30+(i%2)*40) for i in range(12)]
        self.assertEqual(len(select_detections(people, 24)), 12)
        stats = side_statistics([Track(.1,.1,.02,.08,.1+i*.1,.4,.9,0,i) for i in range(6)], PlayerMotionConfig())
        self.assertEqual(stats["count"], 1.)

    def test_missing_players_are_finite_missing_evidence(self):
        detector = Detector()
        extractor = PlayerMotionExtractor(detector)
        values, audit = extractor.push(np.zeros((100,100,3), np.uint8), 0)
        data = dict(zip(FEATURE_NAMES, values, strict=True))
        self.assertEqual(values.shape, (49,))
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(data["player_detector_ran"], 1)
        self.assertEqual(data["player_tracks_available"], 0)
        self.assertEqual(data["player_motion_available"], 0)
        self.assertEqual(audit["nearTracks"], 0)

    def test_four_hz_grid_runs_detector_only_two_hz_in_continuous_scene(self):
        detector = Detector([person()])
        extractor = PlayerMotionExtractor(detector)
        frame = np.zeros((100,100,3), np.uint8)
        observations = [extractor.push(frame, t)[0] for t in [0,.25,.5,.75,1.]]
        self.assertEqual(detector.calls, 3)
        self.assertEqual(extractor.total_tile_calls, 12)
        self.assertEqual(observations[1][FEATURE_NAMES.index("player_observation_age_seconds")], .25)

    def test_camera_only_flow_moves_tracks_without_player_motion(self):
        track = Track(.3,.3,.2,.2,.4,.5,.9,0.,1)
        raw = np.zeros((100,100,2), np.float32)
        raw[...,0] = 2
        moved = propagate_tracks([track], raw, np.zeros_like(raw), .25,.25,PlayerMotionConfig())[0]
        self.assertAlmostEqual(moved.hip_x, .42)
        self.assertEqual(moved.speed, 0)
        self.assertTrue(moved.motion_valid)

    def test_residual_motion_speed_uses_elapsed_time_and_axis_normalization(self):
        track = Track(.3,.3,.2,.2,.4,.5,.9,0.,1)
        raw = np.zeros((100,200,2), np.float32)
        raw[...,0] = 2
        moved = propagate_tracks([track], raw, raw, .25,.25,PlayerMotionConfig())[0]
        self.assertAlmostEqual(moved.velocity_x, .04)
        self.assertAlmostEqual(moved.velocity_y, 0)
        stats = side_statistics([moved], PlayerMotionConfig())
        self.assertEqual(stats["moving_fraction"], 1)
        self.assertEqual(stats["speed_rise_fraction"], 1)

    def test_expired_tracks_and_outside_roi_cannot_leak_observation(self):
        track = Track(.3,.3,.2,.2,.4,.5,.9,0.,1)
        raw = np.zeros((100,100,2), np.float32)
        self.assertEqual(propagate_tracks([track], raw, raw,.25,1.,PlayerMotionConfig()), [])
        detector = Detector([person(-5)])
        values, audit = PlayerMotionExtractor(detector).push(np.zeros((100,100,3),np.uint8), 0)
        self.assertEqual(audit["roiRejected"], 1)
        self.assertEqual(values[FEATURE_NAMES.index("player_tracks_available")], 0)

    def test_gap_and_explicit_reset_break_motion_and_identity(self):
        for reset, next_time in ((True,.25),(False,1.)):
            detector = Detector([person()])
            extractor = PlayerMotionExtractor(detector)
            frame = np.zeros((100,100,3),np.uint8)
            _, first = extractor.push(frame,0)
            values, second = extractor.push(frame,next_time,reset=reset)
            self.assertEqual(values[FEATURE_NAMES.index("player_motion_available")], 0)
            self.assertNotEqual(first["trackIds"], second["trackIds"])

    def test_large_scene_jump_forces_fresh_detection(self):
        extractor = PlayerMotionExtractor(Detector([person()]))
        extractor.push(np.zeros((100,100,3),np.uint8),0)
        values, audit = extractor.push(np.full((100,100,3),255,np.uint8),.25)
        self.assertEqual(values[FEATURE_NAMES.index("player_motion_available")],0)
        self.assertTrue(audit["detectorRan"])

    def test_matching_respects_spatial_scale_and_unique_ownership(self):
        before = Track(.2,.2,.1,.1,.25,.3,.9,0,7,velocity_x=.1,motion_valid=True)
        candidates = [Track(.21,.2,.1,.1,.26,.3,.8,.5,8), Track(.8,.2,.1,.1,.85,.3,.8,.5,9)]
        after, count = match_tracks([before], candidates)
        self.assertEqual(count,1)
        self.assertEqual(after[0].track_id,7)
        self.assertEqual(after[1].track_id,9)
        self.assertEqual(after[0].velocity_x,.1)

    def test_geometry_availability_is_explicit_and_configuration_is_checked(self):
        extractor = PlayerMotionExtractor(Detector(),PlayerMotionConfig(net_y_ratio=.4,net_geometry_supplied=True))
        values,_ = extractor.push(np.zeros((100,100,3),np.uint8),0)
        self.assertEqual(values[FEATURE_NAMES.index("player_net_geometry_supplied")],1)
        for kwargs in ({"net_y_ratio": 0},{"maximum_detections":6},{"detector_fps":4}):
            with self.assertRaises(ValueError):
                PlayerMotionConfig(**kwargs)

    def test_detector_failure_is_not_silently_a_negative(self):
        detector = Detector()
        with patch.object(detector,"detect",side_effect=RuntimeError("model failed")):
            with self.assertRaisesRegex(RuntimeError,"model failed"):
                PlayerMotionExtractor(detector).push(np.zeros((100,100,3),np.uint8),0)

    def test_pts_alignment_handles_vfr_earlier_ties_and_rejects_holes(self):
        np.testing.assert_array_equal(nearest_frame_indexes(np.array([0,.125,.375,.5,.75]),np.array([0,.25,.5,.75])),[0,1,3,4])
        for pts in ([0,.25,.25],[.1,.25,.5],[0,1]):
            with self.assertRaises(ValueError):
                nearest_frame_indexes(np.array(pts),np.array([0,.25,.5]))

    def test_sequential_decode_checks_pts_without_seeks(self):
        pts=np.array([0,.1,.2,.25,.4,.5])
        rows=list(cli.iter_selected_frames(Capture(pts),np.array([0,3,5]),pts,full_recording=True))
        self.assertEqual([row[0] for row in rows],[0,3,5])
        with self.assertRaises(ValueError):
            list(cli.iter_selected_frames(Capture(pts,.001),np.array([0,3,5]),pts,full_recording=True))

    def test_partial_selection_is_not_required_to_decode_entire_source(self):
        pts=np.arange(20)/10
        rows=list(cli.iter_selected_frames(Capture(pts),np.array([0,2,5]),pts,full_recording=False))
        self.assertEqual(len(rows),3)

    def test_roi_crop_respects_native_source_and_rejects_invalid(self):
        frame=np.arange(100*200*3,dtype=np.uint8).reshape(100,200,3)
        np.testing.assert_array_equal(crop_roi(frame,(.25,.2,.5,.6)),frame[20:80,50:150])
        with self.assertRaises(ValueError):
            crop_roi(frame,(.7,0,.5,1))

    def test_sanitized_manifest_rejects_labels_beach_and_protected(self):
        base={"id":"one","video":"/video","contentSha256":"a"*64,"roi":None,
              "sourceGroup":"group","split":"train","environment":"grass",
              "durationSeconds":10,"avCache":{"path":"/av","sha256":"b"*64}}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"manifest.json"
            path.write_text(json.dumps({"records":[base]}))
            self.assertEqual(cli.read_records(path,[])[0]["id"],"one")
            for change in ({"rallies":[]},{"environment":"beach"},{"sourceGroup":private_value('source-group-008')},{"split":"test"}):
                path.write_text(json.dumps({"records":[{**base,**change}]}))
                with self.assertRaises(ValueError):
                    cli.read_records(path,[])

    def test_cache_exact_timestamps_identity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"features.npz"
            times=np.array([0,.250001],np.float64)
            kwargs={"times":times,"values":np.zeros((2,49),np.float32),"metadata":{"trainingEligible":False},
                    "source_pts":np.array([0,.25]),"source_indexes":np.array([0,15]),"frame_hashes":["a","b"],"audit":[{},{}]}
            cli._save_cache(path,**kwargs)
            with np.load(path,allow_pickle=False) as data:
                np.testing.assert_array_equal(data["times"],times)
                self.assertEqual(data["values"].dtype,np.float32)
                self.assertFalse(json.loads(str(data["metadata_json"]))["trainingEligible"])
            with self.assertRaises(FileExistsError):
                cli._save_cache(path,**kwargs)

    def test_complete_streaming_cache_binds_exact_av_times_and_all_features(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video, av, manifest = root/"video.mp4", root/"av.npz", root/"manifest.json"
            video.write_bytes(b"synthetic-byte-identity")
            times = np.array([0.,.25,.5])
            np.savez(av,times=times,values=np.zeros((3,104),np.float32))
            manifest.write_text("{}")
            row={"id":"synthetic","video":str(video),"contentSha256":cli.sha256_file(video),
                 "avCache":{"path":str(av),"sha256":cli.sha256_file(av)},"roi":None,
                 "sourceGroup":"synthetic-group","environment":"indoor","durationSeconds":.75}
            args=SimpleNamespace(output_dir=root,manifest=manifest,max_seconds=None,maximum_detections=24,motion_long_side=64)
            capture=Capture(times)
            capture.isOpened=lambda: True
            capture.release=lambda: None
            with patch.object(cli,"packet_timeline",return_value=(times,{"stream":{"duration":"0.75"}})), \
                 patch.object(cli.cv2,"VideoCapture",return_value=capture):
                result=cli.extract_record(row,args,Detector(),{"path":str(manifest),"sha256":cli.sha256_file(manifest)},{})
            with np.load(result["path"],allow_pickle=False) as data:
                np.testing.assert_array_equal(data["times"],times)
                self.assertEqual(data["values"].shape,(3,49))
                metadata=json.loads(str(data["metadata_json"]))
                self.assertTrue(metadata["trainingEligible"])
                self.assertEqual(metadata["statistics"]["detectorFrames"],2)
                self.assertEqual(metadata["sourceCode"].keys(),{
                    "scripts/precompute-neural-player-features.py","analysis/player_motion_features.py",
                    "analysis/side_switch_player_detector.py","analysis/serving_side_flight.py"})


if __name__ == "__main__":
    unittest.main()

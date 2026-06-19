"""Tests per src/pose/extractor.py — PoseData e PoseExtractor."""
from __future__ import annotations

import pytest

from src.pose.extractor import PoseData, PoseExtractor


# ---------------------------------------------------------------------------
# PoseData
# ---------------------------------------------------------------------------


class TestPoseDataDefaults:
    def test_empty_by_default(self):
        p = PoseData()
        assert p.keypoints == []
        assert p.bones == []
        assert p.confidence == 0.0
        assert p.image_width == 0
        assert p.image_height == 0

    def test_not_valid_when_empty(self):
        assert not PoseData().is_valid

    def test_not_valid_with_zero_confidence(self):
        p = PoseData(keypoints=[[0.5, 0.5, 0.9]], confidence=0.0)
        assert not p.is_valid

    def test_valid_with_keypoints_and_confidence(self):
        p = PoseData(keypoints=[[0.5, 0.5, 0.9]], confidence=0.9)
        assert p.is_valid

    def test_instances_do_not_share_lists(self):
        a = PoseData()
        b = PoseData()
        a.keypoints.append([0.1, 0.2, 0.3])
        assert b.keypoints == []


class TestPoseDataRoundtrip:
    def test_to_dict_keys(self):
        d = PoseData(keypoints=[[0.1, 0.2, 0.9]], confidence=0.9,
                     image_width=512, image_height=768).to_dict()
        assert d["keypoints"] == [[0.1, 0.2, 0.9]]
        assert d["confidence"] == 0.9
        assert d["image_width"] == 512
        assert d["image_height"] == 768

    def test_from_dict_roundtrip(self):
        orig = PoseData(keypoints=[[0.1, 0.2, 0.9], [0.3, 0.4, 0.8]],
                        bones=[[0, 1]], confidence=0.85,
                        image_width=1024, image_height=1024)
        clone = PoseData.from_dict(orig.to_dict())
        assert clone.keypoints == orig.keypoints
        assert clone.bones == orig.bones
        assert clone.confidence == pytest.approx(orig.confidence)
        assert clone.image_width == orig.image_width
        assert clone.image_height == orig.image_height

    def test_from_dict_missing_keys_defaults(self):
        p = PoseData.from_dict({})
        assert p.keypoints == []
        assert p.confidence == 0.0


class TestAsJointsDict:
    def test_indices_match_positions(self):
        p = PoseData(keypoints=[[0.1, 0.2, 0.9], [0.3, 0.4, 0.8]])
        j = p.as_joints_dict()
        assert j[0] == [0.1, 0.2, 0.9]
        assert j[1] == [0.3, 0.4, 0.8]

    def test_empty_keypoints_empty_dict(self):
        assert PoseData().as_joints_dict() == {}


# ---------------------------------------------------------------------------
# PoseExtractor.extract_from_comfy_output
# ---------------------------------------------------------------------------


def _comfy_json(raw: list[float], w: int = 512, h: int = 512) -> dict:
    return {
        "people": [{"pose_keypoints_2d": raw}],
        "canvas_width": w,
        "canvas_height": h,
    }


class TestExtractFromComfyOutput:
    def setup_method(self):
        self.ex = PoseExtractor()

    def test_no_people_returns_empty(self):
        p = self.ex.extract_from_comfy_output({"people": []})
        assert not p.is_valid
        assert p.keypoints == []

    def test_missing_people_key(self):
        p = self.ex.extract_from_comfy_output({})
        assert not p.is_valid

    def test_normalizes_coordinates(self):
        # Punto a (256, 128) su canvas 512x512 → (0.5, 0.25)
        p = self.ex.extract_from_comfy_output(_comfy_json([256, 128, 0.9]))
        assert p.keypoints[0][0] == pytest.approx(0.5)
        assert p.keypoints[0][1] == pytest.approx(0.25)
        assert p.keypoints[0][2] == pytest.approx(0.9)

    def test_canvas_dims_stored(self):
        p = self.ex.extract_from_comfy_output(_comfy_json([10, 10, 0.5], w=1024, h=768))
        assert p.image_width == 1024
        assert p.image_height == 768

    def test_confidence_is_average(self):
        raw = [10, 10, 0.8, 20, 20, 0.4]
        p = self.ex.extract_from_comfy_output(_comfy_json(raw))
        assert p.confidence == pytest.approx(0.6)

    def test_multiple_keypoints_count(self):
        raw = [1, 2, 0.9, 3, 4, 0.9, 5, 6, 0.9]
        p = self.ex.extract_from_comfy_output(_comfy_json(raw))
        assert len(p.keypoints) == 3

    def test_truncated_triplet_ignored(self):
        # 7 valori = 2 triplette complete + 1 valore orfano
        raw = [1, 2, 0.9, 3, 4, 0.9, 5]
        p = self.ex.extract_from_comfy_output(_comfy_json(raw))
        assert len(p.keypoints) == 2

    def test_zero_canvas_does_not_divide_by_zero(self):
        p = self.ex.extract_from_comfy_output(_comfy_json([10, 10, 0.5], w=0, h=0))
        # max(0,1)=1 → coordinate non normalizzate ma niente crash
        assert p.keypoints[0][0] == pytest.approx(10.0)

    def test_first_person_only(self):
        data = {
            "people": [
                {"pose_keypoints_2d": [100, 100, 0.9]},
                {"pose_keypoints_2d": [200, 200, 0.8]},
            ],
            "canvas_width": 400, "canvas_height": 400,
        }
        p = self.ex.extract_from_comfy_output(data)
        assert len(p.keypoints) == 1
        assert p.keypoints[0][0] == pytest.approx(0.25)

    def test_empty_keypoints_list(self):
        p = self.ex.extract_from_comfy_output(_comfy_json([]))
        assert p.keypoints == []
        assert p.confidence == 0.0
        assert not p.is_valid

    def test_result_feeds_categorize_skeleton(self):
        """Integrazione: l'output deve essere accettato da categorize_skeleton."""
        from src.pose.library import categorize_skeleton

        # 17 keypoint COCO fittizi tutti a centro canvas
        raw: list[float] = []
        for _ in range(17):
            raw.extend([256, 256, 0.9])
        p = self.ex.extract_from_comfy_output(_comfy_json(raw))
        cats = categorize_skeleton(p.as_joints_dict())
        assert isinstance(cats, list)

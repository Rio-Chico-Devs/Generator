"""Tests per src/pose/library.py — PoseLibrary, categorize_skeleton e helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pose.library import (
    PoseEntry,
    PoseLibrary,
    _avg_y,
    _parse_joints,
    categorize_skeleton,
    utcnow_iso,
    _entry_dict,
)


# ---------------------------------------------------------------------------
# _parse_joints
# ---------------------------------------------------------------------------


class TestParseJoints:
    def test_empty_returns_empty(self):
        assert _parse_joints({}) == {}

    def test_coco17_string_keys(self):
        j = {"nose": [0.5, 0.1, 0.9], "left_shoulder": [0.3, 0.3, 0.9]}
        r = _parse_joints(j)
        assert "nose" in r
        assert "l_shoulder" in r

    def test_coco17_right_shoulder(self):
        r = _parse_joints({"right_shoulder": [0.7, 0.3, 0.9]})
        assert "r_shoulder" in r

    def test_int_keys_openpose18(self):
        # 0=nose, 5=l_shoulder, 6=r_shoulder (wait: OP18[5]=l_shoulder, OP18[2]=r_shoulder)
        r = _parse_joints({0: [0.5, 0.1, 0.9]})
        assert "nose" in r

    def test_int_key_out_of_range_ignored(self):
        r = _parse_joints({99: [0.5, 0.5, 0.9]})
        assert r == {}

    def test_confidence_below_threshold_excluded(self):
        r = _parse_joints({"nose": [0.5, 0.1, 0.1]})  # conf=0.1 < 0.3
        assert "nose" not in r

    def test_confidence_at_threshold_included(self):
        r = _parse_joints({"nose": [0.5, 0.1, 0.3]})
        assert "nose" in r

    def test_values_as_tuple(self):
        r = _parse_joints({"nose": (0.5, 0.1, 0.9)})
        assert "nose" in r

    def test_values_as_dict(self):
        r = _parse_joints({"nose": {"x": 0.5, "y": 0.1, "conf": 0.9}})
        assert "nose" in r
        assert r["nose"][0] == pytest.approx(0.5)
        assert r["nose"][1] == pytest.approx(0.1)

    def test_dict_value_confidence_key(self):
        r = _parse_joints({"nose": {"x": 0.5, "y": 0.1, "confidence": 0.9}})
        assert "nose" in r

    def test_dict_value_score_key(self):
        r = _parse_joints({"nose": {"x": 0.5, "y": 0.1, "score": 0.9}})
        assert "nose" in r

    def test_2element_list_assumes_conf1(self):
        r = _parse_joints({"nose": [0.5, 0.1]})
        assert "nose" in r

    def test_unknown_string_key_ignored(self):
        r = _parse_joints({"unknown_joint_xyz": [0.5, 0.5, 0.9]})
        assert r == {}

    def test_returns_xy_tuple(self):
        r = _parse_joints({"nose": [0.5, 0.2, 0.9]})
        x, y = r["nose"]
        assert x == pytest.approx(0.5)
        assert y == pytest.approx(0.2)

    def test_shorthand_canonical_keys_pass_through(self):
        # Keys already in canonical form (r_shoulder, l_hip, etc.)
        r = _parse_joints({"r_shoulder": [0.7, 0.3, 0.9], "l_hip": [0.4, 0.6, 0.9]})
        assert "r_shoulder" in r
        assert "l_hip" in r


# ---------------------------------------------------------------------------
# _avg_y
# ---------------------------------------------------------------------------


class TestAvgY:
    def test_single_joint(self):
        j = {"nose": (0.5, 0.2)}
        assert _avg_y(j, "nose") == pytest.approx(0.2)

    def test_two_joints_average(self):
        j = {"l_shoulder": (0.3, 0.3), "r_shoulder": (0.7, 0.5)}
        assert _avg_y(j, "l_shoulder", "r_shoulder") == pytest.approx(0.4)

    def test_none_when_all_absent(self):
        assert _avg_y({}, "nose", "l_eye") is None

    def test_partial_present(self):
        j = {"l_shoulder": (0.3, 0.4)}
        assert _avg_y(j, "l_shoulder", "r_shoulder") == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# categorize_skeleton
# ---------------------------------------------------------------------------


def _standing_frontal() -> dict:
    return {
        "nose": [0.5, 0.08, 0.9],
        "left_eye": [0.42, 0.06, 0.9], "right_eye": [0.58, 0.06, 0.9],
        "left_shoulder": [0.30, 0.30, 0.9], "right_shoulder": [0.70, 0.30, 0.9],
        "left_hip": [0.35, 0.62, 0.9], "right_hip": [0.65, 0.62, 0.9],
        "left_knee": [0.35, 0.76, 0.9], "right_knee": [0.65, 0.76, 0.9],
        "left_ankle": [0.35, 0.92, 0.9], "right_ankle": [0.65, 0.92, 0.9],
    }


class TestCategorizeSkeleton:
    # --- Empty / degenerate ---

    def test_empty_returns_empty(self):
        assert categorize_skeleton({}) == []

    def test_all_low_confidence_returns_empty(self):
        j = {"nose": [0.5, 0.1, 0.05], "left_shoulder": [0.3, 0.3, 0.05]}
        assert categorize_skeleton(j) == []

    def test_only_face_no_posture(self):
        # Just face joints — can't determine posture without shoulders/hips
        j = {"nose": [0.5, 0.1, 0.9], "left_eye": [0.4, 0.09, 0.9], "right_eye": [0.6, 0.09, 0.9]}
        cats = categorize_skeleton(j)
        assert "frontal" in cats
        assert "standing" not in cats

    # --- Postura ---

    def test_standing_full_body(self):
        cats = categorize_skeleton(_standing_frontal())
        assert "standing" in cats

    def test_sitting_knees_at_hip_level(self):
        j = {
            "left_shoulder": [0.3, 0.25, 0.9], "right_shoulder": [0.7, 0.25, 0.9],
            "left_hip": [0.35, 0.55, 0.9], "right_hip": [0.65, 0.55, 0.9],
            # knees very close to hip y → sitting
            "left_knee": [0.25, 0.60, 0.9], "right_knee": [0.75, 0.60, 0.9],
        }
        cats = categorize_skeleton(j)
        assert "sitting" in cats

    def test_lying_horizontal_body(self):
        j = {
            "left_shoulder": [0.05, 0.50, 0.9], "right_shoulder": [0.95, 0.52, 0.9],
            "left_hip": [0.20, 0.53, 0.9], "right_hip": [0.80, 0.54, 0.9],
        }
        cats = categorize_skeleton(j)
        assert "lying" in cats

    def test_standing_not_lying(self):
        cats = categorize_skeleton(_standing_frontal())
        assert "lying" not in cats

    # --- Arms raised ---

    def test_arms_not_raised_by_default(self):
        cats = categorize_skeleton(_standing_frontal())
        assert "arms_raised" not in cats

    def test_arms_raised_when_wrists_above_shoulders(self):
        j = dict(_standing_frontal())
        j["left_wrist"] = [0.2, 0.05, 0.9]   # y=0.05 < shoulder y=0.30
        j["right_wrist"] = [0.8, 0.05, 0.9]
        cats = categorize_skeleton(j)
        assert "arms_raised" in cats

    def test_arms_not_raised_when_wrists_below_shoulders(self):
        j = dict(_standing_frontal())
        j["left_wrist"] = [0.28, 0.65, 0.9]   # y=0.65 > shoulder y=0.30
        j["right_wrist"] = [0.72, 0.65, 0.9]
        cats = categorize_skeleton(j)
        assert "arms_raised" not in cats

    def test_one_wrist_above_enough(self):
        j = dict(_standing_frontal())
        # Only left wrist visible, above shoulder
        j["left_wrist"] = [0.2, 0.05, 0.9]
        cats = categorize_skeleton(j)
        assert "arms_raised" in cats

    # --- Orientamento ---

    def test_frontal_nose_and_both_eyes(self):
        cats = categorize_skeleton(_standing_frontal())
        assert "frontal" in cats

    def test_profile_only_left_eye(self):
        j = dict(_standing_frontal())
        # Remove right eye (below confidence threshold)
        j["right_eye"] = [0.6, 0.06, 0.05]
        cats = categorize_skeleton(j)
        assert "profile" in cats
        assert "frontal" not in cats

    def test_profile_only_right_eye(self):
        j = dict(_standing_frontal())
        j["left_eye"] = [0.42, 0.06, 0.05]
        cats = categorize_skeleton(j)
        assert "profile" in cats

    def test_profile_asymmetric_ears(self):
        j = {
            "left_shoulder": [0.3, 0.3, 0.9], "right_shoulder": [0.7, 0.3, 0.9],
            "left_hip": [0.35, 0.6, 0.9], "right_hip": [0.65, 0.6, 0.9],
            "left_ankle": [0.35, 0.9, 0.9], "right_ankle": [0.65, 0.9, 0.9],
            "left_ear": [0.35, 0.08, 0.9],   # only left ear visible
        }
        cats = categorize_skeleton(j)
        assert "profile" in cats

    def test_posterior_no_face_joints(self):
        j = {
            "left_shoulder": [0.3, 0.3, 0.9], "right_shoulder": [0.7, 0.3, 0.9],
            "left_hip": [0.35, 0.6, 0.9], "right_hip": [0.65, 0.6, 0.9],
            "left_ankle": [0.35, 0.9, 0.9], "right_ankle": [0.65, 0.9, 0.9],
        }
        cats = categorize_skeleton(j)
        assert "posterior" in cats

    def test_posterior_not_frontal(self):
        j = {
            "left_shoulder": [0.3, 0.3, 0.9], "right_shoulder": [0.7, 0.3, 0.9],
            "left_hip": [0.35, 0.6, 0.9], "right_hip": [0.65, 0.6, 0.9],
        }
        cats = categorize_skeleton(j)
        assert "frontal" not in cats

    # --- Int keys (OpenPose-18) ---

    def test_int_keys_standing_frontal(self):
        # OP18: 0=nose, 5=l_shoulder, 2=r_shoulder, 12=l_hip, 9=r_hip, 14=l_ankle, 11=r_ankle
        # 16=l_eye, 15=r_eye
        j = {
            0: [0.5, 0.08, 0.9],    # nose
            16: [0.42, 0.06, 0.9],  # l_eye
            15: [0.58, 0.06, 0.9],  # r_eye
            5: [0.3, 0.30, 0.9],    # l_shoulder
            2: [0.7, 0.30, 0.9],    # r_shoulder
            12: [0.35, 0.62, 0.9],  # l_hip
            9: [0.65, 0.62, 0.9],   # r_hip
            14: [0.35, 0.92, 0.9],  # l_ankle
            11: [0.65, 0.92, 0.9],  # r_ankle
        }
        cats = categorize_skeleton(j)
        assert "standing" in cats
        assert "frontal" in cats

    # --- Multiple categories together ---

    def test_combined_standing_frontal_arms_raised(self):
        j = dict(_standing_frontal())
        j["left_wrist"] = [0.2, 0.02, 0.9]
        j["right_wrist"] = [0.8, 0.02, 0.9]
        cats = categorize_skeleton(j)
        assert "standing" in cats
        assert "frontal" in cats
        assert "arms_raised" in cats

    def test_posterior_with_lying(self):
        j = {
            "left_shoulder": [0.05, 0.5, 0.9], "right_shoulder": [0.9, 0.52, 0.9],
            "left_hip": [0.2, 0.53, 0.9], "right_hip": [0.75, 0.54, 0.9],
        }
        cats = categorize_skeleton(j)
        assert "lying" in cats
        assert "posterior" in cats


# ---------------------------------------------------------------------------
# PoseEntry
# ---------------------------------------------------------------------------


class TestPoseEntry:
    def test_default_values(self):
        e = PoseEntry(id="pose_0001", source_path="sources/pose_0001.jpg")
        assert e.categories == []
        assert e.tags == []
        assert e.person_count == 1
        assert e.confidence == 0.0
        assert e.favorite is False

    def test_instances_dont_share_categories(self):
        a = PoseEntry(id="a", source_path="a.jpg")
        b = PoseEntry(id="b", source_path="b.jpg")
        a.categories.append("standing")
        assert b.categories == []

    def test_custom_values(self):
        e = PoseEntry(
            id="pose_0042",
            source_path="sources/x.png",
            categories=["standing", "frontal"],
            tags=["dynamic"],
            person_count=2,
            confidence=0.87,
            favorite=True,
        )
        assert e.person_count == 2
        assert e.favorite is True
        assert "frontal" in e.categories


# ---------------------------------------------------------------------------
# PoseLibrary — setup helpers
# ---------------------------------------------------------------------------


def _lib(tmp_path: Path) -> PoseLibrary:
    lib = PoseLibrary.__new__(PoseLibrary)
    lib.root = tmp_path / "pose_library"
    lib._poses = {}
    lib._loaded = False
    return lib


def _entry(n: int, **kw) -> PoseEntry:
    return PoseEntry(
        id=f"pose_{n:04d}",
        source_path=f"sources/pose_{n:04d}.jpg",
        **kw,
    )


# ---------------------------------------------------------------------------
# PoseLibrary — directory management
# ---------------------------------------------------------------------------


class TestPoseLibraryDirs:
    def test_ensure_dirs_creates_all(self, tmp_path):
        lib = _lib(tmp_path)
        lib.ensure_dirs()
        assert lib.sources_dir.is_dir()
        assert lib.skeletons_dir.is_dir()
        assert lib.thumbnails_dir.is_dir()

    def test_ensure_dirs_idempotent(self, tmp_path):
        lib = _lib(tmp_path)
        lib.ensure_dirs()
        lib.ensure_dirs()  # no exception
        assert lib.sources_dir.is_dir()

    def test_index_path(self, tmp_path):
        lib = _lib(tmp_path)
        assert lib.index_path == lib.root / "library.json"


# ---------------------------------------------------------------------------
# PoseLibrary — load / save
# ---------------------------------------------------------------------------


class TestPoseLibraryPersistence:
    def test_load_empty_dir(self, tmp_path):
        lib = _lib(tmp_path)
        lib.load()
        assert lib._poses == {}
        assert lib._loaded is True

    def test_save_and_load_roundtrip(self, tmp_path):
        lib = _lib(tmp_path)
        lib.ensure_dirs()
        e = _entry(1, categories=["standing"], tags=["dynamic"], confidence=0.9)
        lib._poses["pose_0001"] = e
        lib.save()

        lib2 = _lib(tmp_path)
        lib2.load()
        assert "pose_0001" in lib2._poses
        loaded = lib2._poses["pose_0001"]
        assert loaded.categories == ["standing"]
        assert loaded.confidence == pytest.approx(0.9)

    def test_save_does_not_include_id_in_dict(self, tmp_path):
        lib = _lib(tmp_path)
        lib.ensure_dirs()
        lib._poses["pose_0001"] = _entry(1)
        lib.save()
        data = json.loads(lib.index_path.read_text())
        assert "id" not in data["poses"]["pose_0001"]

    def test_save_has_schema_version(self, tmp_path):
        lib = _lib(tmp_path)
        lib.ensure_dirs()
        lib.save()
        data = json.loads(lib.index_path.read_text())
        assert "schema_version" in data

    def test_load_multiple_entries(self, tmp_path):
        lib = _lib(tmp_path)
        lib.ensure_dirs()
        for i in range(5):
            lib._poses[f"pose_{i:04d}"] = _entry(i)
        lib.save()

        lib2 = _lib(tmp_path)
        lib2.load()
        assert len(lib2._poses) == 5

    def test_no_tmp_residue_after_save(self, tmp_path):
        lib = _lib(tmp_path)
        lib.ensure_dirs()
        lib.save()
        assert list(lib.root.glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# PoseLibrary — CRUD
# ---------------------------------------------------------------------------


class TestPoseLibraryCRUD:
    def test_add_and_get(self, tmp_path):
        lib = _lib(tmp_path)
        e = _entry(1)
        lib.add(e)
        assert lib.get("pose_0001") is e

    def test_get_missing_returns_none(self, tmp_path):
        lib = _lib(tmp_path)
        assert lib.get("nonexistent") is None

    def test_update_replaces_entry(self, tmp_path):
        lib = _lib(tmp_path)
        lib.add(_entry(1, confidence=0.5))
        updated = _entry(1, confidence=0.9)
        lib.update(updated)
        assert lib.get("pose_0001").confidence == pytest.approx(0.9)

    def test_remove_deletes_entry(self, tmp_path):
        lib = _lib(tmp_path)
        lib.add(_entry(1))
        lib.remove("pose_0001")
        assert lib.get("pose_0001") is None

    def test_remove_missing_no_error(self, tmp_path):
        lib = _lib(tmp_path)
        lib.remove("nonexistent")  # should not raise

    def test_all_returns_list(self, tmp_path):
        lib = _lib(tmp_path)
        for i in range(3):
            lib.add(_entry(i))
        assert len(lib.all()) == 3

    def test_all_empty(self, tmp_path):
        lib = _lib(tmp_path)
        assert lib.all() == []


# ---------------------------------------------------------------------------
# PoseLibrary — search
# ---------------------------------------------------------------------------


class TestPoseLibrarySearch:
    def setup_method(self):
        import shutil
        import tempfile
        self._tmp = Path(tempfile.mkdtemp())
        self.lib = _lib(self._tmp)
        self.lib.add(_entry(1, categories=["standing", "frontal"], tags=["dynamic"], user_notes="eroica"))
        self.lib.add(_entry(2, categories=["sitting", "profile"], tags=["static"], favorite=True))
        self.lib.add(_entry(3, categories=["lying", "posterior"], tags=["relaxed"]))

    def teardown_method(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_search_no_filter_returns_all(self):
        assert len(self.lib.search()) == 3

    def test_search_by_text_tag(self):
        results = self.lib.search(text="dynamic")
        assert len(results) == 1
        assert results[0].id == "pose_0001"

    def test_search_by_text_category(self):
        results = self.lib.search(text="sitting")
        assert len(results) == 1
        assert results[0].id == "pose_0002"

    def test_search_by_text_user_notes(self):
        results = self.lib.search(text="eroica")
        assert len(results) == 1
        assert results[0].id == "pose_0001"

    def test_search_case_insensitive(self):
        results = self.lib.search(text="DYNAMIC")
        assert len(results) == 1

    def test_search_by_category(self):
        results = self.lib.search(categories=["standing"])
        assert len(results) == 1
        assert results[0].id == "pose_0001"

    def test_search_by_multiple_categories_and_logic(self):
        results = self.lib.search(categories=["standing", "frontal"])
        assert len(results) == 1

    def test_search_by_category_no_match(self):
        results = self.lib.search(categories=["jumping"])
        assert len(results) == 0

    def test_search_favorites_only(self):
        results = self.lib.search(favorites_only=True)
        assert len(results) == 1
        assert results[0].id == "pose_0002"

    def test_search_favorites_none(self):
        lib2 = _lib(self._tmp)
        lib2.add(_entry(10))
        assert lib2.search(favorites_only=True) == []

    def test_search_combined_text_and_category(self):
        results = self.lib.search(text="dynamic", categories=["standing"])
        assert len(results) == 1


# ---------------------------------------------------------------------------
# PoseLibrary — all_categories
# ---------------------------------------------------------------------------


class TestAllCategories:
    def test_empty_library(self, tmp_path):
        lib = _lib(tmp_path)
        assert lib.all_categories() == []

    def test_categories_sorted(self, tmp_path):
        lib = _lib(tmp_path)
        lib.add(_entry(1, categories=["standing", "frontal"]))
        lib.add(_entry(2, categories=["lying", "profile"]))
        cats = lib.all_categories()
        assert cats == sorted(cats)

    def test_categories_unique(self, tmp_path):
        lib = _lib(tmp_path)
        lib.add(_entry(1, categories=["standing"]))
        lib.add(_entry(2, categories=["standing", "frontal"]))
        cats = lib.all_categories()
        assert cats.count("standing") == 1

    def test_all_categories_present(self, tmp_path):
        lib = _lib(tmp_path)
        lib.add(_entry(1, categories=["standing", "frontal"]))
        lib.add(_entry(2, categories=["sitting", "profile"]))
        cats = lib.all_categories()
        assert set(cats) == {"standing", "frontal", "sitting", "profile"}


# ---------------------------------------------------------------------------
# PoseLibrary — next_id
# ---------------------------------------------------------------------------


class TestNextId:
    def test_empty_library_starts_at_1(self, tmp_path):
        lib = _lib(tmp_path)
        assert lib.next_id() == "pose_0001"

    def test_increments_after_add(self, tmp_path):
        lib = _lib(tmp_path)
        lib.add(_entry(1))
        assert lib.next_id() == "pose_0002"

    def test_skips_existing_id(self, tmp_path):
        lib = _lib(tmp_path)
        lib._poses["pose_0001"] = _entry(1)
        lib._poses["pose_0002"] = _entry(2)
        assert lib.next_id() == "pose_0003"

    def test_unique_across_multiple_calls(self, tmp_path):
        lib = _lib(tmp_path)
        ids = set()
        for _ in range(10):
            nid = lib.next_id()
            lib.add(_entry(int(nid.split("_")[1])))
            ids.add(nid)
        assert len(ids) == 10


# ---------------------------------------------------------------------------
# _entry_dict helper
# ---------------------------------------------------------------------------


class TestEntryDict:
    def test_no_id_field(self):
        e = _entry(1, categories=["standing"])
        d = _entry_dict(e)
        assert "id" not in d

    def test_has_categories(self):
        e = _entry(1, categories=["lying"])
        d = _entry_dict(e)
        assert d["categories"] == ["lying"]

    def test_has_source_path(self):
        e = _entry(1)
        d = _entry_dict(e)
        assert "source_path" in d


# ---------------------------------------------------------------------------
# utcnow_iso
# ---------------------------------------------------------------------------


class TestUtcnowIso:
    def test_returns_string(self):
        s = utcnow_iso()
        assert isinstance(s, str)

    def test_contains_T_separator(self):
        assert "T" in utcnow_iso()

    def test_ends_with_timezone(self):
        s = utcnow_iso()
        assert s.endswith("+00:00") or s.endswith("Z")

    def test_unique_calls(self):
        import time
        a = utcnow_iso()
        time.sleep(1.01)
        b = utcnow_iso()
        assert a != b

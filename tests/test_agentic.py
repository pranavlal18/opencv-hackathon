import importlib.util
import os

import pytest


def _load(name, relpath):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "scripts", relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ag = _load("searescue_agentic", "agentic_detect.py")

W, H = 1000, 1000
Q_OK = {"sharpness": 500.0, "brightness": 120.0, "contrast": 20.0}
Q_BAD = {"sharpness": 10.0, "brightness": 120.0, "contrast": 9.0}
CFG = {"confident_conf": 0.50, "large_area": 5e-4, "max_extra": 3,
       "zoom_floor": 0.10, "sharp_floor": 100.0, "allow_fallback": True}
CFG_V2 = {"confident_conf": 0.50, "large_area": 5e-4, "max_extra": 2,
          "zoom_floor": 0.15, "sharp_floor": 100.0, "allow_fallback": False,
          "swimmer_only_zoom": True, "max_hires_px": 25e6}


def sw(conf, x1=100, y1=100, x2=200, y2=200):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "class": "swimmer", "class_id": 0, "conf": conf}


def boat(conf=0.8):
    return {"x1": 10, "y1": 10, "x2": 300, "y2": 300,
            "class": "boat", "class_id": 1, "conf": conf}


def test_uncertain_swimmer_triggers_zoom():
    branch, reason, targets = ag.decide([sw(0.30)], W, H, Q_OK, CFG)
    assert branch == ag.ZOOM
    assert len(targets) == 1
    assert "uncertain" in reason


def test_small_swimmer_triggers_zoom_despite_high_conf():
    tiny = sw(0.90, x2=110, y2=110)  # 10x10 px of 1e6 = 1e-4 < 5e-4
    branch, _, targets = ag.decide([tiny], W, H, Q_OK, CFG)
    assert branch == ag.ZOOM and len(targets) == 1


def test_confident_large_detection_skips_reexam():
    big = sw(0.80, x2=400, y2=400)  # 300x300 = 0.09 rel area
    branch, reason, targets = ag.decide([big], W, H, Q_OK, CFG)
    assert branch == ag.KEEP and targets == []
    assert "no re-exam" in reason


def test_no_candidate_fallback_is_bounded():
    branch, reason, targets = ag.decide([boat()], W, H, Q_OK, CFG)
    assert branch == ag.FALLBACK and targets == []
    cfg2 = dict(CFG, allow_fallback=False)
    branch2, _, _ = ag.decide([boat()], W, H, Q_OK, cfg2)
    assert branch2 == ag.KEEP  # deterministic degradation, no unbounded loop


def test_empty_scene_without_cues_keeps():
    branch, _, targets = ag.decide([], W, H, Q_OK, CFG)
    assert branch == ag.KEEP and targets == []


def test_poor_quality_marks_uncertain():
    branch, _, targets = ag.decide([sw(0.30)], W, H, Q_BAD, CFG)
    assert branch == ag.UNCERTAIN and targets == []


def test_budget_caps_zoom_targets():
    many = [sw(0.20 + 0.01 * i, x1=10 * i, y1=10, x2=10 * i + 30, y2=40)
            for i in range(6)]
    branch, _, targets = ag.decide(many, W, H, Q_OK, CFG)
    assert branch == ag.ZOOM
    assert len(targets) == CFG["max_extra"]


def test_crop_mapping_roundtrip():
    det = sw(0.5, x1=100, y1=100, x2=200, y2=200)
    crop = ag.expand_crop(det, W, H, 1.5)  # 150x150 centred -> (75,75,225,225)
    assert crop == (75, 75, 225, 225)
    back = ag.map_to_frame([10.0, 10.0, 20.0, 20.0], crop)
    assert back == [85.0, 85.0, 95.0, 95.0]


def test_crop_clipped_at_border():
    det = sw(0.5, x1=0, y1=0, x2=50, y2=50)
    assert ag.expand_crop(det, W, H, 2.0) == (0, 0, 75, 75)


def test_nms_suppresses_duplicates_keeps_best():
    a = sw(0.9)
    b = sw(0.6)  # identical box, lower conf
    c = sw(0.7, x1=500, y1=500, x2=600, y2=600)
    out = ag.nms([b, c, a], 0.5)
    assert [d["conf"] for d in out] == [0.9, 0.7]


def test_nms_keeps_other_classes_and_regions():
    a = sw(0.9)
    b = boat(0.8)  # same pixels, other class -> kept
    out = ag.nms([a, b], 0.5)
    assert len(out) == 2


def test_v2_zoom_floor_gates_low_candidates():
    cand = sw(0.12)  # >= v1 floor 0.10 but < v2 floor 0.15
    b1, _, t1 = ag.decide([cand], W, H, Q_OK, CFG)
    assert b1 == ag.ZOOM and len(t1) == 1
    b2, _, t2 = ag.decide([cand], W, H, Q_OK, CFG_V2)
    assert b2 == ag.KEEP and t2 == []  # falls through: no eligible zoom target


def test_v2_two_call_maximum():
    many = [sw(0.20 + 0.01 * i, x1=10 * i, y1=10, x2=10 * i + 30, y2=40)
            for i in range(6)]
    _, _, targets = ag.decide(many, W, H, Q_OK, CFG_V2)
    assert len(targets) == 2


def test_v2_no_boat_only_fallback():
    branch, _, targets = ag.decide([boat()], W, H, Q_OK, CFG_V2)
    assert branch == ag.KEEP and targets == []


def test_v2_preserves_uncertain_quality():
    branch, _, targets = ag.decide([sw(0.30)], W, H, Q_BAD, CFG_V2)
    assert branch == ag.UNCERTAIN and targets == []


def test_policy_config_is_deterministic():
    import argparse
    ns = argparse.Namespace(weights="w.pt", conf=0.25, floor=0.10,
                            coarse_imgsz=640, zoom_imgsz=960, confident_conf=0.5,
                            large_area=5e-4, max_extra=None, zoom_floor=None,
                            swimmer_only_zoom=False, max_hires_mpx=None, context=1.5,
                            nms_iou=0.5, sharp_floor=100.0, no_fallback=False,
                            policy="v2")
    c1, c2 = ag.default_cfg(ns), ag.default_cfg(ns)
    assert c1 == c2
    assert c1["max_extra"] == 2 and c1["zoom_floor"] == 0.15
    assert c1["swimmer_only_zoom"] is True and c1["allow_fallback"] is False
    assert c1["max_hires_px"] == 25e6

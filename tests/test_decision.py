import importlib.util
import json
import os

import pytest


def _load(name, relpath):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "scripts", relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ds = _load("searescue_decision", "decision_service.py")
lh = _load("searescue_lambda", "lambda_handler.py")


def _req(dets=None, sharpness=500.0, motion=None, budget=3, stride=5):
    return {"request_id": "r1", "frame_ref": "frame_000001",
            "observation": {"detections": dets or [],
                            "quality": {"sharpness": sharpness,
                                        "brightness": 120.0, "contrast": 20.0},
                            "motion": motion},
            "state": {"stride": stride, "samples_used": 2, "prev_action": None},
            "budget": {"remaining_extra": budget, "max_extra": 3}}


def test_identical_inputs_identical_decisions():
    a = ds.decide_observation(_req([{"class": "swimmer", "conf": 0.3,
                                     "rel_area": 1e-4}]))
    b = ds.decide_observation(_req([{"class": "swimmer", "conf": 0.3,
                                     "rel_area": 1e-4}]))
    assert a == b
    assert a["trace_id"] == b["trace_id"]


def test_uncertain_swimmer_selects_crop_zoom():
    d = ds.decide_observation(_req([{"class": "swimmer", "conf": 0.30,
                                     "rel_area": 1e-4}]))
    assert d["action"] == "crop_zoom"
    assert d["args"]["imgsz"] == 960 and d["review"] is False
    assert d["budget_remaining"] == 2


def test_clear_scene_keeps_sparse():
    d = ds.decide_observation(_req([{"class": "swimmer", "conf": 0.9,
                                     "rel_area": 0.01}]))
    assert d["action"] == "keep_sparse_sampling" and d["review"] is False
    d2 = ds.decide_observation(_req())
    assert d2["action"] == "keep_sparse_sampling"


def test_hot_motion_densifies():
    d = ds.decide_observation(_req(
        [{"class": "boat", "conf": 0.8, "rel_area": 0.02}],
        motion={"diff_mean": 30.0, "flow_mean": 1.2}))
    assert d["action"] == "sample_densely"
    assert d["args"] == {"stride": 1}


def test_budget_blocks_escalation():
    d = ds.decide_observation(_req([{"class": "swimmer", "conf": 0.3,
                                     "rel_area": 1e-4}], budget=0))
    assert d["action"] == "request_human_review" and d["review"] is True


def test_poor_quality_requests_human_review():
    d = ds.decide_observation(_req([{"class": "swimmer", "conf": 0.3,
                                     "rel_area": 1e-4}], sharpness=10.0))
    assert d["action"] == "request_human_review" and d["review"] is True


def test_malformed_rejected():
    with pytest.raises(ds.DecisionError, match="missing field"):
        ds.decide_observation({"request_id": "x"})
    with pytest.raises(ds.DecisionError, match="unknown"):
        ds.decide_observation(_req([{"class": "submarine", "conf": 0.5,
                                     "rel_area": 0.01}]))
    with pytest.raises(ds.DecisionError, match="forbidden key"):
        bad = _req()
        bad["observation"]["image_data"] = "AAAA"
        ds.decide_observation(bad)
    big = _req()
    big["observation"]["pad"] = "x" * (33 * 1024)
    with pytest.raises(ds.DecisionError, match="exceeds"):
        ds.decide_observation(big)


def test_handler_matches_local():
    req = _req([{"class": "swimmer", "conf": 0.3, "rel_area": 1e-4}])
    local = ds.decide_observation(req)
    via_body = json.loads(lh.handler({"body": json.dumps(req)})["body"])
    via_direct = json.loads(lh.handler(req)["body"])
    for got in (via_body, via_direct):
        assert got["action"] == local["action"]
        assert got["trace_id"] == local["trace_id"]
        assert got["persisted"] is False  # no TRACE_TABLE locally


def test_handler_rejects_malformed():
    r = lh.handler({"body": json.dumps({"nope": 1})})
    assert r["statusCode"] == 400
    assert "missing field" in json.loads(r["body"])["detail"]


def test_sequential_trace_uses_updated_evidence():
    first = ds.decide_observation(_req([{"class": "swimmer", "conf": 0.3,
                                         "rel_area": 1e-4}], stride=5))
    assert first["action"] == "crop_zoom"
    follow = _req([{"class": "swimmer", "conf": 0.9, "rel_area": 0.01}],
                  stride=1, budget=first["budget_remaining"])
    follow["state"]["prev_action"] = first["action"]
    second = ds.decide_observation(follow)
    assert second["action"] == "keep_sparse_sampling"
    assert second["trace_id"] != first["trace_id"]

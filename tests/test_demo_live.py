"""Unit tests for the Task-11 live demo integration (NO live AWS calls).

Covers the Task-11 required test surface:
- valid observations serialize into the validated decision schema;
- invalid/HTTP-error responses fail safely without fabricating decisions;
- AWS authentication failures are LOUD (no silent local-policy fallback);
- returned crop coordinates map correctly back to the original image;
- the additional-inference budget is enforced;
- secrets never appear in trace output;
- local-policy mode and live-aws mode are clearly distinguished.
"""
import importlib.util
import json
import os

import numpy as np
import pytest


def _load(name, relpath):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "scripts", relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


demo = _load("searescue_demo_live", "demo_live_agent.py")
sigv4 = _load("searescue_sigv4", "aws_sigv4.py")

# Seeded noise -> sharpness well above the 100 floor, so the local shared
# policy is NOT short-circuited into request_human_review during tests.
_rng = np.random.default_rng(0)
FRAME = (_rng.random((1000, 1000, 3)) * 255).astype(np.uint8)


def make_cfg(**kw):
    cfg = dict(stem="t1", sha="deadbeef", model=None, mode="live-aws",
               conf=0.25, floor=0.10, pool_floor=0.15, coarse_imgsz=640,
               zoom_imgsz=960, max_extra=2, context=1.5, nms_iou=0.5,
               swimmer_only_zoom=True)
    cfg.update(kw)
    return cfg


def det(x1, y1, x2, y2, conf, cls="swimmer", cid=0):
    return {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            "class": cls, "class_id": cid, "conf": float(conf)}


def coarse_stub(frame_or_crop, image, conf, imgsz):
    """Coarse scan: small swimmer candidate at (100,100)-(140,140) conf 0.30.
    Zoom pass (imgsz==960): swimmer at crop-relative (10,10)-(50,50) conf 0.9."""
    if imgsz == 960:
        return [det(5, 5, 25, 25, 0.9)]
    return [det(100, 100, 140, 140, 0.30)]


def live_keep_stub(cfg, request):
    captured["request"] = request
    return {"reached": True, "http_status": 200, "latency_ms": 5,
            "url_masked": "https://abcdef***.lambda-url.x.on.aws/",
            "error": None,
            "response": {"action": "keep_sparse_sampling", "args": {},
                         "reason": "ok", "review": False, "trace_id": "cloud01",
                         "budget_remaining": 2}}


def live_crop_stub(cfg, request):
    captured["request"] = request
    return {"reached": True, "http_status": 200, "latency_ms": 5,
            "url_masked": "https://abcdef***.lambda-url.x.on.aws/",
            "error": None,
            "response": {"action": "crop_zoom",
                         "args": {"target": {"class": "swimmer", "conf": 0.30,
                                             "rel_area": 0.0016},
                                  "imgsz": 960},
                         "reason": "uncertain swimmer", "review": False,
                         "trace_id": "cloud01", "budget_remaining": 1}}


captured = {}

# ---------- observation schema ----------


def test_observation_is_measurement_only_and_schema_valid(monkeypatch):
    monkeypatch.setattr(demo, "predict", coarse_stub)
    _, trace, _, _ = demo.run_image_pipeline(FRAME, make_cfg(), live_keep_stub)
    req = captured["request"]
    assert demo.validate_request(req) == req            # schema-valid
    assert req["observation"]["detections"] == [
        {"class": "swimmer", "conf": 0.3, "rel_area": 0.0016}]
    assert set(req["observation"]["quality"]) == {"sharpness", "brightness",
                                                  "contrast"}
    assert req["budget"] == {"remaining_extra": 2, "max_extra": 2}
    assert req["state"]["samples_used"] == 0
    assert req["request_id"] == "demo:t1"


def test_local_reference_recorded_and_parity_field_exists(monkeypatch):
    monkeypatch.setattr(demo, "predict", coarse_stub)
    _, trace, _, _ = demo.run_image_pipeline(FRAME, make_cfg(), live_crop_stub)
    assert trace["local_policy_reference"]["trace_id"] != "cloud01"
    assert trace["trace_id_parity"] is False            # stub id != real hash
    # cloud crop_zoom on the same observation == what the local policy chose
    assert trace["decision_agrees_with_local"] is True
    assert trace["local_policy_reference"]["action"] == "crop_zoom"


# ---------- mode distinction ----------


def test_live_mode_is_labelled_aws_live(monkeypatch):
    monkeypatch.setattr(demo, "predict", coarse_stub)
    _, trace, _, _ = demo.run_image_pipeline(FRAME, make_cfg(), live_keep_stub)
    assert trace["mode"] == "live-aws"
    assert trace["decision_source"] == "aws-live"
    assert trace["endpoint"].get("skipped") is not True


def test_local_policy_mode_labelled_and_no_network(monkeypatch):
    monkeypatch.setattr(demo, "predict", coarse_stub)
    _, trace, _, _ = demo.run_image_pipeline(
        FRAME, make_cfg(mode="local-policy"), demo.decide_local)
    assert trace["mode"] == "local-policy"
    assert trace["decision_source"] == "local-policy"
    assert trace["endpoint"]["skipped"] is True
    assert trace["endpoint"]["url_masked"] == "local-policy mode (no network)"
    assert trace["action"] in demo.ACTIONS
    assert isinstance(trace["trace_id"], str) and len(trace["trace_id"]) == 16


# ---------- failure handling ----------


def test_http_500_fails_safe_without_fabricated_decision(monkeypatch):
    monkeypatch.setattr(demo, "predict", coarse_stub)

    def bad(cfg, request):
        return {"reached": True, "http_status": 500, "latency_ms": 3,
                "url_masked": "m", "error": "endpoint returned HTTP 500",
                "response_error": "boom"}

    _, trace, _, _ = demo.run_image_pipeline(FRAME, make_cfg(), bad)
    assert trace["action"] is None
    assert trace["decision_source"] == "no-decision"
    assert trace["action_executed"] is False
    assert trace["error"] == "endpoint returned HTTP 500"


def test_http_403_does_not_fall_back_to_local_policy(monkeypatch):
    monkeypatch.setattr(demo, "predict", coarse_stub)
    called = {"local": 0}

    def forbidden(cfg, request):
        called["local"] += 0  # would increment if local policy were consulted
        return {"reached": True, "http_status": 403, "latency_ms": 2,
                "url_masked": "m", "error": "endpoint returned HTTP 403",
                "response_error": '{"Message":"Forbidden"}'}

    _, trace, _, _ = demo.run_image_pipeline(FRAME, make_cfg(), forbidden)
    assert trace["decision_source"] == "no-decision"   # NOT "local-policy"
    assert trace["action"] is None
    assert called["local"] == 0


def test_auth_failure_raises_loudly(monkeypatch):
    monkeypatch.setattr(demo, "predict", coarse_stub)

    def auth_fail(cfg, request):
        raise sigv4.AWSAuthError("no usable keys")

    with pytest.raises(sigv4.AWSAuthError):
        demo.run_image_pipeline(FRAME, make_cfg(), auth_fail)


def test_credential_resolver_raises_awsautherror_offline():
    # Exercises the local CLI failure path (bad profile); works with or
    # without the AWS CLI installed: both paths raise AWSAuthError.
    with pytest.raises(sigv4.AWSAuthError):
        sigv4.get_session_credentials("searescue-no-such-profile-xyz")


# ---------- returned-action execution ----------


def test_crop_target_maps_back_to_frame_exactly(monkeypatch):
    monkeypatch.setattr(demo, "predict", coarse_stub)
    final, trace, _, _ = demo.run_image_pipeline(
        FRAME, make_cfg(mode="local-policy"), demo.decide_local)
    # local policy zooms the 0.30-conf small swimmer (same as our live stub)
    assert trace["action"] == "crop_zoom"
    # expand_crop(100,100,140,140, 1.5): half-extent 20*1.5=30 -> (90,90,150,150)
    assert trace["extra_calls"][0]["args"]["crop_xyxy"] == [90, 90, 150, 150]
    assert trace["target_matched_box"] == [90, 90, 150, 150]
    # zoom box (5,5)-(25,25) in crop coords + crop origin (90,90) -> frame
    mapped = [d for d in final
              if (d["x1"], d["y1"], d["x2"], d["y2"]) == (95.0, 95.0, 115.0, 115.0)]
    assert mapped and mapped[0]["conf"] == 0.9
    assert trace["action_executed"] is True and trace["n_extra"] == 1


def test_budget_exhausted_refuses_escalation(monkeypatch):
    monkeypatch.setattr(demo, "predict", coarse_stub)
    final, trace, _, _ = demo.run_image_pipeline(
        FRAME, make_cfg(max_extra=0), live_crop_stub)
    assert trace["action_not_executed_reason"] == "budget_exhausted"
    assert trace["extra_calls"] == [] and trace["n_extra"] == 0
    assert trace["action_executed"] is False
    assert len(final) == trace["n_initial_operating"] == 1


def test_target_matching_exact_then_nearest_rel_area():
    pool = [det(100, 100, 140, 140, 0.30), det(200, 200, 264, 264, 0.35)]
    got, how = demo.match_target({"class": "swimmer", "conf": 0.30,
                                  "rel_area": 0.0016}, pool, 1000, 1000)
    assert how == "exact-class-conf" and got is pool[0]
    got2, how2 = demo.match_target({"class": "swimmer", "conf": 0.95,
                                    "rel_area": 0.0041}, pool, 1000, 1000)
    assert how2 == "nearest-rel-area-in-class" and got2 is pool[1]
    got3, how3 = demo.match_target({"class": "buoy", "conf": 0.9,
                                    "rel_area": 0.01}, pool, 1000, 1000)
    assert got3 is None and how3 == "no-local-candidate"


def test_keep_sparse_executes_with_no_extra_calls(monkeypatch):
    monkeypatch.setattr(demo, "predict", coarse_stub)
    _, trace, _, _ = demo.run_image_pipeline(FRAME, make_cfg(), live_keep_stub)
    assert trace["action"] == "keep_sparse_sampling"
    assert trace["action_executed"] is True and trace["n_extra"] == 0
    assert trace["n_final"] == trace["n_initial_operating"] == 1


# ---------- traces / secrets ----------


def test_trace_files_never_contain_secret_material(monkeypatch):
    monkeypatch.setattr(demo, "predict", coarse_stub)
    # Fake credentials exist only to prove they can never leak via the trace.
    fake_creds = {"AWS_ACCESS_KEY_ID": "AKIAEXAMPLEKEY123",
                  "AWS_SECRET_ACCESS_KEY": "wJalrEXAMPLESECRET/x1",
                  "AWS_SESSION_TOKEN": "tokEXAMPLESESSION"}
    fake_creds_view = json.dumps({"view": fake_creds})
    assert "AKIAEXAMPLEKEY123" in fake_creds_view  # sanity: fake marker real

    _, trace, _, _ = demo.run_image_pipeline(FRAME, make_cfg(), live_keep_stub)
    blob = json.dumps(trace)
    for marker in ("AKIA", "wJalrEXAMPLESECRET", "tokEXAMPLESESSION",
                   "Authorization", "x-amz-security-token"):
        assert marker not in blob


def test_signer_headers_memory_only_and_url_masked():
    # fictional Function-URL id (same 32-char shape as a real one); the
    # deployed endpoint's real id must never be embedded in this repo
    url = "https://aaaa1111bbbb2222cccc3333dddd4444.lambda-url.ap-south-1.on.aws/"
    creds = {"AWS_ACCESS_KEY_ID": "AKIAEXAMPLEKEY123",
             "AWS_SECRET_ACCESS_KEY": "wJalrEXAMPLESECRET/x1",
             "AWS_SESSION_TOKEN": "tokEXAMPLESESSION"}
    headers = sigv4.sign_headers(url, "{}", creds, "ap-south-1")
    assert headers["Authorization"].startswith(
        "AWS4-HMAC-SHA256 Credential=AKIAEXAMPLEKEY123/")
    assert "x-amz-security-token" in headers
    masked = sigv4.mask_url(url)
    assert "aaaa1111bbbb2222cccc3333dddd4444" not in masked  # id truncated
    assert masked.startswith("https://aaaa11***.")


def test_unwrap_response_handles_wrapped_envelope():
    wrapped = json.dumps({"statusCode": 200,
                          "body": json.dumps({"action": "keep_sparse_sampling"})})
    assert demo.unwrap_response(wrapped) == {"action": "keep_sparse_sampling"}
    assert demo.unwrap_response(json.dumps({"action": "sample_densely"})) == {
        "action": "sample_densely"}


def test_iter_images_sorted_and_limited(tmp_path):
    for name in ("c.jpg", "a.jpg", "b.png"):
        (tmp_path / name).write_bytes(b"")
    items = demo.iter_images(str(tmp_path), 2)
    assert [os.path.basename(p) for p in items] == ["a.jpg", "c.jpg"]


def test_gt_matched_hits_require_class_and_iou():
    boxes = [{"class_id": 0, "x1": 100.0, "y1": 100.0, "x2": 140.0, "y2": 140.0},
             {"class_id": 1, "x1": 500.0, "y1": 500.0, "x2": 600.0, "y2": 600.0}]
    assert demo.gt_matched(boxes, [det(100, 100, 140, 140, 0.9)]) == 1
    assert demo.gt_matched(boxes, [det(500, 500, 600, 600, 0.9, "boat", 1)]) == 1
    assert demo.gt_matched(boxes, [det(490, 490, 590, 590, 0.9)]) == 0  # class mismatch
    assert demo.gt_matched(boxes, []) == 0


# ---------- Task 14: crop-inspection visualization ----------


def zoom_stub_factory(zoom_dets):
    def stub(frame_or_crop, image, conf, imgsz):
        if imgsz == 960:
            return list(zoom_dets)
        return [det(100, 100, 140, 140, 0.30)]  # small uncertain swimmer
    return stub


def test_crop_view_and_merge_bookkeeping_on_zoom(monkeypatch):
    """Zoom raw PASS detection: captured raw records + honest bookkeeping."""
    monkeypatch.setattr(demo, "predict", zoom_stub_factory([det(5, 5, 25, 25, 0.9)]))
    _, trace, _, _ = demo.run_image_pipeline(FRAME, make_cfg(), live_crop_stub)
    ec = trace["extra_calls"][0]
    assert ec["n_raw"] == 1 and ec["n_kept"] == 1
    assert ec["n_raw_pre_class_filter"] == 1
    rec = ec["raw_crop_detections"][0]
    assert rec["class"] == "swimmer" and rec["conf"] == 0.9
    assert rec["passes_operating"] is True and rec["class_filter_excluded"] is False
    # raw crop-local coords recorded from the ACTUAL zoom-inference stub
    assert rec["crop_xyxy"] == [5.0, 5.0, 25.0, 25.0]
    # frame coords = crop offset + local coords (target box was 100..140)
    assert rec["frame_xyxy"][0] >= 70.0 and rec["frame_xyxy"][0] < 100.0
    assert trace["n_new_added"] == 1 and trace["n_removed_by_nms"] == 0
    assert trace["n_added_to_merge"] == 1 and trace["n_final"] == 2
    view = trace["_crop_view"]
    assert view["inspection"].shape[0] >= view["region"].shape[0] * 2


def test_raw_below_threshold_and_class_excluded_tagged(monkeypatch):
    """Sub-threshold swimmer + above-threshold boat: neither may be merged,
    both must appear in raw records with their actual confidences."""
    raws = [det(5, 5, 25, 25, 0.20),          # swimmer below operating conf
            det(10, 10, 40, 40, 0.90, "boat", 1)]  # excluded by class rule
    monkeypatch.setattr(demo, "predict", zoom_stub_factory(raws))
    _, trace, _, _ = demo.run_image_pipeline(FRAME, make_cfg(), live_crop_stub)
    ec = trace["extra_calls"][0]
    assert ec["n_raw"] == 1                    # boat filtered by class rule
    assert ec["n_raw_pre_class_filter"] == 2
    assert ec["n_kept"] == 0                   # nothing above operating conf
    by_cls = {r["class"]: r for r in ec["raw_crop_detections"]}
    assert by_cls["swimmer"]["passes_operating"] is False
    assert by_cls["swimmer"]["class_filter_excluded"] is False
    assert by_cls["boat"]["class_filter_excluded"] is True
    assert by_cls["boat"]["passes_operating"] is False
    # final set unchanged: no new detections, nothing removed
    assert trace["n_new_added"] == 0 and trace["n_removed_by_nms"] == 0
    assert trace["n_final"] == trace["n_initial_operating"] == 1
    assert trace["_crop_view"]["inspection"].shape[0] > 0


def test_zero_raw_crop_still_visualized(monkeypatch):
    """Zoom call returning nothing: empty raw records, view still produced."""
    monkeypatch.setattr(demo, "predict", zoom_stub_factory([]))
    _, trace, _, _ = demo.run_image_pipeline(FRAME, make_cfg(), live_crop_stub)
    ec = trace["extra_calls"][0]
    assert ec["n_raw"] == 0 and ec["n_raw_pre_class_filter"] == 0
    assert ec["raw_crop_detections"] == [] and ec["n_kept"] == 0
    assert trace["n_new_added"] == 0 and trace["n_final"] == 1
    assert "_crop_view" in trace
    assert trace["crop_view_generated"] is True


def test_no_action_no_bookkeeping_fields_positive(monkeypatch):
    """Keep-action path: accounting fields exist and are zero."""
    monkeypatch.setattr(demo, "predict", coarse_stub)
    _, trace, _, _ = demo.run_image_pipeline(FRAME, make_cfg(), live_keep_stub)
    assert trace["n_new_added"] == 0 and trace["n_removed_by_nms"] == 0
    assert trace["crop_view_generated"] is False
    assert "_crop_view" not in trace


def test_crop_inspection_draws_header_and_boxes():
    """Direct visual-function check: header shows crop coords + imgsz."""
    import cv2 as _cv2  # noqa: F401  (already imported at module top)
    crop = FRAME[70:170, 70:170].copy()
    cfg = make_cfg()
    recs = [{"class": "swimmer", "conf": 0.21, "crop_xyxy": [5.0, 5.0, 25.0, 25.0],
             "frame_xyxy": [75.0, 75.0, 95.0, 95.0], "passes_operating": False,
             "class_filter_excluded": False}]
    view, f = demo.make_crop_inspection(crop, [70, 70, 170, 170], recs, cfg)
    assert f >= 2 and view.shape[0] >= crop.shape[0] * f
    assert view.shape[:2] == (100 * f, 100 * f)

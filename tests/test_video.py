import importlib.util
import json
import os

import cv2
import numpy as np
import pytest


def _load(name, relpath):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "scripts", relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


av = _load("searescue_video", "analyze_video.py")

WEIGHTS = os.path.join("weights", "best.pt")
needs_model = pytest.mark.skipif(not os.path.isfile(WEIGHTS),
                                 reason="needs weights (gitignored)")


def _moving_clip(path, n=12, w=320, h=240):
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (w, h))
    assert wr.isOpened()
    for i in range(n):
        img = np.zeros((h, w, 3), dtype=np.uint8) + 30
        x = (i * 20) % (w - 40)
        cv2.rectangle(img, (x, 60), (x + 40, 140), (255, 255, 255), -1)
        wr.write(img)
    wr.release()


def test_motion_densifies_and_quiet_relaxes():
    hot = {"diff_mean": 50.0, "flow_mean": 2.0}
    s, a, r = av.decide_stride(hot, [], 5, 5, 10, 8.0, 0.4)
    assert (s, a) == (1, "densify") and "motion" in r
    quiet = {"diff_mean": 0.5, "flow_mean": 0.01}
    s2, a2, _ = av.decide_stride(quiet, [], 5, 5, 10, 8.0, 0.4)
    assert (s2, a2) == (10, "relax")
    s3, a3, _ = av.decide_stride(None, [], 5, 5, 10, 8.0, 0.4)
    assert (s3, a3) == (10, "relax")  # first frame: no motion yet


def test_uncertain_swimmer_densifies():
    dets = [{"class": "swimmer", "conf": 0.30}]
    s, a, r = av.decide_stride({"diff_mean": 0.1, "flow_mean": 0.01},
                               dets, 5, 5, 10, 8.0, 0.4)
    assert (s, a) == (1, "densify") and "uncertain" in r


def test_confident_other_class_holds_at_max():
    dets = [{"class": "boat", "conf": 0.9}]
    s, a, _ = av.decide_stride({"diff_mean": 0.1, "flow_mean": 0.01},
                               dets, 10, 5, 10, 8.0, 0.4)
    assert (s, a) == (10, "hold")


def test_motion_signal_measures_movement():
    a = np.zeros((100, 100), dtype=np.uint8)
    b = a.copy()
    cv2.rectangle(b, (20, 20), (60, 60), 255, -1)
    still = av.motion_signals(a, a.copy())
    moved = av.motion_signals(a, b)
    assert still["diff_mean"] == 0.0
    assert moved["diff_mean"] > 1.0 and moved["flow_mean"] >= 0.0


@needs_model
def test_video_fixture_end_to_end(tmp_path):
    clip = str(tmp_path / "move.avi")
    _moving_clip(clip)
    rc = av.main(["--input", clip, "--output", str(tmp_path / "out"),
                  "--stride", "2", "--max-stride", "4", "--max-frames", "6"])
    assert rc == 0
    ev = json.load(open(tmp_path / "out" / "events.json"))
    assert ev["fps"] == pytest.approx(5.0)
    assert ev["frames_sampled"] > 0
    f0 = ev["events"][0]
    assert f0["frame"] == 0 and f0["t_s"] == pytest.approx(0.0)
    for e, nxt in zip(ev["events"], ev["events"][1:]):
        assert nxt["t_s"] > e["t_s"]  # timestamps strictly increase
        assert nxt["frame"] > e["frame"]
    for line in open(tmp_path / "out" / "traces.jsonl"):
        t = json.loads(line)
        assert {"frame", "t_s", "decision", "reason",
                "stride_before", "stride_after", "motion"} <= set(t)
        assert 1 <= t["stride_after"] <= 4  # bounded

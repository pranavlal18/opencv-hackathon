import importlib.util
import json
import os

import cv2
import pytest


def _load_detect():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "scripts", "detect.py")
    spec = importlib.util.spec_from_file_location("searescue_detect", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


detect_main = _load_detect().main

SAMPLE = os.path.join("data", "samples", "dl", "data", "images", "val", "shard_000", "10416.jpg")
WEIGHTS = os.path.join("weights", "best.pt")

requires_assets = pytest.mark.skipif(
    not (os.path.isfile(SAMPLE) and os.path.isfile(WEIGHTS)),
    reason="needs sample image + weights (both gitignored, fetched in Task 2)",
)


def test_missing_weights_is_actionable(tmp_path, capsys):
    rc = detect_main(["--input", SAMPLE, "--output", str(tmp_path),
                      "--weights", "weights/does_not_exist.pt"])
    assert rc == 2
    assert "weights not found" in capsys.readouterr().err


def test_missing_input_is_actionable(tmp_path, capsys):
    rc = detect_main(["--input", "nope.jpg", "--output", str(tmp_path)])
    assert rc == 2
    assert "input not found" in capsys.readouterr().err


@requires_assets
def test_image_inference_writes_json_and_annotated(tmp_path):
    rc = detect_main(["--input", SAMPLE, "--output", str(tmp_path)])
    assert rc == 0
    files = os.listdir(tmp_path)
    js = [f for f in files if f.endswith("_detections.json")]
    ann = [f for f in files if f.endswith("_annotated.jpg")]
    assert js and ann
    with open(os.path.join(tmp_path, js[0])) as f:
        payload = json.load(f)
    assert payload["width"] > 0 and payload["height"] > 0
    assert isinstance(payload["detections"], list) and len(payload["detections"]) > 0
    d0 = payload["detections"][0]
    assert {"x1", "y1", "x2", "y2", "class", "class_id", "conf"} <= set(d0)
    assert 0.0 < d0["conf"] <= 1.0
    img = cv2.imread(os.path.join(tmp_path, ann[0]))
    assert img is not None and img.size > 0

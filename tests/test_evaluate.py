import importlib.util
import os

import pytest


def _load_eval():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "scripts", "evaluate.py")
    spec = importlib.util.spec_from_file_location("searescue_evaluate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load_eval()

BOX = [10.0, 10.0, 50.0, 50.0]  # 40x40
SHIFT = [12.0, 12.0, 52.0, 52.0]  # IoU with BOX = 34*34/(1600+1600-1156) ~ 0.565
FAR = [200.0, 200.0, 240.0, 240.0]  # IoU 0


def test_perfect_predictions():
    preds = [(0, BOX, 0.9)]
    gts = [(0, *BOX)]
    tp, fp, matched = ev.match_image(preds, gts, 0.5)
    assert tp == [1] and fp == [0] and matched == {0}
    p, r, f = ev.prf(1, 0, 0)
    assert (p, r, f) == (1.0, 1.0, 1.0)
    assert ev.average_precision([0.9], [1], 1) == pytest.approx(1.0)


def test_unmatched_prediction_is_false_positive():
    preds = [(0, FAR, 0.8)]
    gts = [(0, *BOX)]
    tp, fp, matched = ev.match_image(preds, gts, 0.5)
    assert tp == [0] and fp == [1] and matched == set()


def test_missed_ground_truth_is_false_negative():
    tp, fp, matched = ev.match_image([], [(0, *BOX)], 0.5)
    assert tp == [] and fp == [] and matched == set()
    p, r, f = ev.prf(0, 0, 1)
    assert p == 0.0 and r == 0.0 and f == 0.0


def test_duplicate_predictions_share_one_box():
    preds = [(0, BOX, 0.9), (0, list(BOX), 0.7)]
    gts = [(0, *BOX)]
    tp, fp, matched = ev.match_image(preds, gts, 0.5)
    assert sum(tp) == 1 and sum(fp) == 1 and matched == {0}
    assert tp[0] == 1  # higher-confidence duplicate wins


def test_wrong_class_does_not_match():
    preds = [(1, BOX, 0.9)]
    gts = [(0, *BOX)]
    tp, fp, matched = ev.match_image(preds, gts, 0.5)
    assert tp == [0] and fp == [1] and matched == set()


def test_low_iou_does_not_match():
    preds = [(0, FAR, 0.9)]
    gts = [(0, *BOX)]
    tp, fp, _ = ev.match_image(preds, gts, 0.5)
    assert tp == [0] and fp == [1]
    # but a near-duplicate above threshold does
    tp2, _, _ = ev.match_image([(0, SHIFT, 0.9)], [(0, *BOX)], 0.5)
    assert tp2 == [1]


def test_missing_annotation_is_actionable():
    with pytest.raises(FileNotFoundError, match="missing annotation"):
        ev.load_labels("does/not/exist.txt", 100, 100)


def test_malformed_annotation_is_actionable(tmp_path):
    bad = tmp_path / "x.txt"
    bad.write_text("0 0.5 0.5\n")  # only 3 fields
    with pytest.raises(ValueError, match="malformed line 1"):
        ev.load_labels(str(bad), 100, 100)
    bad2 = tmp_path / "y.txt"
    bad2.write_text("9 0.5 0.5 0.1 0.1\n")  # class id out of range
    with pytest.raises(ValueError, match="out of range"):
        ev.load_labels(str(bad2), 100, 100)


def test_zero_detections_all_fn():
    gts = [(0, *BOX), (1, *FAR)]
    tp, fp, matched = ev.match_image([], gts, 0.5)
    assert matched == set()
    p, r, f = ev.prf(0, 0, len(gts))
    assert (p, r, f) == (0.0, 0.0, 0.0)


def test_conf_split_keeps_ap_floor():
    preds_all = [(0, BOX, 0.9), (0, list(BOX), 0.10)]
    operating = ev.split_at_conf(preds_all, 0.25)
    assert operating == [(0, BOX, 0.9)]
    # the low-confidence duplicate still belongs to the AP pool
    assert len(preds_all) == 2


def test_ap_ranks_globally_by_confidence():
    # High-confidence FP ranked above a low-confidence TP: AP must be 0.5,
    # proving the low-floor TP is pooled (a 0.25-filtered AP would see no TP).
    scores = [0.90, 0.10]
    tp_flags = [0, 1]
    assert ev.average_precision(scores, tp_flags, 1) == pytest.approx(0.5)
    # Perfect global ranking gives 1.0 even with mixed confidences.
    assert ev.average_precision([0.9, 0.1], [1, 1], 2) == pytest.approx(1.0)

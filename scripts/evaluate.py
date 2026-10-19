"""Quantitative detector evaluation on labelled SeaDronesSee val images.

Metric definitions (documented here and in outputs):
- Confidence threshold (--conf, default 0.25): precision/recall/F1 use only
  predictions at or above it.
- AP confidence floor (--ap-conf, default 0.05): mAP@50 pools predictions at
  or above this lower floor and ranks them globally by confidence, per the
  standard definition. A single inference runs at min(conf, ap_conf).
- IoU threshold (--iou, default 0.50): box overlap for a match, xyxy format.
- Matching is per image and per class, greedy by descending confidence:
  each prediction matches the unmatched GT box of the same class with the
  highest IoU, iff IoU >= threshold. One GT matches at most one prediction.
- TP = matched predictions; FP = unmatched predictions; FN = unmatched GT.
- Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1 = harmonic mean (0 if P+R=0).
- mAP@50: mean over classes of all-point interpolated AP (VOC2010 style):
  detections ranked by confidence, AP = sum_k (R_k - R_{k-1}) * max_{j>=k} P_j.
  Classes with zero GT boxes are excluded from the mean (noted in output).
- LIMITATION (documented, not hidden): the AP pool contains only predictions
  at or above --ap-conf (default 0.05); predictions below that floor are
  truncated by construction, so the reported value is mAP@50 over conf>=0.05,
  a lower bound on the textbook all-scores mAP. Recorded as map50_note.
- Zero-detection images are included (their GT becomes FN, no silent discard).

Usage:
  python scripts/evaluate.py --images <dir> --labels <dir> --output <dir>
      [--weights weights/best.pt] [--conf 0.25] [--iou 0.5] [--imgsz 640]
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time

import cv2

CLASS_NAMES = ["swimmer", "boat", "jetski", "life_saving_appliances", "buoy"]


def fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 2


def load_labels(path: str, img_w: int, img_h: int):
    """Parse a YOLO-format label file to (class_id, x1,y1,x2,y2) in pixels."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"missing annotation file: {path} (expected one .txt per image)")
    boxes = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"malformed line {ln} in {path}: expected 5 fields, got {len(parts)}")
            try:
                c = int(parts[0])
                x, y, bw, bh = (float(parts[1]), float(parts[2]),
                                float(parts[3]), float(parts[4]))
            except ValueError:
                raise ValueError(f"malformed line {ln} in {path}: non-numeric value")
            if not 0 <= c < len(CLASS_NAMES):
                raise ValueError(f"malformed line {ln} in {path}: class id {c} out of range")
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                raise ValueError(f"malformed line {ln} in {path}: box out of [0,1] range")
            x1 = (x - bw / 2) * img_w
            y1 = (y - bh / 2) * img_h
            x2 = (x + bw / 2) * img_w
            y2 = (y + bh / 2) * img_h
            boxes.append((c, x1, y1, x2, y2))
    return boxes


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_image(preds, gts, iou_thr):
    """Greedy conf-desc matching. preds: (class_id, box4, conf). gts: (class_id, box4).

    Returns (tp_list, fp_list) aligned with preds order, and set of matched gt idx."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i][2])
    matched_gt = set()
    tp = [0] * len(preds)
    fp = [0] * len(preds)
    for i in order:
        pc, pb, _ = preds[i]
        best, best_iou = -1, iou_thr
        for j, g in enumerate(gts):
            gc, gb = g[0], g[1:]
            if gc != pc or j in matched_gt:
                continue
            v = iou(pb, gb)
            if v >= best_iou:
                best, best_iou = j, v
        if best >= 0:
            matched_gt.add(best)
            tp[i] = 1
        else:
            fp[i] = 1
    return tp, fp, matched_gt


def split_at_conf(preds_all, conf):
    """Split low-floor predictions into the P/R/F1 operating set.

    The AP pool keeps the full low-floor list; only P/R/F1 use this split."""
    return [p for p in preds_all if p[2] >= conf]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def average_precision(scores, tp_flags, n_gt) -> float:
    """All-point interpolated AP (VOC2010+). scores desc-ranked; tp_flags 0/1."""
    if n_gt == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp_cum, fp_cum, prec, rec = 0, 0, [], []
    for i in order:
        if tp_flags[i]:
            tp_cum += 1
        else:
            fp_cum += 1
        prec.append(tp_cum / (tp_cum + fp_cum))
        rec.append(tp_cum / n_gt)
    ap = 0.0
    prev_r = 0.0
    for k in range(len(prec)):
        p_interp = max(prec[k:])
        ap += (rec[k] - prev_r) * p_interp
        prev_r = rec[k]
    return ap


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate SeaRescue detector on labelled images")
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", default=os.path.join("weights", "best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--ap-conf", type=float, default=0.05,
                    help="lower confidence floor for the mAP pool (P/R/F1 still use --conf)")
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--preds-dir", default=None,
                    help="score saved *_final.json predictions instead of running the "
                         "model (same matching/AP code; latency not measured here)")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.images):
        return fail(f"images dir not found: {args.images}")
    if not os.path.isdir(args.labels):
        return fail(f"labels dir not found: {args.labels}")
    if args.preds_dir is None and not os.path.isfile(args.weights):
        return fail(f"weights not found: {args.weights}")
    if not 0.0 < args.conf < 1.0:
        return fail(f"--conf must be in (0,1), got {args.conf}")
    if not 0.0 < args.ap_conf <= args.conf:
        return fail(f"--ap-conf must be in (0,--conf], got {args.ap_conf}")
    os.makedirs(args.output, exist_ok=True)

    try:
        from ultralytics import YOLO
        import torch
    except ImportError as exc:
        return fail(f"missing dependency: {exc} (pip install -r requirements.txt)")

    images = sorted(f for f in os.listdir(args.images) if f.lower().endswith(".jpg"))
    if not images:
        return fail(f"no .jpg images in {args.images}")
    model = None
    if args.preds_dir is None:
        if not os.path.isfile(args.weights):
            return fail(f"weights not found: {args.weights}")
        model = YOLO(args.weights)

        # Warmup (not timed): lets torch/oneDNN init before measurements.
        warm = cv2.imread(os.path.join(args.images, images[0]))
        run_conf = min(args.conf, args.ap_conf)
        for _ in range(2):
            model.predict(source=warm, conf=run_conf, imgsz=args.imgsz, verbose=False)
    elif not os.path.isdir(args.preds_dir):
        return fail(f"preds dir not found: {args.preds_dir}")

    per_image = []
    # per-class accumulators
    stats = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0} for c in range(len(CLASS_NAMES))}
    ap_pool = {c: {"scores": [], "tp": [], "n_gt": 0} for c in range(len(CLASS_NAMES))}
    latencies = []

    for name in images:
        img_path = os.path.join(args.images, name)
        frame = cv2.imread(img_path)
        if frame is None:
            return fail(f"could not decode image: {img_path}")
        h, w = frame.shape[:2]
        base = os.path.splitext(name)[0]
        try:
            gts = load_labels(os.path.join(args.labels, base + ".txt"), w, h)
        except (FileNotFoundError, ValueError) as exc:
            return fail(str(exc))
        t0 = time.time()
        if model is not None:
            res = model.predict(source=frame, conf=run_conf, imgsz=args.imgsz, verbose=False)[0]
            dt = time.time() - t0
            latencies.append(dt)
            preds_all = []
            if res.boxes is not None and len(res.boxes):
                for i in range(len(res.boxes)):
                    preds_all.append((int(res.boxes.cls[i]),
                                      [float(v) for v in res.boxes.xyxy[i].tolist()],
                                      float(res.boxes.conf[i])))
        else:
            pj = os.path.join(args.preds_dir, base + "_final.json")
            if not os.path.isfile(pj):
                return fail(f"missing predictions file: {pj}")
            try:
                with open(pj) as f:
                    saved = json.load(f)
                preds_all = []
                for d in saved["detections"]:
                    cid = int(d.get("class_id", CLASS_NAMES.index(d["class"])))
                    preds_all.append((cid, [float(d["x1"]), float(d["y1"]),
                                            float(d["x2"]), float(d["y2"])],
                                      float(d["conf"])))
            except (KeyError, ValueError, TypeError) as exc:
                return fail(f"malformed predictions file {pj}: {exc}")
            dt = 0.0
        # P/R/F1 operate at --conf; the AP pool keeps the lower floor.
        preds = split_at_conf(preds_all, args.conf)
        tp, fp, matched = match_image(preds, gts, args.iou)
        img_tp, img_fp = sum(tp), sum(fp)
        img_fn = len(gts) - len(matched)
        # AP pool: match the full low-floor set, ranked globally later.
        ap_tp, _, _ = match_image(preds_all, gts, args.iou)
        for c, s in stats.items():
            c_tp = sum(t for k, t in enumerate(tp) if preds[k][0] == c)
            c_fp = sum(f for k, f in enumerate(fp) if preds[k][0] == c)
            c_gt = sum(1 for g in gts if g[0] == c)
            s["tp"] += c_tp
            s["fp"] += c_fp
            s["fn"] += c_gt - sum(1 for j in matched if gts[j][0] == c)
            s["gt"] += c_gt
            s["pred"] += c_tp + c_fp
        for k, (pc, _, conf) in enumerate(preds_all):
            ap_pool[pc]["scores"].append(conf)
            ap_pool[pc]["tp"].append(ap_tp[k])
        for g in gts:
            ap_pool[g[0]]["n_gt"] += 1
        per_image.append({"image": name, "width": w, "height": h,
                          "infer_s": round(dt, 3), "n_gt": len(gts),
                          "n_pred": len(preds), "tp": img_tp,
                          "fp": img_fp, "fn": img_fn})

    per_class = {}
    for c, s in stats.items():
        p, r, f = prf(s["tp"], s["fp"], s["fn"])
        ap50 = average_precision(ap_pool[c]["scores"], ap_pool[c]["tp"], ap_pool[c]["n_gt"])
        per_class[CLASS_NAMES[c]] = {
            "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
            "n_gt": s["gt"], "n_pred": s["pred"],
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "ap50": round(ap50, 4) if ap50 == ap50 else None,
        }
    T = sum(s["tp"] for s in stats.values())
    FP = sum(s["fp"] for s in stats.values())
    FN = sum(s["fn"] for s in stats.values())
    P, R, F = prf(T, FP, FN)
    aps = [v["ap50"] for v in per_class.values() if v["ap50"] is not None]
    lat_sorted = sorted(latencies)
    def pct(q):
        if not lat_sorted:
            return 0.0
        i = min(len(lat_sorted) - 1, int(q * len(lat_sorted)))
        return lat_sorted[i]
    summary = {
        "mode": "preds" if args.preds_dir else "model",
        "weights": args.weights,
        "weights_sha256": (sha256_file(args.weights)
                           if os.path.isfile(args.weights) else None),
        "conf_threshold": args.conf, "ap_conf_floor": args.ap_conf,
        "iou_threshold": args.iou, "imgsz": args.imgsz,
        "n_images": len(images),
        "machine": platform.platform(), "cpu": platform.processor() or platform.machine(),
        "device": "cpu",
        "versions": {"opencv": cv2.__version__,
                     "torch": torch.__version__,
                     "ultralytics": __import__("ultralytics").__version__,
                     "numpy": __import__("numpy").__version__},
        "overall": {"tp": T, "fp": FP, "fn": FN, "n_gt": T + FN, "n_pred": T + FP,
                    "precision": round(P, 4), "recall": round(R, 4), "f1": round(F, 4),
                    "map50": round(sum(aps) / len(aps), 4) if aps else None,
                    "map50_note": f"mAP@50 over predictions conf>={args.ap_conf} "
                                  f"(below-floor predictions truncated; lower bound)"},
        "latency_s": ({"n": len(lat_sorted), "min": round(lat_sorted[0], 3),
                       "median": round(pct(0.5), 3), "p95": round(pct(0.95), 3),
                       "max": round(lat_sorted[-1], 3),
                       "mean": round(sum(lat_sorted) / len(lat_sorted), 3)}
                      if lat_sorted else None),
        "per_class": per_class,
        "per_image": per_image,
    }
    with open(os.path.join(args.output, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output, "results.csv"), "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["class", "tp", "fp", "fn", "n_gt", "n_pred",
                       "precision", "recall", "f1", "ap50"])
        for cname, v in per_class.items():
            wcsv.writerow([cname, v["tp"], v["fp"], v["fn"], v["n_gt"], v["n_pred"],
                           v["precision"], v["recall"], v["f1"], v["ap50"]])
        o = summary["overall"]
        wcsv.writerow(["OVERALL", o["tp"], o["fp"], o["fn"], o["n_gt"], o["n_pred"],
                       o["precision"], o["recall"], o["f1"], o["map50"]])
    print(f"images={len(images)} conf={args.conf} iou={args.iou} imgsz={args.imgsz}")
    o = summary["overall"]
    print(f"OVERALL P={o['precision']} R={o['recall']} F1={o['f1']} mAP@50={o['map50']} "
          f"(TP={o['tp']} FP={o['fp']} FN={o['fn']} GT={o['n_gt']} pred={o['n_pred']})")
    for cname, v in per_class.items():
        print(f"  {cname:22s} P={v['precision']:.3f} R={v['recall']:.3f} "
              f"F1={v['f1']:.3f} AP50={v['ap50']} (TP={v['tp']} FP={v['fp']} FN={v['fn']} GT={v['n_gt']})")
    L = summary["latency_s"]
    if L is not None:
        print(f"latency_s: median={L['median']} p95={L['p95']} mean={L['mean']} (n={L['n']})")
    else:
        print("latency_s: n/a (preds mode; see agent traces for timing)")
    print(f"wrote {args.output}/results.json + results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

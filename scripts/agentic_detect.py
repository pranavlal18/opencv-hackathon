"""Smallest defensible Agentic Vision loop (image-based, deterministic).

State machine: COARSE_SCAN -> DECIDE -> (KEEP | CROP_ZOOM | FULL960 | UNCERTAIN)
-> MERGE -> TRACE. No orchestration framework; plain functions so every
branch is unit-testable with synthetic inputs.

Thresholds are engineering heuristics, NOT calibrated probabilities:
- floor (default 0.10): coarse candidates preserved below the operating point.
- confident_conf (0.50) + large_area (5e-4 rel. area): a swimmer at/above both
  is "clear and large" -> keep coarse results, no extra inference.
- Uncertain/small swimmer candidates -> crop-and-zoom at 960 (context-expanded).
- No swimmer candidate + ambiguous scene (other objects, or low detail)
  -> one bounded full-image 960 fallback.
- Poor quality (sharpness < sharp_floor) with no confident swimmer -> UNCERTAIN.
- Budget: at most --max-extra additional inference calls per image.
- Policies: --policy v1 (Task-4 baseline) or v2 (retuned: zoom needs >= 0.15
  conf, swimmer-only zoom outputs, max 2 extra calls, no boat/detail
  fallback, 25 Mpx hires cap). Explicit flags override policy presets.

Quality signals (OpenCV 5, on a 960x540 proxy): Laplacian variance
(sharpness), mean brightness, std contrast.

Usage:
  single: python scripts/agentic_detect.py --input img.jpg --output out/
  batch:  python scripts/agentic_detect.py --images <dir> --output out/
"""
import argparse
import hashlib
import json
import os
import sys
import time

import cv2

CLASS_NAMES = ["swimmer", "boat", "jetski", "life_saving_appliances", "buoy"]
SWIMMER = 0

# Branches (stable strings, asserted in tests and traces).
KEEP = "confident_keep"
ZOOM = "crop_zoom"
FALLBACK = "full960_fallback"
UNCERTAIN = "uncertain_quality"


def fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 2


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quality_signals(frame) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    proxy = cv2.resize(gray, (960, 540), interpolation=cv2.INTER_AREA)
    return {
        "sharpness": float(cv2.Laplacian(proxy, cv2.CV_64F).var()),
        "brightness": float(proxy.mean()),
        "contrast": float(proxy.std()),
    }


def predict(model, image, conf, imgsz):
    res = model.predict(source=image, conf=conf, imgsz=imgsz, verbose=False)[0]
    dets = []
    if res.boxes is not None and len(res.boxes):
        for i in range(len(res.boxes)):
            xy = [float(v) for v in res.boxes.xyxy[i].tolist()]
            dets.append({"x1": xy[0], "y1": xy[1], "x2": xy[2], "y2": xy[3],
                         "class": CLASS_NAMES[int(res.boxes.cls[i])],
                         "class_id": int(res.boxes.cls[i]),
                         "conf": float(res.boxes.conf[i])})
    return dets


def rel_area(det, img_w, img_h) -> float:
    return max(0.0, det["x2"] - det["x1"]) * max(0.0, det["y2"] - det["y1"]) / (img_w * img_h)


def decide(coarse, img_w, img_h, quality, cfg):
    """Pure decision function -> (branch, reason, targets).

    coarse: candidate detections (>= floor). targets: dets selected for zoom."""
    swimmers = [d for d in coarse if d["class_id"] == SWIMMER]
    for d in coarse:
        d["_area"] = rel_area(d, img_w, img_h)
    confident = [d for d in swimmers
                 if d["conf"] >= cfg["confident_conf"] and d["_area"] >= cfg["large_area"]]
    uncertain = [d for d in swimmers
                 if not (d["conf"] >= cfg["confident_conf"] and d["_area"] >= cfg["large_area"])]
    poor_quality = quality["sharpness"] < cfg["sharp_floor"]
    if poor_quality and not confident:
        return UNCERTAIN, (
            f"sharpness {quality['sharpness']:.1f} < floor {cfg['sharp_floor']} "
            f"and no confident swimmer ({len(confident)}); marking uncertain"), []
    if uncertain:
        # Least-confident / smallest first; zoom floor + budget cap the count.
        eligible = [d for d in uncertain if d["conf"] >= cfg["zoom_floor"]]
        ordered = sorted(eligible, key=lambda d: (d["conf"], d["_area"]))
        targets = ordered[:cfg["max_extra"]]
        if targets:
            return ZOOM, (
                f"{len(uncertain)} uncertain/small swimmer candidate(s), "
                f"{len(eligible)} >= zoom floor {cfg['zoom_floor']}; "
                f"zooming top {len(targets)}"), targets
    if confident:
        biggest = max(d["_area"] for d in confident)
        return KEEP, (
            f"{len(confident)} confident swimmer(s) >= {cfg['confident_conf']} conf "
            f"and area >= {cfg['large_area']:.0e} (max {biggest:.2e}); no re-exam"), []
    others = [d for d in coarse if d["class_id"] != SWIMMER]
    ambiguous = bool(others) or poor_quality
    if ambiguous and cfg["allow_fallback"]:
        why = f"{len(others)} non-swimmer cue(s)" if others else \
            f"low detail (sharpness {quality['sharpness']:.1f})"
        return FALLBACK, f"no swimmer candidate despite ambiguous scene ({why}); one bounded 960 scan", []
    return KEEP, "no swimmer candidate and no ambiguity cues; nothing to re-examine", []


def expand_crop(det, img_w, img_h, context):
    cx = (det["x1"] + det["x2"]) / 2
    cy = (det["y1"] + det["y2"]) / 2
    hw = (det["x2"] - det["x1"]) / 2 * context
    hh = (det["y2"] - det["y1"]) / 2 * context
    x1 = max(0, int(cx - hw))
    y1 = max(0, int(cy - hh))
    x2 = min(img_w, int(cx + hw))
    y2 = min(img_h, int(cy + hh))
    return (x1, y1, x2, y2)


def map_to_frame(box, crop):
    cx1, cy1, _, _ = crop
    return [box[0] + cx1, box[1] + cy1, box[2] + cx1, box[3] + cy1]


def nms(dets, iou_thr):
    """Per-class greedy NMS, conf-desc. Pure function (unit-tested)."""
    out = []
    for c in sorted(set(d["class_id"] for d in dets)):
        group = sorted([d for d in dets if d["class_id"] == c],
                       key=lambda d: -d["conf"])
        kept = []
        for d in group:
            keep = True
            for k in kept:
                ix1, iy1 = max(d["x1"], k["x1"]), max(d["y1"], k["y1"])
                ix2, iy2 = min(d["x2"], k["x2"]), min(d["y2"], k["y2"])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                a = max(0.0, d["x2"] - d["x1"]) * max(0.0, d["y2"] - d["y1"])
                b = max(0.0, k["x2"] - k["x1"]) * max(0.0, k["y2"] - k["y1"])
                u = a + b - inter
                if u > 0 and inter / u > iou_thr:
                    keep = False
                    break
            if keep:
                kept.append(d)
        out.extend(kept)
    return sorted(out, key=lambda d: -d["conf"])


def annotate(frame, dets):
    for d in dets:
        x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        label = f'{d["class"]} {d["conf"]:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        s = max(1, frame.shape[1] // 1280)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return frame


class Agent:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg

    def run(self, img_path):
        t_start = time.time()
        frame = cv2.imread(img_path)
        if frame is None:
            raise ValueError(f"could not decode image: {img_path}")
        img_h, img_w = frame.shape[:2]
        quality = quality_signals(frame)
        t0 = time.time()
        coarse_all = predict(self.model, frame, self.cfg["floor"], self.cfg["coarse_imgsz"])
        coarse_s = time.time() - t0
        coarse = [d for d in coarse_all if d["conf"] >= self.cfg["floor"]]
        operating = [d for d in coarse if d["conf"] >= self.cfg["conf"]]
        branch, reason, targets = decide(coarse, img_w, img_h, quality, self.cfg)

        extra_calls = []  # {tool, args, n_out, seconds}
        zoom_dets = []
        hired_px = 0
        pixel_hit = False

        def px_ok(cost):
            cap = self.cfg.get("max_hires_px")
            return cap is None or hired_px + cost <= cap

        if branch == ZOOM:
            for tgt in targets:
                crop = expand_crop(tgt, img_w, img_h, self.cfg["context"])
                x1, y1, x2, y2 = crop
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                if not px_ok((x2 - x1) * (y2 - y1)):
                    pixel_hit = True
                    break
                crop_img = frame[y1:y2, x1:x2]
                hired_px += (x2 - x1) * (y2 - y1)
                t1 = time.time()
                outs = predict(self.model, crop_img, self.cfg["floor"], self.cfg["zoom_imgsz"])
                dt = time.time() - t1
                for o in outs:
                    o["x1"], o["y1"], o["x2"], o["y2"] = map_to_frame(
                        [o["x1"], o["y1"], o["x2"], o["y2"]], crop)
                    o["x1"] = max(0.0, min(o["x1"], img_w))
                    o["y1"] = max(0.0, min(o["y1"], img_h))
                    o["x2"] = max(0.0, min(o["x2"], img_w))
                    o["y2"] = max(0.0, min(o["y2"], img_h))
                if self.cfg.get("swimmer_only_zoom"):
                    outs = [o for o in outs if o["class_id"] == SWIMMER]
                zoom_dets.extend(o for o in outs if o["conf"] >= self.cfg["floor"])
                extra_calls.append({"tool": "crop_zoom_infer",
                                    "args": {"crop_xyxy": list(crop),
                                             "imgsz": self.cfg["zoom_imgsz"],
                                             "conf_floor": self.cfg["floor"]},
                                    "n_out": len(outs), "seconds": round(dt, 3)})
        elif branch == FALLBACK:
            if not px_ok(img_w * img_h):
                pixel_hit = True
            else:
                t1 = time.time()
                full = predict(self.model, frame, self.cfg["floor"], self.cfg["zoom_imgsz"])
                dt = time.time() - t1
                hired_px += img_w * img_h
                zoom_dets.extend(full)
                extra_calls.append({"tool": "full960_infer",
                                    "args": {"imgsz": self.cfg["zoom_imgsz"],
                                             "conf_floor": self.cfg["floor"]},
                                    "n_out": len(full), "seconds": round(dt, 3)})

        assert len(extra_calls) <= self.cfg["max_extra"], "inference budget exceeded"
        merged = nms([d for d in operating] +
                     [d for d in zoom_dets if d["conf"] >= self.cfg["conf"]],
                     self.cfg["nms_iou"])
        for d in merged:
            d.pop("_area", None)
        total_s = time.time() - t_start
        review = ("uncertain" if branch == UNCERTAIN
                  else ("review" if branch in (ZOOM, FALLBACK) and not merged else "ok"))
        trace = {
            "image": os.path.basename(img_path),
            "sha256": sha256_file(img_path),
            "init": {"policy": self.cfg.get("policy", "v1"),
                     "resolution": self.cfg["coarse_imgsz"], "floor": self.cfg["floor"],
                     "zoom_floor": self.cfg.get("zoom_floor", self.cfg["floor"]),
                     "swimmer_only_zoom": bool(self.cfg.get("swimmer_only_zoom")),
                     "max_extra": self.cfg["max_extra"],
                     "max_hires_px": self.cfg.get("max_hires_px"),
                     "allow_fallback": self.cfg.get("allow_fallback", True),
                     "conf": self.cfg["conf"], "weights": self.cfg["weights"]},
            "observations": [{"class": d["class"], "conf": round(d["conf"], 3),
                              "rel_area": round(rel_area(d, img_w, img_h), 6)}
                             for d in sorted(coarse, key=lambda d: -d["conf"])[:12]],
            "quality": {k: round(v, 2) for k, v in quality.items()},
            "branch": branch, "reason": reason,
            "tool": extra_calls[0]["tool"] if extra_calls else "none",
            "tool_args": [c["args"] for c in extra_calls],
            "extra_calls": extra_calls,
            "n_extra": len(extra_calls),
            "hires_pixels": hired_px,
            "pixel_budget_hit": pixel_hit,
            "final": [{"class": d["class"], "conf": round(d["conf"], 3)} for d in merged],
            "n_final": len(merged),
            "review_status": review,
            "coarse_seconds": round(coarse_s, 3),
            "total_seconds": round(total_s, 3),
        }
        final = {"image": os.path.basename(img_path), "width": img_w, "height": img_h,
                 "detections": merged}
        return final, trace, frame


def default_cfg(args):
    v2 = getattr(args, "policy", "v1") == "v2"
    max_extra = args.max_extra if args.max_extra is not None else (2 if v2 else 3)
    zoom_floor = args.zoom_floor if args.zoom_floor is not None else (0.15 if v2 else args.floor)
    hires = args.max_hires_mpx if args.max_hires_mpx is not None else (25.0 if v2 else None)
    return {"policy": getattr(args, "policy", "v1"),
            "weights": args.weights, "conf": args.conf, "floor": args.floor,
            "zoom_floor": zoom_floor,
            "swimmer_only_zoom": bool(args.swimmer_only_zoom or v2),
            "max_hires_px": (hires * 1e6 if hires is not None else None),
            "coarse_imgsz": args.coarse_imgsz, "zoom_imgsz": args.zoom_imgsz,
            "confident_conf": args.confident_conf, "large_area": args.large_area,
            "max_extra": max_extra, "context": args.context,
            "nms_iou": args.nms_iou, "sharp_floor": args.sharp_floor,
            "allow_fallback": (not args.no_fallback) and not v2}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SeaRescue adaptive vision agent (image-based)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="single image path")
    src.add_argument("--images", help="directory of .jpg images (batch)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", default=os.path.join("weights", "best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--floor", type=float, default=0.10)
    ap.add_argument("--coarse-imgsz", type=int, default=640)
    ap.add_argument("--zoom-imgsz", type=int, default=960)
    ap.add_argument("--confident-conf", type=float, default=0.50)
    ap.add_argument("--large-area", type=float, default=5e-4)
    ap.add_argument("--max-extra", type=int, default=None,
                    help="default: 3 for v1, 2 for v2")
    ap.add_argument("--zoom-floor", type=float, default=None,
                    help="min conf for zoom targets; default: --floor for v1, 0.15 for v2")
    ap.add_argument("--swimmer-only-zoom", action="store_true",
                    help="drop non-swimmer outputs from crop zooms (implied by v2)")
    ap.add_argument("--max-hires-mpx", type=float, default=None,
                    help="hires pixel budget per image in Mpx (v2 default 25)")
    ap.add_argument("--policy", choices=["v1", "v2"], default="v1")
    ap.add_argument("--context", type=float, default=1.5)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--sharp-floor", type=float, default=100.0)
    ap.add_argument("--no-fallback", action="store_true")
    args = ap.parse_args(argv)
    cfg = default_cfg(args)

    if not os.path.isfile(args.weights):
        return fail(f"weights not found: {args.weights}")
    os.makedirs(args.output, exist_ok=True)
    try:
        from ultralytics import YOLO
    except ImportError:
        return fail("ultralytics is not installed (pip install -r requirements.txt)")
    model = YOLO(args.weights)
    # Warmup (not traced).
    import numpy as _np
    model.predict(source=_np.zeros((480, 640, 3), dtype=_np.uint8),
                  conf=cfg["floor"], imgsz=cfg["coarse_imgsz"], verbose=False)
    agent = Agent(model, cfg)

    if args.input:
        items = [args.input]
    else:
        if not os.path.isdir(args.images):
            return fail(f"images dir not found: {args.images}")
        items = [os.path.join(args.images, f) for f in sorted(os.listdir(args.images))
                 if f.lower().endswith(".jpg")]
        if not items:
            return fail(f"no .jpg images in {args.images}")
    counts = {}
    px_total = 0
    trace_path = os.path.join(args.output, "traces.jsonl")
    with open(trace_path, "w") as tf:
        for path in items:
            try:
                final, trace, vis = agent.run(path)
            except (ValueError, RuntimeError) as exc:
                return fail(str(exc))
            base = os.path.splitext(os.path.basename(path))[0]
            with open(os.path.join(args.output, base + "_final.json"), "w") as f:
                json.dump(final, f, indent=2)
            tf.write(json.dumps(trace) + "\n")
            cv2.imwrite(os.path.join(args.output, base + "_annotated.jpg"),
                        annotate(vis, final["detections"]))
            counts[trace["branch"]] = counts.get(trace["branch"], 0) + 1
            px_total += trace["hires_pixels"]
            print(f"{trace['image']}: branch={trace['branch']} extra={trace['n_extra']} "
                  f"final={trace['n_final']} t={trace['total_seconds']}s")
    n = len(items)
    esc = sum(v for k, v in counts.items() if k in (ZOOM, FALLBACK))
    print(f"policy={cfg.get('policy', 'v1')} n={n} branches={counts} escalated={esc} ({esc / n:.0%}) hires_Mpx={px_total / 1e6:.1f}")
    print(f"wrote {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""SeaRescue live end-to-end vision-agent demonstration (Task 11).

Real maritime image -> OpenCV 5 analysis + pretrained detector -> ACTUAL
SigV4-signed POST of a measurement-only observation to the deployed
IAM-authenticated Function URL -> Lambda returns an action -> the returned
action is EXECUTED locally with the same pixel semantics agentic_detect uses
-> annotated before/after evidence + JSONL trace.

Honesty contract:
- mode "live-aws": the decision comes from the deployed Lambda. If AWS auth
  fails the run exits LOUDLY (exit code 3) -- it NEVER falls back silently
  to the local policy. Each trace records whether the request reached the
  actual endpoint, the HTTP status, the returned trace_id, and whether the
  trace id matches what the local shared policy would compute.
- mode "local-policy": the decision comes from scripts/decision_service.py
  on this machine (no network); traces say decision_source=local-policy and
  endpoint skipped=true. This mode exists for offline checks and tests.
- Observations contain measurements only (detector outputs + OpenCV quality
  + state + budget). The client never sends a pre-selected decision.
- No credentials, tokens, private headers, or image bytes are recorded in
  any output file. The endpoint URL is stored masked.
- The <=8-image evidence subset is a DEMO, not an independent benchmark.

Exit codes: 0 ok; 2 usage/environment errors; 3 AWS auth failure (loud);
4 live run finished with at least one non-successful endpoint decision.

Usage:
  live demo (8-image subset):
    python scripts/demo_live_agent.py --images data/samples/dl/data/images/val/shard_000 \
        --limit 8 --output output/demo_live --mode live-aws
  offline check (no network):
    python scripts/demo_live_agent.py --input some.jpg --output output/demo_local --mode local-policy
"""
import argparse
import json
import math
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from agentic_detect import (SWIMMER, annotate, expand_crop, map_to_frame,  # noqa: E402
                            nms, predict, quality_signals, rel_area,
                            sha256_file)
from decision_service import (ACTIONS, DecisionError, decide_observation,  # noqa: E402
                              validate_request)
import aws_sigv4  # noqa: E402

MODES = ("live-aws", "local-policy")


def fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 2


def iter_images(spec, limit):
    """Resolve --input/--images into a sorted list of .jpg paths (<= limit)."""
    if os.path.isfile(spec):
        items = [spec]
    elif os.path.isdir(spec):
        items = [os.path.join(spec, f) for f in sorted(os.listdir(spec))
                 if f.lower().endswith(".jpg")]
        if not items:
            raise ValueError(f"no .jpg images under {spec}")
    else:
        raise ValueError(f"images path not found: {spec}")
    if limit is not None:
        items = items[:limit]
    return items


def parse_gt(path, img_w, img_h):
    """Paired YOLO-normalized txt GT -> pixel boxes (demo-subset stats only)."""
    try:
        with open(path) as f:
            lines = [l.split() for l in f if l.strip()]
    except OSError:
        return []
    out = []
    for parts in lines:
        cls = int(float(parts[0]))
        cx, cy = float(parts[1]) * img_w, float(parts[2]) * img_h
        w, h = float(parts[3]) * img_w, float(parts[4]) * img_h
        out.append({"class_id": cls, "x1": cx - w / 2, "y1": cy - h / 2,
                    "x2": cx + w / 2, "y2": cy + h / 2})
    return out


def gt_matched(gt_boxes, dets, iou=0.5):
    """Class-matched greedy IoU hit count. Demo-subset stat, not a metric."""
    used = set()
    hit = 0
    for g in gt_boxes:
        best, best_iou = None, 0.0
        for j, d in enumerate(dets):
            if j in used or d["class_id"] != g["class_id"]:
                continue
            ix1, iy1 = max(d["x1"], g["x1"]), max(d["y1"], g["y1"])
            ix2, iy2 = min(d["x2"], g["x2"]), min(d["y2"], g["y2"])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            a = max(0.0, d["x2"] - d["x1"]) * max(0.0, d["y2"] - d["y1"])
            b = max(0.0, g["x2"] - g["x1"]) * max(0.0, g["y2"] - g["y1"])
            u = a + b - inter
            i = inter / u if u > 0 else 0.0
            if i > best_iou:
                best, best_iou = j, i
        if best is not None and best_iou >= iou:
            used.add(best)
            hit += 1
    return hit


def match_target(target, pool, img_w, img_h):
    """Map an action target back to OUR local candidate detection.

    The target echoes one of our own observations (conf rounded to 3dp,
    rel_area to 6dp), so the default match is exact on (class, conf);
    fallback is nearest rel_area within the same class. Returns (det, how)
    or (None, reason).
    """
    for d in pool:
        if (d["class"] == target["class"]
                and round(d["conf"], 3) == round(float(target["conf"]), 3)):
            return d, "exact-class-conf"
    same = [d for d in pool if d["class"] == target["class"]]
    if same:
        cand = min(same, key=lambda d: abs(rel_area(d, img_w, img_h)
                                           - float(target["rel_area"])))
        return cand, "nearest-rel-area-in-class"
    return None, "no-local-candidate"


def unwrap_response(text):
    """Accept both the unwrapped Function-URL body and a legacy wrapped
    {"statusCode":..,"body":"<json>"} envelope (tolerates one nesting)."""
    d = json.loads(text)
    for _ in range(2):
        if isinstance(d, dict) and isinstance(d.get("body"), str):
            d = json.loads(d["body"])
        else:
            break
    return d


def _det_key(d):
    """Identity key for detection-diff bookkeeping (merge/NMS accounting)."""
    return (d["class"], round(float(d["conf"]), 3), round(float(d["x1"]), 1),
            round(float(d["y1"]), 1), round(float(d["x2"]), 1),
            round(float(d["y2"]), 1))


def make_crop_inspection(crop_bgr, crop_xyxy, raw_records, cfg):
    """Enlarged inspection view of the ACTUAL crop-inference call.

    Draws every raw detection returned by the crop inference (before the
    operating-confidence filter), labeled with class + confidence + status:
      green  = swimmer at/above the operating threshold (merge candidate)
      orange = raw detection below the operating threshold (not merged)
      gray   = excluded by the swimmer-only class rule (never merged)
    Also prints the exact crop coordinates in the original frame, the crop
    size, the upscale factor and the model input resolution.
    """
    ch, cw = crop_bgr.shape[:2]
    f = min(8, max(2, int(math.ceil(640.0 / max(cw, ch)))))
    view = cv2.resize(crop_bgr, (cw * f, ch * f), interpolation=cv2.INTER_CUBIC)
    n_pass = sum(1 for r in raw_records if r["passes_operating"])
    n_excl = sum(1 for r in raw_records if r["class_filter_excluded"])
    if not raw_records:
        cv2.putText(view, f"no raw detections above conf floor {cfg['floor']}",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                    cv2.LINE_AA)
    last_label = None
    for r in raw_records:
        cx1, cy1, cx2, cy2 = r["crop_xyxy"]
        x1, y1, x2, y2 = (int(round(v * f)) for v in (cx1, cy1, cx2, cy2))
        # Keep sub-pixel sliver detections visible on the enlarged view.
        if x2 - x1 < 8:
            pad = (8 - (x2 - x1)) / 2.0
            x1, x2 = int(x1 - pad), int(x1 - pad) + 8
        if y2 - y1 < 8:
            pad = (8 - (y2 - y1)) / 2.0
            y1, y2 = int(y1 - pad), int(y1 - pad) + 8
        if r["class_filter_excluded"]:
            color, tag = (160, 160, 160), "excluded:class-rule"
        elif r["passes_operating"]:
            color, tag = (0, 200, 0), "PASS"
        else:
            color, tag = (0, 140, 255), "below-operating"
        cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
        label = f'{r["class"]} {r["conf"]:.2f} {tag}'
        ty = max(16, y1 - 6)
        # Stagger labels that would collide with the previous one.
        if last_label is not None:
            lx1, lty, lw = last_label
            if abs(ty - lty) < 20 and x1 < lx1 + lw and x2 > lx1:
                ty = lty + 22
        cv2.putText(view, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2, cv2.LINE_AA)
        last_label = (x1, ty, 14 * len(label))
    header = [
        f"crop xyxy={list(crop_xyxy)}",
        f"size={cw}x{ch} x{f} model_input={cfg['zoom_imgsz']}",
        f"raw={len(raw_records)} pass={n_pass} excluded_class_rule={n_excl}",
        f"conf_operating={cfg['conf']} floor={cfg['floor']}",
    ]
    for i, line in enumerate(header):
        cv2.putText(view, line, (8, view.shape[0] - 10
                                 - (len(header) - 1 - i) * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return view, f


def build_observation(cfg, img_w, img_h, quality, pool, extra_used):
    """Measurement-only observation in the validated decision schema."""
    dets = [{"class": d["class"], "conf": round(d["conf"], 3),
             "rel_area": round(rel_area(d, img_w, img_h), 6)}
            for d in sorted(pool, key=lambda d: -d["conf"])[:20]]
    return {
        "request_id": f"demo:{cfg['stem']}",
        "frame_ref": f"demo:{cfg['stem']}",
        "observation": {"detections": dets, "quality": quality, "motion": None},
        "state": {"stride": cfg["coarse_imgsz"], "samples_used": extra_used,
                  "prev_action": None},
        "budget": {"remaining_extra": cfg["max_extra"] - extra_used,
                   "max_extra": cfg["max_extra"]},
    }


def send_live(cfg, request, endpoint, creds):
    """Signed POST to the deployed Function URL; records endpoint truth.

    AWSAuthError propagates (loud failure; caller exits without fallback).
    Transport errors are RECORDED (reached=false) rather than fabricating
    any local decision.
    """
    try:
        status, text, ms = aws_sigv4.signed_post(
            endpoint["url"], json.dumps(request), creds, endpoint["region"])
    except aws_sigv4.AWSTransportError as exc:
        return {"reached": False, "http_status": None, "latency_ms": None,
                "error": str(exc), "url_masked": aws_sigv4.mask_url(endpoint["url"])}
    out = {"reached": True, "http_status": status, "latency_ms": ms,
           "url_masked": aws_sigv4.mask_url(endpoint["url"]), "error": None}
    if 200 <= status < 300:
        parsed = unwrap_response(text)
        if isinstance(parsed, dict) and parsed.get("action") in ACTIONS:
            out["response"] = parsed
        else:
            out["error"] = "200 response without a valid action"
            out["response_error"] = text[:200]
    else:
        out["error"] = f"endpoint returned HTTP {status}"
        out["response_error"] = text[:200]
    return out


def decide_local(cfg, request):
    """Local shared-policy adapter: same decide_observation the Lambda runs,
    executed on this machine. No network. Clearly labelled in the trace."""
    return {"reached": False, "http_status": None, "latency_ms": None,
            "skipped": True, "url_masked": "local-policy mode (no network)",
            "local_decision": decide_observation(request)}


def run_image_pipeline(frame, cfg, send_fn):
    """One image: measure -> observe -> decide (live or local) -> execute.

    Returns (final_dets, trace, annotated_before, annotated_after).
    send_fn(cfg, request) -> endpoint/decision result dict (live wrapper or
    local adapter) so tests can stub either without touching AWS.
    """
    img_h, img_w = frame.shape[:2]
    quality = quality_signals(frame)
    t0 = time.time()
    coarse_all = predict(cfg["model"], frame, cfg["floor"], cfg["coarse_imgsz"])
    coarse_s = round(time.time() - t0, 3)
    pool = [d for d in coarse_all if d["conf"] >= cfg["pool_floor"]]
    operating = [d for d in coarse_all if d["conf"] >= cfg["conf"]]

    request = build_observation(cfg, img_w, img_h, quality, pool, extra_used=0)
    try:
        validated = validate_request(request)
    except DecisionError as exc:  # schema regression = OUR bug; fail loudly
        raise RuntimeError(f"client produced an invalid observation: {exc}") from exc

    # Local reference decision (same shared policy the Lambda runs) -- used
    # only for agreement/parity evidence, never as a substitute for live AWS.
    local_ref = decide_observation(validated)

    t_ep = time.time()
    ep = send_fn(cfg, validated)          # AWSAuthError propagates (loud)
    ep_ms = round(time.time() - t_ep, 3)

    if cfg["mode"] == "live-aws":
        resp = ep.get("response")
        if isinstance(resp, dict) and resp.get("action") in ACTIONS:
            dec = resp
            decision_source = "aws-live"
        else:
            dec = {"action": None, "args": {}, "reason": None, "review": False}
            decision_source = "no-decision"
        dec_error = ep.get("error")
    else:
        dec = ep.get("local_decision")
        decision_source = "local-policy"
        dec_error = None

    action = dec.get("action")
    action_args = dec.get("args") or {}
    review = bool(dec.get("review")) or action == "request_human_review"

    extra_calls = []
    added = []
    final = list(operating)
    refused = None
    matched_box = None
    matched_how = None
    crop_view = None  # private in-memory crop artifacts (stripped pre-JSON)

    if action in ("crop_zoom", "sample_densely"):
        if cfg["max_extra"] - len(extra_calls) <= 0:
            refused = "budget_exhausted"
        elif action == "crop_zoom":
            target = action_args.get("target")
            if not isinstance(target, dict):
                refused = "action_with_no_target"
            else:
                cand, how = match_target(target, pool, img_w, img_h)
                matched_how = how
                if cand is None:
                    refused = "target_unmatched"
                else:
                    crop = expand_crop(cand, img_w, img_h, cfg["context"])
                    matched_box = list(crop)
                    x1, y1, x2, y2 = crop
                    if x2 - x1 < 8 or y2 - y1 < 8:
                        refused = "crop_too_small"
                    else:
                        t1 = time.time()
                        crop_img = frame[y1:y2, x1:x2]
                        outs_all = predict(cfg["model"], crop_img,
                                           cfg["floor"], cfg["zoom_imgsz"])
                        outs = ([o for o in outs_all
                                 if o["class_id"] == SWIMMER]
                                if cfg["swimmer_only_zoom"] else list(outs_all))
                        raw_records = []
                        for o in outs_all:
                            # Record EVERY raw detection returned by the crop
                            # inference, before the operating-conf filter and
                            # before the swimmer-only class rule, with its
                            # actual crop-local and frame-mapped coordinates.
                            excluded = (cfg["swimmer_only_zoom"]
                                        and o["class_id"] != SWIMMER)
                            fx1, fy1, fx2, fy2 = map_to_frame(
                                [o["x1"], o["y1"], o["x2"], o["y2"]], crop)
                            fx1 = max(0.0, min(fx1, img_w))
                            fy1 = max(0.0, min(fy1, img_h))
                            fx2 = max(0.0, min(fx2, img_w))
                            fy2 = max(0.0, min(fy2, img_h))
                            passes = ((not excluded)
                                      and o["conf"] >= cfg["conf"])
                            raw_records.append({
                                "class": o["class"],
                                "conf": round(float(o["conf"]), 3),
                                "crop_xyxy": [round(float(v), 1) for v in
                                              (o["x1"], o["y1"],
                                               o["x2"], o["y2"])],
                                "frame_xyxy": [round(float(v), 1) for v in
                                               (fx1, fy1, fx2, fy2)],
                                "passes_operating": passes,
                                "class_filter_excluded": excluded})
                            if excluded:
                                continue
                            o["x1"], o["y1"], o["x2"], o["y2"] = fx1, fy1, fx2, fy2
                            if o["conf"] >= cfg["conf"]:
                                added.append(o)
                        final = nms(list(operating) + added, cfg["nms_iou"])
                        insp, _fs = make_crop_inspection(crop_img, crop,
                                                         raw_records, cfg)
                        crop_view = {"region": crop_img.copy(),
                                     "inspection": insp}
                        extra_calls.append({
                            "tool": "crop_zoom_infer",
                            "args": {"crop_xyxy": [x1, y1, x2, y2],
                                     "imgsz": cfg["zoom_imgsz"]},
                            "n_raw": len(outs), "n_kept": len(added),
                            "n_raw_pre_class_filter": len(outs_all),
                            "raw_crop_detections": raw_records,
                            "conf_operating": cfg["conf"],
                            "conf_floor": cfg["floor"],
                            "target_matched_by": how,
                            "seconds": round(time.time() - t1, 3)})
        else:  # sample_densely -> still-image analog: one full-frame 960 scan
            t1 = time.time()
            outs = predict(cfg["model"], frame, cfg["floor"], cfg["zoom_imgsz"])
            added = [o for o in outs if o["conf"] >= cfg["conf"]]
            final = nms(list(operating) + added, cfg["nms_iou"])
            extra_calls.append({
                "tool": "full960_infer",
                "args": {"imgsz": cfg["zoom_imgsz"]},
                "n_raw": len(outs), "n_kept": len(added),
                "note": "still-image analog of denser sampling (stride n/a)",
                "seconds": round(time.time() - t1, 3)})

    # Merge/NMS accounting from ACTUAL executed results (identity-based):
    # n_removed_by_nms = merged input size - final size (NMS suppression),
    # n_new_added = final detections whose identity was not in the initial
    # operating set (i.e., genuinely new detections above the operating
    # threshold contributed by the extra inference).
    if extra_calls:
        n_removed_by_nms = max(0, (len(operating) + len(added)) - len(final))
        _op_identity_keys = {_det_key(d) for d in operating}
        n_new_added = sum(1 for d in final if _det_key(d) not in _op_identity_keys)
    else:
        n_removed_by_nms = 0
        n_new_added = 0

    if action is None:
        action_executed = False
    elif action in ("keep_sparse_sampling", "request_human_review"):
        action_executed = True
    else:
        action_executed = bool(extra_calls)

    before = annotate(frame.copy(), operating)
    after = annotate(frame.copy(), final)
    op_keys = {(d["class"], round(d["conf"], 2), int(d["x1"]), int(d["y1"]))
               for d in operating}
    for d in final:
        if (d["class"], round(d["conf"], 2), int(d["x1"]), int(d["y1"])) not in op_keys:
            cv2.rectangle(after, (int(d["x1"]), int(d["y1"])),
                          (int(d["x2"]), int(d["y2"])), (255, 0, 255), 2)
    if review:
        cv2.putText(after, "HUMAN REVIEW", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3, cv2.LINE_AA)
    if matched_box:
        # Locate the executed crop region on both full-frame views (blue).
        bx1, by1, bx2, by2 = matched_box
        for img_ in (before, after):
            cv2.rectangle(img_, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
            cv2.putText(img_, "crop region", (bx1, max(18, by1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2,
                        cv2.LINE_AA)

    trace = {
        "image": cfg["stem"] + ".jpg",
        "sha256": cfg["sha"],
        "mode": cfg["mode"],
        "decision_source": decision_source,
        "trace_id": dec.get("trace_id"),
        "trace_id_parity": (dec.get("trace_id") == local_ref["trace_id"]
                            if cfg["mode"] == "live-aws" else None),
        "local_policy_reference": {"action": local_ref["action"],
                                   "trace_id": local_ref["trace_id"]},
        "decision_agrees_with_local": dec.get("action") == local_ref["action"],
        "endpoint": ep,
        "observation_sent": validated,
        "quality": {k: round(v, 2) for k, v in quality.items()},
        "action": action,
        "action_args": action_args,
        "reason": dec.get("reason"),
        "error": dec_error if dec_error else None,
        "action_executed": action_executed,
        "action_not_executed_reason": refused,
        "target_matched_box": matched_box,
        "extra_calls": extra_calls,
        "n_extra": len(extra_calls),
        "budget": {"max_extra": cfg["max_extra"],
                   "service_budget_remaining": dec.get("budget_remaining")},
        "review": review,
        "initial_operating": [dict(d) for d in operating],
        "final": final,
        "n_initial_operating": len(operating),
        "n_final": len(final),
        "n_added_to_merge": len(added),
        "n_new_added": n_new_added,
        "n_removed_by_nms": n_removed_by_nms,
        "crop_view_generated": crop_view is not None,
        "coarse_seconds": coarse_s,
        "endpoint_ms": ep_ms,
        "total_seconds": round(time.time() - t0, 3),
    }
    if crop_view:
        trace["_crop_view"] = crop_view  # stripped by main() before JSON dump
    return final, trace, before, after


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SeaRescue live demo agent")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="single image path")
    src.add_argument("--images", help="directory of .jpg images (batch)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=MODES, default="local-policy")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of images (demo subset)")
    ap.add_argument("--profile", default="searescue")
    ap.add_argument("--region", default="ap-south-1")
    ap.add_argument("--function-name",
                    default="searescue-agent-DecideFunction-REPLACEME123456",
                    help="deployed DecideFunction name; replace the "
                         "REPLACEME suffix with your stack's actual suffix")
    ap.add_argument("--weights", default=os.path.join("weights", "best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--floor", type=float, default=0.10)
    ap.add_argument("--pool-floor", type=float, default=0.15,
                    help="conf floor of the observation pool sent to the service")
    ap.add_argument("--coarse-imgsz", type=int, default=640)
    ap.add_argument("--zoom-imgsz", type=int, default=960)
    ap.add_argument("--max-extra", type=int, default=2)
    ap.add_argument("--context", type=float, default=1.5)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--labels", default=os.path.join(
        "data", "samples", "dl", "data", "labels", "val", "shard_000"),
        help="paired YOLO-labels dir (demo-subset stats only)")
    args = ap.parse_args(argv)

    os.makedirs(args.output, exist_ok=True)
    if not os.path.isfile(args.weights):
        return fail(f"weights not found: {args.weights}")
    try:
        from ultralytics import YOLO
    except ImportError:
        return fail("ultralytics is not installed")
    model = YOLO(args.weights)
    import numpy as _np
    model.predict(source=_np.zeros((480, 640, 3), dtype=_np.uint8),
                  conf=args.floor, imgsz=args.coarse_imgsz, verbose=False)

    endpoint = None
    creds = None
    if args.mode == "live-aws":
        try:
            url, auth = aws_sigv4.get_function_url(
                args.profile, args.region, args.function_name)
            creds = aws_sigv4.get_session_credentials(args.profile)
        except aws_sigv4.AWSAuthError as exc:
            fail(f"AWS authentication/configuration failed, NOT falling "
                 f"back: {exc}")
            return 3
        endpoint = {"url": url, "region": args.region, "auth": auth}

    try:
        images = iter_images(args.input or args.images, args.limit)
    except ValueError as exc:
        return fail(str(exc))

    def send_fn(c, request):
        if args.mode == "live-aws":
            return send_live(c, request, endpoint, creds)
        return decide_local(c, request)

    gt_available = os.path.isdir(args.labels)
    rows = []
    t_start = time.time()
    trace_path = os.path.join(args.output, "demo_traces.jsonl")
    with open(trace_path, "w") as tf:
        for path in images:
            stem = os.path.splitext(os.path.basename(path))[0]
            frame = cv2.imread(path)
            if frame is None:
                return fail(f"could not decode image: {path}")
            img_h, img_w = frame.shape[:2]
            cfg = {"stem": stem, "sha": sha256_file(path), "model": model,
                   "mode": args.mode, "conf": args.conf, "floor": args.floor,
                   "pool_floor": args.pool_floor,
                   "coarse_imgsz": args.coarse_imgsz,
                   "zoom_imgsz": args.zoom_imgsz,
                   "max_extra": args.max_extra, "context": args.context,
                   "nms_iou": args.nms_iou, "swimmer_only_zoom": True}

            try:
                final_dets, trace, before, after = run_image_pipeline(
                    frame, cfg, send_fn)
            except aws_sigv4.AWSAuthError as exc:
                fail(f"live endpoint auth failed, NOT falling back: {exc}")
                return 3

            base = os.path.join(args.output, stem)

            if gt_available:
                gt = parse_gt(os.path.join(args.labels, stem + ".txt"),
                              img_w, img_h)
                trace["gt_demo_only"] = {
                    "n_gt": len(gt),
                    "matched_initial": gt_matched(gt, trace["initial_operating"]),
                    "matched_final": gt_matched(gt, final_dets),
                    "note": "demo-subset IoU0.5 hits; NOT a benchmark metric"}

            # Machine-readable crop summary for every executed extra call.
            if trace["extra_calls"]:
                ec0 = trace["extra_calls"][0]
                cs = {
                    "image": trace["image"],
                    "action": trace["action"],
                    "crop_xyxy_in_original_frame": trace["target_matched_box"],
                    "crop_wh": ([ec0["args"]["crop_xyxy"][2]
                                 - ec0["args"]["crop_xyxy"][0],
                                 ec0["args"]["crop_xyxy"][3]
                                 - ec0["args"]["crop_xyxy"][1]]
                                if "crop_xyxy" in ec0["args"] else None),
                    "model_input_resolution": ec0["args"].get("imgsz"),
                    "conf_operating": args.conf, "conf_floor": args.floor,
                    "swimmer_only_zoom": cfg["swimmer_only_zoom"],
                    "n_initial_operating": trace["n_initial_operating"],
                    "n_raw_crop_detections": ec0.get("n_raw_pre_class_filter",
                                                     ec0.get("n_raw")),
                    "n_raw_after_class_filter": ec0.get("n_raw"),
                    "n_retained_after_conf_filter": ec0.get("n_kept"),
                    "n_new_detections_added": trace["n_new_added"],
                    "n_final": trace["n_final"],
                    "n_removed_by_merge_nms": trace["n_removed_by_nms"],
                    "gt_matches_changed": (
                        trace["gt_demo_only"]["matched_final"]
                        - trace["gt_demo_only"]["matched_initial"])
                    if "gt_demo_only" in trace else None,
                    "claim": "records what the executed extra inference "
                             "actually returned; no accuracy improvement is "
                             "claimed when n_new_detections_added == 0"}
                trace["crop_summary"] = cs

            crop_imgs = trace.pop("_crop_view", None)
            tf.write(json.dumps(trace) + "\n")

            cv2.imwrite(base + "_before.jpg", before)
            cv2.imwrite(base + "_after.jpg", after)
            if crop_imgs:
                # Exact pixels sent to the extra inference (unannotated) and
                # the enlarged inspection view with raw detections drawn.
                cv2.imwrite(base + "_crop_region.jpg", crop_imgs["region"])
                cv2.imwrite(base + "_crop_inspection.jpg",
                            crop_imgs["inspection"])
            if trace.get("crop_summary"):
                with open(base + "_crop_summary.json", "w") as f:
                    json.dump(trace["crop_summary"], f, indent=2)
            with open(base + "_final.json", "w") as f:
                json.dump({"image": stem + ".jpg", "width": img_w,
                           "height": img_h, "detections": final_dets}, f, indent=2)

            rows.append({
                "image": trace["image"], "mode": trace["mode"],
                "decision_source": trace["decision_source"],
                "action": trace["action"],
                "reached_endpoint": bool(trace["endpoint"].get("reached")),
                "http_status": trace["endpoint"].get("http_status"),
                "trace_id": trace["trace_id"],
                "trace_id_parity": trace["trace_id_parity"],
                "agrees_with_local": trace["decision_agrees_with_local"],
                "action_executed": trace["action_executed"],
                "not_executed_reason": trace["action_not_executed_reason"],
                "n_extra": trace["n_extra"], "review": trace["review"],
                "n_initial_operating": trace["n_initial_operating"],
                "n_final": trace["n_final"],
                "n_new_added": trace["n_new_added"],
                "n_removed_by_nms": trace["n_removed_by_nms"],
                "n_raw_crop": (trace.get("crop_summary") or {}).get(
                    "n_raw_crop_detections"),
                "n_retained_crop": (trace.get("crop_summary") or {}).get(
                    "n_retained_after_conf_filter"),
                "endpoint_ms": trace["endpoint_ms"],
                "total_seconds": trace["total_seconds"],
                "sha256": trace["sha256"]})
            print(f"{stem}: mode={trace['mode']} src={trace['decision_source']} "
                  f"action={trace['action']} reached={rows[-1]['reached_endpoint']} "
                  f"http={rows[-1]['http_status']} extra={trace['n_extra']} "
                  f"fins={trace['n_final']} new={trace['n_new_added']} "
                  f"nms_rm={trace['n_removed_by_nms']} review={trace['review']} "
                  f"t={trace['total_seconds']}s")

    by_action = {}
    for r in rows:
        key = f"{r['decision_source']}:{r['action']}"
        by_action[key] = by_action.get(key, 0) + 1
    ok = sum(1 for r in rows
             if r["decision_source"] == "aws-live"
             and isinstance(r["http_status"], int)
             and 200 <= r["http_status"] < 300)
    zoom_rows = [r for r in rows if r.get("n_raw_crop") is not None]
    crop_outcomes = {
        "zoom_executed_images": len(zoom_rows),
        "total_raw_crop_detections": sum(r["n_raw_crop"] or 0 for r in zoom_rows),
        "total_retained_after_conf_filter": sum(
            r["n_retained_crop"] or 0 for r in zoom_rows),
        "total_new_detections_added": sum(r["n_new_added"] for r in rows),
        "total_removed_by_merge_nms": sum(r["n_removed_by_nms"] for r in rows),
        "note": "counts of what the executed extra inferences actually "
                "returned; zero new detections means the action did not "
                "change the final detection set"}
    summary = {
        "mode": args.mode,
        "endpoint": ({"url_masked": aws_sigv4.mask_url(endpoint["url"]),
                      "auth": endpoint["auth"],
                      "function": args.function_name}
                     if endpoint else {"skipped": True,
                                       "note": "local-policy mode (no network)"}),
        "n_images": len(rows),
        "images": [r["image"] for r in rows],
        "counts_by_decision_source_action": by_action,
        "endpoint_ok_n": ok,
        "crop_zoom_outcomes": crop_outcomes,
        "agreement_with_local_policy": {
            "agree": sum(1 for r in rows if r["agrees_with_local"]),
            "total": len(rows)},
        "gt_available": gt_available,
        "runtime_seconds": round(time.time() - t_start, 3),
        "weights": os.path.basename(args.weights),
        "limits": {"max_extra_per_image": args.max_extra,
                   "pool_floor": args.pool_floor,
                   "coarse_imgsz": args.coarse_imgsz,
                   "zoom_imgsz": args.zoom_imgsz},
        "honest_notes": [
            "demo subset (<=8 images); NOT an independent benchmark",
            "detection quality and AWS decision-service correctness are "
            "reported separately",
            "traces contain measurements only; endpoint URL stored masked; "
            "no credentials or private headers recorded",
            "crop views show the actual raw results of the executed extra "
            "inference; detections below the operating threshold are drawn "
            "but were not merged"],
    }
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {trace_path} and summary.json")
    if args.mode == "live-aws" and ok < len(rows):
        print(f"WARNING: {len(rows) - ok}/{len(rows)} live decisions were not "
              f"HTTP-2xx server-decided (see traces)", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

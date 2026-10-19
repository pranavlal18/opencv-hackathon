"""Video analysis prototype: sampled detection + motion-gated adaptive sampling.

Image-based agent logic reused per sampled frame; the VIDEO contribution is
the temporal signal changing subsequent behavior:
- OpenCV 5 VideoCapture decode; configurable base stride.
- Per sampled frame: detector outputs + OpenCV motion signals
  (grayscale frame-difference mean, Farneback flow magnitude vs previous
  sampled frame).
- Decision: high motion OR uncertain swimmer (< 0.50 conf) -> dense sampling
  (stride 1) for the next window; quiet + confident -> relax stride up to
  --max-stride. Bounds [1, max_stride]; every decision traced with numbers.
- Outputs: annotated sampled frames, events.json, traces.jsonl
  (frame idx, timestamp, stride, detections, motion, action+reason).

Motion does NOT prove a person is in danger; it only reallocates sampling.
No tracker, no real-time claims.

Usage:
  python scripts/analyze_video.py --input clip.mp4 --output out/ [--stride 5]
"""
import argparse
import json
import os
import sys
import time

import cv2

UNCERTAIN_CONF = 0.50  # heuristic, not calibrated


def fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 2


def motion_signals(prev_gray, gray):
    """OpenCV-5 temporal signals between two sampled frames."""
    diff = cv2.absdiff(gray, prev_gray)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    import numpy as np
    return {"diff_mean": float(diff.mean()),
            "flow_mean": float(np.abs(flow).mean())}


def decide_stride(motion, dets, stride, base, max_stride,
                  diff_thresh, flow_thresh):
    """Pure adaptive-sampling decision -> (next_stride, action, reason)."""
    uncertain = [d for d in dets
                 if d.get("class") == "swimmer" and d.get("conf", 1.0) < UNCERTAIN_CONF]
    hot = (motion is not None
           and (motion["diff_mean"] >= diff_thresh or motion["flow_mean"] >= flow_thresh))
    if hot or uncertain:
        why = []
        if hot:
            why.append(f"motion diff={motion['diff_mean']:.1f}/flow={motion['flow_mean']:.3f}")
        if uncertain:
            why.append(f"{len(uncertain)} uncertain swimmer(s)")
        return 1, "densify", " AND ".join(why) + " -> stride 1"
    if stride < max_stride:
        return min(max_stride, stride + base), "relax", \
            f"quiet (motion below thresh, no uncertain swimmers) -> stride {min(max_stride, stride + base)}"
    return stride, "hold", "quiet and already at max stride"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SeaRescue video sampling prototype")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", default=os.path.join("weights", "best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--stride", type=int, default=5, help="base sampling stride")
    ap.add_argument("--max-stride", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=60, help="max sampled frames")
    ap.add_argument("--diff-thresh", type=float, default=8.0)
    ap.add_argument("--flow-thresh", type=float, default=0.4)
    ap.add_argument("--decision", choices=["internal", "service"], default="internal",
                    help="internal: built-in stride logic; service: shared decision "
                         "service (same function the Lambda handler uses)")
    ap.add_argument("--max-extra", type=int, default=10,
                    help="service mode: zoom-inference budget per clip")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.input):
        return fail(f"input not found: {args.input}")
    if not os.path.isfile(args.weights):
        return fail(f"weights not found: {args.weights}")
    if not (1 <= args.stride <= args.max_stride):
        return fail(f"need 1 <= stride <= max-stride, got {args.stride}/{args.max_stride}")
    os.makedirs(args.output, exist_ok=True)
    try:
        from ultralytics import YOLO
    except ImportError:
        return fail("ultralytics is not installed (pip install -r requirements.txt)")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        cap.release()
        return fail(f"could not open video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    model = YOLO(args.weights)
    import numpy as _np
    model.predict(source=_np.zeros((480, 640, 3), dtype=_np.uint8),
                  conf=args.conf, imgsz=args.imgsz, verbose=False)

    use_service = args.decision == "service"
    svc = None
    if use_service:
        try:
            from scripts.decision_service import decide_observation
            from scripts.agentic_detect import (quality_signals, expand_crop,
                                                map_to_frame)
        except ImportError:  # running as scripts/analyze_video.py: siblings on path
            from decision_service import decide_observation
            from agentic_detect import (quality_signals, expand_crop,
                                        map_to_frame)
        svc = decide_observation
        extra_left = args.max_extra
        prev_action = None

    stride, next_at = args.stride, 0
    idx, sampled = 0, 0
    prev_gray = None
    events, traces = [], []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or sampled >= args.max_frames:
                break
            if idx < next_at:
                idx += 1
                continue
            t0 = time.time()
            res = model.predict(source=frame, conf=args.conf,
                                imgsz=args.imgsz, verbose=False)[0]
            dets = []
            if res.boxes is not None and len(res.boxes):
                for i in range(len(res.boxes)):
                    xy = [float(v) for v in res.boxes.xyxy[i].tolist()]
                    dets.append({"x1": xy[0], "y1": xy[1], "x2": xy[2], "y2": xy[3],
                                 "class": model.names[int(res.boxes.cls[i])],
                                 "conf": float(res.boxes.conf[i])})
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion = motion_signals(prev_gray, gray) if prev_gray is not None else None
            prev_gray = gray
            if not use_service:
                action_stride, action, reason = decide_stride(
                    motion, dets, stride, args.stride, args.max_stride,
                    args.diff_thresh, args.flow_thresh)
                svc_action = None
            else:
                im_h, im_w = frame.shape[:2]
                qual = quality_signals(frame)
                obs_dets = [{"class": d["class"], "conf": d["conf"],
                             "rel_area": max(0.0, d["x2"] - d["x1"])
                             * max(0.0, d["y2"] - d["y1"]) / (im_w * im_h)}
                            for d in dets]
                req = {"request_id": f"{os.path.basename(args.input)}:{idx}",
                       "frame_ref": f"frame_{idx:06d}",
                       "observation": {"detections": obs_dets, "quality": qual,
                                       "motion": motion},
                       "state": {"stride": stride, "samples_used": sampled,
                                 "prev_action": prev_action},
                       "budget": {"remaining_extra": extra_left,
                                  "max_extra": args.max_extra}}
                decision = svc(req)  # shared function: identical to Lambda path
                svc_action = decision["action"]
                reason = f"service:{svc_action}: {decision['reason']}"
                if svc_action == "sample_densely":
                    action, action_stride = "densify", 1
                elif svc_action == "crop_zoom":
                    action, action_stride = "densify", 1
                    sw = sorted([d for d in dets if d["class"] == "swimmer"],
                                key=lambda d: d["conf"])
                    if sw and extra_left > 0:
                        crop = expand_crop(sw[0], im_w, im_h, 1.5)
                        x1, y1, x2, y2 = crop
                        if x2 - x1 >= 8 and y2 - y1 >= 8:
                            zres = model.predict(
                                source=frame[y1:y2, x1:x2], conf=args.conf,
                                imgsz=960, verbose=False)[0]
                            if zres.boxes is not None and len(zres.boxes):
                                for i in range(len(zres.boxes)):
                                    xy = [float(v) for v in zres.boxes.xyxy[i].tolist()]
                                    m = map_to_frame(xy, crop)
                                    dets.append({"x1": m[0], "y1": m[1],
                                                 "x2": m[2], "y2": m[3],
                                                 "class": model.names[int(zres.boxes.cls[i])],
                                                 "conf": float(zres.boxes.conf[i]),
                                                 "via": "service_crop_zoom"})
                            extra_left -= 1
                elif svc_action == "request_human_review":
                    action, action_stride = "hold", stride
                else:
                    action = "relax" if stride < args.max_stride else "hold"
                    action_stride = min(args.max_stride, stride + args.stride)
                prev_action = svc_action
            t_s = idx / fps
            ev = {"frame": idx, "t_s": round(t_s, 3), "stride_used": stride,
                  "motion": motion, "n_det": len(dets),
                  "detections": dets, "action": action,
                  "service_action": svc_action if use_service else None,
                  "review": bool(use_service and svc_action == "request_human_review"),
                  "next_stride": action_stride, "reason": reason,
                  "seconds": round(time.time() - t0, 3)}
            events.append(ev)
            traces.append({"frame": idx, "t_s": round(t_s, 3),
                           "decision": action, "reason": reason,
                           "service_action": svc_action if use_service else None,
                           "prev_action": (traces[-1]["service_action"]
                                           if use_service and traces else None),
                           "stride_before": stride, "stride_after": action_stride,
                           "motion": motion,
                           "uncertain_swimmers": sum(1 for d in dets if d["class"] == "swimmer"
                                                     and d["conf"] < UNCERTAIN_CONF)})
            for d in dets:
                x1, y1, x2, y2 = (int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{d["class"]} {d["conf"]:.2f}', (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"f={idx} t={t_s:.1f}s {action}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(args.output, f"frame_{idx:06d}.jpg"), frame)
            stride = action_stride
            next_at = idx + stride
            sampled += 1
            idx += 1
            print(f"frame={ev['frame']} t={ev['t_s']}s stride={ev['stride_used']} "
                  f"det={len(dets)} -> {action} (next {action_stride}): {reason}")
    finally:
        cap.release()
    with open(os.path.join(args.output, "events.json"), "w") as f:
        json.dump({"input": args.input, "fps": fps, "base_stride": args.stride,
                   "max_stride": args.max_stride, "frames_sampled": sampled,
                   "events": events}, f, indent=2)
    with open(os.path.join(args.output, "traces.jsonl"), "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    print(f"sampled={sampled} wrote {args.output}/events.json + traces.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

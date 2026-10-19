"""Minimal SeaRescue maritime detection baseline.

Backend: Ultralytics YOLO (documented model-card method) for the
SeaDronesSee-fine-tuned YOLOv8n weights. OpenCV 5 does all substantive
image/video work: decode (VideoCapture), frame sampling, preprocessing
(BGR read/resize input), annotation (rectangle/putText), and output
writing (imwrite/VideoWriter).

ONNX note: a pure cv2.dnn path would need a .pt -> .onnx export plus
custom YOLOv8 output decoding (sigmoid + DFL + NMS) and export
validation. That is deferred to keep this step small and verifiable;
this CLI is the smallest workable alternative. See README.

Usage:
  python scripts/detect.py --input <image|video> --output output/ [--weights weights/best.pt] [--conf 0.25]
"""
import argparse
import json
import os
import sys
import time

import cv2

CLASSES_FALLBACK = ["swimmer", "boat", "jetski", "life_saving_appliances", "buoy"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm",
            ".MP4", ".AVI", ".MOV", ".MKV", ".WEBM"}


def fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 2


def load_model(weights: str):
    if not os.path.isfile(weights):
        print(
            f"ERROR: weights not found: {weights}\n"
            "Fetch the feasibility weights (not committed to git) with:\n"
            "  .\\.venv\\Scripts\\python.exe -c \""
            "from huggingface_hub import hf_hub_download; "
            "hf_hub_download('dronefreak/seadronessee-yolov8n', 'best.pt', local_dir='weights')\"\n"
            "See README 'Model & license' before redistributing or releasing.",
            file=sys.stderr,
        )
        return None
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed. Run: pip install -r requirements.txt",
              file=sys.stderr)
        return None
    try:
        return YOLO(weights)
    except Exception as exc:  # noqa: BLE001 - must surface init failures actionably
        print(f"ERROR: failed to initialize model from {weights}: {exc}", file=sys.stderr)
        return None


def annotate(frame, dets, names):
    for d in dets:
        x1, y1, x2, y2 = (int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]))
        label = f'{d["class"]} {d["conf"]:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return frame


def predict_frame(model, frame, conf: float, names):
    res = model.predict(source=frame, conf=conf, verbose=False)[0]
    dets = []
    boxes = res.boxes
    if boxes is not None and len(boxes):
        for i in range(len(boxes)):
            xyxy = [float(v) for v in boxes.xyxy[i].tolist()]
            dets.append({
                "x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3],
                "class": names[int(boxes.cls[i])],
                "class_id": int(boxes.cls[i]),
                "conf": float(boxes.conf[i]),
            })
    return dets


def run_image(model, path, outdir, conf, names):
    frame = cv2.imread(path)
    if frame is None:
        return fail(f"could not decode image (unsupported or corrupt): {path}")
    h, w = frame.shape[:2]
    t0 = time.time()
    dets = predict_frame(model, frame, conf, names)
    dt = time.time() - t0
    annotate(frame, dets, names)
    base = os.path.splitext(os.path.basename(path))[0]
    ann_path = os.path.join(outdir, base + "_annotated.jpg")
    js_path = os.path.join(outdir, base + "_detections.json")
    cv2.imwrite(ann_path, frame)
    with open(js_path, "w") as f:
        json.dump({"input": path, "width": w, "height": h,
                   "conf_threshold": conf, "infer_s": dt, "detections": dets}, f, indent=2)
    print(f"image: {path} ({w}x{h}) infer_s={dt:.2f} n={len(dets)}")
    print(f"  annotated: {ann_path}\n  json: {js_path}")
    return 0


def run_video(model, path, outdir, conf, names, stride, max_frames):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return fail(f"could not open video (unsupported codec or missing file): {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    base = os.path.splitext(os.path.basename(path))[0]
    ann_path = os.path.join(outdir, base + "_annotated.avi")
    js_path = os.path.join(outdir, base + "_detections.json")
    writer = cv2.VideoWriter(ann_path, cv2.VideoWriter_fourcc(*"MJPG"), fps / max(stride, 1), (w, h))
    if not writer.isOpened():
        cap.release()
        return fail("VideoWriter did not open (MJPG/AVI unavailable)")
    frames_out = []
    idx, sampled, total_dt = 0, 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0 and sampled < max_frames:
                t0 = time.time()
                dets = predict_frame(model, frame, conf, names)
                total_dt += time.time() - t0
                annotate(frame, dets, names)
                writer.write(frame)
                frames_out.append({"frame": idx, "t_s": idx / fps, "detections": dets})
                sampled += 1
            idx += 1
    finally:
        cap.release()
        writer.release()
    with open(js_path, "w") as f:
        json.dump({"input": path, "width": w, "height": h, "fps": fps,
                   "stride": stride, "frames_sampled": sampled,
                   "mean_infer_s": (total_dt / sampled if sampled else 0.0),
                   "frames": frames_out}, f, indent=2)
    print(f"video: {path} ({w}x{h} @ {fps:.1f}fps) sampled={sampled} mean_infer_s={(total_dt / sampled if sampled else 0):.2f}")
    print(f"  annotated: {ann_path}\n  json: {js_path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SeaRescue minimal maritime detector")
    ap.add_argument("--input", required=True, help="input image or video path")
    ap.add_argument("--output", required=True, help="output directory")
    ap.add_argument("--weights", default=os.path.join("weights", "best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=1, help="video: process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=300)
    args = ap.parse_args(argv)

    if not os.path.isfile(args.input):
        return fail(f"input not found: {args.input}")
    if not 0.0 < args.conf < 1.0:
        return fail(f"--conf must be in (0,1), got {args.conf}")
    os.makedirs(args.output, exist_ok=True)

    model = load_model(args.weights)
    if model is None:
        return 2
    names = getattr(model, "names", {}) or {}
    if not names:
        names = {i: c for i, c in enumerate(CLASSES_FALLBACK)}

    ext = os.path.splitext(args.input)[1]
    if ext in IMG_EXTS:
        return run_image(model, args.input, args.output, args.conf, names)
    if ext in VID_EXTS:
        return run_video(model, args.input, args.output, args.conf, names,
                         max(1, args.stride), max(1, args.max_frames))
    return fail(f"unknown extension '{ext}' (supported images {sorted(IMG_EXTS)}, videos {sorted(VID_EXTS)})")


if __name__ == "__main__":
    raise SystemExit(main())

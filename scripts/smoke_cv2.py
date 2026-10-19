"""OpenCV 5 smoke test: versions, synthetic video round-trip, basic ops."""
import os
import sys
import tempfile

import cv2
import numpy as np


N_FRAMES = 30
WIDTH, HEIGHT = 320, 240
FPS = 20.0


def make_frame(i: int) -> np.ndarray:
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    x = (i * 8) % (WIDTH - 60)
    cv2.rectangle(img, (x, 80), (x + 60, 160), (255, 255, 255), -1)
    cv2.circle(img, (WIDTH // 2, HEIGHT // 2), 20, (0, 255, 0), -1)
    return img


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"OpenCV: {cv2.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"OpenCV build: {cv2.getBuildInformation().splitlines()[0] if hasattr(cv2, 'getBuildInformation') else 'n/a'}")

    if not cv2.__version__.startswith("5."):
        print(f"FAIL: expected OpenCV 5.x, got {cv2.__version__}")
        return 1

    tmp_path = os.path.join(tempfile.gettempdir(), "searescue_smoke_tmp.avi")
    writer = None
    cap = None
    try:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(tmp_path, fourcc, FPS, (WIDTH, HEIGHT))
        if not writer.isOpened():
            print("FAIL: VideoWriter did not open (MJPG/AVI unavailable)")
            return 1
        for i in range(N_FRAMES):
            writer.write(make_frame(i))
        writer.release()
        writer = None

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            print("FAIL: VideoCapture did not open generated video")
            return 1
        decoded = 0
        first_gray = None
        prev_gray = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if first_gray is None:
                first_gray = gray
            prev_gray = gray
        cap.release()
        cap = None

        if decoded != N_FRAMES:
            print(f"FAIL: decoded {decoded} frames, expected {N_FRAMES}")
            return 1
        print(f"Video round-trip: wrote {N_FRAMES}, decoded {decoded} OK")

        # Basic ops on synthetic frames (deterministic, no video needed).
        a = make_frame(0)
        b = make_frame(5)
        gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray_a, cv2.CV_64F).var())
        diff_mean = float(cv2.absdiff(gray_a, gray_b).mean())
        flow = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_mag = float(np.abs(flow).mean())
        print(f"Grayscale shape: {gray_a.shape} dtype: {gray_a.dtype}")
        print(f"Laplacian variance: {lap_var:.2f}")
        print(f"Frame-diff mean: {diff_mean:.2f}")
        print(f"Optical-flow mean abs: {flow_mag:.4f}")
        if gray_a.shape != (HEIGHT, WIDTH):
            print("FAIL: unexpected grayscale shape")
            return 1
        if lap_var <= 0:
            print("FAIL: Laplacian variance not positive")
            return 1
        if diff_mean <= 0:
            print("FAIL: frame differencing detected no motion on moving-rectangle frames")
            return 1
        if flow.shape != (HEIGHT, WIDTH, 2):
            print(f"FAIL: unexpected flow shape {flow.shape}")
            return 1

        # Sanity on a decoded frame too (proves decode path yields usable pixels).
        if first_gray is None or prev_gray is None:
            print("FAIL: no decoded frames available for pixel checks")
            return 1

        print("PASS: OpenCV 5 smoke test OK")
        return 0
    except Exception as exc:  # noqa: BLE001 - smoke script must report, not raise
        print(f"FAIL: exception: {exc}")
        return 1
    finally:
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as exc:
            print(f"WARNING: could not remove temp file {tmp_path}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())

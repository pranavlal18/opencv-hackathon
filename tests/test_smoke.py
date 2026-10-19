import os
import tempfile

import cv2
import numpy as np


def _make_frame(i, w=320, h=240):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    x = (i * 8) % (w - 60)
    cv2.rectangle(img, (x, 80), (x + 60, 160), (255, 255, 255), -1)
    return img


def test_opencv_major_version_is_5():
    assert cv2.__version__.startswith("5."), f"expected OpenCV 5.x, got {cv2.__version__}"


def test_synthetic_video_roundtrip_frame_count():
    n = 30
    w, h = 320, 240
    tmp = os.path.join(tempfile.gettempdir(), "searescue_pytest_tmp.avi")
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (w, h))
    try:
        assert writer.isOpened(), "VideoWriter did not open (MJPG/AVI unavailable)"
        for i in range(n):
            writer.write(_make_frame(i, w, h))
    finally:
        writer.release()
    cap = cv2.VideoCapture(tmp)
    try:
        assert cap.isOpened(), "VideoCapture did not open generated video"
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            assert frame is not None and frame.size > 0
    finally:
        cap.release()
        if os.path.exists(tmp):
            os.remove(tmp)
    assert count == n, f"decoded {count} frames, expected {n}"


def test_basic_image_processing_ops():
    a = _make_frame(0)
    b = _make_frame(5)
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    assert ga.shape == (240, 320)
    assert float(cv2.Laplacian(ga, cv2.CV_64F).var()) > 0
    assert float(cv2.absdiff(ga, gb).mean()) > 0
    flow = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    assert flow.shape == (240, 320, 2)

"""Thu thập ảnh anchor / positive / verification từ webcam bằng OpenCV.

Mỗi khung hình được cắt vùng vuông ở giữa (mặc định 250×250 - cùng cỡ ảnh LFW)
để phân phối ảnh tự thu gần với ảnh negative.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from .config import ANCHOR_DIR, POSITIVE_DIR, UsageError, VERIFICATION_DIR

KEY_TO_SUBDIR = {"a": ANCHOR_DIR, "p": POSITIVE_DIR, "v": VERIFICATION_DIR}
WINDOW_TITLE = "Thu thap du lieu - a: anchor | p: positive | v: verification | q: thoat"
# Trong ảnh LFW 250×250, khuôn mặt (trán -> cằm) rộng khoảng 110 px, tức ~45% cạnh ảnh và
# nằm ở giữa. Khung hướng dẫn giúp ảnh tự thu có cùng tỉ lệ mặt/khung với dữ liệu huấn luyện.
GUIDE_WIDTH_RATIO = 0.45
GUIDE_HEIGHT_RATIO = 0.55


def center_crop(frame: np.ndarray, size: int) -> np.ndarray:
    """Cắt vùng vuông cạnh ``size`` ở giữa khung hình (thu nhỏ nếu khung hình nhỏ hơn)."""
    height, width = frame.shape[:2]
    size = min(size, height, width)
    top = (height - size) // 2
    left = (width - size) // 2
    return frame[top : top + size, left : left + size]


def guide_box(size: int) -> tuple[int, int, int, int]:
    """Hình chữ nhật (x1, y1, x2, y2) ở giữa ảnh vuông cạnh ``size`` - cỡ mặt nên có."""
    half_w = round(size * GUIDE_WIDTH_RATIO / 2)
    half_h = round(size * GUIDE_HEIGHT_RATIO / 2)
    center = size // 2
    return center - half_w, center - half_h, center + half_w, center + half_h


def draw_guide(image: np.ndarray) -> np.ndarray:
    """Bản sao của ảnh có vẽ khung hướng dẫn (chỉ để hiển thị, không lưu)."""
    import cv2

    preview = image.copy()
    x1, y1, x2, y2 = guide_box(min(image.shape[:2]))
    cv2.ellipse(preview, ((x1 + x2) // 2, (y1 + y2) // 2), ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, (0, 255, 0), 1)
    return preview


def save_image(image: np.ndarray, directory: Path) -> Path:
    """Lưu ảnh JPEG, tên file theo thời điểm chụp (sắp xếp được theo thời gian)."""
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Không mã hoá được ảnh JPEG")
    # Ghi bằng Python thay vì cv2.imwrite để không lỗi với đường dẫn Unicode trên Windows.
    path.write_bytes(encoded.tobytes())
    return path


def collect_samples(data_dir: Path, camera_index: int = 0, crop_size: int = 250) -> dict[str, int]:
    """Mở webcam; nhấn a/p/v để lưu ảnh vào anchor/positive/verification, q để thoát."""
    import cv2

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise UsageError(f"Không mở được camera số {camera_index} - thử --camera 1 hoặc kiểm tra quyền camera.")
    counts = {subdir: 0 for subdir in KEY_TO_SUBDIR.values()}
    print("Phím tắt: [a] lưu anchor, [p] lưu positive, [v] lưu ảnh tham chiếu, [q] thoát")
    print("Đặt khuôn mặt (trán -> cằm) vừa trong hình elip xanh; elip không có trong ảnh lưu.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Không đọc được khung hình từ camera.")
                break
            crop = center_crop(frame, crop_size)
            cv2.imshow(WINDOW_TITLE, draw_guide(crop))
            key = chr(cv2.waitKey(1) & 0xFF).lower()
            if key == "q":
                break
            if key in KEY_TO_SUBDIR:
                subdir = KEY_TO_SUBDIR[key]
                path = save_image(crop, Path(data_dir) / subdir)
                counts[subdir] += 1
                print(f"[{subdir}] {path.name} (đã lưu {counts[subdir]} ảnh trong phiên này)")
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return counts

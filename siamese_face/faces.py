"""Phát hiện và cắt khuôn mặt bằng bộ phân loại Haar cascade của OpenCV (Viola–Jones, 2001).

Ảnh LFW được tạo bằng chính bộ phát hiện Viola–Jones rồi căn khuôn mặt vào giữa ảnh, nên
dùng cùng họ thuật toán ở đây giúp ảnh LFW và ảnh tự chụp được cắt theo cùng một cách:
khuôn mặt nằm giữa và chiếm cùng một tỉ lệ khung hình, bất kể người chụp đứng gần hay xa.

Đây là bước *phát hiện và cắt*, chưa phải *căn chỉnh* (xoay theo vị trí hai mắt).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import UsageError

# Hộp Haar thường chỉ ôm từ trán tới cằm; nới mỗi phía 20% để lấy trọn khuôn mặt.
FACE_MARGIN = 0.2
# Bỏ qua các vùng được phát hiện nhỏ hơn 20% cạnh ngắn của ảnh (thường là nhiễu nền).
MIN_FACE_FRACTION = 0.2
# Không thấy khuôn mặt: cắt vùng vuông ở giữa, cạnh = 60% cạnh ngắn. Với ảnh LFW và ảnh
# chụp theo elip hướng dẫn, vùng này xấp xỉ vùng cắt khi phát hiện được mặt.
FALLBACK_FRACTION = 0.6
CASCADE_FILE = "haarcascade_frontalface_default.xml"

_detector = None


@dataclass(frozen=True)
class FaceCrop:
    """Kết quả cắt: ảnh vuông (uint8, RGB), vị trí ``(x, y, cạnh)`` trong ảnh gốc."""

    image: np.ndarray
    box: tuple[int, int, int]
    detected: bool


def read_image_rgb(path: str | Path) -> np.ndarray:
    """Đọc ảnh thành mảng uint8 ``(cao, rộng, 3)`` hệ màu RGB (an toàn với đường dẫn Unicode)."""
    import cv2

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {path}")
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise UsageError(f"Không giải mã được ảnh: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _cascade():
    global _detector
    if _detector is None:
        import cv2

        detector = cv2.CascadeClassifier(cv2.data.haarcascades + CASCADE_FILE)
        if detector.empty():
            raise RuntimeError(f"Không nạp được {CASCADE_FILE} từ OpenCV")
        _detector = detector
    return _detector


def detect_face(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Hộp ``(x, y, rộng, cao)`` của khuôn mặt gần tâm ảnh nhất, hoặc ``None`` nếu không thấy."""
    import cv2

    gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    min_side = max(20, int(min(gray.shape) * MIN_FACE_FRACTION))
    faces = _cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side)
    )
    return nearest_to_center(faces, gray.shape)


def nearest_to_center(faces, shape) -> tuple[int, int, int, int] | None:
    """Chọn hộp có tâm gần tâm ảnh nhất: chủ thể của ảnh LFW và ảnh tự chụp luôn ở giữa."""
    if len(faces) == 0:
        return None
    height, width = shape[:2]

    def distance(face) -> float:
        x, y, w, h = face
        return (x + w / 2 - width / 2) ** 2 + (y + h / 2 - height / 2) ** 2

    x, y, w, h = min(faces, key=distance)
    return int(x), int(y), int(w), int(h)


def square_around(shape, face: tuple[int, int, int, int] | None) -> tuple[int, int, int]:
    """Vùng vuông ``(x, y, cạnh)`` quanh khuôn mặt (đã nới biên), luôn nằm trọn trong ảnh."""
    height, width = shape[:2]
    if face is None:
        side = round(min(height, width) * FALLBACK_FRACTION)
        center_x, center_y = width / 2, height / 2
    else:
        x, y, w, h = face
        side = round(max(w, h) * (1 + 2 * FACE_MARGIN))
        center_x, center_y = x + w / 2, y + h / 2
    side = max(1, min(int(side), height, width))
    left = min(max(round(center_x - side / 2), 0), width - side)
    top = min(max(round(center_y - side / 2), 0), height - side)
    return int(left), int(top), side


def crop_face(image: np.ndarray) -> FaceCrop:
    """Phát hiện khuôn mặt rồi cắt vùng vuông quanh nó (không thấy mặt thì cắt vùng giữa)."""
    face = detect_face(image)
    left, top, side = square_around(image.shape, face)
    return FaceCrop(image[top : top + side, left : left + side], (left, top, side), face is not None)

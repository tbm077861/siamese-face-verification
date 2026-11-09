"""Hằng số, cấu hình thí nghiệm và tiện ích lưu JSON dùng chung."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

# Cạnh (pixel) của ảnh vuông đưa vào mạng.
IMAGE_SIZE = 100

# Bố cục thư mục dữ liệu, tương đối với data_dir.
ANCHOR_DIR = "anchor"  # ảnh của người cần xác minh
POSITIVE_DIR = "positive"  # ảnh khác của chính người đó
NEGATIVE_DIR = "negative"  # ảnh của người khác (VD: LFW)
VERIFICATION_DIR = "verification"  # ảnh tham chiếu dùng khi suy luận

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})

# Tên file trong thư mục kết quả của một lần chạy (runs/<thời điểm>/).
MODEL_FILENAME = "model.keras"
CONFIG_FILENAME = "config.json"
SPLITS_FILENAME = "splits.json"
METRICS_FILENAME = "metrics.json"


class UsageError(ValueError):
    """Lỗi do dữ liệu hoặc tham số đầu vào; CLI in thông báo gọn thay vì traceback."""


@dataclass
class TrainConfig:
    """Siêu tham số của một lần huấn luyện, lưu kèm mô hình để tái lập thí nghiệm."""

    data_dir: str = "data"
    output_dir: str = "runs"
    image_size: int = IMAGE_SIZE
    embedding_dim: int = 4096
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-4
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    max_images: int | None = None
    augment: bool = False
    patience: int | None = 10
    seed: int = 42

    def save(self, path: str | Path) -> None:
        write_json(asdict(self), path)

    @classmethod
    def load(cls, path: str | Path) -> TrainConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


def write_json(data: Any, path: str | Path) -> None:
    """Ghi JSON dạng UTF-8; NaN/±inf được đổi thành null để file luôn hợp lệ."""
    text = json.dumps(_json_safe(data), indent=2, ensure_ascii=False)
    Path(path).write_text(text, encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()  # số vô hướng NumPy -> kiểu Python
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

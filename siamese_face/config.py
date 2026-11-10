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

# Cách tiền xử lý ảnh trước khi resize: "center" cắt vùng vuông ở giữa ảnh; "face" phát
# hiện khuôn mặt (Haar cascade) rồi cắt vùng vuông quanh nó (xem faces.py).
PREPROCESS_MODES = ("center", "face")

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

    mode: str = "personal"  # "personal" (anchor/positive/negative) hoặc "lfw" (cặp từ LFW)
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
    max_pairs_per_identity: int = 20  # chỉ dùng cho mode="lfw"
    augment: bool = False
    patience: int | None = 10
    seed: int = 42
    preprocess: str = "center"  # một trong PREPROCESS_MODES; lần chạy cũ không có trường này

    def save(self, path: str | Path) -> None:
        write_json(asdict(self), path)

    @classmethod
    def load(cls, path: str | Path) -> TrainConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


def check_preprocess(preprocess: str) -> str:
    if preprocess not in PREPROCESS_MODES:
        raise UsageError(
            f"Cách tiền xử lý không hợp lệ: {preprocess!r} (chọn một trong {', '.join(PREPROCESS_MODES)})"
        )
    return preprocess


def preprocess_for_model(model_path: str | Path, requested: str | None = "auto") -> str:
    """Cách tiền xử lý dùng với một mô hình.

    ``"auto"`` (hoặc ``None``) đọc ``config.json`` nằm cạnh file mô hình - tức thư mục kết
    quả của lệnh ``train`` - để suy luận dùng đúng cách tiền xử lý lúc huấn luyện; không có
    file đó thì dùng ``"center"``.
    """
    if requested not in (None, "auto"):
        return check_preprocess(requested)
    config_path = Path(model_path).parent / CONFIG_FILENAME
    if config_path.is_file():
        return check_preprocess(TrainConfig.load(config_path).preprocess)
    return "center"


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

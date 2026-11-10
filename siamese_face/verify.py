"""Suy luận: xác minh một ảnh so với tập ảnh tham chiếu của người dùng.

Quy tắc quyết định với K ảnh tham chiếu r_1..r_K và ảnh cần xác minh x:
    s_k = p(r_k, x)                          - điểm tương đồng từ mạng Siamese
    ảnh r_k "khớp"  ⇔  s_k ≥ τ_d             - ngưỡng phát hiện
    xác minh thành công ⇔ (#khớp / K) ≥ τ_v  - ngưỡng xác minh
Khi K = 1 quy tắc trở thành s ≥ τ_d, giống phiên bản ứng dụng cũ (một ảnh anchor).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .config import UsageError
from .data import is_image_file, list_images
from .model import model_image_size
from .pipeline import preprocess_image


@dataclass
class VerificationResult:
    references: list[str]
    scores: list[float]
    detection_threshold: float = 0.5
    verification_threshold: float = 0.5
    no_face: list[str] = field(default_factory=list)  # ảnh không phát hiện được khuôn mặt

    @property
    def detections(self) -> int:
        return sum(score >= self.detection_threshold for score in self.scores)

    @property
    def ratio(self) -> float:
        return self.detections / len(self.scores) if self.scores else 0.0

    @property
    def verified(self) -> bool:
        return bool(self.scores) and self.ratio >= self.verification_threshold


def resolve_references(sources: Iterable[str | Path]) -> list[str]:
    """Gom ảnh tham chiếu từ danh sách thư mục và/hoặc file ảnh."""
    references: list[str] = []
    for source in sources:
        path = Path(source)
        if path.is_dir():
            references.extend(list_images(path))
        elif path.is_file() and is_image_file(path):
            references.append(path.as_posix())
        else:
            raise FileNotFoundError(f"Không tìm thấy ảnh/thư mục tham chiếu: {source}")
    if not references:
        raise UsageError("Không có ảnh tham chiếu nào để so sánh.")
    return references


def verify_image(
    model,
    image_path: str | Path,
    references: Sequence[str | Path],
    detection_threshold: float = 0.5,
    verification_threshold: float = 0.5,
    batch_size: int = 32,
    preprocess: str = "center",
) -> VerificationResult:
    """So ảnh ``image_path`` với từng ảnh tham chiếu và áp dụng quy tắc quyết định."""
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh cần xác minh: {image_path}")
    references = [str(reference) for reference in references]
    if not references:
        raise UsageError("Không có ảnh tham chiếu nào để so sánh.")

    size = model_image_size(model)
    no_face: list[str] = []

    def load(path: str) -> np.ndarray:
        image, detected = preprocess_image(path, size, preprocess)
        if detected is False:
            no_face.append(path)
        return image

    probe = load(str(image_path))
    scores: list[float] = []
    for start in range(0, len(references), batch_size):
        chunk = references[start : start + batch_size]
        reference_batch = np.stack([load(path) for path in chunk])
        probe_batch = np.repeat(probe[np.newaxis], len(chunk), axis=0)
        output = model.predict_on_batch([reference_batch, probe_batch])
        scores.extend(float(score) for score in np.asarray(output).reshape(-1))
    return VerificationResult(
        references, scores, detection_threshold, verification_threshold, no_face
    )

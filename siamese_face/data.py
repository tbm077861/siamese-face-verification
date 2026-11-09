"""Quản lý dữ liệu thuần Python: liệt kê ảnh, chia train/val/test và ghép cặp.

Module không phụ thuộc TensorFlow để có thể kiểm thử nhanh và độc lập.

Hai chế độ thực nghiệm (``MODES``):
  - ``personal``: cặp (anchor, positive) / (anchor, negative) - mô hình cá nhân hoá
    cho một người, cần ảnh webcam của người đó.
  - ``lfw``: cặp cùng người / khác người lấy từ chính bộ LFW - mô hình tổng quát,
    tập test gồm những người chưa từng thấy (đúng thiết lập one-shot).

Giao thức chống rò rỉ dữ liệu (xem docs/phuong_phap.md, mục 7):
  1. Chia ảnh thành train/val/test *trước*, rồi mới ghép cặp trong từng tập.
  2. Ảnh negative chia theo *danh tính* - người lạ ở tập test chưa từng xuất hiện.
  3. Chia tất định theo seed và lưu lại danh sách cặp (splits.json).
"""

from __future__ import annotations

import itertools
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .config import (
    ANCHOR_DIR,
    IMAGE_EXTENSIONS,
    NEGATIVE_DIR,
    POSITIVE_DIR,
    UsageError,
    write_json,
)

_LFW_NAME = re.compile(r"^(?P<identity>.+)_\d+$")


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def list_images(directory: str | Path, recursive: bool = False) -> list[str]:
    """Danh sách đường dẫn ảnh (đã sắp xếp, dạng POSIX) trong ``directory``."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(p.as_posix() for p in candidates if p.is_file() and is_image_file(p))


def identity_from_filename(path: str | Path) -> str:
    """Danh tính theo quy ước tên file LFW: ``George_W_Bush_0001.jpg`` -> ``George_W_Bush``."""
    stem = Path(path).stem
    match = _LFW_NAME.match(stem)
    return match.group("identity") if match else stem


def split_items(
    items: Sequence[str],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    group_fn: Callable[[str], str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Chia ``items`` thành (train, val, test) một cách tất định theo ``seed``.

    Khi có ``group_fn``, các phần tử cùng nhóm (VD: cùng một người) luôn nằm chung
    một tập, nhờ đó danh tính trong tập test chưa từng xuất hiện khi huấn luyện.
    """
    if len(ratios) != 3 or min(ratios) < -1e-9 or abs(sum(ratios) - 1.0) > 1e-6:
        raise UsageError(f"Tỉ lệ chia phải gồm 3 số không âm có tổng bằng 1, nhận được {ratios}")

    groups: dict[str, list[str]] = defaultdict(list)
    for item in sorted(items):
        groups[group_fn(item) if group_fn else item].append(item)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    total = len(items)
    train_end = round(total * ratios[0])
    val_end = round(total * (ratios[0] + ratios[1]))
    parts: tuple[list[str], list[str], list[str]] = ([], [], [])
    assigned = 0
    for key in keys:
        index = 0 if assigned < train_end else 1 if assigned < val_end else 2
        parts[index].extend(groups[key])
        assigned += len(groups[key])
    return parts


@dataclass
class PairSet:
    """Tập cặp ảnh (anchor, candidate) với nhãn 1 = cùng người, 0 = khác người."""

    anchors: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not len(self.anchors) == len(self.candidates) == len(self.labels):
            raise ValueError("anchors, candidates và labels phải có cùng độ dài")

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def num_positive(self) -> int:
        return int(sum(self.labels))

    def to_dict(self) -> dict[str, list]:
        return {"anchors": self.anchors, "candidates": self.candidates, "labels": self.labels}

    @classmethod
    def from_dict(cls, data: dict) -> PairSet:
        return cls(
            anchors=list(data["anchors"]),
            candidates=list(data["candidates"]),
            labels=[int(label) for label in data["labels"]],
        )


@dataclass
class PairSplits:
    """Ba tập cặp train/val/test của một thí nghiệm."""

    train: PairSet
    val: PairSet
    test: PairSet

    def items(self) -> list[tuple[str, PairSet]]:
        return [("train", self.train), ("val", self.val), ("test", self.test)]

    def save(self, path: str | Path) -> None:
        write_json({name: pairs.to_dict() for name, pairs in self.items()}, path)

    @classmethod
    def load(cls, path: str | Path) -> PairSplits:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{name: PairSet.from_dict(data[name]) for name in ("train", "val", "test")})


def make_pairs(
    anchors: Sequence[str],
    positives: Sequence[str],
    negatives: Sequence[str],
    seed: int = 42,
) -> PairSet:
    """Ghép cặp cân bằng lớp.

    Tạo n cặp (anchor, positive) nhãn 1 và n cặp (anchor, negative) nhãn 0 với
    n = min(#anchor, #positive, #negative); mỗi ảnh anchor được dùng đúng hai lần.
    """
    rng = random.Random(seed)
    n = min(len(anchors), len(positives), len(negatives))
    chosen_anchors = rng.sample(sorted(anchors), n)
    chosen_positives = rng.sample(sorted(positives), n)
    chosen_negatives = rng.sample(sorted(negatives), n)
    triples = [(a, p, 1) for a, p in zip(chosen_anchors, chosen_positives)]
    triples += [(a, q, 0) for a, q in zip(chosen_anchors, chosen_negatives)]
    rng.shuffle(triples)
    return PairSet(
        anchors=[anchor for anchor, _, _ in triples],
        candidates=[candidate for _, candidate, _ in triples],
        labels=[label for _, _, label in triples],
    )


def make_identity_pairs(
    images: Sequence[str],
    seed: int = 42,
    max_pairs_per_identity: int = 20,
) -> PairSet:
    """Ghép cặp từ ảnh có nhãn danh tính trong tên file (kiểu LFW).

    - Cặp dương: hai ảnh khác nhau của cùng một người; tối đa ``max_pairs_per_identity``
      cặp mỗi người để vài người có rất nhiều ảnh (VD: George_W_Bush, 530 ảnh) không
      áp đảo dữ liệu.
    - Cặp âm: hai ảnh rút ngẫu nhiên đều trên toàn bộ ảnh, thuộc hai người khác nhau;
      số cặp âm bằng số cặp dương (cân bằng lớp).
    """
    rng = random.Random(seed)
    groups: dict[str, list[str]] = defaultdict(list)
    for path in sorted(images):
        groups[identity_from_filename(path)].append(path)

    positives: list[tuple[str, str]] = []
    for identity in sorted(groups):
        combos = list(itertools.combinations(groups[identity], 2))
        if len(combos) > max_pairs_per_identity:
            combos = rng.sample(combos, max_pairs_per_identity)
        positives.extend(combos)

    pool = sorted(images)
    negatives: set[tuple[str, str]] = set()
    attempts, max_attempts = 0, 20 * len(positives) + 100
    while len(groups) >= 2 and len(negatives) < len(positives) and attempts < max_attempts:
        first, second = rng.sample(pool, 2)
        if identity_from_filename(first) != identity_from_filename(second):
            negatives.add((first, second))
        attempts += 1

    # sorted(...) vì thứ tự duyệt set phụ thuộc hash ngẫu nhiên của Python.
    triples = [(a, b, 1) for a, b in positives] + [(a, b, 0) for a, b in sorted(negatives)]
    rng.shuffle(triples)
    return PairSet(
        anchors=[a for a, _, _ in triples],
        candidates=[b for _, b, _ in triples],
        labels=[label for _, _, label in triples],
    )


def _split_ratios(train_ratio: float, val_ratio: float) -> tuple[float, float, float]:
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < -1e-9:
        raise UsageError(f"train_ratio + val_ratio phải ≤ 1 (đang là {train_ratio + val_ratio})")
    return train_ratio, val_ratio, max(test_ratio, 0.0)


def _load_image_list(directory: Path, seed: int, max_images: int | None) -> list[str]:
    found = list_images(directory)
    if not found:
        raise FileNotFoundError(
            f"Không có ảnh nào trong {directory} - xem data/README.md để chuẩn bị dữ liệu."
        )
    if max_images is not None and len(found) > max_images:
        found = sorted(random.Random(seed).sample(found, max_images))
    return found


def build_identity_splits(
    image_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    max_images: int | None = None,
    max_pairs_per_identity: int = 20,
) -> PairSplits:
    """Chế độ ``lfw``: chia *danh tính* thành train/val/test rồi ghép cặp trong từng tập.

    Người xuất hiện ở tập test không bao giờ xuất hiện khi huấn luyện, nên kết quả đo
    được là khả năng xác minh người *mới* - đúng tinh thần one-shot.
    """
    images = _load_image_list(Path(image_dir), seed, max_images)
    parts = split_items(
        images, _split_ratios(train_ratio, val_ratio), seed, group_fn=identity_from_filename
    )
    train, val, test = (
        make_identity_pairs(part, seed=seed + offset, max_pairs_per_identity=max_pairs_per_identity)
        for offset, part in enumerate(parts)
    )
    return PairSplits(train=train, val=val, test=test)


MODES = ("personal", "lfw")


def build_experiment_splits(
    mode: str,
    data_dir: str | Path = "data",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    max_images: int | None = None,
    max_pairs_per_identity: int = 20,
) -> PairSplits:
    """Tạo ba tập cặp theo chế độ thực nghiệm ``mode`` (``personal`` hoặc ``lfw``)."""
    if mode == "personal":
        return build_splits(data_dir, train_ratio, val_ratio, seed, max_images)
    if mode == "lfw":
        return build_identity_splits(
            Path(data_dir) / NEGATIVE_DIR,
            train_ratio,
            val_ratio,
            seed,
            max_images,
            max_pairs_per_identity,
        )
    raise UsageError(f"Chế độ không hợp lệ: {mode} (chọn một trong {MODES})")


def build_splits(
    data_dir: str | Path = "data",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    max_images: int | None = None,
) -> PairSplits:
    """Chế độ ``personal``: đọc ``data_dir/{anchor,positive,negative}`` và tạo ba tập cặp."""
    ratios = _split_ratios(train_ratio, val_ratio)
    images = {
        name: _load_image_list(Path(data_dir) / name, seed, max_images)
        for name in (ANCHOR_DIR, POSITIVE_DIR, NEGATIVE_DIR)
    }

    anchor_parts = split_items(images[ANCHOR_DIR], ratios, seed)
    positive_parts = split_items(images[POSITIVE_DIR], ratios, seed + 1)
    negative_parts = split_items(
        images[NEGATIVE_DIR], ratios, seed + 2, group_fn=identity_from_filename
    )
    train, val, test = (
        make_pairs(anchors, positives, negatives, seed=seed + offset)
        for offset, (anchors, positives, negatives) in enumerate(
            zip(anchor_parts, positive_parts, negative_parts)
        )
    )
    return PairSplits(train=train, val=val, test=test)

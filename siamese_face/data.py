"""Quản lý dữ liệu thuần Python: liệt kê ảnh, chia train/val/test và ghép cặp.

Module không phụ thuộc TensorFlow để có thể kiểm thử nhanh và độc lập.

Giao thức chống rò rỉ dữ liệu (xem docs/phuong_phap.md, mục 7):
  1. Chia ảnh thành train/val/test *trước*, rồi mới ghép cặp trong từng tập.
  2. Ảnh negative chia theo *danh tính* - người lạ ở tập test chưa từng xuất hiện.
  3. Chia tất định theo seed và lưu lại danh sách cặp (splits.json).
"""

from __future__ import annotations

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


def build_splits(
    data_dir: str | Path = "data",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    max_images: int | None = None,
) -> PairSplits:
    """Đọc ``data_dir/{anchor,positive,negative}`` và tạo ba tập cặp không rò rỉ dữ liệu."""
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < -1e-9:
        raise UsageError(f"train_ratio + val_ratio phải ≤ 1 (đang là {train_ratio + val_ratio})")
    ratios = (train_ratio, val_ratio, max(test_ratio, 0.0))

    images: dict[str, list[str]] = {}
    for name in (ANCHOR_DIR, POSITIVE_DIR, NEGATIVE_DIR):
        directory = Path(data_dir) / name
        found = list_images(directory)
        if not found:
            raise FileNotFoundError(
                f"Không có ảnh nào trong {directory} - xem data/README.md để chuẩn bị dữ liệu."
            )
        if max_images is not None and len(found) > max_images:
            found = sorted(random.Random(seed).sample(found, max_images))
        images[name] = found

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

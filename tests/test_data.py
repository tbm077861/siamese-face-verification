from pathlib import Path

import pytest

from siamese_face.data import (
    PairSet,
    PairSplits,
    build_splits,
    identity_from_filename,
    list_images,
    make_pairs,
    split_items,
)


def _touch(directory: Path, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"")


def test_list_images_filters_extensions(tmp_path):
    _touch(tmp_path, ["b.PNG", "a.jpg", "notes.txt", ".gitkeep"])
    _touch(tmp_path / "sub", ["c.jpg"])
    assert [Path(p).name for p in list_images(tmp_path)] == ["a.jpg", "b.PNG"]
    assert len(list_images(tmp_path, recursive=True)) == 3
    assert list_images(tmp_path / "missing") == []


@pytest.mark.parametrize(
    "filename, identity",
    [
        ("Aaron_Eckhart_0001.jpg", "Aaron_Eckhart"),
        ("lfw/George_W_Bush/George_W_Bush_0123.jpg", "George_W_Bush"),
        ("random.jpg", "random"),
    ],
)
def test_identity_from_filename(filename, identity):
    assert identity_from_filename(filename) == identity


def test_split_items_ratios_and_determinism():
    items = [f"img_{i:03d}.jpg" for i in range(100)]
    train, val, test = split_items(items, (0.7, 0.15, 0.15), seed=1)
    assert (len(train), len(val), len(test)) == (70, 15, 15)
    assert set(train) | set(val) | set(test) == set(items)
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    # Kết quả không phụ thuộc thứ tự đầu vào
    assert split_items(list(reversed(items)), (0.7, 0.15, 0.15), seed=1) == (train, val, test)


def test_split_items_keeps_groups_together():
    items = [f"Person{p}_{i:04d}.jpg" for p in range(20) for i in range(1, 4)]
    parts = split_items(items, (0.6, 0.2, 0.2), seed=0, group_fn=identity_from_filename)
    seen: dict[str, set[int]] = {}
    for index, part in enumerate(parts):
        for item in part:
            seen.setdefault(identity_from_filename(item), set()).add(index)
    assert all(len(indices) == 1 for indices in seen.values())
    assert sum(map(len, parts)) == len(items)


@pytest.mark.parametrize("ratios", [(0.8, 0.3, -0.1), (0.5, 0.2, 0.2)])
def test_split_items_rejects_invalid_ratios(ratios):
    with pytest.raises(ValueError):
        split_items(["a"], ratios)


def test_make_pairs_is_balanced():
    anchors = [f"a{i}" for i in range(5)]
    positives = [f"p{i}" for i in range(4)]
    negatives = [f"n{i}" for i in range(10)]
    pairs = make_pairs(anchors, positives, negatives, seed=0)
    assert len(pairs) == 8 and pairs.num_positive == 4
    for anchor, candidate, label in zip(pairs.anchors, pairs.candidates, pairs.labels):
        assert anchor in anchors
        assert candidate in (positives if label == 1 else negatives)


def test_build_splits_has_no_image_leakage(tmp_path):
    _touch(tmp_path / "anchor", [f"a_{i:03d}.jpg" for i in range(20)])
    _touch(tmp_path / "positive", [f"p_{i:03d}.jpg" for i in range(20)])
    _touch(tmp_path / "negative", [f"Person{i}_0001.jpg" for i in range(40)])
    splits = build_splits(tmp_path, train_ratio=0.7, val_ratio=0.15, seed=3)
    images = [set(pairs.anchors) | set(pairs.candidates) for _, pairs in splits.items()]
    assert not (images[0] & images[1] or images[0] & images[2] or images[1] & images[2])
    assert len(splits.train) == 28  # 14 cặp cùng người + 14 cặp khác người
    assert splits.train.num_positive == 14


def test_build_splits_requires_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_splits(tmp_path)


def test_pair_splits_roundtrip(tmp_path):
    splits = PairSplits(PairSet(["a"], ["b"], [1]), PairSet(), PairSet(["c"], ["d"], [0]))
    path = tmp_path / "splits.json"
    splits.save(path)
    assert PairSplits.load(path) == splits


def test_pair_set_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        PairSet(["a"], [], [1])

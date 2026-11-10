"""Chạy thử toàn bộ quy trình train -> evaluate trên dữ liệu giả lập rất nhỏ."""

import json

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from siamese_face.config import TrainConfig  # noqa: E402
from siamese_face.evaluate import evaluate_run, format_report  # noqa: E402
from siamese_face.train import train  # noqa: E402


def _make_dataset(root):
    rng = np.random.default_rng(0)
    layout = {
        "anchor": [f"a_{i}.jpg" for i in range(6)],
        "positive": [f"p_{i}.jpg" for i in range(6)],
        "negative": [f"Person{i}_0001.jpg" for i in range(6)],
    }
    for subdir, names in layout.items():
        (root / subdir).mkdir(parents=True)
        for name in names:
            image = rng.integers(0, 256, size=(120, 120, 3), dtype=np.uint8)
            tf.io.write_file(str(root / subdir / name), tf.io.encode_jpeg(image))


def test_train_then_evaluate_smoke(tmp_path):
    data_dir = tmp_path / "data"
    _make_dataset(data_dir)
    config = TrainConfig(
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "runs"),
        epochs=1,
        batch_size=4,
        embedding_dim=8,
        patience=None,
    )
    run_dir = train(config)
    for name in ("model.keras", "config.json", "splits.json", "history.csv", "training_summary.json"):
        assert (run_dir / name).is_file(), name

    report = evaluate_run(run_dir)
    assert report["test"]["n_pairs"] == 2
    assert 0.0 <= report["test"]["accuracy"] <= 1.0
    saved = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved["threshold"] == pytest.approx(report["threshold"])
    assert "Tập test" in format_report(report)


def test_train_with_face_preprocessing_records_choice(tmp_path):
    pytest.importorskip("cv2")
    data_dir = tmp_path / "data"
    _make_dataset(data_dir)
    config = TrainConfig(
        data_dir=str(data_dir),
        output_dir=str(tmp_path / "runs"),
        epochs=1,
        batch_size=4,
        embedding_dim=8,
        patience=None,
        preprocess="face",
    )
    run_dir = train(config)
    saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert saved["preprocess"] == "face"
    report = evaluate_run(run_dir)  # tự đọc cách tiền xử lý từ config.json
    assert report["preprocess"] == "face"
    assert "Tiền xử lý: face" in format_report(report)

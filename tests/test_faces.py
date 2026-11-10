"""Kiểm thử bước phát hiện/cắt khuôn mặt, đánh giá trên ảnh tự chụp và so sánh hai lần chạy."""

import csv
import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from siamese_face import faces  # noqa: E402
from siamese_face.config import TrainConfig, UsageError, preprocess_for_model  # noqa: E402
from siamese_face.faces import crop_face, nearest_to_center, read_image_rgb, square_around  # noqa: E402


def _write_image(path, array) -> str:
    ok, encoded = cv2.imencode(path.suffix, np.asarray(array, dtype=np.uint8))
    assert ok
    path.write_bytes(encoded.tobytes())
    return str(path)


def _noise_images(directory, names, size=120, seed=0):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    return [
        _write_image(directory / name, rng.integers(0, 256, size=(size, size, 3)))
        for name in names
    ]


# ---------- Hình học vùng cắt ----------

def test_square_around_centers_on_face_with_margin():
    assert square_around((250, 250), (70, 70, 110, 110)) == (48, 48, 154)


def test_square_around_stays_inside_image():
    assert square_around((200, 300), (0, 0, 100, 100)) == (0, 0, 140)
    assert square_around((200, 300), (220, 120, 80, 80)) == (188, 88, 112)


def test_square_around_clamps_to_short_side():
    left, top, side = square_around((250, 400), (0, 0, 300, 300))
    assert side == 250 and top == 0 and 0 <= left <= 150


def test_square_around_without_face_uses_center_fraction():
    assert square_around((250, 250), None) == (50, 50, 150)


def test_nearest_to_center_prefers_central_detection():
    faces_found = [(0, 0, 60, 60), (95, 95, 60, 60), (180, 10, 60, 60)]
    assert nearest_to_center(faces_found, (250, 250)) == (95, 95, 60, 60)
    assert nearest_to_center([], (250, 250)) is None


# ---------- Đọc ảnh và cắt ----------

def test_read_image_rgb_handles_unicode_path_and_channel_order(tmp_path):
    bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    bgr[..., 2] = 255  # đỏ trong hệ BGR của OpenCV
    path = _write_image(tmp_path / "ảnh đỏ.png", bgr)
    rgb = read_image_rgb(path)
    assert rgb.shape == (10, 10, 3)
    assert tuple(rgb[0, 0]) == (255, 0, 0)


def test_read_image_rgb_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_image_rgb(tmp_path / "khong_co.jpg")
    broken = tmp_path / "hong.jpg"
    broken.write_bytes(b"khong phai anh")
    with pytest.raises(UsageError):
        read_image_rgb(broken)


def test_crop_face_falls_back_to_center_when_no_face():
    image = np.full((200, 300, 3), 128, dtype=np.uint8)
    crop = crop_face(image)
    assert not crop.detected
    assert crop.box == (90, 40, 120)
    assert crop.image.shape == (120, 120, 3)


def test_crop_face_uses_detection(monkeypatch):
    image = np.zeros((250, 250, 3), dtype=np.uint8)
    monkeypatch.setattr(faces, "detect_face", lambda _: (70, 70, 110, 110))
    crop = crop_face(image)
    assert crop.detected and crop.box == (48, 48, 154)


# ---------- Chọn cách tiền xử lý theo lần chạy ----------

def test_preprocess_for_model_reads_run_config(tmp_path):
    model_path = tmp_path / "model.keras"
    assert preprocess_for_model(model_path) == "center"  # không có config.json
    TrainConfig(preprocess="face").save(tmp_path / "config.json")
    assert preprocess_for_model(model_path) == "face"
    assert preprocess_for_model(model_path, "center") == "center"  # chỉ định rõ thì ưu tiên
    with pytest.raises(UsageError):
        preprocess_for_model(model_path, "xoay")


def test_old_run_config_without_preprocess_defaults_to_center(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"mode": "lfw", "seed": 42}), encoding="utf-8")
    assert TrainConfig.load(tmp_path / "config.json").preprocess == "center"
    assert preprocess_for_model(tmp_path / "model.keras") == "center"


# ---------- So sánh hai lần chạy ----------

def _write_scores(run_dir, pairs, labels, scores, preprocess):
    run_dir.mkdir(parents=True)
    TrainConfig(preprocess=preprocess).save(run_dir / "config.json")
    with open(run_dir / "test_scores.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["anchor", "candidate", "label", "score"])
        for (anchor, candidate), label, score in zip(pairs, labels, scores):
            writer.writerow([anchor, candidate, label, f"{score:.6f}"])


def test_compare_runs_reports_improvement(tmp_path):
    from siamese_face.compare import compare_runs, format_comparison

    rng = np.random.default_rng(0)
    labels = np.r_[np.ones(60, dtype=int), np.zeros(60, dtype=int)]
    pairs = [(f"data/negative/P{i % 15}_0001.jpg", f"data/negative/Q{i}_0002.jpg") for i in range(120)]
    noisy = rng.random(120)
    better = np.clip(0.6 * labels + 0.4 * rng.random(120), 0, 1)
    _write_scores(tmp_path / "a", pairs, labels, noisy, "center")
    # Đường dẫn khác thư mục gốc nhưng cùng tên file vẫn được coi là cùng một cặp.
    _write_scores(tmp_path / "b", [(f"/x/{a}", f"/x/{c}") for a, c in pairs], labels, better, "face")

    report = compare_runs(tmp_path / "a", tmp_path / "b", n_boot=200)
    auc = report["roc_auc"]
    assert auc["delta"] > 0 and auc["ci_low"] > 0
    assert auc["n_clusters"] == 15
    assert report["preprocess_b"] == "face"
    assert "roc_auc" in format_comparison(report)


def test_compare_runs_rejects_different_test_sets(tmp_path):
    from siamese_face.compare import compare_runs

    labels = [1, 0, 1, 0]
    _write_scores(tmp_path / "a", [("a1", "b1"), ("a2", "b2"), ("a3", "b3"), ("a4", "b4")], labels, [0.9, 0.1, 0.8, 0.2], "center")
    _write_scores(tmp_path / "b", [("a1", "b1"), ("a2", "b2"), ("a3", "b3"), ("a9", "b9")], labels, [0.9, 0.1, 0.8, 0.2], "face")
    with pytest.raises(UsageError):
        compare_runs(tmp_path / "a", tmp_path / "b", n_boot=10)


# ---------- Phần cần TensorFlow: tiền xử lý face và field test ----------

def test_load_images_face_mode(tmp_path):
    pytest.importorskip("tensorflow")
    from siamese_face.pipeline import load_images, preprocess_image

    paths = _noise_images(tmp_path, ["x_1.jpg", "x_2.jpg", "x_3.jpg"])
    images = load_images(paths, (100, 100), preprocess="face")
    assert tuple(images.shape) == (3, 100, 100, 3)
    assert 0.0 <= float(np.min(images)) and float(np.max(images)) <= 1.0
    single, detected = preprocess_image(paths[0], (100, 100), "face")
    np.testing.assert_allclose(single, images[0].numpy(), atol=1e-6)
    assert detected is False  # ảnh nhiễu không có khuôn mặt
    assert preprocess_image(paths[0], (100, 100), "center")[1] is None


def test_field_test_end_to_end(tmp_path):
    pytest.importorskip("tensorflow")
    from siamese_face.data import PairSet, PairSplits
    from siamese_face.field_test import field_test, format_field_test
    from siamese_face.model import build_siamese_model

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model_path = run_dir / "model.keras"
    build_siamese_model((100, 100, 3), embedding_dim=8).save(str(model_path))
    TrainConfig(preprocess="face").save(run_dir / "config.json")
    seen = PairSet(["Person0_0001.jpg"], ["Person1_0001.jpg"], [0])
    PairSplits(train=seen, val=PairSet(), test=PairSet()).save(run_dir / "splits.json")

    references = _noise_images(tmp_path / "refs", ["r_1.jpg", "r_2.jpg"], seed=1)
    probes = _noise_images(tmp_path / "probes", ["p_1.jpg", "p_2.jpg", "p_3.jpg"], seed=2)
    _noise_images(tmp_path / "strangers", [f"Person{i}_0001.jpg" for i in range(6)], seed=3)

    report = field_test(
        model_path,
        [str(tmp_path / "refs")],
        probes,
        tmp_path / "strangers",
        num_impostors=10,
        detection_thresholds=(0.0, 0.5, 1.01),
    )
    assert report["preprocess"] == "face"  # đọc từ config.json của lần chạy
    assert (report["n_references"], report["n_probes"]) == (len(references), 3)
    assert report["n_impostors"] == 4  # Person0 và Person1 đã có trong tập train
    assert report["excluded_identities"] == 2
    for level in ("pairs", "images"):
        low, high = report[level]["roc_auc_ci95"]
        assert 0.0 <= low <= high <= 1.0
    first, _, last = report["decisions"]
    assert first["tar"] == first["far"] == 1.0  # τ_d = 0: chấp nhận tất cả
    assert last["tar"] == last["far"] == 0.0  # τ_d > 1: từ chối tất cả
    assert "τ_d" in format_field_test(report)

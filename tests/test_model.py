"""Kiểm thử phần dùng TensorFlow: mô hình, tiền xử lý, suy luận (tự bỏ qua nếu chưa cài TF)."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from siamese_face.data import PairSet  # noqa: E402
from siamese_face.model import (  # noqa: E402
    L1Dist,
    build_embedding,
    build_siamese_model,
    load_siamese_model,
    model_image_size,
)
from siamese_face.pipeline import load_image, make_dataset  # noqa: E402
from siamese_face.verify import VerificationResult, resolve_references, verify_image  # noqa: E402


def _write_png(path, array) -> str:
    tf.io.write_file(str(path), tf.io.encode_png(np.asarray(array, dtype=np.uint8)))
    return str(path)


def _random_images(n, seed=0):
    return np.random.default_rng(seed).random((n, 100, 100, 3), dtype=np.float32)


# ---------- Mô hình ----------

def test_l1dist_is_absolute_difference():
    a = tf.constant([[1.0, -2.0, 3.0]])
    b = tf.constant([[3.0, 1.0, 3.0]])
    np.testing.assert_allclose(L1Dist()(a, b).numpy(), [[2.0, 3.0, 0.0]])
    np.testing.assert_allclose(L1Dist()([b, a]).numpy(), [[2.0, 3.0, 0.0]])


def test_default_architecture_parameter_count():
    # 38.960.448 tham số mạng nhúng + 4.097 tham số lớp phân loại (xem docs/phuong_phap.md)
    assert build_embedding().count_params() == 38_960_448
    assert build_siamese_model().count_params() == 38_964_545
    assert tuple(build_embedding(embedding_dim=8).outputs[0].shape) == (None, 8)


def test_rejects_too_small_images():
    with pytest.raises(ValueError):
        build_embedding((57, 57, 3))


def test_outputs_are_symmetric_probabilities():
    model = build_siamese_model(embedding_dim=8)
    x1, x2 = _random_images(2, seed=1), _random_images(2, seed=2)
    p12 = model([x1, x2], training=False).numpy()
    p21 = model([x2, x1], training=False).numpy()
    assert p12.shape == (2, 1)
    assert np.all((p12 >= 0) & (p12 <= 1))
    np.testing.assert_allclose(p12, p21, rtol=1e-5, atol=1e-6)


def test_save_and_load_roundtrip(tmp_path):
    model = build_siamese_model(embedding_dim=8)
    x1, x2 = _random_images(2, seed=3), _random_images(2, seed=4)
    expected = model([x1, x2], training=False).numpy()
    path = tmp_path / "model.keras"
    model.save(str(path))
    restored = load_siamese_model(path)
    np.testing.assert_allclose(restored([x1, x2], training=False).numpy(), expected, rtol=1e-5, atol=1e-6)
    assert model_image_size(restored) == (100, 100)


def test_load_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_siamese_model(tmp_path / "missing.keras")


# ---------- Tiền xử lý ----------

def test_load_image_center_crops_resizes_and_normalizes(tmp_path):
    image = np.zeros((60, 120, 3), dtype=np.uint8)
    image[:, 60:] = 255  # nửa phải trắng
    out = load_image(_write_png(tmp_path / "x.png", image), (100, 100)).numpy()
    assert out.shape == (100, 100, 3) and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    # Vùng cắt vuông là cột 30..89 => nửa trái đen, nửa phải trắng
    assert out[:, :40].max() < 0.05 and out[:, 60:].min() > 0.95


def test_load_image_detects_format_from_content(tmp_path):
    # Ảnh PNG mang đuôi .jpg (như anchor của phiên bản cũ) vẫn đọc được
    path = _write_png(tmp_path / "png_named.jpg", np.full((10, 10, 3), 128))
    assert tuple(load_image(path).shape) == (100, 100, 3)


def test_make_dataset_batches_pairs(tmp_path):
    paths = [_write_png(tmp_path / f"{i}.png", np.full((30, 30, 3), i * 50)) for i in range(4)]
    pairs = PairSet(anchors=paths[:2], candidates=paths[2:], labels=[1, 0])
    (anchors, candidates), labels = next(iter(make_dataset(pairs, batch_size=2)))
    assert tuple(anchors.shape) == (2, 100, 100, 3)
    assert tuple(candidates.shape) == (2, 100, 100, 3)
    np.testing.assert_array_equal(labels.numpy(), [[1.0], [0.0]])


def test_make_dataset_reuses_images_and_keeps_order(tmp_path):
    a = _write_png(tmp_path / "a.png", np.zeros((30, 30, 3)))
    b = _write_png(tmp_path / "b.png", np.full((30, 30, 3), 255))
    pairs = PairSet(anchors=[a, a, b], candidates=[b, a, b], labels=[0, 1, 1])
    batches = list(make_dataset(pairs, batch_size=3))
    (first, second), labels = batches[0]
    np.testing.assert_allclose(first.numpy().mean(axis=(1, 2, 3)), [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(second.numpy().mean(axis=(1, 2, 3)), [1.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_array_equal(labels.numpy().ravel(), [0.0, 1.0, 1.0])


def test_make_dataset_rejects_empty_pairs():
    with pytest.raises(ValueError):
        make_dataset(PairSet())


# ---------- Suy luận ----------

def test_verification_rule():
    result = VerificationResult(["a", "b", "c"], [0.9, 0.2, 0.7], 0.5, 0.6)
    assert result.detections == 2
    assert result.ratio == pytest.approx(2 / 3)
    assert result.verified
    result.verification_threshold = 0.7
    assert not result.verified


def test_single_reference_reduces_to_score_threshold():
    assert VerificationResult(["a"], [0.51]).verified
    assert not VerificationResult(["a"], [0.49]).verified


def test_verify_image_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    paths = [_write_png(tmp_path / f"{i}.png", rng.integers(0, 256, (80, 80, 3))) for i in range(4)]
    references = resolve_references([tmp_path])
    assert len(references) == 4
    model = build_siamese_model(embedding_dim=8)
    result = verify_image(model, paths[0], references[1:], detection_threshold=0.0, verification_threshold=1.0)
    assert len(result.scores) == 3
    assert all(0.0 <= score <= 1.0 for score in result.scores)
    assert result.verified  # ngưỡng 0 => mọi ảnh tham chiếu đều "khớp"

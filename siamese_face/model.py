"""Kiến trúc mạng Siamese cho xác minh khuôn mặt (Koch và cộng sự, 2015).

    ảnh x1 ─► f_θ ─┐
                   ├─► L1Dist: |f_θ(x1) − f_θ(x2)| ─► Dense(1, sigmoid) ─► p(cùng người)
    ảnh x2 ─► f_θ ─┘          (hai nhánh dùng chung trọng số θ)
"""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from .config import IMAGE_SIZE, UsageError

layers = tf.keras.layers

DEFAULT_INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
DEFAULT_EMBEDDING_DIM = 4096
# Cạnh ảnh nhỏ nhất để 4 lớp tích chập "valid" và 3 lớp pooling còn kích thước dương.
MIN_IMAGE_SIZE = 58


@tf.keras.utils.register_keras_serializable(package="siamese_face")
class L1Dist(layers.Layer):
    """Khoảng cách L1 theo từng chiều giữa hai vector nhúng: ``|e1 − e2|``.

    Gọi được theo hai cách: ``L1Dist()(e1, e2)`` hoặc ``L1Dist()([e1, e2])``. Tên tham số
    giữ nguyên như phiên bản cũ để vẫn nạp được mô hình ``siamesemodel.h5`` đã huấn luyện.
    """

    def call(self, input_embedding, validation_embedding=None):
        if validation_embedding is None:
            input_embedding, validation_embedding = input_embedding
        return tf.math.abs(_unwrap(input_embedding) - _unwrap(validation_embedding))


def _unwrap(tensor):
    """Keras 3 trả về list một phần tử cho mô hình khai báo ``outputs=[...]`` (mô hình cũ)."""
    while isinstance(tensor, (list, tuple)) and len(tensor) == 1:
        tensor = tensor[0]
    return tensor


def build_embedding(
    input_shape=DEFAULT_INPUT_SHAPE, embedding_dim: int = DEFAULT_EMBEDDING_DIM
) -> tf.keras.Model:
    """Mạng nhúng f_θ: ảnh RGB -> vector đặc trưng ``embedding_dim`` chiều, giá trị trong (0, 1)."""
    if min(input_shape[:2]) < MIN_IMAGE_SIZE:
        raise UsageError(
            f"Ảnh đầu vào phải có cạnh ≥ {MIN_IMAGE_SIZE} pixel, nhận được {tuple(input_shape)}"
        )
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = layers.Conv2D(64, (10, 10), activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="same", name="pool1")(x)
    x = layers.Conv2D(128, (7, 7), activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="same", name="pool2")(x)
    x = layers.Conv2D(128, (4, 4), activation="relu", name="conv3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="same", name="pool3")(x)
    x = layers.Conv2D(256, (4, 4), activation="relu", name="conv4")(x)
    x = layers.Flatten(name="flatten")(x)
    outputs = layers.Dense(embedding_dim, activation="sigmoid", name="embedding_vector")(x)
    # outputs là một tensor (không phải list) để tương thích cả Keras 2 lẫn Keras 3.
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="embedding")


def build_siamese_model(
    input_shape=DEFAULT_INPUT_SHAPE, embedding_dim: int = DEFAULT_EMBEDDING_DIM
) -> tf.keras.Model:
    """Mạng Siamese hoàn chỉnh: (anchor, candidate) -> xác suất hai ảnh là cùng một người."""
    embedding = build_embedding(input_shape, embedding_dim)
    anchor = layers.Input(shape=input_shape, name="anchor")
    candidate = layers.Input(shape=input_shape, name="candidate")
    distance = L1Dist(name="l1_distance")([embedding(anchor), embedding(candidate)])
    similarity = layers.Dense(1, activation="sigmoid", name="similarity")(distance)
    return tf.keras.Model(inputs=[anchor, candidate], outputs=similarity, name="siamese_network")


def load_siamese_model(path: str | Path) -> tf.keras.Model:
    """Nạp mô hình đã lưu: định dạng ``.keras`` (mới) hoặc ``.h5`` (phiên bản cũ)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy mô hình: {path}")
    return tf.keras.models.load_model(str(path), custom_objects={"L1Dist": L1Dist}, compile=False)


def model_image_size(model: tf.keras.Model) -> tuple[int, int]:
    """(cao, rộng) của ảnh đầu vào mà mô hình mong đợi."""
    shape = tuple(model.inputs[0].shape)
    return int(shape[1]), int(shape[2])

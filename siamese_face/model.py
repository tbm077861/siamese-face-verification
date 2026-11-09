"""Kiến trúc mạng Siamese cho xác minh khuôn mặt (Koch và cộng sự, 2015).

    ảnh x1 ─► f_θ ─┐
                   ├─► L1Dist: |f_θ(x1) − f_θ(x2)| ─► Dense(1, sigmoid) ─► p(cùng người)
    ảnh x2 ─► f_θ ─┘          (hai nhánh dùng chung trọng số θ)

Chỉ dùng API Keras 3 (``keras``, ``keras.ops``) nên chạy được trên mọi backend:
TensorFlow, PyTorch hoặc JAX - chọn bằng biến môi trường ``KERAS_BACKEND`` hoặc
tuỳ chọn ``--backend`` của CLI. File ``.keras`` lưu ra dùng chung cho mọi backend.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import keras
from keras import layers, ops

from .config import IMAGE_SIZE, UsageError

DEFAULT_INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
DEFAULT_EMBEDDING_DIM = 4096
# Cạnh ảnh nhỏ nhất để 4 lớp tích chập "valid" và 3 lớp pooling còn kích thước dương.
MIN_IMAGE_SIZE = 58


@keras.saving.register_keras_serializable(package="siamese_face")
class L1Dist(layers.Layer):
    """Khoảng cách L1 theo từng chiều giữa hai vector nhúng: ``|e1 − e2|``.

    Gọi được theo hai cách: ``L1Dist()(e1, e2)`` hoặc ``L1Dist()([e1, e2])``. Tên tham số
    giữ nguyên như phiên bản cũ để vẫn nạp được mô hình ``siamesemodel.h5`` đã huấn luyện.
    """

    def call(self, input_embedding, validation_embedding=None):
        if validation_embedding is None:
            input_embedding, validation_embedding = input_embedding
        return ops.abs(_unwrap(input_embedding) - _unwrap(validation_embedding))


def _unwrap(tensor):
    """Keras 3 trả về list một phần tử cho mô hình khai báo ``outputs=[...]`` (mô hình cũ)."""
    while isinstance(tensor, (list, tuple)) and len(tensor) == 1:
        tensor = tensor[0]
    return tensor


def build_embedding(
    input_shape=DEFAULT_INPUT_SHAPE, embedding_dim: int = DEFAULT_EMBEDDING_DIM
) -> keras.Model:
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
    return keras.Model(inputs=inputs, outputs=outputs, name="embedding")


def build_siamese_model(
    input_shape=DEFAULT_INPUT_SHAPE, embedding_dim: int = DEFAULT_EMBEDDING_DIM
) -> keras.Model:
    """Mạng Siamese hoàn chỉnh: (anchor, candidate) -> xác suất hai ảnh là cùng một người."""
    embedding = build_embedding(input_shape, embedding_dim)
    anchor = layers.Input(shape=input_shape, name="anchor")
    candidate = layers.Input(shape=input_shape, name="candidate")
    distance = L1Dist(name="l1_distance")([embedding(anchor), embedding(candidate)])
    similarity = layers.Dense(1, activation="sigmoid", name="similarity")(distance)
    return keras.Model(inputs=[anchor, candidate], outputs=similarity, name="siamese_network")


def load_siamese_model(path: str | Path) -> keras.Model:
    """Nạp mô hình đã lưu: định dạng ``.keras`` (mới) hoặc ``.h5`` (phiên bản cũ).

    File ``.keras`` do một bản Keras *mới hơn* lưu có thể chứa tham số cấu hình mà bản
    đang cài chưa biết (VD: Keras 3.15 thêm ``input_axes`` cho ``GlorotUniform``, Keras
    3.12 không đọc được). Khi đó, nếu file là mô hình của dự án, kiến trúc được dựng lại
    bằng :func:`build_siamese_model` và chỉ nạp trọng số - phần không phụ thuộc phiên bản.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy mô hình: {path}")
    try:
        return keras.models.load_model(str(path), custom_objects={"L1Dist": L1Dist}, compile=False)
    except (TypeError, ValueError) as error:
        if path.suffix != ".keras":
            raise
        try:
            input_shape, embedding_dim = architecture_from_archive(path)
        except (KeyError, ValueError, zipfile.BadZipFile):
            raise error from None
        print(
            f"Lưu ý: {path.name} được lưu bằng phiên bản Keras khác ({error.__class__.__name__});"
            " dựng lại kiến trúc và chỉ nạp trọng số."
        )
        model = build_siamese_model(input_shape, embedding_dim)
        model.load_weights(str(path))
        return model


def architecture_from_archive(path: str | Path) -> tuple[tuple[int, int, int], int]:
    """Đọc (kích thước ảnh đầu vào, số chiều vector nhúng) từ ``config.json`` của file ``.keras``."""
    with zipfile.ZipFile(path) as archive:
        config = json.loads(archive.read("config.json"))
    found: dict[str, object] = {}

    def visit(node) -> None:
        if isinstance(node, dict):
            layer = node.get("config") if isinstance(node.get("config"), dict) else {}
            if node.get("class_name") == "InputLayer" and layer.get("name") == "anchor":
                found["shape"] = layer.get("batch_shape") or layer.get("batch_input_shape")
            if node.get("class_name") == "Dense" and layer.get("name") == "embedding_vector":
                found["units"] = layer.get("units")
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(config)
    if not found.get("shape") or not found.get("units"):
        raise ValueError(f"{path} không phải mô hình Siamese của dự án")
    shape = tuple(int(dim) for dim in found["shape"][1:])
    return shape, int(found["units"])


def model_image_size(model: keras.Model) -> tuple[int, int]:
    """(cao, rộng) của ảnh đầu vào mà mô hình mong đợi."""
    shape = tuple(model.inputs[0].shape)
    return int(shape[1]), int(shape[2])

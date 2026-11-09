"""Đường ống dữ liệu TensorFlow: đọc ảnh, tiền xử lý, tăng cường và đóng lô."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from .config import IMAGE_SIZE, UsageError
from .data import PairSet

DEFAULT_SIZE = (IMAGE_SIZE, IMAGE_SIZE)


def load_image(path, image_size=DEFAULT_SIZE) -> tf.Tensor:
    """Đọc ảnh và chuẩn hoá thành tensor ``(cao, rộng, 3)`` có giá trị trong [0, 1].

    Các bước: giải mã (JPEG/PNG/BMP, nhận dạng theo nội dung file) -> cắt vùng vuông
    ở giữa -> resize song tuyến tính -> chia 255. Hàm này dùng chung cho huấn luyện,
    đánh giá và suy luận để tránh lệch tiền xử lý giữa các giai đoạn
    (training/serving skew).
    """
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = center_crop_square(image)
    image = tf.image.resize(image, image_size)
    return image / 255.0


def center_crop_square(image: tf.Tensor) -> tf.Tensor:
    """Cắt vùng vuông lớn nhất ở giữa ảnh để resize không làm méo tỉ lệ khuôn mặt."""
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    side = tf.minimum(height, width)
    return tf.image.crop_to_bounding_box(
        image, (height - side) // 2, (width - side) // 2, side, side
    )


def augment_image(image: tf.Tensor) -> tf.Tensor:
    """Tăng cường ngẫu nhiên (chỉ khi huấn luyện): lật ngang, độ sáng, tương phản, bão hoà."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    return tf.clip_by_value(image, 0.0, 1.0)


def make_dataset(
    pairs: PairSet,
    image_size=DEFAULT_SIZE,
    batch_size: int = 16,
    shuffle: bool = False,
    augment: bool = False,
    seed: int | None = None,
    cache: bool = True,
) -> tf.data.Dataset:
    """``tf.data.Dataset`` sinh các lô ``((ảnh_anchor, ảnh_candidate), nhãn)``.

    Khi ``shuffle=False`` thứ tự phần tử giữ nguyên như ``pairs`` (cần cho đánh giá).
    Ảnh đã giải mã được cache *trước* bước tăng cường để mỗi epoch nhận biến thể mới.
    """
    if len(pairs) == 0:
        raise UsageError("Không thể tạo dataset từ tập cặp rỗng.")
    labels = np.asarray(pairs.labels, dtype=np.float32).reshape(-1, 1)
    dataset = tf.data.Dataset.from_tensor_slices((pairs.anchors, pairs.candidates, labels))

    def load_pair(anchor_path, candidate_path, label):
        images = (load_image(anchor_path, image_size), load_image(candidate_path, image_size))
        return images, label

    dataset = dataset.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(len(pairs), seed=seed, reshuffle_each_iteration=True)
    if augment:
        dataset = dataset.map(
            lambda images, label: ((augment_image(images[0]), augment_image(images[1])), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

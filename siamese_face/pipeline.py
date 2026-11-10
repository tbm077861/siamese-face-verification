"""Đường ống dữ liệu TensorFlow: đọc ảnh, tiền xử lý, tăng cường và đóng lô.

``tf.data`` chỉ chạy trên CPU và dùng được với mọi backend Keras; khi backend là
PyTorch/JAX, GPU được ẩn khỏi TensorFlow để nhường toàn bộ cho backend đó.
"""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf

from .config import IMAGE_SIZE, UsageError, check_preprocess
from .data import PairSet

DEFAULT_SIZE = (IMAGE_SIZE, IMAGE_SIZE)

if keras.backend.backend() != "tensorflow":
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:  # TensorFlow đã khởi tạo thiết bị từ trước
        pass


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


def load_face_image(path, image_size=DEFAULT_SIZE) -> tuple[tf.Tensor, bool]:
    """Như :func:`load_image` nhưng cắt vùng vuông quanh khuôn mặt thay vì vùng giữa ảnh.

    Trả về thêm cờ cho biết có phát hiện được khuôn mặt không (không thấy thì cắt vùng giữa
    theo ``faces.FALLBACK_FRACTION``). Bước resize và chuẩn hoá giống hệt ``load_image``.
    """
    from .faces import crop_face, read_image_rgb

    crop = crop_face(read_image_rgb(path))
    image = tf.image.resize(crop.image, image_size)
    return image / 255.0, crop.detected


def preprocess_image(path, image_size=DEFAULT_SIZE, preprocess: str = "center") -> tuple[np.ndarray, bool | None]:
    """Một ảnh -> mảng NumPy ``(cao, rộng, 3)`` dùng khi suy luận.

    Phần tử thứ hai là cờ phát hiện khuôn mặt (chế độ ``"face"``) hoặc ``None`` (``"center"``).
    """
    if check_preprocess(preprocess) == "face":
        image, detected = load_face_image(str(path), image_size)
        return image.numpy(), detected
    return load_image(str(path), image_size).numpy(), None


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


def load_images(
    paths, image_size=DEFAULT_SIZE, batch_size: int = 256, preprocess: str = "center"
) -> tf.Tensor:
    """Nạp nhiều ảnh thành một tensor ``(n, cao, rộng, 3)`` nằm trên RAM CPU."""
    paths = [str(path) for path in paths]
    if check_preprocess(preprocess) == "face":
        images = np.empty((len(paths), *image_size, 3), dtype=np.float32)
        detected = 0
        for i, path in enumerate(paths):
            image, found = load_face_image(path, image_size)
            images[i], detected = image.numpy(), detected + found
        print(
            f"Phát hiện khuôn mặt ở {detected}/{len(paths)} ảnh"
            f" ({len(paths) - detected} ảnh còn lại dùng vùng giữa ảnh)."
        )
        with tf.device("/CPU:0"):
            return tf.constant(images)
    dataset = (
        tf.data.Dataset.from_tensor_slices(list(paths))
        .map(lambda path: load_image(path, image_size), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
    )
    with tf.device("/CPU:0"):  # tf.data đọc tensor từ CPU; tránh đặt khối ảnh lên GPU
        return tf.concat(list(dataset), axis=0)


def make_dataset(
    pairs: PairSet,
    image_size=DEFAULT_SIZE,
    batch_size: int = 16,
    shuffle: bool = False,
    augment: bool = False,
    seed: int | None = None,
    preprocess: str = "center",
) -> tf.data.Dataset:
    """``tf.data.Dataset`` sinh các lô ``((ảnh_anchor, ảnh_candidate), nhãn)``.

    Mỗi ảnh *duy nhất* chỉ được giải mã một lần và giữ trong RAM; các cặp chỉ lưu chỉ
    số tới ảnh. Với LFW (~20 nghìn cặp trên ~13 nghìn ảnh) cách này tốn ~1 GB thay vì
    vài GB nếu cache theo cặp. Khi ``shuffle=False`` thứ tự giữ nguyên như ``pairs``
    (cần cho đánh giá); tăng cường dữ liệu được áp dụng lại ở mỗi epoch.
    """
    if len(pairs) == 0:
        raise UsageError("Không thể tạo dataset từ tập cặp rỗng.")
    paths = sorted(set(pairs.anchors) | set(pairs.candidates))
    position = {path: i for i, path in enumerate(paths)}
    images = load_images(paths, image_size, preprocess=preprocess)

    anchor_index = np.asarray([position[p] for p in pairs.anchors], dtype=np.int32)
    candidate_index = np.asarray([position[p] for p in pairs.candidates], dtype=np.int32)
    labels = np.asarray(pairs.labels, dtype=np.float32).reshape(-1, 1)
    dataset = tf.data.Dataset.from_tensor_slices((anchor_index, candidate_index, labels))
    if shuffle:
        dataset = dataset.shuffle(len(pairs), seed=seed, reshuffle_each_iteration=True)

    def gather_pair(i, j, label):
        first, second = tf.gather(images, i), tf.gather(images, j)
        if augment:
            first, second = augment_image(first), augment_image(second)
        return (first, second), label

    dataset = dataset.map(gather_pair, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

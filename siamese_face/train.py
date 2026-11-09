"""Huấn luyện mạng Siamese: Adam + binary cross-entropy, dừng sớm theo val_loss."""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import keras

from . import plots
from .config import (
    CONFIG_FILENAME,
    MODEL_FILENAME,
    SPLITS_FILENAME,
    TrainConfig,
    UsageError,
    write_json,
)
from .data import build_experiment_splits
from .model import build_siamese_model
from .pipeline import make_dataset


class BestWeights(keras.callbacks.Callback):
    """Ghi nhớ trọng số tốt nhất theo ``monitor`` và khôi phục khi kết thúc huấn luyện.

    Dừng sớm sau ``patience`` epoch không cải thiện (``None`` = không dừng sớm).
    Tự cài đặt thay vì dùng EarlyStopping/ModelCheckpoint để hành vi như nhau trên
    Keras 2 và Keras 3 (luôn khôi phục trọng số tốt nhất, kể cả khi chạy hết số epoch).
    """

    def __init__(self, monitor: str = "val_loss", patience: int | None = None):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.best = float("inf")
        self.best_epoch: int | None = None
        self._best_weights = None
        self._wait = 0

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get(self.monitor)
        if value is None:
            return
        if value < self.best:
            self.best, self.best_epoch, self._wait = float(value), epoch + 1, 0
            self._best_weights = self.model.get_weights()
            return
        self._wait += 1
        if self.patience is not None and self._wait >= self.patience:
            print(f"Dừng sớm: {self.monitor} không cải thiện sau {self.patience} epoch.")
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self._best_weights is not None:
            self.model.set_weights(self._best_weights)


def export_model(model: keras.Model, path: str | Path) -> Path:
    """Lưu kiến trúc + trọng số, *không* kèm trạng thái bộ tối ưu (mô hình chưa compile)."""
    exported = keras.models.clone_model(model)
    exported.set_weights(model.get_weights())
    exported.save(str(path))
    return Path(path)


def create_run_dir(output_dir: str | Path) -> Path:
    """Tạo ``output_dir/<YYYYmmdd-HHMMSS>`` (thêm hậu tố nếu đã tồn tại)."""
    root = Path(output_dir)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir, suffix = root / stamp, 1
    while run_dir.exists():
        run_dir, suffix = root / f"{stamp}-{suffix}", suffix + 1
    run_dir.mkdir(parents=True)
    return run_dir


def train(config: TrainConfig) -> Path:
    """Huấn luyện theo ``config``; trả về thư mục chứa mô hình và nhật ký của lần chạy."""
    keras.utils.set_random_seed(config.seed)
    splits = build_experiment_splits(
        config.mode,
        config.data_dir,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
        max_images=config.max_images,
        max_pairs_per_identity=config.max_pairs_per_identity,
    )
    if len(splits.train) == 0:
        raise UsageError("Tập train rỗng - hãy bổ sung dữ liệu (xem data/README.md).")
    print(f"Chế độ: {config.mode} | backend Keras: {keras.backend.backend()}")
    for name, pairs in splits.items():
        print(f"{name:>5}: {len(pairs):5d} cặp ({pairs.num_positive} cặp cùng người)")

    image_size = (config.image_size, config.image_size)
    model = build_siamese_model(input_shape=(*image_size, 3), embedding_dim=config.embedding_dim)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    run_dir = create_run_dir(config.output_dir)
    print(f"Thư mục kết quả: {run_dir}")
    config.save(run_dir / CONFIG_FILENAME)
    splits.save(run_dir / SPLITS_FILENAME)

    train_ds = make_dataset(
        splits.train,
        image_size=image_size,
        batch_size=config.batch_size,
        shuffle=True,
        augment=config.augment,
        seed=config.seed,
    )
    val_ds = (
        make_dataset(splits.val, image_size=image_size, batch_size=config.batch_size)
        if len(splits.val)
        else None
    )
    tracker = BestWeights(
        monitor="val_loss" if val_ds is not None else "loss", patience=config.patience
    )

    started = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=[tracker],
        shuffle=False,  # đã xáo trộn trong tf.data (mỗi epoch một thứ tự mới)
        verbose=2,
    )
    elapsed = time.perf_counter() - started

    # Lưu bản không kèm trạng thái Adam: nhỏ hơn ~3 lần, đủ cho đánh giá và suy luận.
    export_model(model, run_dir / MODEL_FILENAME)
    _write_history(history.history, run_dir / "history.csv")
    write_json(
        {
            "monitor": tracker.monitor,
            "best_value": tracker.best,
            "best_epoch": tracker.best_epoch,
            "epochs_run": len(history.history.get("loss", [])),
            "training_seconds": round(elapsed, 1),
            "keras_backend": keras.backend.backend(),
            "keras_version": keras.__version__,
        },
        run_dir / "training_summary.json",
    )
    plots.plot_training_history(history.history, run_dir / "training_curves.png")
    return run_dir


def _write_history(history: dict[str, list[float]], path: Path) -> None:
    keys = list(history)
    epochs = len(history[keys[0]]) if keys else 0
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", *keys])
        for i in range(epochs):
            writer.writerow([i + 1, *(history[key][i] for key in keys)])

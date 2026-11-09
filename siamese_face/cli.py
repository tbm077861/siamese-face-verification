"""Giao diện dòng lệnh: ``python -m siamese_face <lệnh> [tuỳ chọn]``.

TensorFlow chỉ được import bên trong lệnh cần đến nó, nên ``--help`` và các lệnh
chuẩn bị dữ liệu chạy nhanh.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .data import MODES
from .config import (
    IMAGE_SIZE,
    METRICS_FILENAME,
    NEGATIVE_DIR,
    VERIFICATION_DIR,
    TrainConfig,
    UsageError,
)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # ẩn bớt log C++ của TensorFlow


def build_parser() -> argparse.ArgumentParser:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        prog="python -m siamese_face",
        description="Xác minh khuôn mặt one-shot bằng mạng nơ-ron Siamese.",
    )
    commands = parser.add_subparsers(dest="command", metavar="<lệnh>")
    commands.required = True

    p = commands.add_parser("prepare-negatives", help="chuẩn bị ảnh negative (VD: LFW)")
    p.add_argument("source", help="file .tgz/.tar/.zip, thư mục ảnh hoặc URL")
    p.add_argument("--data-dir", default=defaults.data_dir, help="thư mục dữ liệu (mặc định: %(default)s)")
    p.add_argument("--limit", type=int, default=None, help="số ảnh tối đa được thêm")
    p.set_defaults(handler=_prepare_negatives)

    p = commands.add_parser("collect", help="thu thập ảnh anchor/positive/verification từ webcam")
    p.add_argument("--data-dir", default=defaults.data_dir, help="thư mục dữ liệu (mặc định: %(default)s)")
    p.add_argument("--camera", type=int, default=0, help="chỉ số camera OpenCV (mặc định: %(default)s)")
    p.add_argument("--crop-size", type=int, default=250, help="cạnh vùng cắt vuông giữa khung hình (mặc định: %(default)s)")
    p.set_defaults(handler=_collect)

    p = commands.add_parser("train", help="huấn luyện mạng Siamese")
    _add_mode_arguments(p, defaults)
    p.add_argument("--data-dir", default=defaults.data_dir, help="(mặc định: %(default)s)")
    p.add_argument("--output-dir", default=defaults.output_dir, help="nơi tạo thư mục cho lần chạy (mặc định: %(default)s)")
    p.add_argument("--epochs", type=int, default=defaults.epochs, help="(mặc định: %(default)s)")
    p.add_argument("--batch-size", type=int, default=defaults.batch_size, help="(mặc định: %(default)s)")
    p.add_argument("--learning-rate", type=float, default=defaults.learning_rate, help="(mặc định: %(default)s)")
    p.add_argument("--embedding-dim", type=int, default=defaults.embedding_dim, help="(mặc định: %(default)s)")
    p.add_argument("--image-size", type=int, default=defaults.image_size, help="(mặc định: %(default)s)")
    p.add_argument("--train-ratio", type=float, default=defaults.train_ratio, help="(mặc định: %(default)s)")
    p.add_argument("--val-ratio", type=float, default=defaults.val_ratio, help="(mặc định: %(default)s)")
    p.add_argument("--max-images", type=int, default=defaults.max_images, help="giới hạn số ảnh mỗi loại")
    p.add_argument("--augment", action="store_true", help="bật tăng cường dữ liệu")
    p.add_argument("--patience", type=int, default=defaults.patience, help="dừng sớm sau N epoch không cải thiện; 0 = tắt (mặc định: %(default)s)")
    p.add_argument("--seed", type=int, default=defaults.seed, help="(mặc định: %(default)s)")
    p.add_argument("--no-eval", action="store_true", help="bỏ qua đánh giá trên tập test sau khi huấn luyện")
    p.set_defaults(handler=_train)

    p = commands.add_parser("evaluate", help="đánh giá mô hình trên tập test")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="thư mục một lần chạy (chứa model.keras và splits.json)")
    source.add_argument("--model", help="file mô hình bất kỳ (.keras/.h5); dữ liệu được chia lại từ --data-dir")
    _add_mode_arguments(p, defaults, suffix=" - chỉ dùng với --model")
    p.add_argument("--data-dir", default=defaults.data_dir, help="(chỉ dùng với --model)")
    p.add_argument("--train-ratio", type=float, default=defaults.train_ratio, help="(chỉ dùng với --model)")
    p.add_argument("--val-ratio", type=float, default=defaults.val_ratio, help="(chỉ dùng với --model)")
    p.add_argument("--seed", type=int, default=defaults.seed, help="(chỉ dùng với --model)")
    p.add_argument("--criterion", choices=("f1", "youden", "eer"), default="f1", help="tiêu chí chọn ngưỡng trên tập val (mặc định: %(default)s)")
    p.add_argument("--output-dir", default=None, help="nơi lưu metrics.json và biểu đồ (mặc định: thư mục --run)")
    p.add_argument("--batch-size", type=int, default=32)
    p.set_defaults(handler=_evaluate)

    p = commands.add_parser("verify", help="xác minh một ảnh so với các ảnh tham chiếu")
    p.add_argument("--model", required=True, help="file mô hình (.keras/.h5)")
    p.add_argument("--image", required=True, help="ảnh cần xác minh")
    p.add_argument("--references", nargs="+", default=[f"data/{VERIFICATION_DIR}"], help="thư mục/file ảnh tham chiếu (mặc định: %(default)s)")
    p.add_argument("--detection-threshold", type=float, default=0.5, help="τ_d (mặc định: %(default)s)")
    p.add_argument("--verification-threshold", type=float, default=0.5, help="τ_v (mặc định: %(default)s)")
    p.set_defaults(handler=_verify)

    p = commands.add_parser("summary", help="in kiến trúc và số tham số của mô hình")
    p.add_argument("--model", default=None, help="file mô hình; bỏ trống để dựng kiến trúc mặc định")
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--embedding-dim", type=int, default=defaults.embedding_dim)
    p.set_defaults(handler=_summary)
    return parser


def _add_mode_arguments(parser, defaults: TrainConfig, suffix: str = "") -> None:
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=defaults.mode,
        help="personal: anchor/positive/negative của một người; lfw: cặp cùng/khác người "
        f"từ data/negative (mặc định: %(default)s){suffix}",
    )
    parser.add_argument(
        "--max-pairs-per-identity",
        type=int,
        default=defaults.max_pairs_per_identity,
        help=f"số cặp dương tối đa mỗi người, chế độ lfw (mặc định: %(default)s){suffix}",
    )


def main(argv: list[str] | None = None) -> int:
    _use_utf8_output()
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (FileNotFoundError, UsageError) as error:
        print(f"Lỗi: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Đã dừng theo yêu cầu.", file=sys.stderr)
        return 130


def _use_utf8_output() -> None:
    """In được tiếng Việt kể cả khi stdout bị chuyển hướng ra file trên Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def _prepare_negatives(args) -> int:
    from .negatives import prepare_negatives

    destination = Path(args.data_dir) / NEGATIVE_DIR
    added = prepare_negatives(args.source, destination, limit=args.limit)
    print(f"Đã thêm {added} ảnh vào {destination}")
    return 0


def _collect(args) -> int:
    from .collect import collect_samples

    counts = collect_samples(Path(args.data_dir), camera_index=args.camera, crop_size=args.crop_size)
    print("Đã lưu: " + ", ".join(f"{name}={count}" for name, count in counts.items()))
    return 0


def _train(args) -> int:
    from .train import train

    config = TrainConfig(
        mode=args.mode,
        max_pairs_per_identity=args.max_pairs_per_identity,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_images=args.max_images,
        augment=args.augment,
        patience=args.patience if args.patience and args.patience > 0 else None,
        seed=args.seed,
    )
    run_dir = train(config)
    print(f"\nĐã lưu mô hình và nhật ký huấn luyện vào: {run_dir}")
    if not args.no_eval:
        from .evaluate import evaluate_run, format_report

        try:
            report = evaluate_run(run_dir)
        except UsageError as error:
            print(f"Bỏ qua đánh giá: {error}")
        else:
            print(format_report(report))
    return 0


def _evaluate(args) -> int:
    from .evaluate import evaluate_model, evaluate_run, format_report

    if args.run:
        report = evaluate_run(
            args.run, criterion=args.criterion, batch_size=args.batch_size, output_dir=args.output_dir
        )
    else:
        from .data import build_experiment_splits

        splits = build_experiment_splits(
            args.mode,
            args.data_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            max_pairs_per_identity=args.max_pairs_per_identity,
        )
        report = evaluate_model(
            args.model, splits, criterion=args.criterion, batch_size=args.batch_size, output_dir=args.output_dir
        )
    print(format_report(report))
    destination = args.output_dir or args.run
    if destination:
        print(f"\nChi tiết: {Path(destination) / METRICS_FILENAME}")
    return 0


def _verify(args) -> int:
    from .model import load_siamese_model
    from .verify import resolve_references, verify_image

    references = resolve_references(args.references)
    model = load_siamese_model(args.model)
    result = verify_image(
        model,
        args.image,
        references,
        detection_threshold=args.detection_threshold,
        verification_threshold=args.verification_threshold,
    )
    for path, score in zip(result.references, result.scores):
        mark = "khớp" if score >= result.detection_threshold else "    "
        print(f"  {score:.4f}  {mark}  {path}")
    print(
        f"Khớp {result.detections}/{len(result.scores)} ảnh tham chiếu "
        f"(tỉ lệ {result.ratio:.2f}; cần ≥ {result.verification_threshold:.2f})"
    )
    print("=> XÁC MINH THÀNH CÔNG" if result.verified else "=> KHÔNG KHỚP")
    return 0


def _summary(args) -> int:
    import tensorflow as tf

    from .model import build_siamese_model, load_siamese_model

    if args.model:
        model = load_siamese_model(args.model)
    else:
        model = build_siamese_model((args.image_size, args.image_size, 3), args.embedding_dim)
    model.summary()
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  # mạng nhúng lồng bên trong
            layer.summary()
    print(f"Tổng số tham số: {model.count_params():,}")
    return 0

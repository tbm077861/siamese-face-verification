"""Đánh giá mô hình: chọn ngưỡng trên tập validation, báo cáo trên tập test.

Ngưỡng quyết định chỉ được chọn bằng tập val rồi giữ cố định khi đo trên tập test,
để số liệu báo cáo không bị "nhìn trộm" dữ liệu kiểm thử.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import plots
from .config import METRICS_FILENAME, MODEL_FILENAME, SPLITS_FILENAME, UsageError, write_json
from .data import PairSet, PairSplits
from .metrics import (
    best_threshold,
    classification_metrics,
    equal_error_rate,
    has_both_classes,
    roc_auc,
)


def score_pairs(model, pairs: PairSet, batch_size: int = 32) -> np.ndarray:
    """Điểm tương đồng (xác suất cùng người) của từng cặp, đúng thứ tự trong ``pairs``."""
    from .model import model_image_size
    from .pipeline import make_dataset

    dataset = make_dataset(pairs, image_size=model_image_size(model), batch_size=batch_size)
    batches = [
        np.asarray(model([anchors, candidates], training=False)).reshape(-1)
        for (anchors, candidates), _ in dataset
    ]
    return np.concatenate(batches)


def summarize(labels, scores, threshold: float) -> dict:
    """Độ đo tại ngưỡng cho trước, kèm ROC-AUC và EER (nếu có đủ hai lớp)."""
    labels = np.asarray(labels).reshape(-1)
    report = classification_metrics(labels, scores, threshold)
    report.update(n_pairs=int(labels.size), n_positive=int(labels.sum()))
    if has_both_classes(labels):
        eer, eer_threshold = equal_error_rate(labels, scores)
        report.update(roc_auc=roc_auc(labels, scores), eer=eer, eer_threshold=eer_threshold)
    return report


def evaluate_model(
    model_path: str | Path,
    splits: PairSplits,
    criterion: str = "f1",
    batch_size: int = 32,
    output_dir: str | Path | None = None,
) -> dict:
    """Đánh giá ``model_path`` trên ``splits`` theo giao thức val -> test."""
    from .model import load_siamese_model

    if len(splits.test) == 0:
        raise UsageError("Tập test rỗng - cần thêm dữ liệu hoặc giảm --train-ratio/--val-ratio.")
    model = load_siamese_model(model_path)
    report: dict = {"model": str(model_path), "criterion": criterion}

    if has_both_classes(splits.val.labels):
        val_scores = score_pairs(model, splits.val, batch_size)
        threshold = best_threshold(splits.val.labels, val_scores, criterion)
        report["threshold_source"] = f"tập val, tiêu chí {criterion}"
        report["val"] = summarize(splits.val.labels, val_scores, threshold)
    else:
        threshold = 0.5
        report["threshold_source"] = "mặc định 0.5 (tập val không đủ hai lớp)"
    report["threshold"] = float(threshold)

    labels = np.asarray(splits.test.labels)
    scores = score_pairs(model, splits.test, batch_size)
    report["test"] = summarize(labels, scores, threshold)
    report["test_at_0.5"] = classification_metrics(labels, scores, 0.5)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(report, output_dir / METRICS_FILENAME)
        _write_scores(splits.test, scores, output_dir / "test_scores.csv")
        if has_both_classes(labels):
            plots.plot_roc(labels, scores, output_dir / "roc_curve.png")
        plots.plot_score_distribution(labels, scores, threshold, output_dir / "score_distribution.png")
    return report


def evaluate_run(
    run_dir: str | Path,
    criterion: str = "f1",
    batch_size: int = 32,
    output_dir: str | Path | None = None,
) -> dict:
    """Đánh giá một lần chạy của lệnh ``train`` trên đúng tập test đã lưu trong splits.json."""
    run_dir = Path(run_dir)
    splits_path = run_dir / SPLITS_FILENAME
    if not splits_path.is_file():
        raise FileNotFoundError(f"Không thấy {splits_path} - đây có phải thư mục kết quả của train?")
    return evaluate_model(
        run_dir / MODEL_FILENAME,
        PairSplits.load(splits_path),
        criterion=criterion,
        batch_size=batch_size,
        output_dir=output_dir or run_dir,
    )


def format_report(report: dict) -> str:
    """Bản tóm tắt dễ đọc để in ra màn hình."""
    lines = [f"Ngưỡng quyết định τ = {report['threshold']:.4f} ({report['threshold_source']})"]
    for name in ("val", "test"):
        m = report.get(name)
        if not m:
            continue
        lines.append(f"\nTập {name}: {m['n_pairs']} cặp ({m['n_positive']} cặp cùng người)")
        lines.append(
            f"  Accuracy {m['accuracy']:.4f} | Precision {m['precision']:.4f}"
            f" | Recall {m['recall']:.4f} | F1 {m['f1']:.4f}"
        )
        line = f"  FAR {m['far']:.4f} | FRR {m['frr']:.4f}"
        if "roc_auc" in m:
            line += f" | ROC-AUC {m['roc_auc']:.4f} | EER {m['eer']:.4f}"
        lines.append(line)
    return "\n".join(lines)


def _write_scores(pairs: PairSet, scores, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["anchor", "candidate", "label", "score"])
        for anchor, candidate, label, score in zip(
            pairs.anchors, pairs.candidates, pairs.labels, scores
        ):
            writer.writerow([anchor, candidate, label, f"{score:.6f}"])

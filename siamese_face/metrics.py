"""Độ đo đánh giá cho bài toán xác minh (phân loại nhị phân trên cặp ảnh).

Cài đặt bằng NumPy thuần để minh bạch công thức và kiểm thử độc lập với TensorFlow.
Quy ước: nhãn 1 = cùng người (lớp dương); dự đoán dương khi ``score >= threshold``.
"""

from __future__ import annotations

import numpy as np


def _validate(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.shape != y_score.shape:
        raise ValueError(f"y_true ({y_true.size}) và y_score ({y_score.size}) phải cùng số phần tử")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("Nhãn phải là 0 hoặc 1")
    return y_true.astype(np.int64), y_score


def has_both_classes(y_true) -> bool:
    labels = np.asarray(y_true).reshape(-1)
    return bool(np.any(labels == 1) and np.any(labels == 0))


def confusion_counts(y_true, y_score, threshold: float = 0.5) -> dict[str, int]:
    """Ma trận nhầm lẫn: TP, FP, TN, FN tại ``threshold``."""
    y_true, y_score = _validate(y_true, y_score)
    predicted = y_score >= threshold
    actual = y_true == 1
    return {
        "tp": int(np.sum(predicted & actual)),
        "fp": int(np.sum(predicted & ~actual)),
        "tn": int(np.sum(~predicted & ~actual)),
        "fn": int(np.sum(~predicted & actual)),
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / denominator if denominator else 0.0


def classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """Accuracy, Precision, Recall, F1, FAR, FRR (kèm TP/FP/TN/FN) tại ``threshold``."""
    counts = confusion_counts(y_true, y_score, threshold)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return {
        "threshold": float(threshold),
        **counts,
        "accuracy": _ratio(tp + tn, tp + fp + tn + fn),
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "far": _ratio(fp, fp + tn),  # False Accept Rate: người lạ được chấp nhận
        "frr": _ratio(fn, fn + tp),  # False Reject Rate: chủ thể bị từ chối
    }


def roc_curve(y_true, y_score) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Đường ROC: trả về (fpr, tpr, thresholds), điểm đầu (0, 0) ứng với ngưỡng +inf.

    Thuật toán: sắp xếp điểm giảm dần, quét ngưỡng qua từng giá trị điểm phân biệt và
    cộng dồn số TP/FP - độ phức tạp O(n log n).
    """
    y_true, y_score = _validate(y_true, y_score)
    if not has_both_classes(y_true):
        raise ValueError("ROC cần có cả cặp cùng người và cặp khác người")
    order = np.argsort(-y_score, kind="mergesort")
    scores, labels = y_score[order], y_true[order]
    last_of_group = np.r_[np.flatnonzero(np.diff(scores)), scores.size - 1]
    tps = np.cumsum(labels)[last_of_group]
    fps = last_of_group + 1 - tps
    tpr = np.r_[0.0, tps / tps[-1]]
    fpr = np.r_[0.0, fps / fps[-1]]
    thresholds = np.r_[np.inf, scores[last_of_group]]
    return fpr, tpr, thresholds


def auc(x, y) -> float:
    """Diện tích dưới đường cong theo quy tắc hình thang."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2.0))


def roc_auc(y_true, y_score) -> float:
    """ROC-AUC; trả về NaN nếu dữ liệu chỉ có một lớp."""
    if not has_both_classes(y_true):
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)


def equal_error_rate(y_true, y_score) -> tuple[float, float]:
    """EER - điểm trên ROC nơi FAR ≈ FRR. Trả về (eer, ngưỡng tương ứng)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    frr = 1.0 - tpr
    # Bỏ điểm đầu (ngưỡng +inf) để ngưỡng trả về luôn là một giá trị hữu hạn.
    i = 1 + int(np.argmin(np.abs(frr[1:] - fpr[1:])))
    return float((fpr[i] + frr[i]) / 2.0), float(thresholds[i])


def best_threshold(y_true, y_score, criterion: str = "f1") -> float:
    """Chọn ngưỡng trên tập validation theo tiêu chí ``f1`` | ``youden`` | ``eer``.

    - ``f1``: cực đại F1.
    - ``youden``: cực đại chỉ số Youden J = TPR - FPR.
    - ``eer``: ngưỡng tại đó FAR ≈ FRR.
    """
    if criterion not in ("f1", "youden", "eer"):
        raise ValueError(f"Tiêu chí chọn ngưỡng không hợp lệ: {criterion}")
    y_true, y_score = _validate(y_true, y_score)
    if not has_both_classes(y_true):
        return 0.5
    if criterion == "eer":
        return equal_error_rate(y_true, y_score)[1]
    best_t, best_value = 0.5, -np.inf
    for t in np.unique(y_score):
        m = classification_metrics(y_true, y_score, t)
        value = m["f1"] if criterion == "f1" else m["recall"] - m["far"]
        if value > best_value:
            best_t, best_value = float(t), value
    return best_t

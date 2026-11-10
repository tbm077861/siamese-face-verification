"""So sánh hai lần huấn luyện trên cùng một tập test (ví dụ: trước và sau khi đổi tiền xử lý).

Đọc ``test_scores.csv`` của hai lần chạy, kiểm tra hai file chấm trên đúng cùng các cặp,
rồi báo cáo ROC-AUC, EER và TAR tại FAR cố định của từng mô hình, kèm hiệu số và khoảng tin
cậy bootstrap theo cụm danh tính (xem :func:`metrics.paired_bootstrap`).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import CONFIG_FILENAME, TrainConfig, UsageError
from .data import identity_from_filename
from .metrics import equal_error_rate, paired_bootstrap, roc_auc, tar_at_far

SCORES_FILENAME = "test_scores.csv"


def read_scores(run_dir: str | Path) -> tuple[list[tuple[str, str]], np.ndarray, np.ndarray]:
    """Đọc ``test_scores.csv``: danh sách cặp (tên file anchor, candidate), nhãn và điểm."""
    path = Path(run_dir) / SCORES_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy {path} - hãy chạy evaluate cho lần chạy này trước.")
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    pairs = [(Path(row["anchor"]).name, Path(row["candidate"]).name) for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows])
    scores = np.asarray([float(row["score"]) for row in rows])
    return pairs, labels, scores


def _preprocess_of(run_dir: Path) -> str:
    config = run_dir / CONFIG_FILENAME
    return TrainConfig.load(config).preprocess if config.is_file() else "?"


STATISTICS = {
    "roc_auc": roc_auc,
    "eer": lambda labels, scores: equal_error_rate(labels, scores)[0],
    "tar_at_far_1%": lambda labels, scores: tar_at_far(labels, scores, 0.01),
    "tar_at_far_10%": lambda labels, scores: tar_at_far(labels, scores, 0.10),
}


def compare_runs(run_a: str | Path, run_b: str | Path, n_boot: int = 2000, seed: int = 0) -> dict:
    """So sánh lần chạy ``run_b`` với ``run_a`` (hiệu số = b - a)."""
    run_a, run_b = Path(run_a), Path(run_b)
    pairs_a, labels, scores_a = read_scores(run_a)
    pairs_b, labels_b, scores_b = read_scores(run_b)
    if pairs_a != pairs_b or not np.array_equal(labels, labels_b):
        raise UsageError(
            "Hai lần chạy không được chấm trên cùng một tập test (khác cặp hoặc khác nhãn) -"
            " cần cùng --mode, --data-dir, tỉ lệ chia và --seed."
        )
    groups = [identity_from_filename(anchor) for anchor, _ in pairs_a]
    report = {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "preprocess_a": _preprocess_of(run_a),
        "preprocess_b": _preprocess_of(run_b),
        "n_pairs": int(labels.size),
        "n_positive": int(labels.sum()),
        "bootstrap": "theo cụm danh tính của ảnh anchor",
    }
    for name, statistic in STATISTICS.items():
        report[name] = paired_bootstrap(
            labels, scores_a, scores_b, statistic, groups=groups, n_boot=n_boot, seed=seed
        )
    return report


def format_comparison(report: dict) -> str:
    any_stat = report["roc_auc"]
    lines = [
        f"A: {report['run_a']} (tiền xử lý {report['preprocess_a']})",
        f"B: {report['run_b']} (tiền xử lý {report['preprocess_b']})",
        f"Tập test chung: {report['n_pairs']} cặp ({report['n_positive']} cặp cùng người),"
        f" {any_stat['n_clusters']} cụm danh tính",
        f"\n{'':16}{'A':>8}{'B':>8}{'B - A':>9}   KTC {any_stat['confidence']:.0%}"
        f" (bootstrap {report['bootstrap']}, {any_stat['n_boot']} lần)",
    ]
    for name in STATISTICS:
        m = report[name]
        lines.append(
            f"{name:16}{m['a']:8.3f}{m['b']:8.3f}{m['delta']:+9.3f}   [{m['ci_low']:+.3f}, {m['ci_high']:+.3f}]"
        )
    return "\n".join(lines)

"""Đánh giá trên ảnh tự chụp (*field test*).

Lệnh ``evaluate`` đo mô hình trên các cặp ảnh LFW. Mô-đun này đo mô hình ở đúng miền dữ
liệu triển khai: ảnh thử của chủ thể và ảnh của người lạ được so với *cùng* bộ ảnh tham
chiếu của chủ thể, theo đúng quy tắc quyết định của lệnh ``verify``.

- Điểm "cùng người": mọi cặp (ảnh tham chiếu, ảnh thử của chủ thể).
- Điểm "khác người": mọi cặp (ảnh tham chiếu, ảnh người lạ). Người lạ được lấy ngẫu nhiên
  từ một thư mục ảnh (mặc định ``data/negative``), *bỏ qua* những danh tính có trong tập
  train/val của mô hình (đọc ``splits.json`` cạnh file mô hình), để mô hình không được lợi
  vì đã thấy họ khi huấn luyện.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .config import SPLITS_FILENAME, UsageError, preprocess_for_model
from .data import PairSplits, identity_from_filename, list_images
from .metrics import bootstrap_ci, equal_error_rate, roc_auc

DEFAULT_DETECTION_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def seen_identities(model_path: str | Path) -> set[str]:
    """Danh tính xuất hiện trong tập train/val của mô hình (rỗng nếu không có splits.json)."""
    path = Path(model_path).parent / SPLITS_FILENAME
    if not path.is_file():
        return set()
    splits = PairSplits.load(path)
    return {
        identity_from_filename(image)
        for pairs in (splits.train, splits.val)
        for image in (*pairs.anchors, *pairs.candidates)
    }


def sample_impostors(directory: str | Path, exclude: set[str], count: int, seed: int = 0) -> list[str]:
    """Lấy ngẫu nhiên (tất định theo ``seed``) tối đa ``count`` ảnh người lạ chưa bị loại."""
    pool = [path for path in list_images(directory) if identity_from_filename(path) not in exclude]
    if not pool:
        raise UsageError(f"Không còn ảnh người lạ nào trong {directory} sau khi loại danh tính đã huấn luyện.")
    if count >= len(pool):
        return pool
    chosen = np.random.default_rng(seed).choice(len(pool), size=count, replace=False)
    return [pool[i] for i in sorted(chosen)]


def score_against_references(model, images: np.ndarray, references: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Ma trận điểm ``(số ảnh, số ảnh tham chiếu)``.

    Cặp được đưa vào mạng theo thứ tự (ảnh tham chiếu, ảnh cần xác minh) như lệnh ``verify``.
    """
    n_images, n_refs = len(images), len(references)
    image_index, reference_index = np.divmod(np.arange(n_images * n_refs), n_refs)
    scores = np.empty(n_images * n_refs, dtype=np.float64)
    for start in range(0, scores.size, batch_size):
        part = slice(start, start + batch_size)
        output = model.predict_on_batch([references[reference_index[part]], images[image_index[part]]])
        scores[part] = np.asarray(output).reshape(-1)
    return scores.reshape(n_images, n_refs)


def separation(genuine: np.ndarray, impostor: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mức tách biệt giữa điểm cùng người và điểm khác người: ROC-AUC (kèm khoảng tin cậy
    95%), EER, trung vị.

    Mỗi hàng của ``genuine``/``impostor`` là các điểm của *một ảnh* (mảng 1 chiều: mỗi phần
    tử một ảnh). Khoảng tin cậy được tính bằng bootstrap lấy lại theo ảnh, phân tầng theo
    lớp, vì các điểm của cùng một ảnh không độc lập và số ảnh của chủ thể thường rất ít.
    """
    genuine = np.asarray(genuine, dtype=np.float64).reshape(len(genuine), -1)
    impostor = np.asarray(impostor, dtype=np.float64).reshape(len(impostor), -1)
    labels = np.r_[np.ones(genuine.size), np.zeros(impostor.size)]
    scores = np.r_[genuine.ravel(), impostor.ravel()]
    images = np.r_[
        np.repeat(np.arange(len(genuine)), genuine.shape[1]),
        len(genuine) + np.repeat(np.arange(len(impostor)), impostor.shape[1]),
    ]
    eer, eer_threshold = equal_error_rate(labels, scores)
    interval = bootstrap_ci(
        labels, scores, roc_auc, groups=images, stratify=True, n_boot=n_boot, seed=seed
    )
    return {
        "roc_auc": roc_auc(labels, scores),
        "roc_auc_ci95": [interval["ci_low"], interval["ci_high"]],
        "eer": eer,
        "eer_threshold": eer_threshold,
        "median_genuine": float(np.median(genuine)),
        "median_impostor": float(np.median(impostor)),
    }


def decision_table(
    genuine: np.ndarray,
    impostor: np.ndarray,
    detection_thresholds: Sequence[float],
    verification_threshold: float,
) -> list[dict]:
    """Tỉ lệ ảnh thử của chủ thể (TAR) và người lạ (FAR) được chấp nhận theo quy tắc τ_d, τ_v."""
    rows = []
    for threshold in detection_thresholds:
        accepted_genuine = (genuine >= threshold).mean(axis=1) >= verification_threshold
        accepted_impostor = (impostor >= threshold).mean(axis=1) >= verification_threshold
        rows.append(
            {
                "detection_threshold": float(threshold),
                "genuine_accepted": int(accepted_genuine.sum()),
                "n_genuine": int(accepted_genuine.size),
                "tar": float(accepted_genuine.mean()),
                "impostors_accepted": int(accepted_impostor.sum()),
                "n_impostors": int(accepted_impostor.size),
                "far": float(accepted_impostor.mean()),
            }
        )
    return rows


def field_test(
    model_path: str | Path,
    references: Sequence[str],
    probes: Sequence[str],
    impostor_dir: str | Path,
    num_impostors: int = 300,
    seed: int = 0,
    detection_thresholds: Sequence[float] = DEFAULT_DETECTION_THRESHOLDS,
    verification_threshold: float = 0.5,
    preprocess: str | None = "auto",
    batch_size: int = 64,
) -> dict:
    """Chạy phép đo và trả về báo cáo (dict, ghi được ra JSON)."""
    from .model import load_siamese_model, model_image_size
    from .pipeline import preprocess_image
    from .verify import resolve_references

    preprocess = preprocess_for_model(model_path, preprocess)
    reference_paths = resolve_references(references)
    probe_paths = resolve_references(probes)
    excluded = seen_identities(model_path)
    impostor_paths = sample_impostors(impostor_dir, excluded, num_impostors, seed)

    model = load_siamese_model(model_path)
    size = model_image_size(model)

    def load(paths: list[str]) -> tuple[np.ndarray, int | None]:
        loaded = [preprocess_image(path, size, preprocess) for path in paths]
        detected = [flag for _, flag in loaded]
        count = None if detected[0] is None else int(sum(detected))
        return np.stack([image for image, _ in loaded]), count

    reference_images, reference_faces = load(reference_paths)
    probe_images, probe_faces = load(probe_paths)
    impostor_images, impostor_faces = load(impostor_paths)

    genuine = score_against_references(model, probe_images, reference_images, batch_size)
    impostor = score_against_references(model, impostor_images, reference_images, batch_size)
    return {
        "model": str(model_path),
        "preprocess": preprocess,
        "n_references": len(reference_paths),
        "n_probes": len(probe_paths),
        "n_impostors": len(impostor_paths),
        "impostor_dir": str(impostor_dir),
        "excluded_identities": len(excluded),
        "seed": seed,
        "faces_detected": {
            "references": reference_faces,
            "probes": probe_faces,
            "impostors": impostor_faces,
        },
        "pairs": separation(genuine, impostor, seed=seed),
        "images": separation(genuine.mean(axis=1), impostor.mean(axis=1), seed=seed),
        "verification_threshold": verification_threshold,
        "decisions": decision_table(genuine, impostor, detection_thresholds, verification_threshold),
    }


def format_field_test(report: dict) -> str:
    """Bản tóm tắt để in ra màn hình."""

    def detected(key: str, total: int) -> str:
        count = report["faces_detected"][key]
        return "-" if count is None else f"{count}/{total}"

    lines = [
        f"Mô hình: {report['model']} | tiền xử lý: {report['preprocess']}",
        f"Ảnh tham chiếu: {report['n_references']} | ảnh thử của chủ thể: {report['n_probes']}"
        f" | người lạ: {report['n_impostors']} (seed {report['seed']};"
        f" đã loại {report['excluded_identities']} danh tính có trong tập train/val)",
    ]
    if report["faces_detected"]["references"] is not None:
        lines.append(
            "Phát hiện khuôn mặt: "
            f"tham chiếu {detected('references', report['n_references'])},"
            f" ảnh thử {detected('probes', report['n_probes'])},"
            f" người lạ {detected('impostors', report['n_impostors'])}"
        )
    for key, title in (("pairs", "Theo từng cặp"), ("images", "Theo từng ảnh (điểm trung bình)")):
        m = report[key]
        low, high = m["roc_auc_ci95"]
        lines.append(
            f"{title}: ROC-AUC {m['roc_auc']:.3f} (KTC 95%: {low:.3f}-{high:.3f}) | EER {m['eer']:.3f}"
            f" | trung vị điểm: chủ thể {m['median_genuine']:.3f}, người lạ {m['median_impostor']:.3f}"
        )
    lines.append(f"\nQuy tắc xác minh với τ_v = {report['verification_threshold']:.2f}:")
    lines.append("   τ_d   chủ thể được chấp nhận   người lạ được chấp nhận")
    for row in report["decisions"]:
        lines.append(
            f"  {row['detection_threshold']:.3f}"
            f"   {row['genuine_accepted']:>3}/{row['n_genuine']:<3} ({row['tar']:6.1%})"
            f"          {row['impostors_accepted']:>4}/{row['n_impostors']:<4} ({row['far']:6.1%})"
        )
    return "\n".join(lines)

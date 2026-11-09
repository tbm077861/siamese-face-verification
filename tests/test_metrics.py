import math

import numpy as np
import pytest

from siamese_face.metrics import (
    auc,
    best_threshold,
    classification_metrics,
    confusion_counts,
    equal_error_rate,
    has_both_classes,
    roc_auc,
    roc_curve,
)


def test_confusion_counts_and_derived_metrics():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.4, 0.6, 0.1]
    assert confusion_counts(labels, scores, 0.5) == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    metrics = classification_metrics(labels, scores, 0.5)
    for key in ("accuracy", "precision", "recall", "f1", "far", "frr"):
        assert metrics[key] == pytest.approx(0.5)


def test_threshold_is_inclusive():
    assert confusion_counts([1], [0.5], 0.5)["tp"] == 1


def test_metrics_handle_zero_denominators():
    metrics = classification_metrics([0, 0], [0.1, 0.2], 0.5)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.0
    assert metrics["accuracy"] == 1.0 and metrics["far"] == 0.0


def test_roc_auc_matches_reference_example():
    # Ví dụ kinh điển (cùng kết quả với sklearn.metrics.roc_auc_score): AUC = 0.75
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)


def test_roc_auc_extremes_and_ties():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)
    assert roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert math.isnan(roc_auc([1, 1], [0.2, 0.9]))


def test_roc_curve_is_monotonic_from_origin_to_one():
    fpr, tpr, thresholds = roc_curve([0, 1, 0, 1], [0.3, 0.7, 0.6, 0.2])
    assert (fpr[0], tpr[0]) == (0.0, 0.0)
    assert (fpr[-1], tpr[-1]) == (1.0, 1.0)
    assert np.isinf(thresholds[0])
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)


def test_auc_trapezoid():
    assert auc([0, 0.5, 1], [0, 1, 1]) == pytest.approx(0.75)


def test_equal_error_rate():
    eer, threshold = equal_error_rate([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert eer == pytest.approx(0.0) and threshold == pytest.approx(0.8)
    eer, threshold = equal_error_rate([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert eer == pytest.approx(0.5) and threshold == pytest.approx(0.4)


@pytest.mark.parametrize("criterion", ["f1", "youden", "eer"])
def test_best_threshold_separates_separable_scores(criterion):
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.05, 0.2, 0.3, 0.7, 0.8, 0.95]
    threshold = best_threshold(labels, scores, criterion)
    assert classification_metrics(labels, scores, threshold)["accuracy"] == 1.0


def test_best_threshold_falls_back_without_both_classes():
    assert best_threshold([1, 1], [0.3, 0.9]) == 0.5


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        confusion_counts([1, 0], [0.5])
    with pytest.raises(ValueError):
        confusion_counts([2], [0.5])
    with pytest.raises(ValueError):
        best_threshold([0, 1], [0.1, 0.9], "abc")


def test_has_both_classes():
    assert has_both_classes([0, 1])
    assert not has_both_classes([1, 1])
    assert not has_both_classes([])

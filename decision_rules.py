"""Threshold selection for binary and confirm/reject/defer decisions."""

import math
from typing import Iterable, List

import numpy as np

from ranked_scan_model import metrics_at_threshold


def candidate_thresholds(scores: Iterable[float]) -> List[float]:
    values = sorted(set(float(score) for score in scores))
    if not values:
        return [0.0]
    thresholds = [values[0] - 1.0]
    thresholds.extend((left + right) / 2.0 for left, right in zip(values[:-1], values[1:]))
    thresholds.append(values[-1] + 1.0)
    return thresholds


def constrained_confirm_threshold(
    train_scores: List[dict],
    max_controller_fcr: float,
) -> float:
    thresholds = candidate_thresholds(row["score"] for row in train_scores)
    selected = thresholds[-1]
    selected_key = (-1.0, -1.0, -1.0)
    controller_rows = [row for row in train_scores if row["controller_only"] == 1]

    for threshold in thresholds:
        metrics = metrics_at_threshold(train_scores, threshold)
        controller_fcr = metrics["controller_only_false_confirmation"]
        if math.isnan(controller_fcr):
            controller_fcr = 0.0
        if controller_rows and controller_fcr > max_controller_fcr + 1e-12:
            continue
        key = (
            metrics["linked_tpr"],
            metrics["balanced_accuracy"],
            metrics["negative_tnr"],
        )
        if key > selected_key:
            selected_key = key
            selected = threshold
    return selected


def constrained_reject_threshold(
    train_scores: List[dict],
    max_linked_reject: float,
) -> float:
    thresholds = candidate_thresholds(row["score"] for row in train_scores)
    selected = thresholds[0] - 1.0
    selected_key = (-1.0, -1.0)
    labels = np.asarray([row["linked"] for row in train_scores], dtype=int)
    scores = np.asarray([row["score"] for row in train_scores], dtype=float)
    linked = labels == 1
    negative = labels == 0

    for threshold in thresholds:
        rejected = scores <= threshold
        linked_reject = float(np.mean(rejected[linked])) if np.any(linked) else 0.0
        if linked_reject > max_linked_reject + 1e-12:
            continue
        negative_reject = float(np.mean(rejected[negative])) if np.any(negative) else 0.0
        key = (negative_reject, threshold)
        if key > selected_key:
            selected_key = key
            selected = threshold
    return selected

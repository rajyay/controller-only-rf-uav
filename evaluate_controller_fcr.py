#!/usr/bin/env python3
"""Run the main LORO analyses for controller-only false confirmation."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, List

import numpy as np

from ranked_scan_model import (
    FEATURE_SETS,
    SUBSETS,
    class_stats,
    metrics_at_threshold,
    policy_steps,
    read_rows,
    scaler,
    scan_scores,
    subband_id,
)
from decision_rules import (
    candidate_thresholds,
    constrained_confirm_threshold,
    constrained_reject_threshold,
)


SUBSET_ORDER = ["all", "phantom", "hubsan", "mavic", "hubsan_mavic"]
BAND_ORDER = ["2.4_only", "5.8_only", "dual"]
BAND_FILTERS = {
    "2.4_only": {"2.4"},
    "5.8_only": {"5.8"},
    "dual": {"2.4", "5.8"},
}
BAND_DWELL_SECONDS = {
    "2.4_only": 16.0,
    "5.8_only": 24.0,
    "dual": 24.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser.parse_args()


def select_subset(rows: List[dict], subset: str) -> List[dict]:
    states = SUBSETS[subset]
    if states is None:
        return list(rows)
    return [row for row in rows if row["state_label"] in states]


def conservative_balanced_threshold(score_rows: List[dict]) -> float:
    """Maximize balanced accuracy, breaking ties toward fewer negatives flagged."""
    best_threshold = 0.0
    best_key = (-1.0, -1.0, -1.0, float("-inf"))
    for threshold in candidate_thresholds(row["score"] for row in score_rows):
        metrics = metrics_at_threshold(score_rows, threshold)
        key = (
            metrics["balanced_accuracy"],
            metrics["negative_tnr"],
            metrics["linked_tpr"],
            threshold,
        )
        if key > best_key:
            best_key = key
            best_threshold = threshold
    return best_threshold


def ordered_predictions(
    score_rows: Iterable[dict],
    confirm_threshold: float,
    reject_threshold: float | None = None,
) -> List[dict]:
    predictions = []
    for row in score_rows:
        confirm = int(float(row["score"]) >= confirm_threshold)
        if reject_threshold is None:
            reject = int(not confirm)
            defer = 0
        else:
            reject = int((not confirm) and float(row["score"]) <= reject_threshold)
            defer = int(not confirm and not reject)
        predictions.append({
            **row,
            "confirm_threshold": confirm_threshold,
            "reject_threshold": "" if reject_threshold is None else reject_threshold,
            "confirm": confirm,
            "reject": reject,
            "defer": defer,
        })
    return predictions


def rate(rows: List[dict], numerator: Callable[[dict], bool]) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([bool(numerator(row)) for row in rows]))


def audit_metrics(rows: List[dict]) -> dict:
    linked = [row for row in rows if int(row["linked"]) == 1]
    ambient = [row for row in rows if int(row["ambient"]) == 1]
    controller = [row for row in rows if int(row["controller_only"]) == 1]
    linked_tpr = rate(linked, lambda row: int(row["confirm"]) == 1)
    ambient_fpr = rate(ambient, lambda row: int(row["confirm"]) == 1)
    controller_fcr = rate(controller, lambda row: int(row["confirm"]) == 1)
    return {
        "benchmark_balanced_accuracy": 0.5 * (linked_tpr + (1.0 - ambient_fpr)),
        "linked_tpr": linked_tpr,
        "ambient_fpr": ambient_fpr,
        "controller_only_fcr": controller_fcr,
        "linked_confirmed": sum(int(row["confirm"]) for row in linked),
        "linked_n": len(linked),
        "ambient_false_confirmed": sum(int(row["confirm"]) for row in ambient),
        "ambient_n": len(ambient),
        "controller_false_confirmed": sum(int(row["confirm"]) for row in controller),
        "controller_n": len(controller),
    }


def operating_metrics(rows: List[dict]) -> dict:
    linked = [row for row in rows if int(row["linked"]) == 1]
    ambient = [row for row in rows if int(row["ambient"]) == 1]
    controller = [row for row in rows if int(row["controller_only"]) == 1]
    negative = [row for row in rows if int(row["linked"]) == 0]
    linked_tpr = rate(linked, lambda row: int(row["confirm"]) == 1)
    negative_tnr = rate(negative, lambda row: int(row["confirm"]) == 0)
    return {
        "balanced_accuracy": 0.5 * (linked_tpr + negative_tnr),
        "linked_tpr": linked_tpr,
        "controller_only_fcr": rate(controller, lambda row: int(row["confirm"]) == 1),
        "ambient_fpr": rate(ambient, lambda row: int(row["confirm"]) == 1),
        "defer_rate": rate(rows, lambda row: int(row["defer"]) == 1),
        "linked_reject_rate": rate(linked, lambda row: int(row["reject"]) == 1),
        "negative_reject_rate": rate(negative, lambda row: int(row["reject"]) == 1),
        "linked_confirmed": sum(int(row["confirm"]) for row in linked),
        "linked_n": len(linked),
        "controller_false_confirmed": sum(int(row["confirm"]) for row in controller),
        "controller_n": len(controller),
        "ambient_false_confirmed": sum(int(row["confirm"]) for row in ambient),
        "ambient_n": len(ambient),
        "deferred": sum(int(row["defer"]) for row in rows),
        "total_n": len(rows),
    }


def bootstrap_intervals(
    rows: List[dict],
    metric_function: Callable[[List[dict]], dict],
    metric_names: List[str],
    resamples: int,
    rng: np.random.Generator,
) -> dict:
    by_round = defaultdict(list)
    for row in rows:
        by_round[row["round_id"]].append(row)
    round_ids = sorted(by_round)
    draws = {name: [] for name in metric_names}
    for _ in range(resamples):
        sampled = rng.choice(round_ids, size=len(round_ids), replace=True)
        replicate = []
        for round_id in sampled:
            replicate.extend(by_round[str(round_id)])
        metrics = metric_function(replicate)
        for name in metric_names:
            draws[name].append(float(metrics[name]))
    intervals = {}
    for name, values in draws.items():
        intervals[f"{name}_ci_low"] = float(np.quantile(values, 0.025))
        intervals[f"{name}_ci_high"] = float(np.quantile(values, 0.975))
    return intervals


def group_rows(rows: List[dict], fields: List[str]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    return groups


def model_scores(
    train_rows: List[dict],
    eval_rows: List[dict],
    feature_set: str,
    steps: List[List[str]],
) -> tuple[List[dict], List[dict]]:
    features = FEATURE_SETS[feature_set]
    scale = scaler(train_rows, features)
    stats = class_stats(train_rows, features, scale)
    return (
        scan_scores(train_rows, steps, stats, features, scale),
        scan_scores(eval_rows, steps, stats, features, scale),
    )


def evaluate_omitted_negative(rows: List[dict]) -> List[dict]:
    predictions = []
    rounds = sorted({row["round_id"] for row in rows})
    for subset in SUBSET_ORDER:
        subset_data = select_subset(rows, subset)
        for held_out_round in rounds:
            train_all = [row for row in subset_data if row["round_id"] != held_out_round]
            eval_all = [row for row in subset_data if row["round_id"] == held_out_round]
            train_rows = [
                row for row in train_all
                if int(row["linked_uav_target"]) == 1 or row["state_label"] == "ambient"
            ]
            if not train_rows or not eval_all:
                continue
            features = FEATURE_SETS["energy"]
            scale = scaler(train_rows, features)
            stats = class_stats(train_rows, features, scale)
            for policy in ["dual_as_collected", "dual_ranked"]:
                steps = policy_steps(policy, train_rows, stats)
                train_scores = scan_scores(train_rows, steps, stats, features, scale)
                eval_scores = scan_scores(eval_all, steps, stats, features, scale)
                for observed_steps in range(1, len(steps) + 1):
                    train_k = [
                        row for row in train_scores
                        if int(row["observed_steps"]) == observed_steps
                    ]
                    eval_k = [
                        row for row in eval_scores
                        if int(row["observed_steps"]) == observed_steps
                    ]
                    threshold = conservative_balanced_threshold(train_k)
                    for prediction in ordered_predictions(eval_k, threshold):
                        predictions.append({
                            **prediction,
                            "analysis": "omitted_negative_audit",
                            "subset": subset,
                            "feature_set": "energy",
                            "policy": policy,
                            "observed_steps": observed_steps,
                            "nominal_dwell_s": observed_steps * 2.0,
                        })
    return predictions


def summarize_omitted_negative(
    predictions: List[dict],
    bootstrap_resamples: int,
    rng: np.random.Generator,
) -> tuple[List[dict], List[dict]]:
    fields = ["subset", "feature_set", "policy", "observed_steps", "nominal_dwell_s"]
    summaries = []
    selected = []
    ci_metrics = [
        "benchmark_balanced_accuracy",
        "linked_tpr",
        "ambient_fpr",
        "controller_only_fcr",
    ]
    for key, group in sorted(group_rows(predictions, fields).items()):
        summary = {field: value for field, value in zip(fields, key)}
        summary.update(audit_metrics(group))
        summaries.append(summary)
        if summary["policy"] == "dual_ranked" and int(summary["observed_steps"]) == 4:
            selected_row = dict(summary)
            selected_row.update(bootstrap_intervals(
                group, audit_metrics, ci_metrics, bootstrap_resamples, rng,
            ))
            selected.append(selected_row)
    return summaries, selected


def evaluate_operating_points(rows: List[dict], alpha: float) -> List[dict]:
    predictions = []
    rounds = sorted({row["round_id"] for row in rows})
    for subset in SUBSET_ORDER:
        subset_data = select_subset(rows, subset)
        for held_out_round in rounds:
            train_rows = [row for row in subset_data if row["round_id"] != held_out_round]
            eval_rows = [row for row in subset_data if row["round_id"] == held_out_round]
            if not train_rows or not eval_rows:
                continue
            features = FEATURE_SETS["energy"]
            scale = scaler(train_rows, features)
            stats = class_stats(train_rows, features, scale)
            steps = policy_steps("dual_ranked", train_rows, stats)
            train_scores = scan_scores(train_rows, steps, stats, features, scale)
            eval_scores = scan_scores(eval_rows, steps, stats, features, scale)
            for observed_steps in range(1, len(steps) + 1):
                train_k = [
                    row for row in train_scores
                    if int(row["observed_steps"]) == observed_steps
                ]
                eval_k = [
                    row for row in eval_scores
                    if int(row["observed_steps"]) == observed_steps
                ]
                binary_threshold = conservative_balanced_threshold(train_k)
                confirm_threshold = constrained_confirm_threshold(train_k, alpha)
                reject_threshold = constrained_reject_threshold(train_k, alpha)
                modes = [
                    ("binary_balanced", binary_threshold, None),
                    ("controller_fcr_constrained", confirm_threshold, reject_threshold),
                ]
                for mode, confirm_th, reject_th in modes:
                    for prediction in ordered_predictions(eval_k, confirm_th, reject_th):
                        predictions.append({
                            **prediction,
                            "analysis": "same_subset_operating_point",
                            "subset": subset,
                            "feature_set": "energy",
                            "policy": "dual_ranked",
                            "operating_point": mode,
                            "alpha": alpha,
                            "observed_steps": observed_steps,
                            "nominal_dwell_s": observed_steps * 2.0,
                        })
    return predictions


def summarize_operating_points(
    predictions: List[dict],
    bootstrap_resamples: int,
    rng: np.random.Generator,
) -> tuple[List[dict], List[dict]]:
    fields = [
        "subset",
        "feature_set",
        "policy",
        "operating_point",
        "alpha",
        "observed_steps",
        "nominal_dwell_s",
    ]
    summaries = []
    selected = []
    ci_metrics = [
        "balanced_accuracy",
        "linked_tpr",
        "controller_only_fcr",
        "ambient_fpr",
        "defer_rate",
    ]
    for key, group in sorted(group_rows(predictions, fields).items()):
        summary = {field: value for field, value in zip(fields, key)}
        summary.update(operating_metrics(group))
        summaries.append(summary)
        if int(summary["observed_steps"]) == 4:
            selected_row = dict(summary)
            selected_row.update(bootstrap_intervals(
                group, operating_metrics, ci_metrics, bootstrap_resamples, rng,
            ))
            selected.append(selected_row)
    return summaries, selected


def full_band_step(rows: List[dict]) -> List[List[str]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            0 if row["band"] == "2.4" else 1,
            int(float(row.get("capture_index") or 0)),
            int(float(row["center_freq_hz"])),
        ),
    )
    return [list(dict.fromkeys(subband_id(row) for row in ordered))]


def evaluate_band_ablation(rows: List[dict], alpha: float) -> List[dict]:
    predictions = []
    rounds = sorted({row["round_id"] for row in rows})
    for feature_set in ["energy", "energy_spectral"]:
        for subset in SUBSET_ORDER:
            subset_data = select_subset(rows, subset)
            for held_out_round in rounds:
                train_all = [row for row in subset_data if row["round_id"] != held_out_round]
                eval_all = [row for row in subset_data if row["round_id"] == held_out_round]
                for band_selection in BAND_ORDER:
                    allowed_bands = BAND_FILTERS[band_selection]
                    train_rows = [row for row in train_all if row["band"] in allowed_bands]
                    eval_rows = [row for row in eval_all if row["band"] in allowed_bands]
                    if not train_rows or not eval_rows:
                        continue
                    steps = full_band_step(train_rows)
                    train_scores, eval_scores = model_scores(
                        train_rows, eval_rows, feature_set, steps,
                    )
                    confirm_threshold = constrained_confirm_threshold(train_scores, alpha)
                    reject_threshold = constrained_reject_threshold(train_scores, alpha)
                    for prediction in ordered_predictions(
                        eval_scores, confirm_threshold, reject_threshold,
                    ):
                        predictions.append({
                            **prediction,
                            "analysis": "band_ablation",
                            "subset": subset,
                            "feature_set": feature_set,
                            "band_selection": band_selection,
                            "alpha": alpha,
                            "nominal_dwell_s": BAND_DWELL_SECONDS[band_selection],
                        })
    return predictions


def summarize_band_ablation(
    predictions: List[dict],
    bootstrap_resamples: int,
    rng: np.random.Generator,
) -> List[dict]:
    fields = ["subset", "feature_set", "band_selection", "alpha", "nominal_dwell_s"]
    ci_metrics = [
        "balanced_accuracy",
        "linked_tpr",
        "controller_only_fcr",
        "ambient_fpr",
        "defer_rate",
    ]
    summaries = []
    for key, group in sorted(group_rows(predictions, fields).items()):
        summary = {field: value for field, value in zip(fields, key)}
        summary.update(operating_metrics(group))
        summary.update(bootstrap_intervals(
            group, operating_metrics, ci_metrics, bootstrap_resamples, rng,
        ))
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        raise SystemExit(f"No rows generated for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.features_csv))
    out_dir = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)

    audit_predictions = evaluate_omitted_negative(rows)
    audit_summary, audit_selected = summarize_omitted_negative(
        audit_predictions, args.bootstrap_resamples, rng,
    )
    write_csv(out_dir / "omitted_negative_predictions.csv", audit_predictions)
    write_csv(out_dir / "omitted_negative_audit.csv", audit_summary)
    write_csv(out_dir / "omitted_negative_audit_8s.csv", audit_selected)

    operating_predictions = evaluate_operating_points(rows, args.alpha)
    operating_summary, operating_selected = summarize_operating_points(
        operating_predictions, args.bootstrap_resamples, rng,
    )
    write_csv(out_dir / "operating_point_predictions.csv", operating_predictions)
    write_csv(out_dir / "operating_point_tradeoff.csv", operating_summary)
    write_csv(out_dir / "operating_point_tradeoff_8s.csv", operating_selected)

    band_predictions = evaluate_band_ablation(rows, args.alpha)
    band_summary = summarize_band_ablation(
        band_predictions, args.bootstrap_resamples, rng,
    )
    write_csv(out_dir / "band_ablation_predictions.csv", band_predictions)
    write_csv(out_dir / "band_ablation.csv", band_summary)

    print(f"[DONE] wrote controller-FCR analyses under {out_dir}")


if __name__ == "__main__":
    main()

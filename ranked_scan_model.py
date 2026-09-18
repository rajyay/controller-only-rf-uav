#!/usr/bin/env python3
"""Fit and evaluate the ranked-scan Gaussian LLR model."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


TRAIN_DEFAULT = [f"r{i:02d}" for i in range(1, 15)]
EVAL_DEFAULT = [f"r{i:02d}" for i in range(15, 21)]

FEATURE_SETS = {
    "energy": ["mean_power_db"],
    "spectral_shape": [
        "psd_peak_to_median_db",
        "spectral_entropy",
        "occupied_fraction_6db",
    ],
    "energy_spectral": [
        "mean_power_db",
        "psd_peak_to_median_db",
        "spectral_entropy",
        "occupied_fraction_6db",
    ],
}

SUBSETS = {
    "all": None,
    "phantom": {
        "ambient",
        "phantom_controller_only",
        "phantom_linked",
        "phantom_controller_only_phone_connected",
        "phantom_linked_phone_connected",
    },
    "hubsan": {"ambient", "hubsan_controller_only", "hubsan_linked"},
    "mavic": {"ambient", "mavic_controller_only", "mavic_linked"},
    "hubsan_mavic": {
        "ambient",
        "hubsan_controller_only",
        "hubsan_linked",
        "mavic_controller_only",
        "mavic_linked",
    },
}

EPS = 1e-9
VAR_FLOOR = 1e-4


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--mode", choices=["split", "looro"], default="split")
    ap.add_argument("--feature-set", action="append", choices=sorted(FEATURE_SETS),
                    default=None)
    ap.add_argument("--subset", action="append", choices=sorted(SUBSETS),
                    default=None)
    ap.add_argument("--train-rounds", nargs="+", default=TRAIN_DEFAULT)
    ap.add_argument("--eval-rounds", nargs="+", default=EVAL_DEFAULT)
    ap.add_argument("--dwell-seconds", type=float, default=2.0)
    return ap.parse_args()


def read_rows(path: Path) -> List[dict]:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise SystemExit("No feature rows found.")
    return rows


def subband_id(row: dict) -> str:
    return f"{row.get('receiver_id') or row.get('band')}:{row['center_freq_hz']}"


def label_by_scan(rows: Iterable[dict]) -> Dict[str, dict]:
    labels = {}
    for row in rows:
        sid = row["scan_cycle_id"]
        labels.setdefault(sid, {
            "scan_cycle_id": sid,
            "round_id": row["round_id"],
            "state_label": row["state_label"],
            "linked": int(row["linked_uav_target"]),
            "controller_only": int(row["controller_only_target"]),
            "ambient": 1 if row["state_label"] == "ambient" else 0,
        })
    return labels


def grouped_rows(rows: Iterable[dict]) -> Dict[str, List[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["scan_cycle_id"]].append(row)
    return groups


def scaler(rows: List[dict], features: List[str]) -> Dict[str, tuple[float, float]]:
    params = {}
    for feat in features:
        vals = np.asarray([float(r[feat]) for r in rows], dtype=float)
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        params[feat] = (mu, max(sd, EPS))
    return params


def z_value(row: dict, feature: str, scale: Dict[str, tuple[float, float]]) -> float:
    mu, sd = scale[feature]
    return (float(row[feature]) - mu) / sd


def class_stats(
    rows: List[dict],
    features: List[str],
    scale: Dict[str, tuple[float, float]],
) -> Dict[str, dict]:
    vals = defaultdict(lambda: {0: defaultdict(list), 1: defaultdict(list)})
    for row in rows:
        sid = subband_id(row)
        y = int(row["linked_uav_target"])
        for feat in features:
            vals[sid][y][feat].append(z_value(row, feat, scale))

    stats = {}
    for sid, by_class in vals.items():
        if not by_class[0] or not by_class[1]:
            continue
        stats[sid] = {0: {}, 1: {}, "separation": 0.0}
        sep_terms = []
        for feat in features:
            if not by_class[0][feat] or not by_class[1][feat]:
                continue
            neg = np.asarray(by_class[0][feat], dtype=float)
            pos = np.asarray(by_class[1][feat], dtype=float)
            neg_mean = float(np.mean(neg))
            pos_mean = float(np.mean(pos))
            neg_var = float(max(np.var(neg), VAR_FLOOR))
            pos_var = float(max(np.var(pos), VAR_FLOOR))
            stats[sid][0][feat] = (neg_mean, neg_var)
            stats[sid][1][feat] = (pos_mean, pos_var)
            sep_terms.append(((pos_mean - neg_mean) ** 2) / max(pos_var + neg_var, VAR_FLOOR))
        stats[sid]["separation"] = math.sqrt(float(sum(sep_terms))) if sep_terms else 0.0
    return stats


def gaussian_logpdf(x: float, mean: float, var: float) -> float:
    return -0.5 * (math.log(2.0 * math.pi * var) + ((x - mean) ** 2) / var)


def row_llr(
    row: dict,
    stats: Dict[str, dict],
    features: List[str],
    scale: Dict[str, tuple[float, float]],
) -> float:
    sid = subband_id(row)
    if sid not in stats:
        return 0.0
    score = 0.0
    for feat in features:
        if feat not in stats[sid][0] or feat not in stats[sid][1]:
            continue
        x = z_value(row, feat, scale)
        pos_mean, pos_var = stats[sid][1][feat]
        neg_mean, neg_var = stats[sid][0][feat]
        score += gaussian_logpdf(x, pos_mean, pos_var) - gaussian_logpdf(x, neg_mean, neg_var)
    return score


def sorted_by_capture(rows: Iterable[dict], band: str | None = None) -> List[str]:
    selected = [r for r in rows if band is None or r.get("band") == band]
    selected.sort(key=lambda r: (int(float(r.get("capture_index") or 0)), int(float(r["center_freq_hz"]))))
    return [subband_id(r) for r in selected]


def ranked_subbands(rows: List[dict], stats: Dict[str, dict], band: str | None = None) -> List[str]:
    seen = {}
    for row in rows:
        if band is not None and row.get("band") != band:
            continue
        sid = subband_id(row)
        seen[sid] = stats.get(sid, {}).get("separation", 0.0)
    return [sid for sid, _ in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))]


def policy_steps(policy: str, train_rows: List[dict], stats: Dict[str, dict]) -> List[List[str]]:
    if policy == "serial_ranked":
        ranked = ranked_subbands(train_rows, stats)
        return [[sid] for sid in ranked]

    if policy == "dual_ranked":
        order24 = ranked_subbands(train_rows, stats, band="2.4")
        order58 = ranked_subbands(train_rows, stats, band="5.8")
    elif policy == "dual_as_collected":
        order24 = list(dict.fromkeys(sorted_by_capture(train_rows, band="2.4")))
        order58 = list(dict.fromkeys(sorted_by_capture(train_rows, band="5.8")))
    else:
        raise ValueError(f"Unknown policy: {policy}")

    steps = []
    for i in range(max(len(order24), len(order58))):
        step = []
        if i < len(order24):
            step.append(order24[i])
        if i < len(order58):
            step.append(order58[i])
        steps.append(step)
    return steps


def scan_scores(
    rows: List[dict],
    steps: List[List[str]],
    stats: Dict[str, dict],
    features: List[str],
    scale: Dict[str, tuple[float, float]],
) -> List[dict]:
    groups = grouped_rows(rows)
    labels = label_by_scan(rows)
    out = []
    for k in range(1, len(steps) + 1):
        observed = {sid for step in steps[:k] for sid in step}
        for scan_id, scan_rows in groups.items():
            score = sum(
                row_llr(r, stats, features, scale)
                for r in scan_rows
                if subband_id(r) in observed
            )
            row = dict(labels[scan_id])
            row.update({
                "observed_steps": k,
                "observed_subbands": len(observed),
                "score": score,
            })
            out.append(row)
    return out


def metrics_at_threshold(score_rows: List[dict], threshold: float) -> dict:
    if not score_rows:
        return {
            "balanced_accuracy": float("nan"),
            "accuracy": float("nan"),
            "linked_tpr": float("nan"),
            "negative_tnr": float("nan"),
            "controller_only_false_confirmation": float("nan"),
            "ambient_false_confirmation": float("nan"),
        }

    y = np.asarray([r["linked"] for r in score_rows], dtype=int)
    pred = np.asarray([1 if r["score"] >= threshold else 0 for r in score_rows], dtype=int)
    pos = y == 1
    neg = y == 0
    ctrl = np.asarray([r["controller_only"] for r in score_rows], dtype=int) == 1
    ambient = np.asarray([r["ambient"] for r in score_rows], dtype=int) == 1

    linked_tpr = float(np.mean(pred[pos] == 1)) if np.any(pos) else float("nan")
    negative_tnr = float(np.mean(pred[neg] == 0)) if np.any(neg) else float("nan")
    return {
        "balanced_accuracy": float(np.nanmean([linked_tpr, negative_tnr])),
        "accuracy": float(np.mean(pred == y)),
        "linked_tpr": linked_tpr,
        "negative_tnr": negative_tnr,
        "controller_only_false_confirmation": float(np.mean(pred[ctrl] == 1)) if np.any(ctrl) else float("nan"),
        "ambient_false_confirmation": float(np.mean(pred[ambient] == 1)) if np.any(ambient) else float("nan"),
    }


def best_threshold(train_scores: List[dict]) -> float:
    scores = sorted({r["score"] for r in train_scores})
    if not scores:
        return 0.0
    candidates = [scores[0] - 1.0]
    candidates.extend((a + b) / 2.0 for a, b in zip(scores[:-1], scores[1:]))
    candidates.append(scores[-1] + 1.0)

    best = candidates[0]
    best_bal = -1.0
    for th in candidates:
        bal = metrics_at_threshold(train_scores, th)["balanced_accuracy"]
        if bal > best_bal:
            best_bal = bal
            best = th
    return best


def subset_rows(rows: List[dict], subset: str) -> List[dict]:
    states = SUBSETS[subset]
    if states is None:
        return rows
    return [r for r in rows if r["state_label"] in states]


def evaluate_split(
    rows: List[dict],
    subset: str,
    feature_set: str,
    train_rounds: set[str],
    eval_rounds: set[str],
    dwell_seconds: float,
) -> List[dict]:
    rows = subset_rows(rows, subset)
    train_rows = [r for r in rows if r["round_id"] in train_rounds]
    eval_rows = [r for r in rows if r["round_id"] in eval_rounds]
    if not train_rows or not eval_rows:
        return []
    return evaluate_one_train_eval(
        train_rows=train_rows,
        eval_rows=eval_rows,
        subset=subset,
        feature_set=feature_set,
        split_id="split",
        dwell_seconds=dwell_seconds,
    )


def evaluate_one_train_eval(
    train_rows: List[dict],
    eval_rows: List[dict],
    subset: str,
    feature_set: str,
    split_id: str,
    dwell_seconds: float,
) -> List[dict]:
    features = FEATURE_SETS[feature_set]
    scale = scaler(train_rows, features)
    stats = class_stats(train_rows, features, scale)
    if not stats:
        return []
    out = []
    for policy in ["dual_as_collected", "dual_ranked", "serial_ranked"]:
        steps = policy_steps(policy, train_rows, stats)
        train_scores = scan_scores(train_rows, steps, stats, features, scale)
        eval_scores = scan_scores(eval_rows, steps, stats, features, scale)
        for k in range(1, len(steps) + 1):
            train_k = [r for r in train_scores if r["observed_steps"] == k]
            eval_k = [r for r in eval_scores if r["observed_steps"] == k]
            threshold = best_threshold(train_k)
            metrics = metrics_at_threshold(eval_k, threshold)
            observed_subbands = len({sid for step in steps[:k] for sid in step})
            out.append({
                "mode": "split" if split_id == "split" else "looro_fold",
                "split_id": split_id,
                "subset": subset,
                "feature_set": feature_set,
                "policy": policy,
                "observed_steps": k,
                "latency_s": f"{k * dwell_seconds:.3f}",
                "observed_subbands": observed_subbands,
                "threshold": f"{threshold:.6f}",
                "train_scan_cycles": len(grouped_rows(train_rows)),
                "eval_scan_cycles": len(grouped_rows(eval_rows)),
                **{name: f"{value:.6f}" for name, value in metrics.items()},
            })
    return out


def evaluate_looro(
    rows: List[dict],
    subset: str,
    feature_set: str,
    dwell_seconds: float,
) -> List[dict]:
    rows = subset_rows(rows, subset)
    rounds = sorted({r["round_id"] for r in rows})
    fold_rows = []
    for eval_round in rounds:
        train_rows = [r for r in rows if r["round_id"] != eval_round]
        eval_rows = [r for r in rows if r["round_id"] == eval_round]
        fold_rows.extend(evaluate_one_train_eval(
            train_rows=train_rows,
            eval_rows=eval_rows,
            subset=subset,
            feature_set=feature_set,
            split_id=eval_round,
            dwell_seconds=dwell_seconds,
        ))
    return aggregate_looro(fold_rows)


def aggregate_looro(fold_rows: List[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in fold_rows:
        key = (
            row["subset"],
            row["feature_set"],
            row["policy"],
            row["observed_steps"],
            row["latency_s"],
            row["observed_subbands"],
        )
        grouped[key].append(row)

    out = []
    metric_names = [
        "balanced_accuracy",
        "accuracy",
        "linked_tpr",
        "negative_tnr",
        "controller_only_false_confirmation",
        "ambient_false_confirmation",
    ]
    for key, rows in grouped.items():
        subset, feature_set, policy, observed_steps, latency_s, observed_subbands = key
        # Each fold has one scan cycle per state in the subset, so a macro mean
        # over folds is equivalent to equal round weighting.
        metrics = {}
        for name in metric_names:
            vals = np.asarray([float(r[name]) for r in rows], dtype=float)
            finite = vals[np.isfinite(vals)]
            metrics[name] = float(np.mean(finite)) if finite.size else float("nan")
        out.append({
            "mode": "looro",
            "split_id": "leave_one_round_out",
            "subset": subset,
            "feature_set": feature_set,
            "policy": policy,
            "observed_steps": observed_steps,
            "latency_s": latency_s,
            "observed_subbands": observed_subbands,
            "threshold": "",
            "train_scan_cycles": "",
            "eval_scan_cycles": sum(int(r["eval_scan_cycles"]) for r in rows),
            **{name: f"{value:.6f}" for name, value in metrics.items()},
        })
    return sorted(out, key=lambda r: (
        r["subset"],
        r["feature_set"],
        r["policy"],
        int(r["observed_steps"]),
    ))


def write_results(path: Path, rows: List[dict]) -> None:
    if not rows:
        raise SystemExit("No results generated.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "mode",
        "split_id",
        "subset",
        "feature_set",
        "policy",
        "observed_steps",
        "latency_s",
        "observed_subbands",
        "threshold",
        "train_scan_cycles",
        "eval_scan_cycles",
        "balanced_accuracy",
        "accuracy",
        "linked_tpr",
        "negative_tnr",
        "controller_only_false_confirmation",
        "ambient_false_confirmation",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.features_csv))
    feature_sets = args.feature_set or ["energy", "spectral_shape", "energy_spectral"]
    subsets = args.subset or ["all", "phantom", "hubsan", "mavic", "hubsan_mavic"]

    results = []
    if args.mode == "split":
        train_rounds = set(args.train_rounds)
        eval_rounds = set(args.eval_rounds)
        for subset in subsets:
            for feature_set in feature_sets:
                results.extend(evaluate_split(
                    rows=rows,
                    subset=subset,
                    feature_set=feature_set,
                    train_rounds=train_rounds,
                    eval_rounds=eval_rounds,
                    dwell_seconds=args.dwell_seconds,
                ))
    else:
        for subset in subsets:
            for feature_set in feature_sets:
                results.extend(evaluate_looro(
                    rows=rows,
                    subset=subset,
                    feature_set=feature_set,
                    dwell_seconds=args.dwell_seconds,
                ))

    write_results(Path(args.out_csv), results)
    print(f"[DONE] wrote {len(results)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()

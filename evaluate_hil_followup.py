#!/usr/bin/env python3
"""Score follow-up hardware-in-loop captures using an r01-r20 trained model."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List

from ranked_scan_model import (
    FEATURE_SETS,
    SUBSETS,
    best_threshold,
    class_stats,
    row_llr,
    scaler,
    subband_id,
)
from decision_rules import (
    constrained_confirm_threshold,
    constrained_reject_threshold,
)


DEFAULT_SUBSETS = ["all", "phantom", "hubsan", "mavic", "hubsan_mavic"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-features-csv", required=True)
    ap.add_argument("--eval-features-csv", required=True)
    ap.add_argument("--timing-csv", default="")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--feature-set", default="energy", choices=sorted(FEATURE_SETS))
    ap.add_argument("--constraint", type=float, default=0.05)
    ap.add_argument("--dwell-seconds", type=float, default=2.0)
    return ap.parse_args()


def read_rows(path: Path) -> List[dict]:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise SystemExit(f"No rows found in {path}")
    return rows


def subset_rows(rows: Iterable[dict], subset: str) -> List[dict]:
    states = SUBSETS[subset]
    if states is None:
        return list(rows)
    return [r for r in rows if r["state_label"] in states]


def grouped(rows: Iterable[dict]) -> dict[str, List[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["scan_cycle_id"]].append(row)
    return groups


def labels_for(rows: List[dict]) -> dict:
    row = rows[0]
    return {
        "round_id": row["round_id"],
        "scan_cycle_id": row["scan_cycle_id"],
        "state_label": row["state_label"],
        "linked": int(row["linked_uav_target"]),
        "controller_only": int(row["controller_only_target"]),
        "ambient": 1 if row["state_label"] == "ambient" else 0,
    }


def scan_kind(scan_cycle_id: str) -> str:
    for token in ["full20", "ranked8s", "ranked10s", "ranked12s"]:
        if scan_cycle_id.endswith(f"_{token}") or f"_{token}_" in scan_cycle_id:
            return token
    return "unknown"


def platform_from_state(state_label: str) -> str:
    if state_label == "ambient":
        return "ambient"
    return state_label.split("_", 1)[0]


def score_scan(rows: List[dict], stats: dict, features: List[str], scale: dict, observed_sids: set[str]) -> float:
    return sum(
        row_llr(row, stats, features, scale)
        for row in rows
        if subband_id(row) in observed_sids
    )


def train_score_rows(
    train_rows: List[dict],
    stats: dict,
    features: List[str],
    scale: dict,
    observed_sids: set[str],
) -> List[dict]:
    out = []
    for _, rows in grouped(train_rows).items():
        label = labels_for(rows)
        label["score"] = score_scan(rows, stats, features, scale, observed_sids)
        out.append(label)
    return out


def decision(score: float, confirm_th: float, reject_th: float) -> str:
    if score >= confirm_th:
        return "confirm_linked"
    if score <= reject_th:
        return "reject_negative"
    return "defer"


def observed_steps(rows: List[dict]) -> int:
    counts = defaultdict(int)
    for row in rows:
        key = row.get("receiver_id") or row.get("band")
        counts[key] += 1
    return max(counts.values()) if counts else 0


def read_timing(path: str) -> dict[str, dict]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return {row["timing_id"]: row for row in csv.DictReader(p.open())}


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        raise SystemExit("No scoring rows produced.")
    fields = [
        "subset",
        "feature_set",
        "round_id",
        "scan_cycle_id",
        "scan_kind",
        "platform",
        "state_label",
        "linked",
        "controller_only",
        "ambient",
        "observed_subbands",
        "observed_steps",
        "dwell_budget_s",
        "wall_clock_s",
        "receiver24_elapsed_s",
        "receiver58_elapsed_s",
        "receiver24_exit_status",
        "receiver58_exit_status",
        "score",
        "best_threshold",
        "binary_pred_linked",
        "constraint",
        "confirm_threshold",
        "reject_threshold",
        "constrained_decision",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    train_all = read_rows(Path(args.train_features_csv))
    eval_all = read_rows(Path(args.eval_features_csv))
    timing = read_timing(args.timing_csv)
    features = FEATURE_SETS[args.feature_set]

    out = []
    for subset in DEFAULT_SUBSETS:
        train_rows = subset_rows(train_all, subset)
        eval_rows = subset_rows(eval_all, subset)
        if not train_rows or not eval_rows:
            continue
        scale = scaler(train_rows, features)
        stats = class_stats(train_rows, features, scale)
        for scan_id, scan_rows in sorted(grouped(eval_rows).items()):
            observed_sids = {subband_id(row) for row in scan_rows}
            train_scores = train_score_rows(train_rows, stats, features, scale, observed_sids)
            if not train_scores:
                continue
            best_th = best_threshold(train_scores)
            confirm_th = constrained_confirm_threshold(train_scores, args.constraint)
            reject_th = constrained_reject_threshold(train_scores, args.constraint)
            score = score_scan(scan_rows, stats, features, scale, observed_sids)
            label = labels_for(scan_rows)
            kind = scan_kind(scan_id)
            time_row = timing.get(scan_id, {})
            steps = observed_steps(scan_rows)
            out.append({
                "subset": subset,
                "feature_set": args.feature_set,
                "round_id": label["round_id"],
                "scan_cycle_id": scan_id,
                "scan_kind": kind,
                "platform": platform_from_state(label["state_label"]),
                "state_label": label["state_label"],
                "linked": str(label["linked"]),
                "controller_only": str(label["controller_only"]),
                "ambient": str(label["ambient"]),
                "observed_subbands": str(len(observed_sids)),
                "observed_steps": str(steps),
                "dwell_budget_s": f"{steps * args.dwell_seconds:.3f}",
                "wall_clock_s": time_row.get("wall_clock_s", ""),
                "receiver24_elapsed_s": time_row.get("receiver24_elapsed_s", ""),
                "receiver58_elapsed_s": time_row.get("receiver58_elapsed_s", ""),
                "receiver24_exit_status": time_row.get("receiver24_exit_status", ""),
                "receiver58_exit_status": time_row.get("receiver58_exit_status", ""),
                "score": f"{score:.8g}",
                "best_threshold": f"{best_th:.8g}",
                "binary_pred_linked": str(1 if score >= best_th else 0),
                "constraint": f"{args.constraint:.2f}",
                "confirm_threshold": f"{confirm_th:.8g}",
                "reject_threshold": f"{reject_th:.8g}",
                "constrained_decision": decision(score, confirm_th, reject_th),
            })

    write_csv(Path(args.out_csv), out)
    print(f"[DONE] wrote {len(out)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()

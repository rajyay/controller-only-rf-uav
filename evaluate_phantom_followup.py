#!/usr/bin/env python3
"""Evaluate the phone-connected Phantom follow-up with LORO folds."""

import argparse
from pathlib import Path

from ranked_scan_model import evaluate_looro, read_rows, write_results


PHONE_STATES = {
    "phantom_controller_only_phone_connected",
    "phantom_linked_phone_connected",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--exclude-round", default="p01")
    parser.add_argument("--dwell-seconds", type=float, default=2.0)
    return parser.parse_args()


def evaluate(rows: list[dict], dwell_seconds: float) -> list[dict]:
    phone_rows = [row for row in rows if row["state_label"] in PHONE_STATES]
    if not phone_rows:
        raise SystemExit("No phone-connected Phantom rows found.")
    return evaluate_looro(
        rows=phone_rows,
        subset="phantom",
        feature_set="energy_spectral",
        dwell_seconds=dwell_seconds,
    )


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.features_csv))
    out_dir = Path(args.out_dir)

    write_results(
        out_dir / "phantom_phone_looro.csv",
        evaluate(rows, args.dwell_seconds),
    )

    sensitivity_rows = [
        row for row in rows if row["round_id"] != args.exclude_round
    ]
    write_results(
        out_dir / "phantom_phone_looro_without_p01.csv",
        evaluate(sensitivity_rows, args.dwell_seconds),
    )

    print(f"[DONE] wrote Phantom follow-up results under {out_dir}")


if __name__ == "__main__":
    main()

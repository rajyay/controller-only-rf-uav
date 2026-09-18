#!/usr/bin/env python3
"""Extract per-subband energy and spectral features from IQ captures."""

import argparse
import csv
import math
from pathlib import Path
from typing import List

import numpy as np
from scipy import signal


EPS = 1e-12


OUT_FIELDS = [
    "round_id",
    "scan_cycle_id",
    "session_id",
    "scan_id",
    "state_label",
    "linked_uav_target",
    "controller_only_target",
    "band",
    "receiver_id",
    "capture_index",
    "center_freq_hz",
    "sample_rate_hz",
    "duration_s",
    "iq_path",
    "iq_format",
    "loaded_seconds",
    "mean_power_db",
    "rms",
    "peak_abs",
    "dc_power_ratio",
    "psd_peak_db",
    "psd_median_db",
    "psd_peak_to_median_db",
    "spectral_entropy",
    "occupied_fraction_6db",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--max-seconds", type=float, default=0.25)
    ap.add_argument("--nperseg", type=int, default=4096)
    ap.add_argument("--strict", action="store_true",
                    help="Fail if any IQ file is missing or malformed.")
    return ap.parse_args()


def round_id(session_id: str) -> str:
    return session_id.split("_", 1)[0]


def scan_kind(session_id: str) -> str:
    for token in session_id.split("_"):
        if token in {"full20", "ranked8s", "ranked10s", "ranked12s"}:
            return token
    return ""


def scan_cycle_id(row: dict) -> str:
    kind = scan_kind(row["session_id"])
    if kind:
        return f"{round_id(row['session_id'])}_{row['state_label']}_{kind}"
    return f"{round_id(row['session_id'])}_{row['state_label']}"


def iq_format(row: dict) -> str:
    return row.get("iq_format") or "complex_float32_interleaved"


def load_iq(path: Path, fs: float, max_seconds: float, fmt: str) -> np.ndarray:
    count = int(round(fs * max_seconds)) * 2
    dtype = np.int16 if fmt == "complex_int16_interleaved" else np.float32
    raw = np.fromfile(path, dtype=dtype, count=count)
    if raw.size < 2:
        raise RuntimeError(f"No IQ samples in {path}")
    if raw.size % 2:
        raw = raw[:-1]
    if dtype == np.int16:
        raw = raw.astype(np.float32) / 32768.0
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64, copy=False)


def spectral_features(iq: np.ndarray, fs: float, nperseg: int) -> dict:
    nperseg = min(nperseg, len(iq))
    if nperseg < 16:
        raise RuntimeError("Too few samples for PSD features")
    _, pxx = signal.welch(
        iq - np.mean(iq),
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        nfft=nperseg,
        return_onesided=False,
        detrend=False,
        scaling="density",
    )
    pxx = np.fft.fftshift(np.real(pxx).astype(np.float64))
    pxx = np.maximum(pxx, EPS)
    p_db = 10.0 * np.log10(pxx)
    peak = float(np.max(p_db))
    median = float(np.median(p_db))
    probs = pxx / np.sum(pxx)
    entropy = float(-np.sum(probs * np.log(probs + EPS)) / math.log(len(probs)))
    occupied = float(np.mean(p_db >= peak - 6.0))
    return {
        "psd_peak_db": peak,
        "psd_median_db": median,
        "psd_peak_to_median_db": peak - median,
        "spectral_entropy": entropy,
        "occupied_fraction_6db": occupied,
    }


def row_features(row: dict, args) -> dict:
    fs = float(row["sample_rate_hz"])
    fmt = iq_format(row)
    iq = load_iq(Path(row["iq_path"]), fs, args.max_seconds, fmt)
    mag = np.abs(iq)
    mean_power = float(np.mean(mag ** 2))
    rms = float(np.sqrt(mean_power))
    peak_abs = float(np.max(mag))
    dc_power_ratio = float((abs(np.mean(iq)) ** 2) / max(mean_power, EPS))
    out = {
        "round_id": round_id(row["session_id"]),
        "scan_cycle_id": scan_cycle_id(row),
        "session_id": row.get("session_id", ""),
        "scan_id": row.get("scan_id", ""),
        "state_label": row.get("state_label", ""),
        "linked_uav_target": row.get("linked_uav_target", ""),
        "controller_only_target": row.get("controller_only_target", ""),
        "band": row.get("band", ""),
        "receiver_id": row.get("receiver_id", ""),
        "capture_index": row.get("capture_index", ""),
        "center_freq_hz": row.get("center_freq_hz", ""),
        "sample_rate_hz": row.get("sample_rate_hz", ""),
        "duration_s": row.get("duration_s", ""),
        "iq_path": row.get("iq_path", ""),
        "iq_format": fmt,
        "loaded_seconds": f"{min(args.max_seconds, len(iq) / fs):.6f}",
        "mean_power_db": f"{10.0 * math.log10(mean_power + EPS):.6f}",
        "rms": f"{rms:.8g}",
        "peak_abs": f"{peak_abs:.8g}",
        "dc_power_ratio": f"{dc_power_ratio:.8g}",
    }
    out.update({k: f"{v:.6f}" for k, v in spectral_features(iq, fs, args.nperseg).items()})
    return out


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(Path(args.metadata_csv).open()))
    if not rows:
        raise SystemExit("No manifest rows found.")

    out_rows = []
    skipped = 0
    for idx, row in enumerate(rows, start=1):
        try:
            feats = row_features(row, args)
        except Exception as exc:
            if args.strict:
                raise
            skipped += 1
            print(f"[SKIP {idx}/{len(rows)}] {row.get('iq_path', '')}: {exc}", flush=True)
            continue
        out_rows.append(feats)
        if idx % 25 == 0:
            print(f"[FEATURES] processed {idx}/{len(rows)}", flush=True)

    if not out_rows:
        raise SystemExit("No feature rows were generated.")
    write_csv(Path(args.out_csv), out_rows)
    print(f"[DONE] wrote {len(out_rows)} rows to {args.out_csv}; skipped={skipped}")


if __name__ == "__main__":
    main()

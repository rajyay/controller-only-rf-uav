#!/usr/bin/env python3
"""Capture one multi-frequency scan cycle with a USRP B210."""

import argparse
import csv
import fcntl
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List


CSV_FIELDS = [
    "session_id",
    "scan_id",
    "capture_index",
    "timestamp_utc",
    "state_label",
    "vendor",
    "platform",
    "link_state",
    "linked_uav_target",
    "controller_only_target",
    "expected_rf_architecture",
    "band",
    "center_freq_hz",
    "sample_rate_hz",
    "gain_db",
    "duration_s",
    "channel",
    "usrp_args",
    "antenna",
    "antenna_gain_dbi",
    "receiver",
    "receiver_id",
    "capture_host",
    "iq_format",
    "iq_path",
    "capture_stdout_path",
    "capture_stderr_path",
    "notes",
]


STATES = {
    "ambient": {
        "vendor": "ambient",
        "platform": "ambient",
        "link_state": "ambient",
        "linked_uav_target": "0",
        "controller_only_target": "0",
        "expected_rf_architecture": "ambient",
    },
    "phantom_controller_only": {
        "vendor": "dji",
        "platform": "phantom3_4k",
        "link_state": "controller_only",
        "linked_uav_target": "0",
        "controller_only_target": "1",
        "expected_rf_architecture": "split_link_expected",
    },
    "phantom_linked": {
        "vendor": "dji",
        "platform": "phantom3_4k",
        "link_state": "linked",
        "linked_uav_target": "1",
        "controller_only_target": "0",
        "expected_rf_architecture": "split_link_expected",
    },
    "phantom_controller_only_phone_connected": {
        "vendor": "dji",
        "platform": "phantom3_4k",
        "link_state": "controller_only_phone_connected",
        "linked_uav_target": "0",
        "controller_only_target": "1",
        "expected_rf_architecture": "split_link_phone_connected",
    },
    "phantom_linked_phone_connected": {
        "vendor": "dji",
        "platform": "phantom3_4k",
        "link_state": "linked_phone_connected_video_active",
        "linked_uav_target": "1",
        "controller_only_target": "0",
        "expected_rf_architecture": "split_link_phone_connected",
    },
    "hubsan_controller_only": {
        "vendor": "hubsan",
        "platform": "h501s",
        "link_state": "controller_only",
        "linked_uav_target": "0",
        "controller_only_target": "1",
        "expected_rf_architecture": "split_link_expected",
    },
    "hubsan_linked": {
        "vendor": "hubsan",
        "platform": "h501s",
        "link_state": "linked",
        "linked_uav_target": "1",
        "controller_only_target": "0",
        "expected_rf_architecture": "split_link_expected",
    },
    "mavic_controller_only": {
        "vendor": "dji",
        "platform": "mavic_mini_mr1ss5",
        "link_state": "controller_only",
        "linked_uav_target": "0",
        "controller_only_target": "1",
        "expected_rf_architecture": "same_band_expected",
    },
    "mavic_linked": {
        "vendor": "dji",
        "platform": "mavic_mini_mr1ss5",
        "link_state": "linked",
        "linked_uav_target": "1",
        "controller_only_target": "0",
        "expected_rf_architecture": "same_band_expected",
    },
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--metadata-csv", required=True, type=str)
    ap.add_argument("--state-label", required=True, choices=sorted(STATES))
    ap.add_argument("--session-id", required=True, type=str,
                    help="Stable round/state id, e.g. r01_mavic_linked.")
    ap.add_argument("--freq", action="append", type=float, default=None,
                    help="Center frequency in Hz. Repeat for each subband.")
    ap.add_argument("--freq-grid", type=str, default=None,
                    help="CSV with center_freq_hz column. Used if --freq is omitted.")
    ap.add_argument("--rate", type=float, default=20e6)
    ap.add_argument("--gain", type=float, default=25.0)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--usrp-args", type=str, default=None)
    ap.add_argument("--stream-args", type=str, default=None,
                    help="Additional UHD/GNU Radio stream args, e.g. spp=2000.")
    ap.add_argument("--channel", type=int, default=None)
    ap.add_argument("--antenna", type=str, default="VERT2450")
    ap.add_argument("--antenna-gain-dbi", type=float, default=3.0)
    ap.add_argument("--receiver", type=str, default="USRP B210")
    ap.add_argument("--receiver-id", type=str, default="",
                    help="Stable receiver label, e.g. b210a_24 or b210b_58.")
    ap.add_argument("--output-shorts", action="store_true",
                    help="Write interleaved int16 IQ instead of complex float32.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Allow overwriting existing IQ/log files for this session.")
    ap.add_argument("--retry-on-runtime-issue", type=int, default=0,
                    help="Retry a dwell this many times if UHD reports overflow/underflow.")
    ap.add_argument("--retry-delay-seconds", type=float, default=0.5,
                    help="Delay between overflow/underflow retry attempts.")
    ap.add_argument("--keep-bad-attempts", action="store_true",
                    help="Keep IQ files from failed retry attempts instead of deleting them.")
    ap.add_argument("--notes", type=str, default="")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def infer_band(freq_hz: float) -> str:
    if 2.3e9 <= freq_hz <= 2.5e9:
        return "2.4"
    if 5.6e9 <= freq_hz <= 5.9e9:
        return "5.8"
    return "other"


def load_freqs(args) -> List[float]:
    if args.freq:
        return args.freq
    if not args.freq_grid:
        raise ValueError("Provide either --freq or --freq-grid.")
    with Path(args.freq_grid).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "center_freq_hz" not in rows[0]:
        raise ValueError("--freq-grid must contain a center_freq_hz column.")
    return [float(r["center_freq_hz"]) for r in rows]


def run_capture(
    out_path: Path,
    freq: float,
    rate: float,
    gain: float,
    seconds: float,
    usrp_args: str | None,
    stream_args: str | None,
    channel: int | None,
    output_shorts: bool,
) -> subprocess.CompletedProcess[str]:
    num_samps = int(round(rate * seconds))
    if num_samps <= 0:
        raise ValueError("seconds too small; num_samps <= 0")

    cmd: List[str] = [
        "uhd_rx_cfile",
        "--freq", str(freq),
        "-r", str(rate),
        "--gain", str(gain),
        "-N", str(num_samps),
        str(out_path),
    ]
    if output_shorts:
        cmd.insert(-1, "-s")
    if stream_args:
        cmd[1:1] = ["--stream-args", stream_args]
    if usrp_args:
        cmd[1:1] = ["--args", usrp_args]
    if channel is not None:
        cmd[1:1] = ["--channel", str(channel)]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print(f"[WARN] uhd_rx_cfile failed with returncode={proc.returncode}.", flush=True)
        return proc
    stderr_lines = [line.strip() for line in proc.stderr.splitlines()]
    if "overflow" in proc.stderr.lower() or any(line == "O" for line in stderr_lines):
        print("[WARN] UHD stderr may indicate overflow. Inspect capture quality.")
    if "underflow" in proc.stderr.lower() or any(line == "U" for line in stderr_lines):
        print("[WARN] UHD stderr may indicate underflow. Inspect capture quality.")
    return proc


def runtime_issue(proc: subprocess.CompletedProcess[str]) -> bool:
    text = f"{proc.stdout}\n{proc.stderr}".lower()
    stderr_lines = [line.strip() for line in proc.stderr.splitlines()]
    stdout_lines = [line.strip() for line in proc.stdout.splitlines()]
    return (
        proc.returncode != 0
        or
        "overflow" in text
        or "underflow" in text
        or any(line in {"O", "U"} for line in stderr_lines + stdout_lines)
    )


def raise_capture_failure(
    proc: subprocess.CompletedProcess[str],
    freq_mhz: float,
    attempts: int,
) -> None:
    raise RuntimeError(
        f"uhd_rx_cfile failed after {attempts} attempts at {freq_mhz:.3f} MHz.\n"
        f"Return code: {proc.returncode}\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}\n"
    )


def store_bad_attempt(
    out_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    proc: subprocess.CompletedProcess[str],
    attempt: int,
    keep_iq: bool,
) -> None:
    bad_base = out_path.with_suffix(out_path.suffix + f".attempt{attempt:02d}.bad")
    bad_stdout = bad_base.with_suffix(bad_base.suffix + ".stdout.txt")
    bad_stderr = bad_base.with_suffix(bad_base.suffix + ".stderr.txt")
    bad_stdout.write_text(proc.stdout)
    bad_stderr.write_text(proc.stderr)

    if out_path.exists():
        if keep_iq:
            out_path.replace(bad_base)
        else:
            out_path.unlink()
    for path in [stdout_path, stderr_path]:
        if path.exists():
            path.unlink()


def append_rows(csv_path: Path, rows: List[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0, 2)
        write_header = f.tell() == 0
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main():
    args = parse_args()
    state = STATES[args.state_label]
    freqs = load_freqs(args)

    scan_id = f"{args.session_id}_{args.state_label}"
    scan_dir = Path(args.out_dir) / args.session_id / scan_id
    scan_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv = Path(args.metadata_csv)
    host = platform.node()

    print(f"[SCAN] session_id={args.session_id}")
    print(f"[SCAN] scan_id={scan_id}")
    print(f"[SCAN] state={args.state_label} linked_uav_target={state['linked_uav_target']}")
    print(f"[CONFIG] freqs={len(freqs)} rate={args.rate:.0f} gain={args.gain} seconds={args.seconds}")

    rows = []
    for idx, freq in enumerate(freqs, start=1):
        timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        freq_mhz = freq / 1e6
        filename = f"{idx:02d}_{freq_mhz:.3f}MHz.cfile"
        out_path = scan_dir / filename
        print(f"[CAPTURE {idx:02d}/{len(freqs)}] {freq_mhz:.3f} MHz -> {out_path}")

        stdout_path = out_path.with_suffix(out_path.suffix + ".stdout.txt")
        stderr_path = out_path.with_suffix(out_path.suffix + ".stderr.txt")
        if not args.dry_run:
            if not args.overwrite:
                existing = [p for p in [out_path, stdout_path, stderr_path] if p.exists()]
                if existing:
                    names = "\n".join(str(p) for p in existing)
                    raise RuntimeError(
                        "Refusing to overwrite existing capture files. "
                        "Use a new session-id, clean raw_iq, or pass --overwrite.\n"
                        f"Existing files:\n{names}"
                    )
            max_attempts = 1 + max(args.retry_on_runtime_issue, 0)
            proc = None
            for attempt in range(1, max_attempts + 1):
                if attempt > 1:
                    print(
                        f"[RETRY {attempt}/{max_attempts}] {freq_mhz:.3f} MHz after UHD runtime issue",
                        flush=True,
                    )
                proc = run_capture(
                    out_path=out_path,
                    freq=freq,
                    rate=args.rate,
                    gain=args.gain,
                    seconds=args.seconds,
                    usrp_args=args.usrp_args,
                    stream_args=args.stream_args,
                    channel=args.channel,
                    output_shorts=args.output_shorts,
                )
                if not runtime_issue(proc):
                    break
                if attempt < max_attempts:
                    store_bad_attempt(
                        out_path=out_path,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        proc=proc,
                        attempt=attempt,
                        keep_iq=args.keep_bad_attempts,
                    )
                    time.sleep(max(args.retry_delay_seconds, 0.0))
            assert proc is not None
            stdout_path.write_text(proc.stdout)
            stderr_path.write_text(proc.stderr)
            if proc.returncode != 0:
                raise_capture_failure(proc, freq_mhz, max_attempts)

        row = {
            "session_id": args.session_id,
            "scan_id": scan_id,
            "capture_index": str(idx),
            "timestamp_utc": timestamp_utc,
            "state_label": args.state_label,
            "vendor": state["vendor"],
            "platform": state["platform"],
            "link_state": state["link_state"],
            "linked_uav_target": state["linked_uav_target"],
            "controller_only_target": state["controller_only_target"],
            "expected_rf_architecture": state["expected_rf_architecture"],
            "band": infer_band(freq),
            "center_freq_hz": f"{freq:.0f}",
            "sample_rate_hz": f"{args.rate:.0f}",
            "gain_db": f"{args.gain:.2f}",
            "duration_s": f"{args.seconds:.3f}",
            "channel": "" if args.channel is None else str(args.channel),
            "usrp_args": "" if args.usrp_args is None else args.usrp_args,
            "antenna": args.antenna,
            "antenna_gain_dbi": f"{args.antenna_gain_dbi:.2f}",
            "receiver": args.receiver,
            "receiver_id": args.receiver_id,
            "capture_host": host,
            "iq_format": "complex_int16_interleaved" if args.output_shorts else "complex_float32_interleaved",
            "iq_path": str(out_path),
            "capture_stdout_path": "" if args.dry_run else str(stdout_path),
            "capture_stderr_path": "" if args.dry_run else str(stderr_path),
            "notes": args.notes,
        }
        rows.append(row)

    append_rows(metadata_csv, rows)
    print(f"[DONE] appended {len(rows)} rows to {metadata_csv}")


if __name__ == "__main__":
    main()

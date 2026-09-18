#!/usr/bin/env python3
"""Run simultaneous 2.4/5.8 GHz captures and record wall-clock timing."""

import argparse
import csv
import fcntl
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List


DEFAULT_RECEIVER_24 = "b210a_24"
DEFAULT_RECEIVER_58 = "b210b_58"

TIMING_FIELDS = [
    "timing_id",
    "timestamp_utc",
    "session_prefix",
    "state_label",
    "scan_kind",
    "wall_clock_s",
    "receiver24_elapsed_s",
    "receiver58_elapsed_s",
    "receiver24_exit_status",
    "receiver58_exit_status",
    "grid24",
    "grid58",
    "out_dir",
    "metadata_csv",
    "rate_hz",
    "gain_db",
    "duration_s",
    "stream_args",
    "retry_on_runtime_issue",
    "capture_host",
    "notes",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--metadata-csv", required=True)
    ap.add_argument("--timing-csv", required=True)
    ap.add_argument("--session-prefix", required=True,
                    help="Prefix without receiver id, e.g. hil01_hubsan_linked.")
    ap.add_argument("--state-label", required=True)
    ap.add_argument("--scan-kind", required=True,
                    help="Label such as full20 or ranked8s.")
    ap.add_argument("--grid-24", required=True)
    ap.add_argument("--grid-58", required=True)
    ap.add_argument("--serial-24", required=True)
    ap.add_argument("--serial-58", required=True)
    ap.add_argument("--receiver-id-24", default=DEFAULT_RECEIVER_24)
    ap.add_argument("--receiver-id-58", default=DEFAULT_RECEIVER_58)
    ap.add_argument("--rate", type=int, default=20_000_000)
    ap.add_argument("--gain", type=float, default=25.0)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--stream-args", default="spp=2000")
    ap.add_argument("--retry-on-runtime-issue", type=int, default=5)
    ap.add_argument("--retry-delay-seconds", type=float, default=0.5)
    ap.add_argument("--output-shorts", action="store_true", default=True)
    ap.add_argument("--notes", default="")
    return ap.parse_args()


def scan_cmd(
    args: argparse.Namespace,
    receiver_id: str,
    serial: str,
    grid: str,
) -> List[str]:
    session_id = f"{args.session_prefix}_{args.scan_kind}_{receiver_id}"
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("capture_scan.py")),
        "--out-dir", args.out_dir,
        "--metadata-csv", args.metadata_csv,
        "--session-id", session_id,
        "--state-label", args.state_label,
        "--freq-grid", grid,
        "--seconds", f"{args.seconds:g}",
        "--rate", str(args.rate),
        "--gain", f"{args.gain:g}",
        "--usrp-args", f"serial={serial}",
        "--stream-args", args.stream_args,
        "--receiver-id", receiver_id,
        "--retry-on-runtime-issue", str(args.retry_on_runtime_issue),
        "--retry-delay-seconds", f"{args.retry_delay_seconds:g}",
        "--notes", args.notes,
    ]
    if args.output_shorts:
        cmd.append("--output-shorts")
    return cmd


def launch(cmd: List[str]) -> tuple[subprocess.Popen[str], float]:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return proc, time.monotonic()


def collect(name: str, proc: subprocess.Popen[str], start: float, output: dict) -> None:
    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        text = line.rstrip()
        print(f"[{name}] {text}", flush=True)
        lines.append(line)
    status = proc.wait()
    elapsed = time.monotonic() - start
    output[name] = {
        "status": status,
        "elapsed": elapsed,
        "text": "".join(lines),
    }


def append_timing(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0, 2)
        write_header = f.tell() == 0
        writer = csv.DictWriter(f, fieldnames=TIMING_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main() -> None:
    args = parse_args()
    timing_id = f"{args.session_prefix}_{args.scan_kind}"
    cmd24 = scan_cmd(args, args.receiver_id_24, args.serial_24, args.grid_24)
    cmd58 = scan_cmd(args, args.receiver_id_58, args.serial_58, args.grid_58)

    print(f"[DUAL] timing_id={timing_id}")
    print(f"[DUAL] state={args.state_label} scan_kind={args.scan_kind}")
    print(f"[DUAL] grid24={args.grid_24}")
    print(f"[DUAL] grid58={args.grid_58}")

    wall_start = time.monotonic()
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    proc24, start24 = launch(cmd24)
    proc58, start58 = launch(cmd58)

    outputs = {}
    thread24 = threading.Thread(target=collect, args=(args.receiver_id_24, proc24, start24, outputs))
    thread58 = threading.Thread(target=collect, args=(args.receiver_id_58, proc58, start58, outputs))
    thread24.start()
    thread58.start()
    thread24.join()
    thread58.join()
    wall_elapsed = time.monotonic() - wall_start
    status24 = int(outputs[args.receiver_id_24]["status"])
    status58 = int(outputs[args.receiver_id_58]["status"])
    elapsed24 = float(outputs[args.receiver_id_24]["elapsed"])
    elapsed58 = float(outputs[args.receiver_id_58]["elapsed"])

    row = {
        "timing_id": timing_id,
        "timestamp_utc": timestamp_utc,
        "session_prefix": args.session_prefix,
        "state_label": args.state_label,
        "scan_kind": args.scan_kind,
        "wall_clock_s": f"{wall_elapsed:.3f}",
        "receiver24_elapsed_s": f"{elapsed24:.3f}",
        "receiver58_elapsed_s": f"{elapsed58:.3f}",
        "receiver24_exit_status": str(status24),
        "receiver58_exit_status": str(status58),
        "grid24": args.grid_24,
        "grid58": args.grid_58,
        "out_dir": args.out_dir,
        "metadata_csv": args.metadata_csv,
        "rate_hz": str(args.rate),
        "gain_db": f"{args.gain:g}",
        "duration_s": f"{args.seconds:g}",
        "stream_args": args.stream_args,
        "retry_on_runtime_issue": str(args.retry_on_runtime_issue),
        "capture_host": platform.node(),
        "notes": args.notes,
    }
    append_timing(Path(args.timing_csv), row)

    print(f"[DUAL DONE] wall_clock_s={wall_elapsed:.3f}")
    print(f"[DUAL DONE] receiver24_status={status24} receiver58_status={status58}")
    print(f"[DUAL DONE] appended timing to {args.timing_csv}")
    if status24 != 0 or status58 != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

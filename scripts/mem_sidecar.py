#!/usr/bin/env python3
"""
mem_sidecar.py — Continuous 1-second memory monitor for OOM diagnosis.

Writes one JSON line per sample to an output file so that peak memory
is captured regardless of *when* during a long operation the spike occurs.
The existing mem_sidecar.sh is too coarse (30s) and unstructured (free text).

Usage:
    # Watch a specific process (e.g. the isolation test):
    python scripts/mem_sidecar.py --target-pid 12345 --out /tmp/mem.jsonl

    # System-only mode (no target pid), used when the sidecar is launched
    # before the target process starts and you will supply the pid later:
    python scripts/mem_sidecar.py --system-only --out /tmp/mem.jsonl

    # Mark epochs in the output so you can split by test case:
    python scripts/mem_sidecar.py --target-pid 12345 --out /tmp/mem.jsonl \\
        --epoch-pipe /tmp/epoch.pipe

Each JSON line contains:
    t           : float  — epoch seconds (time.time())
    epoch       : str    — current epoch label (set via --epoch-pipe, or "default")
    mem_avail   : int    — MemAvailable kB  ← the key leading indicator
    mem_free    : int    — MemFree kB
    mem_used    : int    — MemTotal - MemAvailable kB  (real pressure)
    mem_total   : int    — MemTotal kB
    anon_pages  : int    — AnonPages kB
    cached      : int    — Cached kB
    slab        : int    — Slab kB
    proc_vmrss  : int    — target pid VmRSS kB  (0 if system-only)
    proc_rssanon: int    — target pid RssAnon kB
    proc_vmsize : int    — target pid VmSize kB
    proc_threads: int    — target pid Threads
    proc_pss    : int    — target pid Pss kB from smaps_rollup (0 if unavailable)
    proc_priv_anon: int  — target pid Private_Anon kB from smaps_rollup

Additionally prints a one-line human summary to stderr every --print-every samples
so you can eyeball it in the tmux pane without reading the JSONL.

Exits cleanly on SIGTERM, SIGINT, or if the target pid disappears.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional


# ── /proc parsing helpers ─────────────────────────────────────────────────────

def _parse_kv_file(path: str) -> dict[str, int]:
    """Parse a file of 'Key: N kB' lines (e.g. /proc/meminfo, /proc/PID/status)."""
    result: dict[str, int] = {}
    try:
        with open(path) as f:
            for line in f:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts:
                    try:
                        result[key.strip()] = int(parts[0])
                    except ValueError:
                        pass
    except (OSError, IOError):
        pass
    return result


def _parse_smaps_rollup(pid: int) -> dict[str, int]:
    """Parse /proc/<pid>/smaps_rollup for Pss and Private_Anon (kB)."""
    return _parse_kv_file(f"/proc/{pid}/smaps_rollup")


def read_meminfo() -> dict[str, int]:
    return _parse_kv_file("/proc/meminfo")


def read_proc_status(pid: int) -> dict[str, int]:
    return _parse_kv_file(f"/proc/{pid}/status")


def pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


# ── epoch pipe ────────────────────────────────────────────────────────────────

def _open_epoch_pipe(path: str) -> Optional[int]:
    """
    Open a named pipe (FIFO) for reading epoch labels.
    Returns the file descriptor, or None if the pipe can't be opened.
    """
    try:
        if not os.path.exists(path):
            os.mkfifo(path)
        # O_NONBLOCK so we don't block waiting for a writer
        return os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return None


def _read_epoch_label(fd: int, current: str) -> str:
    """Try to read a new epoch label from the pipe; return current if nothing new."""
    if fd is None:
        return current
    try:
        data = os.read(fd, 256)
        if data:
            return data.decode().strip() or current
    except BlockingIOError:
        pass
    except OSError:
        pass
    return current


# ── main loop ─────────────────────────────────────────────────────────────────

_RUNNING = True


def _handle_signal(signum, frame):
    global _RUNNING
    _RUNNING = False


def main() -> None:
    global _RUNNING

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-pid", type=int, default=None,
                        help="PID to monitor (omit for system-only mode)")
    parser.add_argument("--system-only", action="store_true",
                        help="Skip per-process /proc/<pid> reads")
    parser.add_argument("--out", required=True,
                        help="Path to output JSONL file (appended, not overwritten)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Polling interval in seconds (default: 1.0)")
    parser.add_argument("--print-every", type=int, default=10,
                        help="Print one-line status to stderr every N samples")
    parser.add_argument("--epoch", default="default",
                        help="Initial epoch label for this run")
    parser.add_argument("--epoch-pipe", default=None,
                        help="Path to named pipe for dynamic epoch label updates")
    args = parser.parse_args()

    pid: Optional[int] = args.target_pid
    system_only: bool = args.system_only or (pid is None)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    epoch_fd = _open_epoch_pipe(args.epoch_pipe) if args.epoch_pipe else None
    current_epoch = args.epoch

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sample_n = 0
    print(f"[mem_sidecar] started  pid={os.getpid()}  target={pid}  out={out_path}",
          file=sys.stderr)

    with open(out_path, "a") as out_f:
        while _RUNNING:
            t0 = time.time()

            # Check target still exists
            if pid is not None and not pid_exists(pid):
                print(f"[mem_sidecar] target pid {pid} gone — exiting", file=sys.stderr)
                break

            # Check for epoch update via pipe
            if epoch_fd is not None:
                current_epoch = _read_epoch_label(epoch_fd, current_epoch)

            # System memory
            mi = read_meminfo()
            mem_total = mi.get("MemTotal", 0)
            mem_free = mi.get("MemFree", 0)
            mem_avail = mi.get("MemAvailable", 0)
            mem_used = mem_total - mem_avail  # real pressure (excl reclaimable cache)
            anon_pages = mi.get("AnonPages", 0)
            cached = mi.get("Cached", 0)
            slab = mi.get("Slab", 0)

            # Per-process
            proc_vmrss = proc_rssanon = proc_vmsize = proc_threads = 0
            proc_pss = proc_priv_anon = 0
            if pid is not None and not system_only:
                ps = read_proc_status(pid)
                proc_vmrss = ps.get("VmRSS", 0)
                proc_rssanon = ps.get("RssAnon", 0)
                proc_vmsize = ps.get("VmSize", 0)
                proc_threads = ps.get("Threads", 0)

                sr = _parse_smaps_rollup(pid)
                proc_pss = sr.get("Pss", 0)
                proc_priv_anon = sr.get("Private_Anon", 0)

            record = {
                "t": round(t0, 3),
                "epoch": current_epoch,
                "mem_avail_kb": mem_avail,
                "mem_free_kb": mem_free,
                "mem_used_kb": mem_used,
                "mem_total_kb": mem_total,
                "anon_pages_kb": anon_pages,
                "cached_kb": cached,
                "slab_kb": slab,
                "proc_vmrss_kb": proc_vmrss,
                "proc_rssanon_kb": proc_rssanon,
                "proc_vmsize_kb": proc_vmsize,
                "proc_threads": proc_threads,
                "proc_pss_kb": proc_pss,
                "proc_priv_anon_kb": proc_priv_anon,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            sample_n += 1
            if sample_n % args.print_every == 0:
                avail_gib = mem_avail / (1024 * 1024)
                used_gib = mem_used / (1024 * 1024)
                rss_gib = proc_vmrss / (1024 * 1024)
                pss_gib = proc_pss / (1024 * 1024)
                print(
                    f"[mem_sidecar +{sample_n:5d}s] "
                    f"epoch={current_epoch}  "
                    f"used={used_gib:.1f}GiB  avail={avail_gib:.1f}GiB  "
                    f"anon={anon_pages/(1024*1024):.1f}GiB  "
                    f"proc_rss={rss_gib:.1f}GiB  proc_pss={pss_gib:.1f}GiB",
                    file=sys.stderr,
                )

            # Sleep for the remainder of the interval
            elapsed = time.time() - t0
            sleep_time = max(0.0, args.interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    if epoch_fd is not None:
        try:
            os.close(epoch_fd)
        except OSError:
            pass
    print(f"[mem_sidecar] done  samples={sample_n}", file=sys.stderr)


if __name__ == "__main__":
    main()

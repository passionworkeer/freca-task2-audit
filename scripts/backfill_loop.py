"""Backfill driver: run remaining method/case gaps until quota (429) stops us.

Idempotent for single-shot methods (run_case.py skips already-valid units).
verify_audit is NOT idempotent -> its case list excludes already-run cases.
Stops on first PROBE_429 (account-tier quota; backoff won't help - wait window).
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

PY = r"D:/Data/Desktop/contest (2)/contest/contest/Task2/.venv/Scripts/python.exe"
ROOT = Path(r"D:/Data/Desktop/contest (2)/contest/contest/Task2/.worktrees/feature-direct-llm-experiments")
LOG = ROOT / "build" / "logs" / "backfill.log"
ENV = {**os.environ, "PYTHONPATH": "src"}

# cheap -> expensive; samples (not full 100) for the 4 slow methods — stage_audit
# is 53min/case so full 100 is infeasible; ~10 representative cases suffice for
# multi-method consensus. Done cases skip idempotently. verify_audit not idempotent.
PLAN = [
    ("element_full", list(range(91, 101))),           # already complete; idempotent skip
    ("stage_audit", list(range(4, 14))),              # sample: cases 4–13
    ("agent_audit", list(range(4, 14))),              # sample: cases 4–13
    ("verify_audit", [3] + list(range(4, 14))),       # sample: case 3 + 4–13
    ("checkpoint_full", list(range(5, 15))),          # sample: cases 5–14
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def probe() -> tuple[str, bool]:
    """Probe quota; retry transient blips (NET_ERROR / 5XX / timeout) up to 3×,
    bail immediately on 429/4XX (real quota stop — wait for daily reset)."""
    out = ""
    for attempt in range(3):
        try:
            r = subprocess.run(
                [PY, "scripts/probe_quota.py"], cwd=str(ROOT),
                capture_output=True, text=True, env=ENV, timeout=90,
            )
        except subprocess.TimeoutExpired:
            out = "PROBE_TIMEOUT >90s"
        else:
            out = (r.stdout or "").strip()
        if out.startswith("PROBE_OK"):
            return out, True
        if out.startswith("PROBE_429") or out.startswith("PROBE_4XX"):
            return out, False  # real quota stop, do not retry
        if attempt < 2:  # NET_ERROR / 5XX / timeout — transient, retry
            time.sleep(15)
            continue
    return out, False


def run_case(case_id: int, method: str) -> None:
    cmd = [PY, "scripts/run_case.py", "--case-id", str(case_id),
           "--methods", method, "--inter-call-delay", "1.0"]
    log(f"RUN {method} case-{case_id:03d} start")
    t0 = time.monotonic()
    r = subprocess.run(cmd, cwd=str(ROOT), env=ENV, capture_output=True, text=True)
    dt = time.monotonic() - t0
    log(f"RUN {method} case-{case_id:03d} done rc={r.returncode} elapsed={dt:.0f}s")
    for line in reversed((r.stdout or "").splitlines()):
        if '"summary"' in line:
            log("  summary: " + line[:500])
            break
    if r.returncode != 0:
        log("  STDERR tail: " + (r.stderr or "")[-500:])


def main() -> int:
    log("=== backfill start ===")
    total = 0
    for method, cases in PLAN:
        for cid in cases:
            out, ok = probe()
            if not ok:
                log(f"STOP probe={out}  (completed {total} case-methods this window)")
                log("=== quota window likely exhausted; re-run after reset ===")
                return 0
            run_case(cid, method)
            total += 1
    log(f"=== PLAN exhausted; ran {total} case-methods ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

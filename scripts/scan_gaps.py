"""Scan build/experiments and report real per-method/case completion.

Reads result.json files on disk (summary.json is unreliable - it's overwritten
per-run at the method level). Used by the hourly loop to find incomplete work.

Layouts:
- single-shot (case_full/element_full/checkpoint_full/automatic_retrieval):
    <method>/case-NNN/track3-raw/unit-NNN/result.json   (done if valid && verdicts)
- plan-runner (stage_audit/agent_audit):
    <method>/case-NNN/track3-raw/cp-NNN/result.json      (done if valid)
- verify_audit:
    <method>/case-NNN/track3-raw/case-NNN-unit-000/result.json (base)
    <method>/case-NNN/track3-raw/case-NNN-unit-000/verify-cp-NNN/result.json

Expected unit counts per case (from case-001 baseline):
    case_full=1, element_full=4, checkpoint_full=41, automatic_retrieval=41,
    stage_audit=41, agent_audit=41, verify_audit=42 (1 base + 41 verify-cp)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EXPECTED = {
    "case_full": ("unit", 1),
    "element_full": ("unit", 4),
    "checkpoint_full": ("unit", 41),
    "automatic_retrieval": ("unit", 41),
    "stage_audit": ("cp", 41),
    "agent_audit": ("cp", 41),
    "verify_audit": ("verify", 42),
}


def _load_valid(path: Path) -> bool:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(d, dict):
        return False
    if not d.get("valid"):
        return False
    # single-shot units need non-empty verdicts to count as done
    if "verdicts" in d and not d.get("verdicts"):
        return False
    return True


def scan_method(root: Path, method: str):
    kind, expected = EXPECTED[method]
    mdir = root / method
    if not mdir.is_dir():
        return {"method": method, "cases": [], "total_cases": 0, "complete": 0, "partial": 0, "missing_units": 0}
    cases = []
    for case_dir in sorted(mdir.glob("case-*")):
        try:
            case_id = int(case_dir.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        track = case_dir / "track3-raw"
        done = 0
        if kind == "verify":
            unit_dir = track / f"case-{case_id:03d}-unit-000"
            base = unit_dir / "result.json"
            if _load_valid(base):
                done += 1
            for cp_dir in sorted(unit_dir.glob("verify-cp-*")):
                if _load_valid(cp_dir / "result.json"):
                    done += 1
        else:
            prefix = kind  # "unit" or "cp"
            for d in sorted(track.glob(f"{prefix}-*")):
                if d.is_dir() and _load_valid(d / "result.json"):
                    done += 1
        missing = max(0, expected - done)
        cases.append({"case_id": case_id, "done": done, "expected": expected, "missing": missing})
    complete = sum(1 for c in cases if c["missing"] == 0)
    partial = len(cases) - complete
    missing_units = sum(c["missing"] for c in cases)
    return {
        "method": method,
        "cases": cases,
        "total_cases": len(cases),
        "complete": complete,
        "partial": partial,
        "missing_units": missing_units,
        "expected_per_case": expected,
    }


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/experiments")
    print(f"# gap scan: {root}")
    grand_missing = 0
    grand_cases_done = 0
    grand_cases_target = 0
    work_items = []  # (method, case_id, missing) for incomplete cases
    for method in EXPECTED:
        r = scan_method(root, method)
        exp = r["expected_per_case"]
        target = 100
        done_cases = r["complete"]
        grand_cases_done += done_cases
        grand_cases_target += target
        grand_missing += r["missing_units"]
        print(
            f"{method:22s} cases={r['total_cases']:3d} complete={done_cases:3d} "
            f"partial={r['partial']:3d} missing_units={r['missing_units']:4d} "
            f"(target {target} cases x {exp} units)"
        )
        for c in r["cases"]:
            if c["missing"] > 0:
                work_items.append((method, c["case_id"], c["missing"], c["done"], c["expected"]))
    # next cases not yet started (case dirs missing) per method - extend toward 100
    print()
    print("## next unstarted case per method (extend toward 100):")
    for method in EXPECTED:
        mdir = root / method
        started = set()
        if mdir.is_dir():
            for cd in mdir.glob("case-*"):
                try:
                    started.add(int(cd.name.split("-")[1]))
                except Exception:
                    pass
        nxt = None
        for i in range(1, 101):
            if i not in started:
                nxt = i
                break
        print(f"  {method:22s} next_unstarted=case-{nxt:03d}" if nxt else f"  {method:22s} all 100 started")
    print()
    print(f"## incomplete cases (started but missing units): {len(work_items)}")
    work_items.sort(key=lambda x: (x[0], x[1]))
    for method, case_id, missing, done, exp in work_items[:60]:
        print(f"  {method:22s} case-{case_id:03d} done={done}/{exp} missing={missing}")
    print()
    print(f"TOTAL: cases_complete={grand_cases_done}/{grand_cases_target}  missing_units={grand_missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

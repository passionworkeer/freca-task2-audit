"""Compare per-CP verdicts across methods for given case ids.

Usage:
    PYTHONPATH=src ../../.venv/Scripts/python.exe scripts/compare_methods.py 23 35 38 65 74

For each case, loads every method that has results, builds CP -> verdict (plus
reason/citations), prints a per-case disagreement report, and dumps a JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

METHODS = [
    "case_full",
    "element_full",
    "checkpoint_full",
    "automatic_retrieval",
    "stage_audit",
    "agent_audit",
    "verify_audit",
]
ROOT = Path("build/experiments")


def load_case_verdicts(method: str, case_id: int) -> dict[str, dict]:
    """Return {cp_id: {verdict, reason, citation_ids, unit}} for one method/case."""
    case_dir = ROOT / method / f"case-{case_id:03d}" / "track3-raw"
    if not case_dir.is_dir():
        return {}
    out: dict[str, dict] = {}
    for rf in sorted(case_dir.rglob("result.json")):
        # unit dir name (unit-NNN / cp-NNN / verify-cp-NNN)
        unit = rf.parent.name
        try:
            d = json.loads(rf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not d.get("valid"):
            continue
        for v in d.get("verdicts", []):
            cp = v.get("cp_id")
            if not cp:
                continue
            out.setdefault(cp, {
                "verdict": v.get("verdict"),
                "reason": v.get("reason", ""),
                "citation_ids": v.get("citation_ids", []),
                "unit": unit,
            })
    return out


def main(case_ids: list[int]) -> int:
    cps = {c["cp_id"]: c for c in json.loads(Path("build/parsed/checkpoints.json").read_text(encoding="utf-8"))}
    cp_order = [f"CP{i}" for i in range(1, 42)]
    report = {}
    for cid in case_ids:
        per_method = {m: load_case_verdicts(m, cid) for m in METHODS}
        per_method = {m: v for m, v in per_method.items() if v}
        report[cid] = per_method
        present = list(per_method.keys())
        print(f"\n{'='*90}\nCASE {cid:03d}  | methods with results: {present}\n{'='*90}")
        if not present:
            print("  (no method results)")
            continue
        # disagreement scan
        disagreements = []
        for cp in cp_order:
            row = {m: per_method[m].get(cp, {}).get("verdict") for m in present}
            vals = {v for v in row.values() if v is not None}
            if len(vals) >= 2:
                disagreements.append((cp, row))
        # verdict tallies per method
        for m in present:
            vd = per_method[m]
            ones = sum(1 for c in vd if vd[c]["verdict"] == "1")
            zeros = sum(1 for c in vd if vd[c]["verdict"] == "0")
            nas = sum(1 for c in vd if vd[c]["verdict"] not in ("0", "1"))
            print(f"  {m:22s} CPs={len(vd):2d}  1={ones:2d} 0={zeros:2d} N/A={nas:2d}")
        print(f"  --- disagreements across methods: {len(disagreements)} CPs ---")
        for cp, row in disagreements:
            txt = cps.get(cp, {}).get("text", "")[:70]
            cells = "  ".join(f"{m[:4]}={row[m]}" for m in present)
            print(f"    {cp}  {cells}")
            print(f"         rule: {txt}")
    out = Path("build/experiments/method_comparison.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main([int(x) for x in sys.argv[1:]]))

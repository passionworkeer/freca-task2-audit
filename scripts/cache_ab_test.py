"""A/B probe: does reordering the prompt + adding cache_control actually make
the MiniMax-M3 /anthropic endpoint hit the case-material prefix cache across
different checkpoints?

This script is READ-ONLY with respect to the experiment harness: it builds its
own raw HTTP payloads (it does NOT call build_prompt / run_execution), so it
cannot pollute the persisted results. It only burns a handful of model calls.

DESIGN INTENT was to prove that restructuring the prompt (material block first
+ cache_control) lets the case material hit the prefix cache across CPs.

╔══════════════════════════════════════════════════════════════════════════╗
║  MEASURED OUTCOME (2026-08-10, MiniMax-M3): the optimisation does NOT   ║
║  work. Keep this script as the evidence / regression check.             ║
╠══════════════════════════════════════════════════════════════════════════╣
║  • NEW CP1 -> CP2 (different cp_block): cache_read stays 128. Miss.     ║
║  • Same CP sent twice (identical payload): cache_read 128 -> 168,468.   ║
║    Hit — but ONLY because the whole request is byte-identical.          ║
║  • cache_creation_input_tokens is ALWAYS 0, even with cache_control +   ║
║    anthropic-beta: prompt-caching-2024-07-31 header.                    ║
║  => The MiniMax-M3 /anthropic endpoint ignores Anthropic-style prefix   ║
║    breakpoints and only deduplicates BYTE-IDENTICAL whole requests. The │
║    case material can never hit across CPs/cases, so restructuring the   ║
║    prompt cannot raise the hit rate. Don't pursue prompt-reorder cache  ║
║    optimisations on M3. Re-run this script if the model changes (M2.x   ║
║    documents active cache_control support and may behave differently).  ║
╚══════════════════════════════════════════════════════════════════════════╝

For each structure we send CP1 then CP2 and print the usage of every call.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

from freca.config import PipelineConfig
from freca.cp import load_checkpoints
from freca.env_loader import apply_env_file, find_env_file
from freca.experiments.materials import load_material_snapshot_from_parsed
from freca.experiments.models import ExperimentMethod, ExecutionUnit, Track3Condition
from freca.experiments.prompts import SYSTEM_PROMPT, VERDICT_SCHEMA, build_prompt

ENDPOINT = "https://api.minimaxi.com/anthropic/v1/messages"


def _send(payload: dict, api_key: str) -> dict:
    with httpx.Client(timeout=120) as client:
        resp = client.post(
            ENDPOINT,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
        )
    body = resp.json()
    if resp.status_code >= 400:
        return {"http_status": resp.status_code, "error": body}
    return body


def _print_usage(tag: str, body: dict) -> None:
    usage = body.get("usage", {})
    cr = usage.get("cache_read_input_tokens", 0)
    cc = usage.get("cache_creation_input_tokens", 0)
    it = usage.get("input_tokens", 0)
    ot = usage.get("output_tokens", 0)
    denom = cr + cc + it
    rate = (cr / denom * 100) if denom else 0
    print(
        f"  [{tag}] cache_read={cr:,} cache_create={cc:,} "
        f"input={it:,} output={ot:,} hit_rate={rate:.1f}%"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", type=int, default=23)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args(argv)

    env_file = find_env_file()
    if env_file is not None:
        apply_env_file(env_file)
    os.environ.setdefault("FRECA_AUDIT_BASE_URL", "https://api.minimaxi.com/anthropic")

    config = PipelineConfig.from_yaml(args.config)
    object.__setattr__(
        config.models,
        "audit",
        config.models.audit.model_copy(update={"model": "MiniMax-M3"}),
    )
    api_key = os.environ.get(config.models.audit.api_key_env)
    if not api_key:
        print(f"ERROR: env var {config.models.audit.api_key_env} unset", file=sys.stderr)
        return 2
    checkpoints = load_checkpoints(config.paths.checkpoints_xlsx)
    parsed_dir = config.paths.build_dir / "parsed"
    material = load_material_snapshot_from_parsed(
        parsed_dir=parsed_dir,
        case_id=args.case_id,
        checkpoints=list(checkpoints),
        track3_condition=Track3Condition.RAW,
    )

    cp_ids = [cp.cp_id for cp in checkpoints][:2]
    units = {
        cp_id: ExecutionUnit(
            case_id=args.case_id,
            method=ExperimentMethod.CHECKPOINT_FULL,
            checkpoint_ids=(cp_id,),
        )
        for cp_id in cp_ids
    }

    # Pre-build the material-only JSON string reused by the NEW structure. This
    # is the stable prefix we want the cache to retain across CPs.
    material_block_text = json.dumps(
        {
            "official_material_chunks": [c.model_dump(mode="json") for c in material.chunks],
            "allowed_citation_ids": list(material.chunk_ids),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    material_chars = len(material_block_text)
    print(
        f"case-{args.case_id:03d}: material chunks={len(material.chunks)} "
        f"material_block_chars={material_chars:,} (~{material_chars // 4:,} tokens)"
    )
    print(f"CPs under test: {cp_ids}")
    print()

    max_tokens = 4096

    # ─────────────────────────────────────────────────────────────────────
    # OLD structure: build_prompt as-is (sort_keys, no cache_control).
    # ─────────────────────────────────────────────────────────────────────
    print("=== OLD (current shape: sort_keys, CP before material, no cache_control) ===")
    for cp_id in cp_ids:
        prompt = build_prompt(unit=units[cp_id], material=material)
        payload = {
            "model": "MiniMax-M3",
            "system": prompt.system,
            "messages": [{"role": "user", "content": prompt.text}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        body = _send(payload, api_key)
        if "error" in body:
            print(f"  [OLD cp={cp_id}] ERROR: {body}")
            return 1
        _print_usage(f"OLD cp={cp_id}", body)
        time.sleep(2)
    print()

    # ─────────────────────────────────────────────────────────────────────
    # NEW structure: material block first with cache_control, CP block after.
    # CP1 writes the cache, CP2 (different CP) should read the material prefix.
    # ─────────────────────────────────────────────────────────────────────
    print("=== NEW (material block + cache_control first, CP block after breakpoint) ===")
    for cp_id in cp_ids:
        checkpoint_by_id = {cp.cp_id: cp for cp in material.checkpoints}
        cp_block_text = json.dumps(
            {
                "case_id": args.case_id,
                "method": "checkpoint_full",
                "official_checkpoints": [
                    checkpoint_by_id[cp_id].model_dump(mode="json")
                ],
                "official_image_paths": list(material.image_paths),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        payload = {
            "model": "MiniMax-M3",
            "system": [{"type": "text", "text": SYSTEM_PROMPT}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": material_block_text,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": cp_block_text},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        body = _send(payload, api_key)
        if "error" in body:
            print(f"  [NEW cp={cp_id}] ERROR: {body}")
            return 1
        _print_usage(f"NEW cp={cp_id}", body)
        time.sleep(2)

    print()
    print("Verdict: compare NEW cp2 cache_read vs OLD cp2 cache_read.")
    print("If NEW cp2 cache_read ~ material tokens and OLD cp2 ~ 0, restructuring wins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

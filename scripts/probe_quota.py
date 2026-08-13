"""Quick MiniMax 429 probe: a single tiny call, no retry, short timeout.

Loads the same .env / env overrides as scripts/run_case.py so the probe hits
the exact same endpoint+key+model as a real run. Prints a one-line status:

    PROBE_OK status=200 elapsed=0.8s            -> quota available, safe to run
    PROBE_429 elapsed=0.3s retry_after=...       -> quota exhausted, skip this loop
    PROBE_5XX/PROBE_4XX/PROBE_NET_ERROR/PROBE_NO_KEY

Exit code: 0 = ok, 1 = throttled/error (skip running), 2 = no key.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freca.env_loader import apply_env_file, find_env_file  # noqa: E402


def main() -> int:
    env_file = find_env_file()
    if env_file is not None:
        apply_env_file(env_file)
    os.environ.setdefault("FRECA_AUDIT_BASE_URL", "https://api.minimaxi.com/anthropic")

    api_key = os.environ.get("FRECA_AUDIT_API_KEY")
    base_url = os.environ.get("FRECA_AUDIT_BASE_URL", "https://api.minimaxi.com/anthropic")
    model = os.environ.get("FRECA_AUDIT_MODEL", "MiniMax-M3")
    if not api_key:
        print("PROBE_NO_KEY FRECA_AUDIT_API_KEY unset")
        return 2

    url = f"{base_url.rstrip('/')}/v1/messages"
    payload = {
        "model": model,
        "system": "Reply with the single word OK.",
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0,
        "max_tokens": 8,
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        print(f"PROBE_NET_ERROR {type(exc).__name__}: {str(exc)[:160]}")
        return 1
    elapsed = round(time.monotonic() - t0, 2)
    status = r.status_code
    body = r.text[:200].replace("\n", " ")
    if status == 429:
        print(f"PROBE_429 elapsed={elapsed}s retry_after={r.headers.get('retry-after')} body={body}")
        return 1
    if status >= 500:
        print(f"PROBE_5XX status={status} elapsed={elapsed}s body={body}")
        return 1
    if status >= 400:
        print(f"PROBE_4XX status={status} elapsed={elapsed}s body={body}")
        return 1
    print(f"PROBE_OK status={status} elapsed={elapsed}s model={model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

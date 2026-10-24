"""Shared Earth Engine helpers for Stage 0 probes.

Run `python scripts/ee_auth.py` FIRST (it opens a browser for authentication).
Each probe then uses ee_init() and writes its JSON report into data/stage0/.
Reports are the facts that Gate G0 turns into the feasibility note - nothing
in Phase 1+ may rely on an assumption that a probe did not measure.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import ee  # optional until `pip install -e ".[gee,dev]"` has been run
except ModuleNotFoundError as exc:  # pragma: no cover
    print("earthengine-api is not installed. Run:\n  python -m pip install -e \".[gee,dev]\"")
    raise SystemExit(1) from exc

REPORT_DIR = ROOT / "data" / "stage0"


def ee_init():
    """Initialise Earth Engine; gives actionable output if auth is missing."""
    sys.path.insert(0, str(ROOT))
    from no2map.config import load_config

    cfg = load_config()
    try:
        ee.Initialize(project=cfg["quota"]["project_id"])
    except Exception:
        project = cfg["quota"]["project_id"]
        hint = (
            "Run `python scripts/ee_auth.py` to authenticate first."
            if project is None
            else f'Check quota.project_id="{project}" in configs/default.yaml, then run scripts/ee_auth.py.'
        )
        raise SystemExit(f"Earth Engine not initialised: {hint}")
    return cfg


def save_report(name: str, payload: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"{name}.json"
    payload = {"generated_utc": datetime.now(timezone.utc).isoformat(), **payload}
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[report] wrote {out}")
    return out

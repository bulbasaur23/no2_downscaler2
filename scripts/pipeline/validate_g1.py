"""Gate G1: validate every in-window monthly export for one city (Phase 1).

Usage:
  python scripts/pipeline/validate_g1.py --city delhi_ncr

Pulls the window (start/end/exclude) from configs/default.yaml, derives the
expected cell set from the locked bbox via no2map.grid, loads each manifest-
done month's parquet, runs every hard/warn check (no2map.monthcheck), prints
a per-month table plus the G1 verdict, and writes the full report to
data/exports/g1_report_<city>.json. Exit code 0 iff G1 PASS.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from no2map.grid import cells_in_bbox  # noqa: E402
from no2map.monthcheck import gate_g1  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def expected_cells_from_config(cfg: dict, city: str) -> list[str]:
    c = cfg["study_areas"][city]
    return [cell.id for cell in
            cells_in_bbox(c["lat_min"], c["lat_max"], c["lon_min"], c["lon_max"])]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    tr = cfg["training"]
    cells = expected_cells_from_config(cfg, args.city)
    print(f"[g1] {args.city}: {len(cells)} grid cells, window "
          f"{tr['window_start']}..{tr['window_end']} "
          f"(excluded transition: {tr.get('exclude_months', [])})")

    manifest_path = ROOT / "data" / "exports" / "manifest.json"
    entries_raw: dict[str, dict | None] = {}
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        for k, e in m.get("entries", {}).items():
            if k.startswith(f"{args.city}/"):
                entries_raw[k.split("/", 1)[1]] = e

    frames: dict[str, pd.DataFrame] = {}
    for month, entry in entries_raw.items():
        if entry and entry.get("status") == "done":
            p = ROOT / "data" / "exports" / args.city / f"{args.city}_{month}.parquet"
            if p.exists():
                frames[month] = pd.read_parquet(p)

    ys, ye = int(tr["window_start"][:4]), int(tr["window_end"][:4])
    window_months = [f"{y}-{m:02d}"
                     for y in range(ys, ye + 1)
                     for m in range(1, 13)
                     if tr["window_start"][:7] <= f"{y}-{m:02d}" <= tr["window_end"][:7]]

    report = gate_g1(
        entries_raw, frames, cells,
        window_start=tr["window_start"], window_end=tr["window_end"],
        exclude_months=list(tr.get("exclude_months") or []),
    )

    print(f"\n[g1] {'MONTH':8s} {'STATUS':10s} {'VS':4s} NOTES")
    n_fail = n_warn = 0
    for month in window_months:
        info = report["months"][month]
        status = info.get("status") or "missing"
        ver = ""
        if month in frames:
            pv = (entries_raw.get(month) or {}).get("processor_versions") or []
            ver = ",".join(pv)
        note = "; ".join(filter(None, [
            ",".join(info.get("failed", [])) or None,
            "warn:" + ",".join(info.get("warned", [])) if info.get("warned") else None,
            info.get("detail") or None,
        ])) or ("ok" if info["ok"] else "FAIL")
        if not info["ok"]:
            n_fail += 1
        elif info.get("warned"):
            n_warn += 1
        flag = "PASS" if info["ok"] else "FAIL"
        print(f"     {month:8s} {status:10s} {ver:8s} {flag} {note}")

    out = ROOT / "data" / "exports" / f"g1_report_{args.city}.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n[g1] verdict: {report['verdict']}  "
          f"(months: {len(window_months)}, fail {n_fail}, warn-only {n_warn})")
    print(f"[g1] report: {out}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Gate G5 runner: leave-one-city-out transfer + OOD evidence (Phase 5).

Usage:
  # leave-one-city-out across the three training cities
  python scripts/pipeline/transfer_g5.py \
      --cities delhi_ncr,bengaluru,kolkata

  # untouched case study: train on all three, score Hyderabad only
  python scripts/pipeline/transfer_g5.py \
      --cities delhi_ncr,bengaluru,kolkata,hyderabad --test-cities hyderabad

Every fold fits on the training cities only (blocked CV + conformal there),
then predicts the held-out city frozen. Poor transfer is written to the report
as poor; nothing is tuned on the test city. The OOD block says whether cells
flagged by static-feature distance actually show higher held-out error - a
`None`/false there is reported, never hidden.

Writes data/exports/g5_transfer.json and returns 0. If a city has no exported
months it prints the export commands and exits 2 (a transfer claim needs data).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from no2map.dataset import (done_months, load_training_panel,  # noqa: E402
                            missing_window_months)
from no2map.features import FEATURE_GROUPS  # noqa: E402
from no2map.model import PRED_COLS  # noqa: E402
from no2map.transfer import transfer_report  # noqa: E402

CONFIG = ROOT / "configs" / "default.yaml"
MANIFEST = ROOT / "data" / "exports" / "manifest.json"
EXPORTS = ROOT / "data" / "exports"
OUT = EXPORTS / "g5_transfer.json"


def _fmt(v, spec: str = ".3f") -> str:
    return "n/a" if v is None or (isinstance(v, float) and v != v) else format(v, spec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cities", required=True,
                    help="comma-separated city ids available for training")
    ap.add_argument("--test-cities", default=None,
                    help="comma-separated cities to score (default: leave-one-out "
                         "over --cities)")
    ap.add_argument("--groups", default=",".join(FEATURE_GROUPS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--block-m", type=float, default=89_278.0,
                    help="G2 block size for the training-city CV")
    ap.add_argument("--buffer-m", type=float, default=30_000.0)
    ap.add_argument("--ood-quantile", type=float, default=0.95)
    ap.add_argument("--ood-k", type=int, default=5)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    test_cities = ([c.strip() for c in args.test_cities.split(",") if c.strip()]
                   if args.test_cities else None)
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    # pre-flight: every requested city needs exported months before a transfer
    # claim can be made; never silently shrink the table
    missing = []
    for city in sorted(set(cities) | set(test_cities or [])):
        months = done_months(MANIFEST, city)
        if not months:
            missing.append(city)
    if missing:
        print(f"[g5] cannot run: no exported months for {missing}")
        print("[g5] export each city first (one process per city at a time):")
        for city in missing:
            print(f"  python scripts\\pipeline\\export_month.py --city {city} "
                  f"--window auto")
        return 2

    # the training pool must have the complete configured window; a partial
    # export would silently shrink the training set behind every score
    if test_cities:
        train_pool = [c for c in cities if c not in set(test_cities)]
    else:
        train_pool = list(cities)  # every city trains in some fold
    incomplete = {c: missing_window_months(CONFIG, MANIFEST, c)
                  for c in train_pool}
    incomplete = {c: ms for c, ms in incomplete.items() if ms}
    if incomplete:
        print("[g5] cannot run: training window incomplete")
        for city, months in incomplete.items():
            print(f"  {city}: {len(months)} month(s) missing "
                  f"({months[0]}..{months[-1]})")
        print("[g5] finish those exports first (never train a transfer claim "
              "on a short window):")
        for city in incomplete:
            print(f"  python scripts\\pipeline\\export_month.py --city {city} "
                  f"--window auto")
        return 2

    panels = {}
    for city in cities:
        panel = load_training_panel(CONFIG, MANIFEST, EXPORTS, city)
        print(f"[g5] {city}: {len(panel):,} rows, months "
              f"{panel.attrs['months'][0]}..{panel.attrs['months'][-1]}")
        panels[city] = panel

    rep = transfer_report(
        panels, groups, test_cities=test_cities, seed=args.seed,
        alpha=args.alpha, block_m=args.block_m, buffer_m=args.buffer_m,
        ood_quantile=args.ood_quantile, ood_k=args.ood_k,
    )

    print("\n[g5] per-city data diagnostics:")
    for city, d in rep["cities"].items():
        print(f"  {city:<12s} cells={d['n_cells']:5d} obs={d['observed_fraction']:.3f} "
              f"median={_fmt(d['target_median'], '.1f')} umol/m2 "
              f"water_p95={_fmt(d['water_fraction_p95'])}")

    print("\n[g5] transfer (frozen on the training cities):")
    for city, f in rep["folds"].items():
        print(f"  test {city} (train: {', '.join(f['train_cities'])})")
        for col in PRED_COLS:
            s = f["scores"].get(col)
            if not s:
                continue
            print(f"    {col:<18s} rmse={s['rmse']:7.3f} bias={s['bias']:7.3f} "
                  f"level_r={_fmt(s['level_r'], '.3f'):>6s} "
                  f"pattern_r={_fmt(s['pattern_mean_r'], '.3f'):>6s} "
                  f"cov={_fmt(s['coverage'], '.3f'):>6s}")
        o = f["ood"]
        print(f"    OOD threshold={o['reference']['threshold']:.3f} "
              f"flagged_rows={_fmt(o['flagged_row_fraction'], '.3f')} "
              f"rmse_flagged={_fmt(o['rmse_flagged'])} "
              f"rmse_unflagged={_fmt(o['rmse_unflagged'])} "
              f"flag_higher_error={o['flagged_higher_error']}")
        if o["flagged_higher_error"] is False:
            print("    NOTE: the OOD flag does NOT separate higher-error cells "
                  "here; report that, do not tune the flag")

    OUT.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print(f"\n[g5] saved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

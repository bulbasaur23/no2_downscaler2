"""G2: empirical semivariogram on exported months -> advisory block size.

Usage:
  python scripts/pipeline/variogram_g2.py --city delhi_ncr [--dates 80]

Reads observed rows from the config training window (manifest-driven),
draws `--dates` distinct dates spread evenly (pairs explode with date
count; the variogram is same-date by design), fits the exponential range,
and prints the advisory block side with 0.5x and 1.0x buffer gaps plus the
G2 feasibility verdict against the NCR extent. Writes
data/exports/g2_variogram_<city>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from no2map.dataset import load_training_panel          # noqa: E402
from no2map.features import TARGET                     # noqa: E402
from no2map.variogram import (                          # noqa: E402
    TROPOMI_CELL_M, advisory_block_size, empirical_semivariogram,
    gate_g2_block_verdict, ncr_width_m)

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", required=True)
    ap.add_argument("--dates", type=int, default=80,
                    help="distinct dates to sample (same-date pairing)")
    ap.add_argument("--n-bins", type=int, default=16)
    args = ap.parse_args()

    panel = load_training_panel(
        ROOT / "configs" / "default.yaml",
        ROOT / "data" / "exports" / "manifest.json",
        ROOT / "data" / "exports", args.city)
    obs = panel[(panel["state"] == "observed")][["date", "cell_id", TARGET]]
    all_dates = np.array(sorted(obs["date"].unique()))
    rng = np.random.default_rng(0)
    pick = all_dates[np.sort(rng.choice(len(all_dates),
                                        size=min(args.dates, len(all_dates)),
                                        replace=False))]
    sub = obs[obs["date"].isin(set(pick))]
    print(f"[g2] {args.city}: {len(sub):,} observed rows | "
          f"{len(pick)} dates | {sub['cell_id'].nunique()} cells")

    res = empirical_semivariogram(sub, TARGET, n_bins=args.n_bins)
    print(f"[g2] exponential fit: range={res.range_m:,.0f} m "
          f"({res.range_m / 1000:.1f} km), sill={res.sill:.2f}")
    print("[g2] bins (m): gamma / pairs")
    for c, g, n in zip(res.bin_centers_m, res.gamma, res.n_pairs):
        print(f"       {c:9,.0f} {g:8.3f} {n:10d}")

    lat_w, lon_w = ncr_width_m()
    widths = (lon_w, lat_w)
    out = {"city": args.city, "n_dates": len(pick), "n_rows": len(sub),
           "range_m": res.range_m, "sill": res.sill,
           "tropomi_cell_m": TROPOMI_CELL_M,
           "bins": {"centers_m": res.bin_centers_m.tolist(),
                    "gamma": res.gamma.tolist(),
                    "n_pairs": res.n_pairs.tolist()},
           "advisories": {}}
    for buf in (0.5, 1.0):
        block = advisory_block_size(res, buffer=buf)
        verdict = gate_g2_block_verdict(block, min(widths))
        out["advisories"][f"buffer_{buf}"] = {
            "block_m": block, "verdict": verdict}
        feas = "feasible" if verdict["blocking_feasible"] else "IMPRACTICAL (G2 fallback)"
        print(f"[g2] buffer {buf:>3}: block={block:,.0f} m -> {feas}")

    dst = ROOT / "data" / "exports" / f"g2_variogram_{args.city}.json"
    dst.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"[g2] saved: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

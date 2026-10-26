"""Feature-group ablation CLI (Phase 3; honest skill attribution).

Usage:
  python scripts/pipeline/ablation.py --city delhi_ncr [--block-m ...]

Runs the cumulative ladder (static -> +instant weather -> +ventilation lags
-> +seasonal harmonics) through the identical blocked-fold machinery, so the
per-stage delta on RMSE is attributable to the ADDED feature family alone.
Prints the table and saves data/runs/<city>_ablation_<ts>/metrics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from no2map.dataset import load_training_panel  # noqa: E402
from no2map.model import run_ablation  # noqa: E402
from no2map.variogram import (gate_g2_block_verdict,   # noqa: E402
                              study_area_width_m)

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", required=True)
    ap.add_argument("--block-m", type=float, default=30_000.0)
    ap.add_argument("--buffer-m", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    panel = load_training_panel(
        ROOT / "configs" / "default.yaml",
        ROOT / "data" / "exports" / "manifest.json",
        ROOT / "data" / "exports", args.city)
    if args.max_rows:
        panel = panel.sample(min(args.max_rows, len(panel)), random_state=args.seed)
    print(f"[ablate] panel: {len(panel):,} rows | "
          f"months {panel.attrs['months'][0]}..{panel.attrs['months'][-1]}")

    g2 = gate_g2_block_verdict(args.block_m, min(study_area_width_m(args.city)))
    print(f"[ablate] blocking: {'feasible' if g2['blocking_feasible'] else 'G2 FALLBACK'} "
          f"(block {args.block_m:.0f} m)")

    stages = run_ablation(panel, block_m=args.block_m, buffer_m=args.buffer_m,
                          seed=args.seed, alpha=args.alpha)

    print("\n[ablate] features                                   model_rmse  delta_m  city_rmse")
    base = None
    for st in stages:
        name = " + ".join(st["groups"])
        delta = ("" if base is None else f"{st['delta_model_m']:+8.3f}")
        if base is None:
            base = st["model_rmse"]
        print(f"         {name:<50s} {st['model_rmse']:10.3f} {delta:>8s} "
              f"{st['city_mean_rmse']:10.3f}")

    out = ROOT / "data" / "runs" / (
        f"{args.city}_ablation_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}Z")
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps(
        {"stages": stages, "config": vars(args)}, indent=2, default=float),
        encoding="utf-8")
    print(f"[ablate] saved: {out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

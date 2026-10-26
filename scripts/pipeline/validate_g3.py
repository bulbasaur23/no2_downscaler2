"""G3: Model A vs per-cell climatology on both protocols, with seed noise.

The plan's G3 gate: "Model A beats per-cell climatology on blocked and
chronological splits by more than seed-to-seed noise, or the result is
reported honestly as it stands. If it does not beat climatology: report skill
on anomalies; check the static-only baseline."

This CLI produces that evidence in one JSON:

- blocked seed sweep (default seeds 0,1,2): per-method rmse/mae/bias/r2,
  level and pattern scores, anomaly correlation and skill-vs-climatology,
  plus the model-minus-climatology RMSE delta per seed and its spread;
- chronological seed sweep: same metrics, intervals calibrated inside the
  training period and frozen on the held-out future;
- static-only blocked run: the plan's "static-only baseline" (groups=static);
- a verdict block stating, per split, whether the delta exceeds the seed
  spread, and the standing action when it does not (report honestly; the app
  labels estimates as climatology-anchored).

Usage:
  python scripts/pipeline/validate_g3.py --city delhi_ncr \
      --block-m 89278 --buffer-m 30000 [--seeds 0,1,2] [--no-static]
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from no2map.dataset import load_training_panel                    # noqa: E402
from no2map.model import (PRED_COLS, run_blocked_evaluation,      # noqa: E402
                          run_chronological_evaluation)

ROOT = Path(__file__).resolve().parents[2]
MODEL = "pred_model_a"
CLIM = "pred_climatology"

# metric keys carried through to the report per method
METRIC_KEYS = ("n", "rmse", "mae", "bias", "r2", "level_rmse", "level_r",
               "level_n_dates", "pattern_mean_r", "pattern_n_dates",
               "r_anomaly", "skill_vs_climatology", "coverage", "mean_width")


def summarize(rep: dict) -> dict:
    return {m: {k: rep[m][k] for k in METRIC_KEYS if k in rep[m]}
            for m in PRED_COLS if m in rep}


def run_blocked(panel, groups, args, seed: int) -> dict:
    t0 = time.time()
    _, rep = run_blocked_evaluation(
        panel, groups, block_m=args.block_m, buffer_m=args.buffer_m,
        seed=seed, alpha=args.alpha)
    print(f"  [blocked seed={seed}] {time.time() - t0:6.0f}s "
          f"model={rep[MODEL]['rmse']:.3f} clim={rep[CLIM]['rmse']:.3f}",
          flush=True)
    return summarize(rep)


def run_chron(panel, groups, args, seed: int) -> dict:
    t0 = time.time()
    _, rep = run_chronological_evaluation(
        panel, groups, seed=seed, alpha=args.alpha,
        block_m=args.block_m, buffer_m=args.buffer_m)
    print(f"  [chron   seed={seed}] {time.time() - t0:6.0f}s "
          f"model={rep[MODEL]['rmse']:.3f} clim={rep[CLIM]['rmse']:.3f} "
          f"| test {rep['test_dates'][0]}..{rep['test_dates'][1]}",
          flush=True)
    return summarize(rep)


def delta_summary(runs: list[dict]) -> dict:
    """Model-minus-climatology RMSE delta across seeds, with the noise
    yardstick. 'beyond_noise' demands BOTH a consistent direction and a mean
    delta larger than one standard deviation of the per-seed deltas - a
    deliberately conservative reading, since three seeds cannot support a
    formal test and the plan only asks for 'more than seed-to-seed noise'."""
    deltas = [r[MODEL]["rmse"] - r[CLIM]["rmse"] for r in runs]
    model = [r[MODEL]["rmse"] for r in runs]
    clim = [r[CLIM]["rmse"] for r in runs]
    mean_d = stats.fmean(deltas)
    sd_d = stats.stdev(deltas) if len(deltas) > 1 else 0.0
    same_sign = all(d < 0 for d in deltas) or all(d > 0 for d in deltas)
    return {
        "deltas_by_seed": deltas,
        "delta_mean": mean_d,
        "delta_sd": sd_d,
        "model_rmse_by_seed": model,
        "climatology_rmse_by_seed": clim,
        "model_rmse_sd": stats.stdev(model) if len(model) > 1 else 0.0,
        "climatology_rmse_sd": stats.stdev(clim) if len(clim) > 1 else 0.0,
        "model_beats_climatology_mean": mean_d < 0,
        "beyond_seed_noise": bool(same_sign and abs(mean_d) > sd_d and mean_d < 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", required=True)
    ap.add_argument("--block-m", type=float, default=89_278.0,
                    help="variogram-advised block side (G2)")
    ap.add_argument("--buffer-m", type=float, default=30_000.0,
                    help="0.5x variogram range buffer (G2)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--groups", default="static,weather_inst,weather_lags,temporal")
    ap.add_argument("--no-static", action="store_true",
                    help="skip the static-only baseline run")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    panel = load_training_panel(
        ROOT / "configs" / "default.yaml",
        ROOT / "data" / "exports" / "manifest.json",
        ROOT / "data" / "exports", args.city)
    out = {
        "city": args.city,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "groups": groups, "block_m": args.block_m,
            "buffer_m": args.buffer_m, "alpha": args.alpha, "seeds": seeds,
            "months": panel.attrs["months"],
            "n_rows": int(len(panel)),
            "n_cells": int(panel["cell_id"].nunique()),
        },
        "blocked": {}, "chronological": {}, "static_only": None,
    }

    print(f"[g3] {args.city}: {len(panel):,} rows, "
          f"{panel['cell_id'].nunique()} cells, block {args.block_m:,.0f} m",
          flush=True)

    print("[g3] blocked protocol: seed sweep", flush=True)
    blocked_runs = [run_blocked(panel, groups, args, s) for s in seeds]
    out["blocked"] = {"runs": {str(s): r for s, r in zip(seeds, blocked_runs)},
                      "seed_delta": delta_summary(blocked_runs)}

    print("[g3] chronological protocol: seed sweep", flush=True)
    chron_runs = [run_chron(panel, groups, args, s) for s in seeds]
    out["chronological"] = {
        "runs": {str(s): r for s, r in zip(seeds, chron_runs)},
        "seed_delta": delta_summary(chron_runs)}

    if not args.no_static:
        print("[g3] static-only baseline (blocked, seed=0)", flush=True)
        static = run_blocked(panel, ["static"], args, seeds[0])
        out["static_only"] = {
            "blocked_seed0": static,
            "model_minus_full_model_rmse":
                static[MODEL]["rmse"] - blocked_runs[0][MODEL]["rmse"],
        }

    b, c = out["blocked"]["seed_delta"], out["chronological"]["seed_delta"]
    both = b["beyond_seed_noise"] and c["beyond_seed_noise"]
    actions = []
    if not b["beyond_seed_noise"]:
        actions.append(
            "blocked split does not beat per-cell climatology beyond seed "
            "noise: report as it stands, report anomaly skill, and keep "
            "climatology-anchored estimates clearly labelled in the app")
    if not c["beyond_seed_noise"]:
        actions.append(
            "chronological split does not beat per-cell climatology beyond "
            "seed noise: report as it stands (no tuning on the future)")
    out["verdict"] = {
        "beats_blocked_beyond_noise": b["beyond_seed_noise"],
        "beats_chronological_beyond_noise": c["beyond_seed_noise"],
        "g3_mode": "both_splits" if both else "honest_report",
        "required_actions": actions,
        "note": ("G3 permits the honest-report path; the plan forbids tuning "
                 "on the test period either way."),
    }

    dst = ROOT / "data" / "exports" / f"g3_report_{args.city}.json"
    dst.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")

    print(f"\n[g3] blocked : model {b['delta_mean']:+.2f} vs climatology "
          f"(sd {b['delta_sd']:.2f}) | beyond noise: {b['beyond_seed_noise']}"
          f" | anomaly r={blocked_runs[0][MODEL]['r_anomaly']:.3f} "
          f"skill={blocked_runs[0][MODEL]['skill_vs_climatology']:+.3f}")
    print(f"[g3] chron   : model {c['delta_mean']:+.2f} vs climatology "
          f"(sd {c['delta_sd']:.2f}) | beyond noise: {c['beyond_seed_noise']}"
          f" | anomaly r={chron_runs[0][MODEL]['r_anomaly']:.3f} "
          f"skill={chron_runs[0][MODEL]['skill_vs_climatology']:+.3f}")
    if out["static_only"]:
        print(f"[g3] static-only model rmse={out['static_only']['blocked_seed0'][MODEL]['rmse']:.3f} "
              f"(full model +{out['static_only']['model_minus_full_model_rmse']:+.3f})")
    print(f"[g3] verdict: {out['verdict']['g3_mode']}")
    for a in actions:
        print(f"[g3] action: {a}")
    print(f"[g3] report: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

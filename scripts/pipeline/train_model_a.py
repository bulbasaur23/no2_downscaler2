"""Train Model A (XGBoost) and compare with baselines (Phase 3, Gate G3).

Usage:
  python scripts/pipeline/train_model_a.py --city delhi_ncr \
      [--block-m 30000] [--buffer-m 0] [--groups static,weather_inst,...] [--chron]

Flow: dataset module assembles the config-window training panel (manifest-
driven); run_blocked_evaluation scores Model A + Ridge + climatology +
daily-city-mean through identical spatial-block folds (variogram-backed
block size from --block-m, with the plan's buffer gap); the OOF frame plus
the conformal calibration are written for reporting. With --chron the
train-early/test-late protocol runs too, because G3 requires both. The
version-shift test is separate and must never be used to tune.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from no2map.artifact import save_model_artifact        # noqa: E402
from no2map.conformal import SplitConformal            # noqa: E402
from no2map.dataset import load_training_panel         # noqa: E402
from no2map.features import (TARGET, build_features,  # noqa: E402
                             clean_training_frame, feature_block)
from no2map.model import (PRED_COLS, XGBConfig, season_key,          # noqa: E402
                          run_blocked_evaluation,
                          run_chronological_evaluation)
from no2map.variogram import gate_g2_block_verdict, ncr_width_m  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", required=True)
    ap.add_argument("--groups", default="static,weather_inst,weather_lags,temporal",
                    help="comma-separated feature groups (see features.py)")
    ap.add_argument("--block-m", type=float, default=30_000.0,
                    help="spatial block side in metres (variogram-advised)")
    ap.add_argument("--buffer-m", type=float, default=5_000.0,
                    help="buffer gap around validation blocks (metres)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="1 - interval coverage target")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="optional cap on training rows (debug runs)")
    ap.add_argument("--chron", action="store_true",
                    help="also run the train-early/test-late protocol (G3 "
                         "requires both blocked and chronological)")
    args = ap.parse_args()
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    panel = load_training_panel(
        ROOT / "configs" / "default.yaml",
        ROOT / "data" / "exports" / "manifest.json",
        ROOT / "data" / "exports",
        args.city,
    )
    if args.max_rows:
        panel = panel.sample(min(args.max_rows, len(panel)), random_state=args.seed)
    print(f"[train] panel: {len(panel):,} rows | "
          f"{panel['cell_id'].nunique()} cells | "
          f"months: {panel.attrs['months'][0]}..{panel.attrs['months'][-1]} "
          f"({len(panel.attrs['months'])} months)")

    # conservative: the narrower NCR extent (lat, ~191 km) gates feasibility
    g2 = gate_g2_block_verdict(args.block_m, min(ncr_width_m()))
    advice = ("SPATIAL BLOCKED" if g2["blocking_feasible"] else
              "G2 fallback: block too large for NCR - chronological + coarse "
              "blocks; spatial claim via leave-one-city-out")
    print(f"[train] block {args.block_m:.0f} m, buffer {args.buffer_m:.0f} m -> {advice}")

    oof, report = run_blocked_evaluation(
        panel, groups, block_m=args.block_m, buffer_m=args.buffer_m,
        seed=args.seed, alpha=args.alpha)
    report["config"] = {
        "city": args.city, "groups": groups,
        "block_m": args.block_m, "buffer_m": args.buffer_m,
        "seed": args.seed, "alpha": args.alpha,
        "months": panel.attrs["months"],
        "blocking_protocol": advice,
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    # advisory_block_size (variogram range) needs a fitted semivariogram on
    # real data; the CLI takes the size directly via --block-m instead.
    report["g2_verdict"] = g2

    run_dir = ROOT / "data" / "runs" / (
        f"{args.city}_modelA_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}Z")
    run_dir.mkdir(parents=True)

    # serving bundle (G4: predict without retraining): refit on ALL window
    # rows; conformal reuses the blocked OOF residuals (same quantiles the
    # report shows - oof carries 'residual' and 'month')
    final_rows = build_features(clean_training_frame(panel))
    fe_cols = feature_block(final_rows, groups)
    final_model = XGBConfig().make(args.seed)
    final_model.fit(final_rows[fe_cols], final_rows[TARGET])
    conf = SplitConformal(alpha=args.alpha, key_fn=season_key).fit(oof)
    save_model_artifact(
        run_dir, final_model, conf, fe_cols, groups,
        meta={"city": args.city, "months": panel.attrs["months"],
              "block_m": args.block_m, "buffer_m": args.buffer_m,
              "protocol": advice, "ran_at": report["config"]["ran_at"]})
    print(f"[train] artifact saved (model + conformal + {len(fe_cols)} fe cols)")

    # chronological co-report: the plan requires BOTH protocols for G3, so the
    # claim cannot rest on whichever split happens to look better. Everything
    # here is fitted on the earlier dates only; the future stays untouched.
    if args.chron:
        print("[train] running chronological protocol (train-early / test-late) ...")
        chron_oof, chron_report = run_chronological_evaluation(
            panel, groups, seed=args.seed, alpha=args.alpha,
            block_m=args.block_m, buffer_m=args.buffer_m)
        report["chronological"] = chron_report
        chron_oof.to_parquet(run_dir / "chron_oof.parquet", index=False)
        tr_lo, tr_hi = chron_report["train_dates"]
        te_lo, te_hi = chron_report["test_dates"]
        print(f"[chron] train {tr_lo}..{tr_hi} ({chron_report['n_train_dates']} dates) "
              f"-> test {te_lo}..{te_hi} ({chron_report['n_test_dates']} dates)")
        print("[chron] RMSE / interval coverage (frozen on the future):")
        for c in PRED_COLS:
            if c not in chron_report:
                continue
            s = chron_report[c]
            print(f"  {c:<18s} {s['rmse']:8.3f}  cov={s['coverage']:.3f}  "
                  f"width={s['mean_width']:.3f}")
        chron_rows = {c: chron_report[c]["rmse"] for c in PRED_COLS
                      if c in chron_report}
        report["chronological_best"] = min(chron_rows, key=chron_rows.get)
        print(f"[chron] best method: {report['chronological_best']}")

    oof.to_parquet(run_dir / "oof.parquet", index=False)
    (run_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8")

    rows = [(c, report[c]["rmse"]) for c in PRED_COLS if c in report]
    print("\n[train] blocked-CV RMSE (lower is better; cells held out by block):")
    for c, r in sorted(rows, key=lambda t: t[1]):
        print(f"  {c:<18s} {r:8.3f}")
    conf_rep = report["conformal"]
    print(f"[train] conformal alpha={conf_rep['alpha']}: "
          f"q_pooled={conf_rep['q_pooled']:.3f}, q_by_season={conf_rep['q_by_season']}")
    print(f"[train] saved: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Version-shift test runner (Phase 3; G0 processor-drift mitigation).

Usage:
  python scripts/pipeline/version_shift_test.py --city delhi_ncr \
      --eval-months 2025-11,2025-12

Trains frozen on the config training window (2023-04..2025-09, transitions
excluded) through the blocked-CV machinery, then predicts the eval months
(new processor 02.09.01) without refitting anything and reports cellwise +
two-level (level/pattern) scores plus frozen-interval coverage. Hard guard:
eval months' processor versions must not overlap training versions. The
shift set must NEVER be used to tune - this script only reports; it writes
data/shift/<city>_version_shift.json and returns 0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from no2map.dataset import load_training_panel  # noqa: E402
from no2map.features import FEATURE_GROUPS  # noqa: E402
from no2map.version_shift import version_shift_report  # noqa: E402
from no2map.model import PRED_COLS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", required=True)
    ap.add_argument("--eval-months", required=True,
                    help="comma-separated new-processor months, e.g. 2025-11,2025-12")
    ap.add_argument("--groups", default=",".join(FEATURE_GROUPS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--block-m", type=float, default=30_000.0)
    ap.add_argument("--buffer-m", type=float, default=0.0)
    args = ap.parse_args()
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    eval_months = [m.strip() for m in args.eval_months.split(",") if m.strip()]

    manifest_path = ROOT / "data" / "exports" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["entries"]

    def versions_of(month: str) -> list[str]:
        e = manifest.get(f"{args.city}/{month}") or {}
        return list(e.get("processor_versions") or [])

    train_panel = load_training_panel(
        ROOT / "configs" / "default.yaml", manifest_path,
        ROOT / "data" / "exports", args.city)
    train_versions = sorted({v for mo in train_panel.attrs["months"]
                             for v in versions_of(mo)})
    print(f"[shift] train panel: {len(train_panel):,} rows, versions {train_versions}")

    eval_frames = {}
    eval_versions = {}
    for month in eval_months:
        e = manifest.get(f"{args.city}/{month}") or {}
        if e.get("status") != "done":
            raise FileNotFoundError(
                f"eval month {month} is not exported/done (status="
                f"{e.get('status')}); export it first, do not tune on it")
        p = ROOT / "data" / "exports" / args.city / f"{args.city}_{month}.parquet"
        eval_frames[month] = pd.read_parquet(p)
        eval_versions[month] = list(e.get("processor_versions") or [])
        print(f"[shift] eval {month}: {len(eval_frames[month]):,} rows, "
              f"versions {eval_versions[month]}")

    rep = version_shift_report(
        train_panel, eval_frames, train_versions, eval_versions,
        groups, seed=args.seed, alpha=args.alpha,
    )

    print("\n[shift] per-month, per-method (rmse | bias | level_r | "
          "pattern_r | coverage):")
    for month, info in rep["per_month"].items():
        print(f"  {month} (versions {info['versions']}):")
        for col in PRED_COLS:
            s = info["scores"][col]
            print(f"    {col:<18s} {s['rmse']:7.3f} {s['bias']:7.3f} "
                  f"{s['level_r']:6.3f} {s['pattern_mean_r']:6.3f} "
                  f"cov={s['coverage']:.3f} w={s['mean_width']:.2f}")

    out_dir = ROOT / "data" / "shift"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.city}_version_shift.json"
    out.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print(f"[shift] saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

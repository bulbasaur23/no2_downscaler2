"""Version-shift test runner (Phase 3; G0 processor-drift mitigation).

Usage:
  python scripts/pipeline/version_shift_test.py --city delhi_ncr
  python scripts/pipeline/version_shift_test.py --city delhi_ncr \
      --eval-months 2025-11,2025-12

Trains frozen on the config training window (2023-04..2025-09, transitions
excluded) through the blocked-CV machinery, then predicts the eval months
(new processor 02.09.01) without refitting anything and reports cellwise +
two-level (level/pattern) scores plus frozen-interval coverage. `--eval-months`
defaults to the config's version_shift_test window (two months from its start).
The eval set is chosen PER DAY on the processor version measured over the
city's overpass window: a day that still carries a training version (measured:
2025-11 days 4-21, plus one mixed day) or two new versions is excluded and
counted, because it cannot attribute a change to one update. That exclusion is
on the treatment, never on outcomes. Hard guard: eval months' processor
versions must not overlap training versions. The shift set must NEVER be used
to tune - this script only reports; it writes
data/shift/<city>_version_shift.json and returns 0. If the eval months are not
exported yet it says so and exits 2 (it never silently shrinks the test).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from no2map.dataset import (TrainingConfig, load_training_panel,  # noqa: E402
                            missing_window_months)
from no2map.features import FEATURE_GROUPS  # noqa: E402
from no2map.version_shift import (partition_eval_days,  # noqa: E402
                                  version_shift_report)
from no2map.model import PRED_COLS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def default_eval_months(start: str, n_months: int = 2) -> list[str]:
    """The n calendar months starting at `start` ('YYYY-MM-DD' or 'YYYY-MM')."""
    year, month = int(start[:4]), int(start[5:7])
    out = []
    for _ in range(n_months):
        out.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return out


def day_versions_for(city: str, dates: list[str], hours: list[int],
                     cache_path: Path) -> dict[str, list[str]]:
    """Per-day processor versions within the city's overpass window (cached).

    The update does not land on a month boundary, so the eval set is chosen
    per DAY on the treatment (which processor produced it), never on outcomes.
    Cached to disk: the answer cannot change for a past day and a rerun must
    not spend EE quota again."""
    cache: dict[str, list[str]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    todo = [d for d in dates if d not in cache]
    if todo:
        from no2map.acquisition import day_processor_versions
        from no2map.eerun import initialize_ee

        initialize_ee()
        for d in todo:
            cache[d] = day_processor_versions(d, hours)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True),
                              encoding="utf-8")
    return {d: cache[d] for d in dates}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", required=True)
    ap.add_argument("--eval-months", default=None,
                    help="comma-separated new-processor months; default = two "
                         "months from config training.version_shift_test.start")
    ap.add_argument("--groups", default=",".join(FEATURE_GROUPS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--block-m", type=float, default=30_000.0)
    ap.add_argument("--buffer-m", type=float, default=0.0)
    ap.add_argument("--include-mixed-days", action="store_true",
                    help="score every day of the eval months, including days "
                         "that still carry a training processor version "
                         "(off by default: such days cannot test the shift)")
    args = ap.parse_args()
    # long fit: keep the log readable while running under a redirect
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    config_path = ROOT / "configs" / "default.yaml"
    if args.eval_months:
        eval_months = [m.strip() for m in args.eval_months.split(",") if m.strip()]
    else:
        shift_start = TrainingConfig.from_config(config_path).version_shift_start
        if not shift_start:
            print("[shift] no --eval-months given and "
                  "training.version_shift_test.start is not configured")
            return 2
        eval_months = default_eval_months(shift_start)
    print(f"[shift] eval months: {', '.join(eval_months)}")

    manifest_path = ROOT / "data" / "exports" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["entries"]

    # pre-flight: report every month that is not usable yet and stop, rather
    # than loading the full training panel and then dying mid-way. The shift
    # set is never partial: a missing month means the test cannot be claimed.
    missing = []
    for month in eval_months:
        entry = manifest.get(f"{args.city}/{month}") or {}
        parquet = (ROOT / "data" / "exports" / args.city
                   / f"{args.city}_{month}.parquet")
        if entry.get("status") != "done" or not parquet.exists():
            missing.append((month, entry.get("status")))
    if missing:
        print(f"[shift] cannot run: {len(missing)} eval month(s) not exported")
        for month, status in missing:
            print(f"  {month}: status={status or 'not in manifest'}")
        print("[shift] export them first (never tune on them):")
        for month, _ in missing:
            print(f"  python scripts\\pipeline\\export_month.py --city "
                  f"{args.city} --month {month}")
        return 2

    def versions_of(month: str) -> list[str]:
        e = manifest.get(f"{args.city}/{month}") or {}
        return list(e.get("processor_versions") or [])

    # a quietly short training window would invalidate the whole comparison
    missing_train = missing_window_months(config_path, manifest_path, args.city)
    if missing_train:
        print(f"[shift] cannot run: training window incomplete "
              f"({len(missing_train)} month(s) not done): {missing_train}")
        print(f"  python scripts\\pipeline\\export_month.py --city "
              f"{args.city} --window auto")
        return 2

    train_panel = load_training_panel(
        config_path, manifest_path, ROOT / "data" / "exports", args.city)
    train_versions = sorted({v for mo in train_panel.attrs["months"]
                             for v in versions_of(mo)})
    print(f"[shift] train panel: {len(train_panel):,} rows, versions {train_versions}")

    eval_frames = {}
    eval_versions = {}
    excluded_by_month: dict[str, dict] = {}
    kept_days_by_month: dict[str, list[str]] = {}
    day_versions: dict[str, list[str]] = {}

    from no2map.acquisition import overpass_utc_hours
    from no2map.config import load_config

    cfg = load_config()
    area = cfg["study_areas"][args.city]
    hours = overpass_utc_hours(area["lon_min"], area["lon_max"])
    cache_path = ROOT / "data" / "shift" / f"day_versions_{args.city}.json"

    for month in eval_months:
        p = ROOT / "data" / "exports" / args.city / f"{args.city}_{month}.parquet"
        frame = pd.read_parquet(p)
        days = sorted(frame["date"].str.slice(0, 10).unique())
        day_versions.update(day_versions_for(args.city, days, hours, cache_path))

        if args.include_mixed_days:
            keep, excluded = days, {}
        else:
            keep, excluded = partition_eval_days(
                {d: day_versions[d] for d in days}, train_versions)
        if not keep:
            print(f"[shift] {month}: no version-pure eval days "
                  f"(excluded {len(excluded)}) - month dropped")
            excluded_by_month[month] = excluded
            continue

        eval_frames[month] = frame[frame["date"].isin(keep)].copy()
        eval_versions[month] = sorted({v for d in keep for v in day_versions[d]})
        excluded_by_month[month] = excluded
        kept_days_by_month[month] = keep
        print(f"[shift] eval {month}: kept {len(keep)}/{len(days)} days "
              f"({len(eval_frames[month]):,} rows), "
              f"versions {eval_versions[month]}")
        if excluded:
            reasons: dict[str, int] = {}
            for v in excluded.values():
                reasons[v["reason"]] = reasons.get(v["reason"], 0) + 1
            print(f"[shift]   excluded days (treatment-based, never outcome-"
                  f"based): {reasons}")

    if not eval_frames:
        print("[shift] no eval month has a version-pure day: nothing to score")
        return 2

    print(f"[shift] protocol: blocked CV block={args.block_m:.0f} m "
          f"buffer={args.buffer_m:.0f} m, seed={args.seed}, alpha={args.alpha} "
          f"(frozen; eval days never enter the fit)")
    try:
        rep = version_shift_report(
            train_panel, eval_frames, train_versions, eval_versions,
            groups, seed=args.seed, alpha=args.alpha,
            block_m=args.block_m, buffer_m=args.buffer_m,
        )
    except ValueError as exc:
        # the premise guard: a mixed eval set cannot test a processor shift
        print(f"[shift] cannot score: {exc}")
        return 2

    print("\n[shift] per-month, per-method (rmse | bias | level_r | "
          "pattern_r | coverage):")
    for month, info in rep["per_month"].items():
        purity = "version-pure" if info["version_pure"] else "MIXED-VERSION"
        print(f"  {month} (versions {info['versions']}, {purity}):")
        for col in PRED_COLS:
            s = info["scores"][col]
            print(f"    {col:<18s} {s['rmse']:7.3f} {s['bias']:7.3f} "
                  f"{s['level_r']:6.3f} {s['pattern_mean_r']:6.3f} "
                  f"cov={s['coverage']:.3f} w={s['mean_width']:.2f}")

    if not rep["eval_version_pure"]:
        print(f"[shift] NOTE mixed-version months: "
              f"{rep['mixed_version_months']}; their scores reflect an "
              "unknown blend of processor versions")

    rep["eval_protocol"] = {
        "pure_days_only": not args.include_mixed_days,
        "kept_days": kept_days_by_month,
        "excluded_days": excluded_by_month,
        "day_versions": {d: day_versions[d] for m in kept_days_by_month
                         for d in kept_days_by_month[m]},
    }

    out_dir = ROOT / "data" / "shift"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.city}_version_shift.json"
    out.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print(f"[shift] saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

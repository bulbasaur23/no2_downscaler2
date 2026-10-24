"""Training-dataset assembly (Phase 3 / Phase 4 shared loader).

One definition of "the training table": config-pinned window, transition
months excluded, manifest-driven month files, per-city flat panel. Both the
train CLI and (later) the live app resolve inputs through this module so the
train/serve parity rule extends to dataset construction, not just features.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from no2map.manifest import ExportManifest

STATUS_DONE = "done"


@dataclass
class TrainingConfig:
    window_start: str
    window_end: str
    exclude_months: list[str]
    version_shift_start: str | None = None

    @staticmethod
    def from_config(path: str | Path) -> "TrainingConfig":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        tr = cfg["training"]
        shift = tr.get("version_shift_test") or {}
        return TrainingConfig(
            window_start=tr["window_start"],
            window_end=tr["window_end"],
            exclude_months=list(tr.get("exclude_months") or []),
            version_shift_start=shift.get("start"),
        )

    def month_in_window(self, month: str) -> bool:
        """'YYYY-MM' inside [window_start, window_end] and not excluded."""
        if month in self.exclude_months:
            return False
        return self.window_start[:7] <= month <= self.window_end[:7]


def done_months(manifest_path: str | Path, city: str) -> list[str]:
    m = ExportManifest.load(Path(manifest_path))
    return sorted(k.split("/")[1] for k, e in m.entries.items()
                  if e.status == STATUS_DONE and k.startswith(f"{city}/"))


def month_files(exports_dir: str | Path, city: str, months: list[str]) -> list[Path]:
    base = Path(exports_dir) / city
    files = [(mo, base / f"{city}_{mo}.parquet") for mo in months]
    missing = [mo for mo, p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"manifest says done but parquet missing for {missing}")
    return [p for _, p in files]


def load_training_panel(
    config_path: str | Path,
    manifest_path: str | Path,
    exports_dir: str | Path,
    city: str,
) -> pd.DataFrame:
    """Flat panel of the config window: concatenated done-month parquets,
    restricted in code to the window (a stale parquet outside it is dropped,
    not trusted, because the manifest does not carry the window)."""
    tc = TrainingConfig.from_config(config_path)
    months = [mo for mo in done_months(manifest_path, city) if tc.month_in_window(mo)]
    if not months:
        raise RuntimeError(f"no done months inside window for {city!r}")
    frames = [pd.read_parquet(p) for p in
              month_files(exports_dir, city, months)]
    panel = pd.concat(frames, ignore_index=True)
    seen_months = sorted(panel["date"].str.slice(0, 7).unique()) if "date" in panel else []
    if seen_months and set(seen_months) != set(months):
        bad = set(seen_months) ^ set(months)
        raise ValueError(f"parquet months mismatch with manifest: {sorted(bad)}")
    panel.attrs["months"] = list(months)
    return panel

"""Evaluation splits (Phase 3, plan section 6).

Two split families feed every model comparison:

1. `assign_spatial_blocks` / `spatial_block_folds` - blocks sized by the
   variogram range (Phase 2) with an OPTIONAL buffer gap: rows within
   `buffer_m` of a validation block are REMOVED from training for that fold,
   because cells just outside the block boundary are near-duplicates of
   validation cells (the leakage the plan explicitly forbids).
2. `chronological_split` - train on an earlier date range, test on a later
   one. Dates are ordered, disjoint, and there is no shuffling.

No random splitting of neighbouring cells exists here at all - the honest-
validation rule (plan section 6) is enforced by construction, not by care.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from no2map.grid import parse_cell_id, cell_from_row_col

_EARTH_RADIUS_M = 6371008.8
M_PER_DEG_LAT = _EARTH_RADIUS_M * math.pi / 180.0


def cell_positions(cell_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Lat/lon centroid arrays in the given order (no re-ordering)."""
    lat = np.empty(len(cell_ids))
    lon = np.empty(len(cell_ids))
    for i, cid in enumerate(cell_ids):
        c = cell_from_row_col(*parse_cell_id(cid))
        lat[i] = c.center_lat
        lon[i] = c.center_lon
    return lat, lon


@dataclass
class BlockPartition:
    cell_ids: list[str]
    block_of: dict[str, str]
    n_blocks_x: int
    n_blocks_y: int
    block_m: float

    @property
    def block_ids(self) -> list[str]:
        return sorted(set(self.block_of.values()))


def assign_spatial_blocks(cell_ids: list[str], block_m: float) -> BlockPartition:
    """Tile the cells' extent into square blocks of side `block_m`.

    Blocks use cell CENTROIDS: a cell belongs to exactly one block, which
    makes fold construction unambiguous (no cell in two folds). The lon->m
    conversion fixes cos(mid-lat): at NCR lattitudes this matters ~12%.
    """
    if block_m <= 0:
        raise ValueError("block_m must be positive")
    if not cell_ids:
        raise ValueError("cell_ids must be non-empty")
    lat, lon = cell_positions(cell_ids)
    lat0, lon0 = float(lat.min()), float(lon.min())
    mid_lat = float(pd.Series(lat).mean())
    m_per_deg_lon = M_PER_DEG_LAT * math.cos(math.radians(mid_lat))

    bx = {}
    for cid, la, lo in zip(cell_ids, lat, lon):
        biy = int((la - lat0) * M_PER_DEG_LAT // block_m)
        bix = int((lo - lon0) * m_per_deg_lon // block_m)
        bx[cid] = f"b{biy:02d}_{bix:02d}"
    nx = 1 + max(int(b[5:7]) for b in bx.values())
    ny = 1 + max(int(b[1:3]) for b in bx.values())
    return BlockPartition(list(cell_ids), bx, nx, ny, block_m)


def _haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    l1, l2 = np.radians(lon1), np.radians(lon2)
    dp = p2[:, None] - p1[None, :]
    dl = l2[:, None] - l1[None, :]
    a = np.sin(dp / 2) ** 2 + np.cos(p1)[None, :] * np.cos(p2)[:, None] * np.sin(dl / 2) ** 2
    return 2.0 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def spatial_block_folds(
    cell_ids: list[str],
    block_m: float,
    buffer_m: float = 0.0,
) -> list[tuple[str, list[str], list[str]]]:
    """Leave-one-block-out folds with a distance buffer.

    Returns [(block_id, train_cell_ids, val_cell_ids), ...]: each block takes
    one turn as the validation region; training cells within `buffer_m` of it
    are dropped (the plan's buffer gap). Raises if the tiling gives <2 blocks
    (degenerate: blocking cannot leak anything -> block size too large).
    """
    part = assign_spatial_blocks(cell_ids, block_m)
    if len(part.block_ids) < 2:
        raise ValueError(
            f"block_m={block_m} tiles the region into 1 block; blocking is "
            "degenerate. With a variogram-backed size this should not happen; "
            "the plan's G2 fallback (chronological + coarse blocks) applies.")
    lat, lon = cell_positions(cell_ids)
    folds = []
    for blk in part.block_ids:
        val_mask = np.array([part.block_of[cid] == blk for cid in cell_ids])
        if buffer_m > 0:
            train_mask = np.ones(len(cell_ids), dtype=bool)
            # rows = val cells, cols = all cells: for each candidate training
            # cell take its distance to the NEAREST val cell
            d = _haversine(lat[val_mask], lon[val_mask], lat[train_mask], lon[train_mask])
            train_mask = (d.min(axis=1) > buffer_m) & ~val_mask
        else:
            train_mask = ~val_mask
        train = [c for c, m in zip(cell_ids, train_mask) if m]
        if len(train) < max(2, len(cell_ids) // 5):
            raise ValueError(
                f"buffer is swallowing training data for block {blk} "
                f"({len(train)} cells left); shrink block_m or buffer_m")
        folds.append((blk, train, [c for c in cell_ids if c and part.block_of[c] == blk]))
    return folds


def chronological_split(
    dates: list[str],
    train_frac: float = 0.75,
) -> tuple[list[str], list[str]]:
    """Ordered, disjoint date split: the earliest `train_frac` of sorted
    unique dates train, the rest test. No shuffling, no leakage window."""
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be in (0, 1)")
    uniq = sorted(set(dates))
    if len(uniq) < 2:
        raise ValueError("need at least two distinct dates for a split")
    k = int(round(train_frac * len(uniq)))
    k = min(max(k, 1), len(uniq) - 1)
    return uniq[:k], uniq[k:]

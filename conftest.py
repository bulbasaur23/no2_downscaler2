"""Repo-wide pytest configuration.

Windows fixes for pytest's temp machinery (both encountered in practice):

1. The default base in %TEMP% creates a `pytest-current` symlink that
   non-admin sessions cannot unlink during cleanup (WinError 5 Access denied,
   crashing AFTER a fully green run).
2. A fixed repo-local basetemp (.pytest_tmp) breaks when two pytest processes
   run concurrently (e.g. an agent background run and a user run): their
   teardown rmtree calls race and fail.

Solution: each pytest process gets its own `.pytest_tmp/p<PID>` base, and
stale directories from other runs are swept best-effort at configure time.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / ".pytest_tmp"


def pytest_configure(config):
    mine = BASE / f"p{os.getpid()}"
    try:
        BASE.mkdir(exist_ok=True)
    except OSError:
        return  # keep default behaviour rather than failing the session
    # Sweep only STALE directories (older than a day) and only directories:
    # deleting every other child would rmtree the live basetemp of a pytest
    # process that is running right now in another session (CodeRabbit
    # review finding on this file). Age-based is dependency-free; growth is
    # still bounded because each run sweeps its own leftovers next day.
    import time
    cutoff = time.time() - 24 * 3600
    for child in BASE.iterdir():
        try:
            if child.is_dir() and child.name != mine.name and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue
    config.option.basetemp = str(mine)

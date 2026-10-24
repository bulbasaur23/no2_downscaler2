"""Contract tests for weather module constants (offline invariants).

The module itself is EE-bound; what can be pinned without a network are the
band names and column schema that the Parquet schema and the live path both
depend on.
"""
from __future__ import annotations

import no2map.weather as weather


class TestBandContracts:
    def test_precip_band_is_the_deaccumulated_hourly_band(self):
        # Regression pin: ERA5-Land's raw `total_precipitation` accumulates
        # from midnight and resets daily, so lag-window sums across midnight
        # over-count. GEE's `total_precipitation_hourly` is the difference of
        # consecutive forecast steps and is the only sum-safe band.
        # (Found by CodeRabbit review; burned 4 exported months before fix.)
        assert weather.LAND_PRECIP_BAND == "total_precipitation_hourly"

    def test_raw_accumulating_band_is_not_used_anywhere(self):
        src = open(weather.__file__, encoding="utf-8").read()
        assert src.count('"total_precipitation"') == 0

    def test_collections_are_the_reanalysis_datasets(self):
        assert weather.ERA5_LAND == "ECMWF/ERA5_LAND/HOURLY"
        assert weather.ERA5_HOURLY == "ECMWF/ERA5/HOURLY"

    def test_weather_columns_unique_and_known(self):
        cols = weather.WEATHER_COLUMNS
        assert len(cols) == len(set(cols))
        assert {"precip_l24_m", "precip_l48_m"} <= set(cols)

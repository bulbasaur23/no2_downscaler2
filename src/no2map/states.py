"""Error taxonomy (decision D6).

Four distinct outcomes, never mixed into one "value":

  OBSERVED      cell covered by valid TROPOMI pixels
  RECONSTRUCTED cell filled by Model B (never presented as observation)
  INSUFFICIENT  satellite coverage too poor - carries a binary-feel reason code
                (CLOUDS / HAZE / WATER / LOW_COVERAGE)
  SYSTEM_ERROR  GEE, OSM, weather or outage failure - data-source problem,
                NEVER cached as data, always retryable elsewhere
  NO_OVERPASS   the requested date genuinely has no overpass data - a data
                fact, not a failure and not "insufficient"

Mixing SYSTEM_ERROR into data (e.g. caching it) silently corrupts the target
the model trains on; keeping them separate is one of the "never cut" items in
the plan's cut ladder (section 10).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CellState(str, Enum):
    OBSERVED = "observed"
    RECONSTRUCTED = "reconstructed"
    INSUFFICIENT = "insufficient"
    SYSTEM_ERROR = "system_error"
    NO_OVERPASS = "no_overpass"


class InsufficientReason(str, Enum):
    CLOUDS = "clouds"
    HAZE = "haze"  # aerosol flagged as cloud (risk 5); winter-haze gaps cluster by season
    WATER = "water"  # water cell / masked
    LOW_COVERAGE = "low_coverage"  # valid-pixel fraction below the D2 threshold
    NO_OVERPASS = "no_overpass"  # modelled as its own state for dates with zero orbit data


class SystemErrorSource(str, Enum):
    GEE = "gee"  # Google Earth Engine quota, job or auth failure (risk 3)
    OSM = "osm"  # Overpass / osmnx failure while fetching road data (risk 14)
    WEATHER = "weather"  # ERA5 source slow or unavailable (risk 2)
    SATELLITE_OUTAGE = "satellite_outage"  # missing S5P orbits (risk 4)
    UNKNOWN = "unknown"


@dataclass
class CellStatus:
    """One cell's status, always explicit about which codes it carries."""

    state: CellState
    reason: InsufficientReason | None = None
    error_source: SystemErrorSource | None = None

    def is_ok(self) -> bool:
        return self.state in (CellState.OBSERVED, CellState.RECONSTRUCTED)

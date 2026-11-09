"""
Regression tests for the timing fixes:
  - progressions/solar arc must start from each candidate's own birth moment
  - progressed angles via solar arc in RA
  - primary directions computed in mundo (RA for MC/IC, oblique ascension for ASC/DSC)
"""

from datetime import date

import pytest

from src.ephemeris import (
    birth_to_jd, generate_candidate_grid, progressed_angles, angle_diff,
)
from src.models import HouseSystem, LifeEvent, EventType, EventWeight
from src.scoring.primary_directions import (
    ecliptic_to_ra, ecliptic_to_dec, obliquity_at_jd, mundane_arcs_to_angles,
    horizon_hour_angles,
)
from src.scoring.progressions import ProgressionScorer

LAT, LON, TZ = -23.5225, -46.1861, -2.0
BIRTH = date(1992, 2, 5)


@pytest.fixture(scope="module")
def grid():
    return generate_candidate_grid(BIRTH, LAT, LON, TZ, interval_minutes=60)


def test_grid_carries_birth_latitude(grid):
    assert all(c.geo_latitude == LAT for c in grid)


def test_progressed_moon_depends_on_birth_time(grid):
    """
    Put a sensitive point exactly on the progressed Moon of the 00:00 candidate.
    That candidate must hit it at ~0° orb; the 12:00 candidate (Moon ~6.5° further
    on) must not — previously both used the noon placeholder and scored the same.
    """
    import swisseph as swe
    from src.ephemeris import MOSHIER_FLAG

    event = LifeEvent(description="x", event_type=EventType.MARRIAGE,
                      date=date(2020, 1, 1), weight=EventWeight.ANCHOR)
    natal_noon = birth_to_jd(BIRTH, 720, TZ)
    early, late = grid[0], grid[12]
    age = (event.date - BIRTH).days / 365.25
    target = swe.calc_ut(early.julian_day + age, swe.MOON, MOSHIER_FLAG)[0][0]

    def with_asc_at(c):
        return c.model_copy(update={"ascendant": target, "mc": (target - 90) % 360,
                                    "house_cusps": [(target + i * 30) % 360 for i in range(12)]})

    scorer = ProgressionScorer()
    def asc_orb(c):
        hits = [h for h in scorer.score_event(with_asc_at(c), event, natal_noon)
                if h.natal_point == "ProgMOON→ASC"]
        return hits[0].orb if hits else None

    assert asc_orb(early) is not None and asc_orb(early) < 0.05
    assert asc_orb(late) is None or asc_orb(late) > 2.0


def test_progressed_angles_at_age_zero_equal_natal(grid):
    c = grid[5]
    asc, mc = progressed_angles(c.julian_day, c.mc, LAT, 0.0, "P")
    assert angle_diff(asc, c.ascendant) < 0.01
    assert angle_diff(mc, c.mc) < 0.01


def test_progressed_mc_moves_about_one_degree_per_year(grid):
    c = grid[5]
    _, mc = progressed_angles(c.julian_day, c.mc, LAT, 30.0, "P")
    assert 25.0 < (mc - c.mc) % 360 < 36.0


@pytest.mark.parametrize("which", ["ascendant", "mc", "dsc", "ic"])
def test_point_on_an_angle_has_zero_arc_to_it(grid, which):
    c = grid[3]
    obl = obliquity_at_jd(c.julian_day)
    armc = ecliptic_to_ra(c.mc, 0.0, obl)
    lon = getattr(c, which)
    arcs = mundane_arcs_to_angles(ecliptic_to_ra(lon, 0, obl), ecliptic_to_dec(lon, 0, obl), armc, LAT)
    name = {"ascendant": "ASC", "mc": "MC", "dsc": "DSC", "ic": "IC"}[which]
    direct, converse = arcs[name]
    assert min(direct, 360 - direct) < 1e-6
    assert min(converse, 360 - converse) < 1e-6


def test_asc_arc_includes_ascensional_difference():
    """At non-zero latitude and declination, the ASC arc is not RA − (ARMC + 90)."""
    armc, ra, dec = 100.0, 200.0, 20.0
    arcs = mundane_arcs_to_angles(ra, dec, armc, 45.0)
    ra_only = (ra - (armc + 90)) % 360
    assert abs(arcs["ASC"][0] - ra_only) > 10.0


def test_circumpolar_body_has_no_horizon_arcs():
    assert horizon_hour_angles(dec=60.0, geo_lat=60.0) is None
    arcs = mundane_arcs_to_angles(0.0, 60.0, 0.0, 60.0)
    assert set(arcs) == {"MC", "IC"}

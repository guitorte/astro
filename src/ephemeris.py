"""
Ephemeris calculations using pyswisseph (Swiss Ephemeris).
Uses built-in Moshier ephemeris — no external data files needed.
"""

from datetime import date, datetime
import swisseph as swe
import numpy as np
from .models import HouseSystem, CandidateChart

# Use Moshier built-in ephemeris (no external files required)
MOSHIER_FLAG = swe.FLG_MOSEPH

PLANET_CODES: dict[str, int] = {
    "SUN": swe.SUN,
    "MOON": swe.MOON,
    "MERCURY": swe.MERCURY,
    "VENUS": swe.VENUS,
    "MARS": swe.MARS,
    "JUPITER": swe.JUPITER,
    "SATURN": swe.SATURN,
    "URANUS": swe.URANUS,
    "NEPTUNE": swe.NEPTUNE,
    "PLUTO": swe.PLUTO,
}

OUTER_PLANETS = ["SATURN", "URANUS", "NEPTUNE", "PLUTO"]
ALL_PLANETS = list(PLANET_CODES.keys())


def date_to_jd(d: date, hour_ut: float = 12.0) -> float:
    """Convert a calendar date and UT hour to Julian Day number."""
    return swe.julday(d.year, d.month, d.day, hour_ut)


def birth_to_jd(birth_date: date, time_minutes: int, tz_offset: float) -> float:
    """
    Convert birth date + local time (minutes from midnight) to Julian Day (UT).

    Args:
        birth_date: calendar date of birth
        time_minutes: local time expressed as minutes from midnight (0–1439)
        tz_offset: hours east of UTC (e.g. -3.0 for Brazil/São Paulo historical)
    """
    hour_local = time_minutes / 60.0
    hour_ut = hour_local - tz_offset
    return swe.julday(birth_date.year, birth_date.month, birth_date.day, hour_ut)


def calc_planet_positions(jd: float) -> dict[str, float]:
    """
    Calculate ecliptic longitudes for all 10 planets at given Julian Day.
    Returns dict: planet_name → longitude (0–360°).
    """
    positions = {}
    for name, code in PLANET_CODES.items():
        result, retflag = swe.calc_ut(jd, code, MOSHIER_FLAG)
        positions[name] = result[0] % 360
    return positions


def calc_planet_latitudes(jd: float) -> dict[str, float]:
    """
    Calculate ecliptic latitudes for all 10 planets at given Julian Day.
    Returns dict: planet_name → latitude (degrees, negative = south of ecliptic).
    Latitudes are needed for accurate RA conversion in primary directions.
    """
    latitudes = {}
    for name, code in PLANET_CODES.items():
        result, _ = swe.calc_ut(jd, code, MOSHIER_FLAG)
        latitudes[name] = result[1]  # index 1 = ecliptic latitude
    return latitudes


def calc_planet_speed(jd: float, planet_name: str) -> float:
    """Return daily motion (degrees/day) of a planet. Negative = retrograde."""
    code = PLANET_CODES[planet_name]
    result, _ = swe.calc_ut(jd, code, MOSHIER_FLAG | swe.FLG_SPEED)
    return result[3]  # longitude speed in deg/day


def calc_houses(jd: float, lat: float, lon: float,
                house_system: HouseSystem) -> tuple[list[float], float, float]:
    """
    Calculate house cusps, ASC, and MC.

    Returns:
        (cusps_1_to_12, ascendant, midheaven)
    """
    cusps, ascmc = swe.houses(jd, lat, lon, house_system.value.encode())
    return list(cusps[0:12]), ascmc[0], ascmc[1]


def calc_full_chart(jd: float, lat: float, lon: float,
                    house_system: HouseSystem) -> dict:
    """
    Return a complete chart dict with planets, angles, cusps, and planetary latitudes.
    """
    planets = calc_planet_positions(jd)
    planet_lats = calc_planet_latitudes(jd)
    cusps, asc, mc = calc_houses(jd, lat, lon, house_system)
    return {
        "planets": planets,
        "planet_latitudes": planet_lats,
        "cusps": cusps,
        "asc": asc,
        "mc": mc,
        "dsc": (asc + 180) % 360,
        "ic": (mc + 180) % 360,
    }


def angle_diff(a: float, b: float) -> float:
    """Shortest angular distance between two ecliptic longitudes (0–180°)."""
    diff = abs((a - b) % 360)
    return min(diff, 360 - diff)


def generate_candidate_grid(
    birth_date: date,
    lat: float,
    lon: float,
    tz_offset: float,
    interval_minutes: int = 15,
    rising_signs: list[int] | None = None,
    house_system: HouseSystem = HouseSystem.PLACIDUS,
) -> list[CandidateChart]:
    """
    Generate all candidate charts for a 24-hour day at the given interval.

    Args:
        birth_date: date of birth
        lat/lon: geographic coordinates
        tz_offset: historical timezone offset in hours
        interval_minutes: time resolution (15 = 96 candidates, 5 = 288, 1 = 1440)
        rising_signs: if provided, only include candidates with these rising signs (1–12)
        house_system: house system to use for cusp calculations
    """
    candidates = []
    for t in range(0, 1440, interval_minutes):
        jd = birth_to_jd(birth_date, t, tz_offset)
        chart = calc_full_chart(jd, lat, lon, house_system)

        asc_sign = int(chart["asc"] / 30) + 1  # 1–12

        if rising_signs is not None and asc_sign not in rising_signs:
            continue

        candidates.append(
            CandidateChart(
                time_minutes=t,
                julian_day=jd,
                ascendant=chart["asc"],
                mc=chart["mc"],
                house_cusps=chart["cusps"],
                house_system=house_system,
                planets=chart["planets"],
                planet_latitudes=chart["planet_latitudes"],
            )
        )
    return candidates


def secondary_progressed_jd(natal_jd: float, age_years: float) -> float:
    """
    Secondary progression: one day of ephemeris time = one year of life.
    Returns the Julian Day corresponding to the progressed chart.
    """
    return natal_jd + age_years


def solar_arc_for_age(natal_jd: float, age_years: float) -> float:
    """
    Calculate solar arc (degrees to add to all natal points).
    Uses the Naibod key: rate = mean Sun motion ≈ 0.9856°/year.
    Approximated here via the actual progressed Sun position.
    """
    # Progressed Sun position at natal_jd + age_years days
    prog_jd = secondary_progressed_jd(natal_jd, age_years)
    natal_sun, _ = swe.calc_ut(natal_jd, swe.SUN, MOSHIER_FLAG)
    prog_sun, _ = swe.calc_ut(prog_jd, swe.SUN, MOSHIER_FLAG)
    arc = (prog_sun[0] - natal_sun[0]) % 360
    return arc


def profected_asc(natal_asc: float, age_years: int) -> float:
    """
    Annual profection: Ascendant moves one sign (30°) per year.
    Returns the profected ASC longitude.
    """
    return (natal_asc + age_years * 30) % 360


def sign_ruler(longitude: float) -> str:
    """Return the traditional ruler of the sign containing the given longitude."""
    # Traditional (pre-modern) rulerships
    rulers = [
        "MARS",    # Aries
        "VENUS",   # Taurus
        "MERCURY", # Gemini
        "MOON",    # Cancer
        "SUN",     # Leo
        "MERCURY", # Virgo
        "VENUS",   # Libra
        "MARS",    # Scorpio
        "JUPITER", # Sagittarius
        "SATURN",  # Capricorn
        "SATURN",  # Aquarius
        "JUPITER", # Pisces
    ]
    sign_idx = int(longitude / 30) % 12
    return rulers[sign_idx]


def validate_timezone(
    birth_date: date,
    latitude: float,
    longitude: float,
    provided_offset: float,
) -> str | None:
    """
    Warn if the provided UTC offset differs from what zoneinfo reports for the
    given location and date. Returns a warning string or None if offset looks correct.

    Uses Python's `zoneinfo` (stdlib since 3.9) + `timezonefinder` if available.
    Falls back to a longitude-based approximation (+1hr per 15° east) if the package
    is not installed.

    This is advisory only — historical timezone changes (DST, political shifts) mean
    the correct historical offset may differ from what modern databases report.
    """
    try:
        from timezonefinder import TimezoneFinder  # type: ignore
        from zoneinfo import ZoneInfo
        from datetime import datetime as dt

        tf = TimezoneFinder()
        tz_name = tf.timezone_at(lat=latitude, lng=longitude)
        if tz_name:
            tz = ZoneInfo(tz_name)
            dt_naive = dt(birth_date.year, birth_date.month, birth_date.day, 12, 0, 0)
            dt_aware = dt_naive.replace(tzinfo=tz)
            utc_offset_hours = dt_aware.utcoffset().total_seconds() / 3600  # type: ignore
            diff = abs(utc_offset_hours - provided_offset)
            if diff >= 0.5:
                return (
                    f"Timezone warning: provided offset {provided_offset:+.1f}h, "
                    f"but {tz_name} on {birth_date} is UTC{utc_offset_hours:+.1f}. "
                    f"Difference: {diff:.1f}h. Verify DST and historical timezone rules."
                )
    except ImportError:
        # timezonefinder not installed — use longitude approximation
        approx_offset = round(longitude / 15)
        diff = abs(approx_offset - provided_offset)
        if diff >= 2:
            return (
                f"Timezone warning: provided offset {provided_offset:+.1f}h, "
                f"but longitude {longitude:.1f}° suggests ~UTC{approx_offset:+d}. "
                f"Install 'timezonefinder' for accurate DST-aware validation."
            )
    except Exception:
        pass  # Don't let validation errors break the main workflow

    return None

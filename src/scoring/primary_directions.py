"""
Technique 1 (highest weight): Ptolemaic primary directions.

Primary directions are the oldest and most mathematically precise timing
technique in traditional astrology. The celestial sphere rotates at the rate
of ~1° of Right Ascension per year (Naibod key: 0.9856° RA/year, or 59'8"/year).

Method:
    - Convert natal ASC and MC to Right Ascension (RAMC-based)
    - Compute the arc (in RA degrees) between a natal point and a promittor
    - That arc in degrees = years of life until the event

Key directions:
    - Planets directed (promissor) to natal ASC/MC (significator)
    - Natal ASC/MC directed to natal planets

The Naibod arc key is used here (most common traditional choice).
Arcs are computed in mundo from hour angles: right ascension for the MC/IC and
oblique ascension/descension (via the ascensional difference at the birth
latitude) for the ASC/DSC. Both direct and converse directions are scored.

Since primary directions use a completely different mathematical framework
(Right Ascension, not ecliptic longitude), they are treated as genuinely
independent from progressions and solar arc. Highest technique weight.
"""

import math
import swisseph as swe
from datetime import date as date_type
from ..models import CandidateChart, LifeEvent, TechniqueScore
from .base import BaseScorer, orb_score, cap_hits
from ..ephemeris import MOSHIER_FLAG

# Primary directions: fully independent mathematical framework (diurnal rotation,
# right ascension / oblique ascension), so it keeps the highest technique weight.
TECHNIQUE_WEIGHT = 2.0

# Naibod arc key: degrees of RA per year
NAIBOD_RATE = 0.9856  # degrees RA per year

# Directions beyond this arc (≈ age 100 under Naibod) are ignored.
MAX_ARC = 110.0


def ecliptic_to_ra(lon: float, lat: float, obliquity: float) -> float:
    """Convert ecliptic longitude/latitude to Right Ascension (degrees)."""
    lon_r = math.radians(lon)
    lat_r = math.radians(lat)
    obl_r = math.radians(obliquity)
    y = math.sin(lon_r) * math.cos(obl_r) - math.tan(lat_r) * math.sin(obl_r)
    x = math.cos(lon_r)
    return math.degrees(math.atan2(y, x)) % 360


def ecliptic_to_dec(lon: float, lat: float, obliquity: float) -> float:
    """Convert ecliptic longitude/latitude to declination (degrees)."""
    lon_r, lat_r, obl_r = map(math.radians, (lon, lat, obliquity))
    sin_dec = (math.sin(lat_r) * math.cos(obl_r)
               + math.cos(lat_r) * math.sin(obl_r) * math.sin(lon_r))
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_dec))))


def obliquity_at_jd(jd: float) -> float:
    """Return obliquity of ecliptic at given Julian Day (degrees)."""
    eps, _ = swe.calc_ut(jd, swe.ECL_NUT, MOSHIER_FLAG)
    return eps[0]


def horizon_hour_angles(dec: float, geo_lat: float) -> tuple[float, float] | None:
    """
    Hour angles (degrees, measured westward from the meridian, 0–360) at which a body
    of the given declination rises and sets: (rising, setting).
    Returns None for circumpolar / never-rising bodies.
    """
    x = -math.tan(math.radians(geo_lat)) * math.tan(math.radians(dec))
    if abs(x) > 1.0:
        return None
    dsa = math.degrees(math.acos(x))  # diurnal semi-arc
    return (360.0 - dsa) % 360, dsa


def mundane_arcs_to_angles(
    ra: float, dec: float, armc: float, geo_lat: float
) -> dict[str, tuple[float, float]]:
    """
    Primary arcs (in mundo) for a body to reach each angle by diurnal rotation.

    The body's hour angle is H = ARMC − RA. As the sphere turns, H grows by the
    same amount as ARMC, so the direct arc to an angle with target hour angle Ht is
    (Ht − H) mod 360 and the converse arc is (H − Ht) mod 360.

      MC:  Ht = 0         IC: Ht = 180
      ASC: Ht = rising hour angle   DSC: Ht = setting hour angle
    The ASC/DSC targets depend on the body's declination and the birth latitude
    (ascensional difference) — this is what an RA-only shortcut gets wrong.

    Returns {angle: (direct_arc, converse_arc)}.
    """
    h = (armc - ra) % 360
    targets = {"MC": 0.0, "IC": 180.0}
    hz = horizon_hour_angles(dec, geo_lat)
    if hz is not None:
        targets["ASC"], targets["DSC"] = hz
    return {
        name: ((ht - h) % 360, (h - ht) % 360) for name, ht in targets.items()
    }


class PrimaryDirectionScorer(BaseScorer):
    """
    Score Ptolemaic primary directions using the Naibod arc key.
    Planets directed to angles, and angles directed to planets.
    """

    name = "primary_direction"

    def __init__(self, tight: bool = False):
        self.tight = tight
        # Tolerance in years (translates from arc degrees via Naibod rate)
        self.year_orb = 1.5 if not tight else 0.75

    def _age_at_event(self, birth_date: date_type, event_date: date_type) -> float:
        return (event_date - birth_date).days / 365.25

    def score_event(
        self,
        candidate: CandidateChart,
        event: LifeEvent,
        natal_jd: float,
    ) -> list[TechniqueScore]:
        scores: list[TechniqueScore] = []

        birth_parts = swe.revjul(natal_jd)
        birth_date = date_type(birth_parts[0], birth_parts[1], int(birth_parts[2]))
        age = self._age_at_event(birth_date, event.date)
        if age <= 0:
            return scores

        obl = obliquity_at_jd(candidate.julian_day)
        armc = ecliptic_to_ra(candidate.mc, 0.0, obl)
        # Legacy callers without a latitude get an equatorial approximation
        # (ASC/DSC arcs then equal RA arcs ± 90°).
        geo_lat = candidate.geo_latitude if candidate.geo_latitude is not None else 0.0

        expected_arc = age * NAIBOD_RATE
        arc_tolerance = self.year_orb * NAIBOD_RATE

        for planet_name, planet_lon in candidate.planets.items():
            planet_lat = candidate.planet_latitudes.get(planet_name, 0.0)
            ra = ecliptic_to_ra(planet_lon, planet_lat, obl)
            dec = ecliptic_to_dec(planet_lon, planet_lat, obl)
            arcs = mundane_arcs_to_angles(ra, dec, armc, geo_lat)
            for angle_name, (direct, converse) in arcs.items():
                for label, arc in (("d", direct), ("c", converse)):
                    if arc > MAX_ARC:
                        continue
                    diff = abs(arc - expected_arc)
                    if diff > arc_tolerance:
                        continue
                    orb_equiv = diff / NAIBOD_RATE  # in years
                    raw = orb_score(min(orb_equiv, 2.0), tight=self.tight)
                    if raw <= 0:
                        continue
                    scores.append(
                        TechniqueScore(
                            technique=self.name,
                            event_description=event.description,
                            natal_point=f"PD_{planet_name}→{angle_name}({label})",
                            orb=round(orb_equiv, 4),
                            score=raw * TECHNIQUE_WEIGHT * event.weight_multiplier(),
                            held_out=event.held_out,
                            time_minutes=candidate.time_minutes,
                        )
                    )

        return cap_hits(scores)

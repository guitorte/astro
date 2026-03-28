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
True solar arc key is a close alternative but requires more computation.

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

# Primary directions: fully independent mathematical framework.
# Weight reduced to 1.0 (from 2.0) because this implementation uses a simplified
# RA approximation (ecliptic lat=0, no oblique ascension). A proper implementation
# using OA + ascensional difference would warrant the full 2.0 weight.
TECHNIQUE_WEIGHT = 1.0

# Naibod arc key: degrees of RA per year
NAIBOD_RATE = 0.9856  # degrees RA per year


def ecliptic_to_ra(lon: float, lat: float, obliquity: float) -> float:
    """Convert ecliptic longitude/latitude to Right Ascension (degrees)."""
    lon_r = math.radians(lon)
    lat_r = math.radians(lat)
    obl_r = math.radians(obliquity)

    # RA formula
    y = math.sin(lon_r) * math.cos(obl_r) - math.tan(lat_r) * math.sin(obl_r)
    x = math.cos(lon_r)
    ra = math.degrees(math.atan2(y, x)) % 360
    return ra


def obliquity_at_jd(jd: float) -> float:
    """Return mean obliquity of ecliptic at given Julian Day (degrees)."""
    eps, _ = swe.calc_ut(jd, swe.ECL_NUT, MOSHIER_FLAG)
    return eps[0]  # mean obliquity


def primary_arc(
    promissor_ra: float, significator_ra: float, is_upper: bool = True
) -> float:
    """
    Compute the primary arc (in degrees RA) between promissor and significator
    under the Regiomontanus or simplified RA method.
    Returns arc in degrees (>= 0), sign = direction of motion.
    """
    if is_upper:
        arc = (significator_ra - promissor_ra) % 360
    else:
        arc = (promissor_ra - significator_ra) % 360
    # Normalize: use the shorter arc
    if arc > 180:
        arc = 360 - arc
    return arc


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

        obl = obliquity_at_jd(natal_jd)

        # Convert angles to RA
        # ASC and MC are already in ecliptic longitude — convert to RA
        asc_ra = ecliptic_to_ra(candidate.ascendant, 0.0, obl)
        mc_ra = ecliptic_to_ra(candidate.mc, 0.0, obl)

        angle_ras = {
            "ASC": asc_ra,
            "MC": mc_ra,
            "DSC": (asc_ra + 180) % 360,
            "IC": (mc_ra + 180) % 360,
        }

        # Convert natal planets to RA using actual ecliptic latitudes.
        # Planets have non-zero ecliptic latitude (up to ~17° for Pluto, ~8° for Moon);
        # using lat=0 causes systematic RA errors that generate spurious tight-orb hits.
        planet_ras: dict[str, float] = {}
        for planet_name, planet_lon in candidate.planets.items():
            planet_lat = candidate.planet_latitudes.get(planet_name, 0.0)
            planet_ras[planet_name] = ecliptic_to_ra(planet_lon, planet_lat, obl)

        # Expected arc = age * NAIBOD_RATE
        expected_arc = age * NAIBOD_RATE
        arc_tolerance = self.year_orb * NAIBOD_RATE  # degrees of arc ≈ years × rate

        # Planets (promissor) directed to angles (significator)
        for planet_name, planet_ra in planet_ras.items():
            for angle_name, angle_ra in angle_ras.items():
                arc = primary_arc(planet_ra, angle_ra)
                diff = abs(arc - expected_arc)
                if diff <= arc_tolerance:
                    # Convert arc difference to orb in degrees
                    orb_equiv = diff / NAIBOD_RATE  # years ≈ degrees equivalent
                    raw = orb_score(min(orb_equiv, 2.0), tight=self.tight)
                    if raw > 0:
                        scores.append(
                            TechniqueScore(
                                technique=self.name,
                                event_description=event.description,
                                natal_point=f"PD_{planet_name}→{angle_name}",
                                orb=round(orb_equiv, 4),
                                score=raw * TECHNIQUE_WEIGHT * event.weight_multiplier(),
                                held_out=event.held_out,
                                time_minutes=candidate.time_minutes,
                            )
                        )

        # Angles (promissor) directed to planets (significator)
        for angle_name, angle_ra in angle_ras.items():
            for planet_name, planet_ra in planet_ras.items():
                arc = primary_arc(angle_ra, planet_ra)
                diff = abs(arc - expected_arc)
                if diff <= arc_tolerance:
                    orb_equiv = diff / NAIBOD_RATE
                    raw = orb_score(min(orb_equiv, 2.0), tight=self.tight)
                    if raw > 0:
                        scores.append(
                            TechniqueScore(
                                technique=self.name,
                                event_description=event.description,
                                natal_point=f"PD_{angle_name}→{planet_name}",
                                orb=round(orb_equiv, 4),
                                score=raw * TECHNIQUE_WEIGHT * event.weight_multiplier(),
                                held_out=event.held_out,
                                time_minutes=candidate.time_minutes,
                            )
                        )

        return cap_hits(scores)

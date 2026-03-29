"""
Technique 2 (weighted second): Outer-planet transits to natal angles.

Outer planets: Saturn, Uranus, Neptune, Pluto.
Only conjunctions and oppositions to the 4 natal angles (ASC/MC/DSC/IC) score.

Methodology ("Rifle, Not Shotgun"):
- Angles are the only birth-time-sensitive targets (~4°/hour movement).
  House cusps add noise without discriminating power.
- Only conjunction and opposition aspects score — these are the classical
  "hard aspect" contacts with clear astrological meaning and unambiguous timing.
- Weight raised to 2.0: this is the second-strongest birth-time discriminator
  after the progressed Moon.
- Orbs tightened to 1.5° standard / 0.75° tight.
"""

import swisseph as swe
from ..models import CandidateChart, LifeEvent, TechniqueScore
from .base import BaseScorer, angle_diff, get_angles, cap_hits

OUTER_PLANET_CODES: dict[str, int] = {
    "SATURN": swe.SATURN,
    "URANUS": swe.URANUS,
    "NEPTUNE": swe.NEPTUNE,
    "PLUTO": swe.PLUTO,
}

MOSHIER_FLAG = swe.FLG_MOSEPH

# Raised to 2.0 — transits to angles are the second-strongest birth-time discriminator.
TECHNIQUE_WEIGHT = 2.0

# Tightened orbs: 1.5° standard / 0.75° tight (was 2.0° / 1.0°)
_ORB_STANDARD = 1.5
_ORB_TIGHT = 0.75


def _transit_orb_score(orb: float, tight: bool) -> float:
    """Award points based on orb size with tightened thresholds."""
    limit = _ORB_TIGHT if tight else _ORB_STANDARD
    if orb > limit:
        return 0.0
    # Linearly scale: full score at orb=0, zero at orb=limit
    fraction = 1.0 - (orb / limit)
    # Discretise into 3 levels to match orb_score() pattern
    if fraction >= 0.67:
        return 3.0
    elif fraction >= 0.33:
        return 2.0
    else:
        return 1.0


class TransitScorer(BaseScorer):
    """Score outer-planet transits (conjunction + opposition only) to natal angles."""

    name = "transit"

    def __init__(self, tight: bool = False):
        self.tight = tight

    def score_event(
        self,
        candidate: CandidateChart,
        event: LifeEvent,
        natal_jd: float,
    ) -> list[TechniqueScore]:
        scores: list[TechniqueScore] = []
        event_jd = swe.julday(
            event.date.year, event.date.month, event.date.day, 12.0
        )

        # Only the 4 angles — angles move ~4°/hour, making them birth-time-sensitive.
        # House cusps excluded: they add combinatorial noise without extra discrimination.
        angles = get_angles(candidate)

        for planet_name, planet_code in OUTER_PLANET_CODES.items():
            transit_pos, _ = swe.calc_ut(event_jd, planet_code, MOSHIER_FLAG)
            transit_lon = transit_pos[0]

            for angle_name, angle_lon in angles.items():
                # Conjunction check
                conj_orb = angle_diff(transit_lon, angle_lon)
                raw = _transit_orb_score(conj_orb, tight=self.tight)
                if raw > 0:
                    scores.append(
                        TechniqueScore(
                            technique=self.name,
                            event_description=event.description,
                            natal_point=f"{planet_name}⊙{angle_name}",
                            orb=round(conj_orb, 4),
                            score=raw * TECHNIQUE_WEIGHT * event.weight_multiplier(),
                            held_out=event.held_out,
                            time_minutes=candidate.time_minutes,
                        )
                    )

                # Opposition check: shift transit by 180° and recheck
                opp_lon = (transit_lon + 180.0) % 360
                opp_orb = angle_diff(opp_lon, angle_lon)
                raw_opp = _transit_orb_score(opp_orb, tight=self.tight)
                if raw_opp > 0:
                    scores.append(
                        TechniqueScore(
                            technique=self.name,
                            event_description=event.description,
                            natal_point=f"{planet_name}☍{angle_name}",
                            orb=round(opp_orb, 4),
                            score=raw_opp * TECHNIQUE_WEIGHT * event.weight_multiplier(),
                            held_out=event.held_out,
                            time_minutes=candidate.time_minutes,
                        )
                    )

        return cap_hits(scores)

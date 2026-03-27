"""
Technique 5 (weighted lowest): Outer-planet transits to natal angles and
event-relevant house cusps.

Outer planets: Saturn, Uranus, Neptune, Pluto.
Only transits to *time-sensitive* points (angles, ruled cusps) score.
Transits to natal planets that are not angles are excluded — they fire for
everyone born in the same year and carry zero discriminating power.
"""

import swisseph as swe
from ..models import CandidateChart, LifeEvent, TechniqueScore
from .base import BaseScorer, angle_diff, orb_score, get_event_sensitive_points

OUTER_PLANET_CODES: dict[str, int] = {
    "SATURN": swe.SATURN,
    "URANUS": swe.URANUS,
    "NEPTUNE": swe.NEPTUNE,
    "PLUTO": swe.PLUTO,
}

MOSHIER_FLAG = swe.FLG_MOSEPH

# Correlation penalty: transits are one independent witness.
TECHNIQUE_WEIGHT = 1.0


class TransitScorer(BaseScorer):
    """Score outer-planet transits to natal angles and event-relevant cusps."""

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
        sensitive_points = get_event_sensitive_points(candidate, event)

        for planet_name, planet_code in OUTER_PLANET_CODES.items():
            transit_pos, _ = swe.calc_ut(event_jd, planet_code, MOSHIER_FLAG)
            transit_lon = transit_pos[0]

            for point_name, point_lon in sensitive_points.items():
                orb = angle_diff(transit_lon, point_lon)
                raw = orb_score(orb, tight=self.tight)
                if raw > 0:
                    final_score = raw * TECHNIQUE_WEIGHT * event.weight_multiplier()
                    scores.append(
                        TechniqueScore(
                            technique=self.name,
                            event_description=event.description,
                            natal_point=f"{planet_name}→{point_name}",
                            orb=round(orb, 4),
                            score=final_score,
                            held_out=event.held_out,
                            time_minutes=candidate.time_minutes,
                        )
                    )
        return scores

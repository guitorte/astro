"""
Technique 3: Solar arc directions.

Every point in the natal chart is advanced by the same arc (the arc
the progressed Sun has moved since birth). The rate is ~0.9856°/year (Naibod key).
Angular hits — directed planet reaching within orb of natal ASC/MC — are the
primary signal.

Correlation penalty applied: solar arc and secondary progressions share the
progressed Sun arc as their basis. Their combined weight is capped so they
don't double-count the same underlying signal.
"""

import swisseph as swe
from datetime import date as date_type
from ..models import CandidateChart, LifeEvent, TechniqueScore
from .base import BaseScorer, angle_diff, orb_score, get_angles
from ..ephemeris import solar_arc_for_age, MOSHIER_FLAG

# Naibod mean rate (degrees per year) — used as fallback
NAIBOD_RATE = 0.9856

# Correlation penalty with progressions: cap at 1.2
TECHNIQUE_WEIGHT = 1.2


class SolarArcScorer(BaseScorer):
    """Score solar arc directions: directed planets to natal angles, and vice versa."""

    name = "solar_arc"

    def __init__(self, tight: bool = False, use_naibod: bool = False):
        self.tight = tight
        self.use_naibod = use_naibod

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

        # Compute solar arc
        if self.use_naibod:
            arc = age * NAIBOD_RATE
        else:
            arc = solar_arc_for_age(natal_jd, age)

        angles = get_angles(candidate)

        # Directed natal planets → natal angles (most powerful)
        for planet_name, planet_lon in candidate.planets.items():
            directed_lon = (planet_lon + arc) % 360
            for angle_name, angle_lon in angles.items():
                orb = angle_diff(directed_lon, angle_lon)
                raw = orb_score(orb, tight=self.tight)
                if raw > 0:
                    scores.append(
                        TechniqueScore(
                            technique=self.name,
                            event_description=event.description,
                            natal_point=f"SA_{planet_name}→{angle_name}",
                            orb=round(orb, 4),
                            score=raw * TECHNIQUE_WEIGHT * event.weight_multiplier(),
                            held_out=event.held_out,
                            time_minutes=candidate.time_minutes,
                        )
                    )

        # Directed angles → natal planets (also informative)
        directed_asc = (candidate.ascendant + arc) % 360
        directed_mc = (candidate.mc + arc) % 360

        for planet_name, planet_lon in candidate.planets.items():
            for dir_name, dir_lon in [("SA_ASC", directed_asc), ("SA_MC", directed_mc)]:
                orb = angle_diff(dir_lon, planet_lon)
                raw = orb_score(orb, tight=self.tight)
                if raw > 0:
                    scores.append(
                        TechniqueScore(
                            technique=self.name,
                            event_description=event.description,
                            natal_point=f"{dir_name}→{planet_name}",
                            orb=round(orb, 4),
                            score=raw * TECHNIQUE_WEIGHT * event.weight_multiplier(),
                            held_out=event.held_out,
                            time_minutes=candidate.time_minutes,
                        )
                    )

        return scores

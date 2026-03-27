"""
Technique 4: Annual profections.

The Ascendant advances one whole sign (30°) per year. The sign it lands in
activates its ruler (the "lord of the year") and any planets in that sign.
Scoring: if the lord of the year is also activated by a transit, progression,
or solar arc during the event year, it confirms the event.

As an independent technique (different time scale, different mathematical basis
from directions), profections receive a standard weight. They act as a
corroborating filter — the lord of the year SHOULD be active by another
technique during significant events.
"""

import swisseph as swe
from datetime import date as date_type
from ..models import CandidateChart, LifeEvent, TechniqueScore
from .base import BaseScorer, angle_diff, orb_score
from ..ephemeris import profected_asc, sign_ruler, MOSHIER_FLAG

# Profections are independent (different time scale) → full weight
TECHNIQUE_WEIGHT = 1.0


class ProfectionScorer(BaseScorer):
    """
    Score annual profections: active lord testing.

    A hit is recorded when the lord of the profected year is:
    1. Also the natal chart ruler (confirming baseline activation), OR
    2. In hard aspect (conjunction/opposition) to a transiting outer planet
       on the event date.
    """

    name = "profection"

    def _age_at_event(self, birth_date: date_type, event_date: date_type) -> int:
        """Return floor age (complete years) at event date."""
        age = event_date.year - birth_date.year
        if (event_date.month, event_date.day) < (birth_date.month, birth_date.day):
            age -= 1
        return max(age, 0)

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

        # Profected ASC: natal ASC + age * 30°
        prof_asc_lon = profected_asc(candidate.ascendant, age)
        lord = sign_ruler(prof_asc_lon)
        prof_sign_idx = int(prof_asc_lon / 30) % 12

        # Check if lord of year is activated by outer planet transit on event date
        event_jd = swe.julday(event.date.year, event.date.month, event.date.day, 12.0)

        outer_planet_codes = {
            "SATURN": swe.SATURN,
            "JUPITER": swe.JUPITER,  # Jupiter included for profections
            "URANUS": swe.URANUS,
            "NEPTUNE": swe.NEPTUNE,
            "PLUTO": swe.PLUTO,
        }

        # Lord of year natal position
        lord_natal_lon = candidate.planets.get(lord)
        if lord_natal_lon is None:
            return scores

        for outer_name, outer_code in outer_planet_codes.items():
            transit_pos, _ = swe.calc_ut(event_jd, outer_code, MOSHIER_FLAG)
            transit_lon = transit_pos[0]

            # Outer planet conjunct or opposite the natal lord of year
            orb = angle_diff(transit_lon, lord_natal_lon)
            raw = orb_score(orb)
            if raw > 0:
                scores.append(
                    TechniqueScore(
                        technique=self.name,
                        event_description=event.description,
                        natal_point=f"ProfASC(age{age})→lord={lord}×{outer_name}",
                        orb=round(orb, 4),
                        score=raw * TECHNIQUE_WEIGHT * event.weight_multiplier(),
                        held_out=event.held_out,
                        time_minutes=candidate.time_minutes,
                    )
                )

            # Outer planet conjunct the profected sign itself
            prof_sign_lon = prof_sign_idx * 30 + 15  # midpoint of profected sign
            orb2 = angle_diff(transit_lon, prof_sign_lon)
            raw2 = orb_score(orb2)
            if raw2 > 0 and orb2 < 3.0:  # slightly wider orb for sign activation
                scores.append(
                    TechniqueScore(
                        technique=self.name + "_sign",
                        event_description=event.description,
                        natal_point=f"ProfSign(age{age})×{outer_name}",
                        orb=round(orb2, 4),
                        score=raw2 * 0.5 * event.weight_multiplier(),  # half weight
                        held_out=event.held_out,
                        time_minutes=candidate.time_minutes,
                    )
                )

        return scores

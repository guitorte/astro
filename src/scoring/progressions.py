"""
Technique 2 (weighted second highest): Secondary progressions.

One day of ephemeris time after natal = one year of life.
The progressed Moon is the most discriminating signal:
  - Its house ingresses are sensitive to birth time (±10 min shifts ingress ~2.5 days).
  - Its aspects to natal angles pinpoint emotional/life events.
Progressed angles to natal planets are also scored.

Correlation note: secondary progressions share mathematical structure with
solar arc (both derived from the progressed Sun arc). A correlation penalty
is applied so they don't count as fully independent witnesses.
"""

import swisseph as swe
from ..models import CandidateChart, LifeEvent, TechniqueScore
from .base import BaseScorer, angle_diff, orb_score, get_event_sensitive_points, cap_hits
from ..ephemeris import secondary_progressed_jd, MOSHIER_FLAG

# Progressed Moon is the strongest birth-time discriminator:
# ~13°/day in ephemeris = ~0.54°/hour of birth time.
# A 30-min birth-time shift moves the progressed Moon ~0.27° — significant within 1.5° orb.
# Weight raised to 3.0 in the "Rifle, Not Shotgun" methodology revision.
TECHNIQUE_WEIGHT = 3.0

# Event-type to house mapping for ingress relevance scoring
_INGRESS_HOUSE_MAP: dict[str, list[int]] = {
    "marriage": [7, 5],
    "divorce": [7, 12],
    "death_of_parent": [4, 10],
    "career_peak": [10, 6],
    "accident": [1, 8, 12],
    "illness": [6, 1, 12],
    "relocation": [4, 9],
    "custody_loss": [5, 4, 12],
    "legal_restriction": [4, 10, 12],
    "hospitalization": [12, 6, 1],
    "birth_of_child": [5, 11],
}


class ProgressionScorer(BaseScorer):
    """Score secondary progressions: progressed Moon and progressed angles."""

    name = "progression"

    def __init__(self, tight: bool = False):
        self.tight = tight

    def _age_at_event(self, birth_date, event_date) -> float:
        """Calculate fractional age in years at event date."""
        delta = (event_date - birth_date).days
        return delta / 365.25

    def score_event(
        self,
        candidate: CandidateChart,
        event: LifeEvent,
        natal_jd: float,
    ) -> list[TechniqueScore]:
        from datetime import date as date_type
        import math

        scores: list[TechniqueScore] = []

        # Derive birth date from natal_jd (day-level precision)
        birth_parts = swe.revjul(natal_jd)
        birth_date = date_type(birth_parts[0], birth_parts[1], int(birth_parts[2]))

        age = self._age_at_event(birth_date, event.date)
        if age <= 0:
            return scores

        # CRITICAL: Use the candidate's actual birth JD, not the noon reference JD.
        # "One day after birth = one year of life" requires the exact birth moment.
        # Using noon (natal_jd) instead of the actual birth time shifts the progressed
        # Moon by ~5° for a birth near midnight — catastrophic for 2° orb scoring,
        # and it makes the progressed Moon identical across all candidates, destroying
        # the Moon's discriminating power (which is the strongest timing signal).
        prog_jd = secondary_progressed_jd(candidate.julian_day, age)

        # --- Progressed Moon aspects to natal sensitive points ---
        prog_moon, _ = swe.calc_ut(prog_jd, swe.MOON, MOSHIER_FLAG)
        prog_moon_lon = prog_moon[0]

        # Tighter Moon orbs: conjunction/opposition only
        # Standard: 1.5°, Tight: 0.75° (vs general 2.0°/1.0°)
        moon_orb_limit = 0.75 if self.tight else 1.5

        sensitive_points = get_event_sensitive_points(candidate, event)
        for point_name, point_lon in sensitive_points.items():
            orb = angle_diff(prog_moon_lon, point_lon)
            if orb <= moon_orb_limit:
                # Score within tighter Moon-specific orb
                raw = orb_score(orb, tight=self.tight)
                if raw > 0:
                    scores.append(
                        TechniqueScore(
                            technique=self.name,
                            event_description=event.description,
                            natal_point=f"ProgMOON→{point_name}",
                            orb=round(orb, 4),
                            score=raw * TECHNIQUE_WEIGHT * event.weight_multiplier(),
                            held_out=event.held_out,
                            time_minutes=candidate.time_minutes,
                        )
                    )

        # --- Progressed Moon house ingress check ---
        # Detect if the progressed Moon crossed a house cusp around the event
        # by checking position ±30 days (in progressed time)
        days_window = 30
        moon_before, _ = swe.calc_ut(prog_jd - days_window / 365.25, swe.MOON, MOSHIER_FLAG)
        moon_after, _ = swe.calc_ut(prog_jd + days_window / 365.25, swe.MOON, MOSHIER_FLAG)

        event_houses = _INGRESS_HOUSE_MAP.get(event.event_type.value, [])

        for h, cusp_lon in enumerate(candidate.house_cusps):
            # Check if the Moon crossed this cusp in the window
            lon_b = moon_before[0]
            lon_a = moon_after[0]
            # Normalise movement direction
            motion = (lon_a - lon_b) % 360
            dist_b = (cusp_lon - lon_b) % 360
            if 0 < motion < 180 and dist_b < motion:
                house_num = h + 1
                # Enhanced ingress scoring: relevant house gets 3.0, others 1.0
                if house_num in event_houses:
                    ingress_pts = 3.0 * event.weight_multiplier()
                else:
                    ingress_pts = 1.0 * event.weight_multiplier()
                scores.append(
                    TechniqueScore(
                        technique=self.name + "_ingress",
                        event_description=event.description,
                        natal_point=f"ProgMOON→H{house_num}",
                        orb=0.0,
                        score=ingress_pts,
                        held_out=event.held_out,
                        time_minutes=candidate.time_minutes,
                    )
                )
                break  # at most one ingress per event

        # NOTE: Progressed angles (ProgASC/ProgMC) removed — the crude approximation
        # (ASC + age * 1.0°/year) is mathematically incorrect and adds noise.
        # Proper progressed angles require full RAMC-based computation.

        return cap_hits(scores)

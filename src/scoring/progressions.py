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

# Partial correlation with solar arc → weight reduced to 1.5 (vs 2.0 for primaries)
TECHNIQUE_WEIGHT = 1.5


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

        # Derive birth date from natal_jd (reverse engineering)
        birth_parts = swe.revjul(natal_jd)
        birth_date = date_type(birth_parts[0], birth_parts[1], int(birth_parts[2]))

        age = self._age_at_event(birth_date, event.date)
        if age <= 0:
            return scores

        prog_jd = secondary_progressed_jd(natal_jd, age)

        # --- Progressed Moon aspects to natal sensitive points ---
        prog_moon, _ = swe.calc_ut(prog_jd, swe.MOON, MOSHIER_FLAG)
        prog_moon_lon = prog_moon[0]

        sensitive_points = get_event_sensitive_points(candidate, event)
        for point_name, point_lon in sensitive_points.items():
            orb = angle_diff(prog_moon_lon, point_lon)
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
        # by checking position ±30 days
        days_window = 30
        moon_before, _ = swe.calc_ut(prog_jd - days_window / 365.25, swe.MOON, MOSHIER_FLAG)
        moon_after, _ = swe.calc_ut(prog_jd + days_window / 365.25, swe.MOON, MOSHIER_FLAG)

        for h, cusp_lon in enumerate(candidate.house_cusps):
            # Check if the Moon crossed this cusp in the window
            lon_b = moon_before[0]
            lon_a = moon_after[0]
            # Normalise movement direction
            motion = (lon_a - lon_b) % 360
            dist_b = (cusp_lon - lon_b) % 360
            if 0 < motion < 180 and dist_b < motion:
                ingress_score = 1.5 * event.weight_multiplier()
                scores.append(
                    TechniqueScore(
                        technique=self.name + "_ingress",
                        event_description=event.description,
                        natal_point=f"ProgMOON→H{h+1}",
                        orb=0.0,
                        score=ingress_score,
                        held_out=event.held_out,
                        time_minutes=candidate.time_minutes,
                    )
                )
                break  # at most one ingress per event

        # --- Progressed ASC/MC aspects to natal planets --- (continued)
        prog_asc_approx = (candidate.ascendant + age * 1.0) % 360  # rough arc
        prog_mc_approx = (candidate.mc + age * 1.0) % 360

        for planet_name, planet_lon in candidate.planets.items():
            for prog_angle_name, prog_angle_lon in [
                ("ProgASC", prog_asc_approx),
                ("ProgMC", prog_mc_approx),
            ]:
                orb = angle_diff(prog_angle_lon, planet_lon)
                raw = orb_score(orb, tight=self.tight)
                if raw > 0:
                    scores.append(
                        TechniqueScore(
                            technique=self.name,
                            event_description=event.description,
                            natal_point=f"{prog_angle_name}→{planet_name}",
                            orb=round(orb, 4),
                            score=raw * TECHNIQUE_WEIGHT * event.weight_multiplier(),
                            held_out=event.held_out,
                            time_minutes=candidate.time_minutes,
                        )
                    )

        return cap_hits(scores)

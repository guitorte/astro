"""
Angle-first scoring, after Carol Tebbs, "The Complete Book of Chart Rectification"
(Llewellyn), chapters 1, 2 and 8 and appendices B and C.

What is taken from that book:
  - Only the four angles are scored. House cusps are left out: they depend on the
    house system and are described as rarely useful for rectification.
  - Solar arc directions lead the search. Secondary progressions and outer-planet
    transits are used to confirm.
  - The orb is 1 degree (0.5 in tight mode), and a closer contact counts for more.
  - Painful, dramatic events are weighted above pleasant ones (the book estimates
    four to five times as helpful).

What is a choice of this implementation, not of the book:
  - Each technique contributes at most one contact per event (its closest), so a
    chart cannot win by piling up loose coincidences.
  - Only conjunctions with the four angles are scored. Because ASC/DSC and MC/IC are
    opposite pairs this covers oppositions too; squares are not scored.
  - How the three techniques are combined (see AngleFirstConfig). The book reports
    Dobyns' rule that an event should show in all three systems; `mode="strict"`
    implements that literally, the default does not (see README for measurements).
"""

from dataclasses import dataclass
from datetime import date as date_type
from functools import lru_cache

import numpy as np
import swisseph as swe

from ..ephemeris import MOSHIER_FLAG, solar_arc_for_age
from ..models import CandidateChart, EventType, LifeEvent, TechniqueScore
from .base import BaseScorer

OUTER_TRANSIT_BODIES = {
    "SATURN": swe.SATURN, "URANUS": swe.URANUS,
    "NEPTUNE": swe.NEPTUNE, "PLUTO": swe.PLUTO,
}
PROGRESSED_BODIES = {
    "SUN": swe.SUN, "MOON": swe.MOON, "MERCURY": swe.MERCURY,
    "VENUS": swe.VENUS, "MARS": swe.MARS,
}

# Event types counted as painful or dramatic losses (Tebbs ch. 1, first list).
PAINFUL_EVENTS = {
    EventType.DIVORCE, EventType.DEATH_OF_PARENT, EventType.DEATH_OF_SIBLING,
    EventType.ACCIDENT, EventType.ILLNESS, EventType.SURGERY, EventType.ARREST,
    EventType.CUSTODY_LOSS, EventType.LEGAL_RESTRICTION, EventType.HOSPITALIZATION,
    EventType.ELECTION_LOSS,
}

ANGLE_NAMES = ("ASC", "MC", "DSC", "IC")


@dataclass(frozen=True)
class AngleFirstConfig:
    orb: float = 1.0                 # degrees; a contact scores 1 - orb_found/orb
    mode: str = "additive"           # "sa_only" | "additive" | "confirm" | "strict"
    transit_weight: float = 1.0
    progression_weight: float = 1.0
    stack_bonus: float = 0.0         # extra per additional technique that also hits
    painful_weight: float = 4.0      # multiplier for PAINFUL_EVENTS


def _sep(a, b):
    """Shortest angular distance, elementwise (degrees, 0–180)."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360.0)
    return np.minimum(d, 360.0 - d)


@lru_cache(maxsize=4096)
def _transit_longitudes(event_ordinal: int) -> tuple[float, ...]:
    d = date_type.fromordinal(event_ordinal)
    jd = swe.julday(d.year, d.month, d.day, 12.0)
    return tuple(swe.calc_ut(jd, code, MOSHIER_FLAG)[0][0] for code in OUTER_TRANSIT_BODIES.values())


class AngleFirstScorer(BaseScorer):
    """One scorer that applies solar arcs, progressions and transits to the angles."""

    name = "angle_first"

    def __init__(self, tight: bool = False, config: AngleFirstConfig | None = None):
        self.config = config or AngleFirstConfig()
        self.orb = self.config.orb * (0.5 if tight else 1.0)

    # -- the three techniques: each returns (strength 0..1, label, orb) ----------

    def _strength(self, orbs: np.ndarray, labels: list[str]) -> tuple[float, str, float]:
        i = int(np.argmin(orbs))
        orb = float(orbs.flat[i])
        if orb > self.orb:
            return 0.0, "", orb
        return 1.0 - orb / self.orb, labels[i], orb

    def _solar_arc(self, c: CandidateChart, age: float, angles: np.ndarray):
        arc = solar_arc_for_age(c.julian_day, age)
        names = list(c.planets)
        lons = np.array([c.planets[n] for n in names])
        # directed planets reach natal angles; directed angles reach natal planets
        p2a = _sep((lons[:, None] + arc), angles[None, :])
        a2p = _sep((angles[None, :] + arc), lons[:, None])
        labels = (
            [f"SA_{p}→{a}" for p in names for a in ANGLE_NAMES]
            + [f"SA_{a}→{p}" for p in names for a in ANGLE_NAMES]
        )
        return self._strength(np.concatenate([p2a.ravel(), a2p.ravel()]), labels)

    def _transits(self, event: LifeEvent, angles: np.ndarray):
        t = np.array(_transit_longitudes(event.date.toordinal()))
        orbs = _sep(t[:, None], angles[None, :])
        labels = [f"Tr_{p}→{a}" for p in OUTER_TRANSIT_BODIES for a in ANGLE_NAMES]
        return self._strength(orbs.ravel(), labels)

    def _progressions(self, c: CandidateChart, age: float, angles: np.ndarray):
        jd = c.julian_day + age  # a day for a year, from the candidate's own moment
        prog = {n: swe.calc_ut(jd, code, MOSHIER_FLAG)[0][0] for n, code in PROGRESSED_BODIES.items()}
        p = np.array(list(prog.values()))
        orbs = _sep(p[:, None], angles[None, :]).ravel()
        labels = [f"Prog_{n}→{a}" for n in prog for a in ANGLE_NAMES]
        # The progressed Moon to natal planets also depends on the birth time.
        others = [n for n in c.planets if n != "MOON"]
        moon_orbs = _sep(prog["MOON"], np.array([c.planets[n] for n in others]))
        orbs = np.concatenate([orbs, moon_orbs])
        labels += [f"Prog_MOON→{n}" for n in others]
        return self._strength(orbs, labels)

    # -- public ---------------------------------------------------------------

    def components(self, c: CandidateChart, event: LifeEvent, natal_jd: float):
        """(solar arc, transit, progression) results for one event; each is
        (strength, label, orb). Exposed for the benchmark."""
        y, m, d, _ = swe.revjul(natal_jd)
        age = (event.date - date_type(y, m, int(d))).days / 365.25
        if age <= 0:
            return None
        angles = np.array([c.ascendant, c.mc, c.dsc, c.ic])
        return (
            self._solar_arc(c, age, angles),
            self._transits(event, angles),
            self._progressions(c, age, angles),
        )

    def event_weight(self, event: LifeEvent) -> float:
        w = event.weight_multiplier()
        if event.event_type in PAINFUL_EVENTS:
            w *= self.config.painful_weight
        return w

    def score_event(self, candidate, event, natal_jd) -> list[TechniqueScore]:
        comps = self.components(candidate, event, natal_jd)
        if comps is None:
            return []
        cfg = self.config
        sa, tr, pr = comps
        parts = [("solar_arc", sa, 1.0)]
        if cfg.mode != "sa_only":
            parts += [("transit", tr, cfg.transit_weight), ("progression", pr, cfg.progression_weight)]
        hit = [(n, r, w) for n, r, w in parts if r[0] > 0]
        if cfg.mode == "strict" and len(hit) < 3:
            return []
        w_event = self.event_weight(event)
        if cfg.mode == "confirm":
            # Transits and progressions count only on top of a solar arc contact,
            # scaled by how close that contact is. They never score on their own.
            if sa[0] <= 0:
                return []
            hit = [(n, r if n == "solar_arc" else (r[0] * sa[0], r[1], r[2]), w) for n, r, w in hit]
        bonus = 1.0 + cfg.stack_bonus * max(0, len(hit) - 1)
        return [
            TechniqueScore(
                technique=name, event_description=event.description, natal_point=label,
                orb=round(orb, 4), score=strength * w * w_event * bonus,
                held_out=event.held_out, time_minutes=candidate.time_minutes,
            )
            for name, (strength, label, orb), w in hit
        ]

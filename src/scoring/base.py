"""Base scorer and shared scoring utilities."""

from abc import ABC, abstractmethod
from ..models import CandidateChart, LifeEvent, TechniqueScore, EVENT_HOUSE_RULERSHIPS

# Orb within which a house cusp is considered the same point as an angle
_CUSP_ANGLE_DEDUP_ORB = 2.0

# Maximum hits to keep per (technique, event) pair — prevents combinatorial noise
MAX_HITS_PER_EVENT = 3


def angle_diff(a: float, b: float) -> float:
    """Shortest angular distance between two ecliptic longitudes (0–180°)."""
    diff = abs((a - b) % 360)
    return min(diff, 360 - diff)


def orb_score(orb: float, tight: bool = False) -> float:
    """
    Award points based on orb size.

    Standard mode (tight=False):
      ≤0.5° → 3 pts, ≤1.0° → 2 pts, ≤2.0° → 1 pt, >2.0° → 0

    Tight mode (tight=True, used in Loop 2):
      ≤0.25° → 3 pts, ≤0.5° → 2 pts, ≤1.0° → 1 pt, >1.0° → 0
    """
    if tight:
        if orb <= 0.25:
            return 3.0
        elif orb <= 0.5:
            return 2.0
        elif orb <= 1.0:
            return 1.0
        return 0.0
    else:
        if orb <= 0.5:
            return 3.0
        elif orb <= 1.0:
            return 2.0
        elif orb <= 2.0:
            return 1.0
        return 0.0


def get_angles(candidate: CandidateChart) -> dict[str, float]:
    """Return the four major angles of a chart."""
    return {
        "ASC": candidate.ascendant,
        "MC": candidate.mc,
        "DSC": candidate.dsc,
        "IC": candidate.ic,
    }


def get_event_sensitive_points(
    candidate: CandidateChart, event: LifeEvent
) -> dict[str, float]:
    """
    Return all time-sensitive points relevant to an event type:
    the four angles + relevant house cusps (deduplicated against angles).

    Deduplication: ASC ≡ H1, MC ≡ H10, DSC ≡ H7, IC ≡ H4 in most house systems.
    If a cusp is within _CUSP_ANGLE_DEDUP_ORB degrees of any angle, the cusp is
    omitted to prevent double-counting the same geometric point.
    """
    points = get_angles(candidate)
    angle_lons = list(points.values())

    event_key = event.event_type.value
    ruled_houses = EVENT_HOUSE_RULERSHIPS.get(event_key, [1])
    for h in ruled_houses:
        if 1 <= h <= 12:
            idx = h - 1
            if idx < len(candidate.house_cusps):
                cusp_lon = candidate.house_cusps[idx]
                # Skip if this cusp is effectively coincident with an existing angle
                if any(angle_diff(cusp_lon, a) < _CUSP_ANGLE_DEDUP_ORB for a in angle_lons):
                    continue
                points[f"H{h}"] = cusp_lon
    return points


def cap_hits(hits: list[TechniqueScore], max_hits: int = MAX_HITS_PER_EVENT) -> list[TechniqueScore]:
    """
    Keep only the top `max_hits` hits (by score) from a single technique/event pair.

    Prevents combinatorial noise: with 80+ checks per event, many coincidental
    tight-orb hits accumulate and overwhelm genuine signal. Keeping only the
    highest-scoring hits enforces that only the strongest geometric relationships count.
    """
    if len(hits) <= max_hits:
        return hits
    return sorted(hits, key=lambda h: h.score, reverse=True)[:max_hits]


class BaseScorer(ABC):
    """Abstract base for all scoring techniques."""

    name: str = "base"

    @abstractmethod
    def score_event(
        self,
        candidate: CandidateChart,
        event: LifeEvent,
        natal_jd: float,
    ) -> list[TechniqueScore]:
        """Score one life event against one candidate chart. Return scored hits."""

    def score_all_events(
        self,
        candidate: CandidateChart,
        events: list[LifeEvent],
        natal_jd: float,
    ) -> list[TechniqueScore]:
        """Score all events and return the combined hit list."""
        results = []
        for event in events:
            results.extend(self.score_event(candidate, event, natal_jd))
        return results

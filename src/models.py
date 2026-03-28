"""Core data models for the astrological rectification system."""

from datetime import date
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class HouseSystem(str, Enum):
    PLACIDUS = "P"
    KOCH = "K"
    WHOLE_SIGN = "W"
    REGIOMONTANUS = "R"


SIGN_NAMES = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]


class EventType(str, Enum):
    MARRIAGE = "marriage"
    DIVORCE = "divorce"
    DEATH_OF_PARENT = "death_of_parent"
    DEATH_OF_SIBLING = "death_of_sibling"
    CAREER_PEAK = "career_peak"
    ACCIDENT = "accident"
    ILLNESS = "illness"
    RELOCATION = "relocation"
    PUBLICATION = "publication"
    ELECTION_WIN = "election_win"
    ELECTION_LOSS = "election_loss"
    ARREST = "arrest"
    EMIGRATION = "emigration"
    SURGERY = "surgery"
    BIRTH_OF_CHILD = "birth_of_child"
    CUSTODY_LOSS = "custody_loss"          # losing custody of children
    LEGAL_RESTRICTION = "legal_restriction" # conservatorship, guardianship by family
    HOSPITALIZATION = "hospitalization"     # psychiatric or medical admission
    OTHER = "other"


# Which house cusps and rulers are relevant per event type
EVENT_HOUSE_RULERSHIPS: dict[str, list[int]] = {
    "marriage": [7, 5],
    "divorce": [7, 12],
    "death_of_parent": [4, 10],
    "death_of_sibling": [3, 9],
    "career_peak": [10, 6],
    "accident": [1, 8, 12],
    "illness": [6, 1, 12],
    "relocation": [4, 9],
    "publication": [9, 3],
    "election_win": [10, 11],
    "election_loss": [10, 12],
    "arrest": [12, 7],
    "emigration": [9, 4],
    "surgery": [8, 12, 6],
    "birth_of_child": [5, 11],
    "custody_loss": [5, 4, 12],            # children (5), home/family (4), loss (12)
    "legal_restriction": [4, 10, 12, 7],   # father/Saturn (4), legal authority (10), confinement (12)
    "hospitalization": [12, 6, 1],         # hidden/institutions (12), health (6), body (1)
    "other": [1, 4, 7, 10],               # test all four angles for unclassified events
}


class EventWeight(str, Enum):
    ANCHOR = "anchor"   # Hard-dated, high discriminating power
    SOFT = "soft"       # Approximate date, lower weight


class LifeEvent(BaseModel):
    description: str
    event_type: EventType
    date: date
    date_certainty_days: int = Field(default=1, ge=1)  # uncertainty window
    weight: EventWeight = EventWeight.ANCHOR
    held_out: bool = False  # reserved for cross-validation

    def weight_multiplier(self) -> float:
        return 1.0 if self.weight == EventWeight.ANCHOR else 0.4


class BirthData(BaseModel):
    name: str
    birth_date: date
    birth_city: str
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    timezone_offset: float = Field(ge=-14.0, le=14.0)  # hours from UTC, historical
    biography: str = ""
    physical_description: str = ""


class CandidateChart(BaseModel):
    time_minutes: int = Field(ge=0, le=1439)  # minutes from midnight
    julian_day: float
    ascendant: float          # degrees 0–360
    mc: float                 # degrees 0–360
    house_cusps: list[float]  # 12 cusps in degrees
    house_system: HouseSystem
    planets: dict[str, float] = Field(default_factory=dict)           # name → ecliptic longitude
    planet_latitudes: dict[str, float] = Field(default_factory=dict)  # name → ecliptic latitude

    @property
    def dsc(self) -> float:
        return (self.ascendant + 180) % 360

    @property
    def ic(self) -> float:
        return (self.mc + 180) % 360

    def rising_sign(self) -> int:
        """Return rising sign as integer 1–12."""
        return int(self.ascendant / 30) + 1

    def rising_sign_name(self) -> str:
        return SIGN_NAMES[self.rising_sign() - 1]

    def time_label(self) -> str:
        h, m = divmod(self.time_minutes, 60)
        return f"{h:02d}:{m:02d}"


class TechniqueScore(BaseModel):
    technique: str
    event_description: str
    natal_point: str
    orb: float
    score: float
    held_out: bool
    time_minutes: int = 0


class CandidateScore(BaseModel):
    time_minutes: int
    total_score: float
    technique_scores: list[TechniqueScore]
    posterior_probability: float = 0.0

    def training_score(self) -> float:
        return sum(s.score for s in self.technique_scores if not s.held_out)

    def held_out_score(self) -> float:
        return sum(s.score for s in self.technique_scores if s.held_out)


class RisingSignPrior(BaseModel):
    """Probability distribution over 12 rising signs from Morin pre-filter."""
    probabilities: dict[int, float]  # sign (1-12) → probability

    def top_signs(self, threshold: float = 0.05) -> list[int]:
        """Return signs above probability threshold."""
        return [s for s, p in self.probabilities.items() if p >= threshold]

    def excluded_signs(self, threshold: float = 0.05) -> list[int]:
        return [s for s, p in self.probabilities.items() if p < threshold]


class RectificationResult(BaseModel):
    rectified_time_minutes: int
    uncertainty_minutes: int
    confidence_score: float = Field(ge=0.0, le=1.0)
    house_system_consensus: int = Field(ge=0, le=4)  # how many systems agree
    is_provisional: bool
    evidence_ledger: list[TechniqueScore]
    ranked_candidates: list[CandidateScore] = Field(default_factory=list)
    notes: str = ""

    def time_label(self) -> str:
        h, m = divmod(self.rectified_time_minutes, 60)
        return f"{h:02d}:{m:02d}"

    def rising_sign_name(self) -> str:
        sign = int(self.rectified_time_minutes / 5)  # placeholder
        return "Unknown"

    def summary(self) -> str:
        status = "PROVISIONAL" if self.is_provisional else "CONFIRMED"
        return (
            f"Rectified time: {self.time_label()} ±{self.uncertainty_minutes}min "
            f"| Confidence: {self.confidence_score:.2f} "
            f"| House consensus: {self.house_system_consensus}/4 "
            f"| {status}"
        )

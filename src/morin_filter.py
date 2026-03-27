"""
Loop 0 — Morin structural pre-filter.

Jean-Baptiste Morin de Villefranche's method: analyze the native's biography
for life themes that map to probable house configurations and rising signs.
This eliminates 7–9 of the 12 possible rising signs before ephemeris
calculations begin, reducing the candidate pool from 288 → 72.

In this implementation the pre-filter runs two modes:
  1. Rule-based: keyword mapping from biography text to probable house emphases
  2. LLM-assisted (optional): send biography to Claude for structured analysis

Output is a RisingSignPrior: probability vector over 12 signs.
Signs below a threshold (default 5%) are excluded from the candidate grid.
"""

import re
from .models import RisingSignPrior, SIGN_NAMES

# Keyword patterns → house emphases → probable rising signs
# Maps life-theme keywords to likely house emphases
THEME_HOUSE_MAP: dict[str, list[int]] = {
    # 1st house themes (self, body, identity)
    r"\b(athlete|sportsman|sportswoman|physical|body|health|fitness)\b": [1],
    # 3rd house (communication, writing, siblings)
    r"\b(journalist|writer|author|blogger|communicat|sibling)\b": [3],
    # 4th house (home, family, roots)
    r"\b(real estate|family|roots|ancestr|homeland|domestic)\b": [4],
    # 5th house (creativity, children, romance, entertainment)
    r"\b(actor|actress|entertainer|performer|artist|creative|child|romance|gambl)\b": [5],
    # 6th house (work, health, service)
    r"\b(doctor|nurse|healer|service|employee|daily work|health worker)\b": [6],
    # 7th house (partnerships, law, public)
    r"\b(lawyer|attorney|partner|diplomat|marriage|public relations)\b": [7],
    # 8th house (transformation, death, finance, occult)
    r"\b(occult|transform|crisis|death|inheritance|banker|financ|psycholog)\b": [8],
    # 9th house (philosophy, law, religion, travel, academia)
    r"\b(philosopher|professor|academic|university|religion|spiritual|travel|foreign|international|publish)\b": [9],
    # 10th house (career, authority, public fame)
    r"\b(politic|president|prime minister|ceo|director|executive|famous|celebrity|public figure|authority|career peak)\b": [10],
    # 11th house (groups, causes, networks)
    r"\b(activist|humanitarian|group|community|network|social cause|revolution)\b": [11],
    # 12th house (isolation, institutions, hidden)
    r"\b(prison|monastery|hospital|isolation|hidden|secret|exile|addict)\b": [12],
}

# House emphasis → most probable rising signs
# Logic: if house H is prominent, the rising sign is likely a sign that puts
# that house's natural sign on the Ascendant or its ruler on the ASC.
HOUSE_TO_RISING_SIGNS: dict[int, list[int]] = {
    1:  [1, 5, 8, 11],   # Aries/Leo/Scorpio/Aquarius rising → strong 1st
    3:  [10, 3, 6, 1],   # Gemini/Virgo/Sagittarius/Pisces rising → busy 3rd
    4:  [9, 10, 4, 7],   # Cancer rising → prominent 4th; Libra/Cap/Aries also
    5:  [4, 8, 11, 1],   # Leo ascending → natural 5th house emphasis
    6:  [5, 2, 8, 11],   # Virgo/Pisces rising common for health vocations
    7:  [1, 4, 7, 10],   # Libra rising → strong 7th
    8:  [1, 4, 8, 11],   # Scorpio rising → natural 8th emphasis
    9:  [4, 7, 10, 3],   # Sagittarius rising → strong 9th
    10: [4, 1, 7, 10],   # Capricorn rising → prominent 10th
    11: [3, 6, 9, 12],   # Aquarius rising → active 11th
    12: [2, 5, 8, 11],   # Pisces/Virgo rising common for 12th house themes
}


def rule_based_prior(biography: str) -> dict[int, float]:
    """
    Assign rising-sign probabilities using keyword matching on biography text.
    Returns a dict mapping sign (1–12) → raw score.
    """
    text = biography.lower()
    house_scores: dict[int, float] = {h: 0.0 for h in range(1, 13)}

    for pattern, houses in THEME_HOUSE_MAP.items():
        matches = len(re.findall(pattern, text))
        if matches > 0:
            for house in houses:
                house_scores[house] += matches

    # Convert house scores to rising sign scores
    sign_scores: dict[int, float] = {s: 0.1 for s in range(1, 13)}  # small uniform prior
    for house, score in house_scores.items():
        if score > 0:
            candidate_signs = HOUSE_TO_RISING_SIGNS.get(house, [])
            for sign in candidate_signs:
                sign_scores[sign] += score

    # Normalize to probabilities
    total = sum(sign_scores.values())
    return {s: v / total for s, v in sign_scores.items()}


def normalize_prior(raw: dict[int, float]) -> dict[int, float]:
    """Normalize a raw score dict to a proper probability distribution."""
    total = sum(raw.values())
    if total == 0:
        return {s: 1 / 12 for s in range(1, 13)}
    return {s: v / total for s, v in raw.items()}


def build_morin_prior(
    biography: str,
    physical_description: str = "",
    llm_prior: dict[int, float] | None = None,
    llm_weight: float = 0.6,
) -> RisingSignPrior:
    """
    Combine rule-based keyword analysis with optional LLM-derived prior.

    Args:
        biography: biography text
        physical_description: physical appearance description
        llm_prior: optional pre-computed LLM prior over 12 signs
        llm_weight: blending weight for LLM prior (0.0 = pure rule-based)
    """
    rule_prior = rule_based_prior(biography + " " + physical_description)

    if llm_prior is not None:
        # Blend: weighted average of rule-based and LLM priors
        blended: dict[int, float] = {}
        for sign in range(1, 13):
            blended[sign] = (
                (1 - llm_weight) * rule_prior.get(sign, 1 / 12)
                + llm_weight * llm_prior.get(sign, 1 / 12)
            )
        probs = normalize_prior(blended)
    else:
        probs = normalize_prior(rule_prior)

    return RisingSignPrior(probabilities=probs)


def uniform_prior() -> RisingSignPrior:
    """Return a uniform prior over all 12 rising signs (no information case)."""
    return RisingSignPrior(probabilities={s: 1 / 12 for s in range(1, 13)})

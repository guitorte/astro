"""
LLM-assisted event extraction and Morin pre-filter.

Uses the Anthropic Claude API to:
1. Extract hard-dated life events from biography text
2. Generate a rising sign probability distribution from life theme analysis

This module is optional — the system works without it using rule-based fallbacks.
"""

import json
import os
from datetime import date
from typing import Optional

from .models import LifeEvent, EventType, EventWeight, SIGN_NAMES

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


def _get_client() -> Optional["anthropic.Anthropic"]:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or not _ANTHROPIC_AVAILABLE:
        return None
    return anthropic.Anthropic(api_key=api_key)


EVENT_EXTRACTION_PROMPT = """You are an expert astrological researcher. Given a biography, extract ALL hard-dated life events that can be used for birth time rectification.

RULES:
- Only include events with a known date to within 1 week (ideally exact date)
- Classify each event by type
- Mark events with uncertain dates as "soft", certain dates as "anchor"
- Exclude vague events like "rose to prominence" or "experienced a creative peak"
- Include: marriages, divorces, accidents, surgeries, deaths of close relatives, career milestones with exact dates, arrests, emigration, birth of children

Return a JSON array of events:
[
  {
    "description": "Married Jane Smith",
    "event_type": "marriage",
    "date": "1997-06-15",
    "date_certainty_days": 1,
    "weight": "anchor"
  },
  ...
]

Valid event_type values: marriage, divorce, death_of_parent, death_of_sibling, career_peak, accident, illness, relocation, publication, election_win, election_loss, arrest, emigration, surgery, birth_of_child, other

Biography:
{biography}

Return ONLY the JSON array, no other text."""


MORIN_FILTER_PROMPT = """You are an expert traditional astrologer specializing in birth time rectification using the Morin method.

Given the following biography and physical description, analyze the native's life themes and assign a probability (0.0–1.0) for each of the 12 rising signs.

Consider:
- Career type and 10th house themes
- Relationship patterns and 7th house themes
- Major life crises (8th, 12th house)
- International travel/philosophy (9th house)
- Physical appearance and typical Ascendant descriptions
- Dominant house emphasis throughout life

Return a JSON object with rising sign probabilities (must sum to ~1.0):
{
  "1": 0.05,   // Aries rising
  "2": 0.02,   // Taurus rising
  "3": 0.15,   // Gemini rising
  ...
  "12": 0.08   // Pisces rising
}

Biography:
{biography}

Physical description:
{physical_description}

Return ONLY the JSON object."""


def extract_events_llm(
    biography: str,
    held_out_fraction: float = 0.2,
) -> list[LifeEvent]:
    """
    Use Claude to extract hard-dated life events from biography text.

    Args:
        biography: raw biography text
        held_out_fraction: fraction of events to mark as held-out for CV

    Returns:
        List of LifeEvent objects, some flagged held_out=True.
    """
    client = _get_client()
    if client is None:
        return []

    prompt = EVENT_EXTRACTION_PROMPT.format(biography=biography)

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        events_data = json.loads(raw)
    except Exception as exc:
        print(f"[LLM] Event extraction failed: {exc}")
        return []

    events: list[LifeEvent] = []
    for i, item in enumerate(events_data):
        try:
            event_date = date.fromisoformat(item["date"])
            event = LifeEvent(
                description=item["description"],
                event_type=EventType(item.get("event_type", "other")),
                date=event_date,
                date_certainty_days=int(item.get("date_certainty_days", 7)),
                weight=EventWeight(item.get("weight", "anchor")),
                held_out=False,
            )
            events.append(event)
        except Exception as exc:
            print(f"[LLM] Skipping event {i}: {exc}")

    # Reserve last `held_out_fraction` of events as test set
    n_held = max(1, int(len(events) * held_out_fraction))
    for e in events[-n_held:]:
        object.__setattr__(e, "held_out", True)

    return events


def build_llm_morin_prior(
    biography: str, physical_description: str = ""
) -> Optional[dict[int, float]]:
    """
    Use Claude to generate a Morin rising sign prior.

    Returns:
        Dict mapping sign (1–12) → probability, or None if unavailable.
    """
    client = _get_client()
    if client is None:
        return None

    prompt = MORIN_FILTER_PROMPT.format(
        biography=biography,
        physical_description=physical_description or "Not provided.",
    )

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        raw_dict = json.loads(raw)
        # Normalize keys to int
        prior = {int(k): float(v) for k, v in raw_dict.items()}
        # Ensure all 12 signs present
        for s in range(1, 13):
            prior.setdefault(s, 0.01)
        # Normalize
        total = sum(prior.values())
        return {s: v / total for s, v in prior.items()}
    except Exception as exc:
        print(f"[LLM] Morin prior failed: {exc}")
        return None

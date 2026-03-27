"""
FastAPI service layer for the astrological rectification system.

Endpoints:
  POST /rectify          — run full rectification for a celebrity
  POST /extract-events   — extract life events from biography via LLM
  POST /morin-filter     — compute Morin rising sign prior
  GET  /health           — health check
"""

from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .models import (
    BirthData, LifeEvent, EventType, EventWeight,
    RectificationResult, RisingSignPrior,
)
from .rectifier import Rectifier
from .morin_filter import build_morin_prior, uniform_prior
from .llm_extractor import extract_events_llm, build_llm_morin_prior

app = FastAPI(
    title="Astrology Rectifier",
    description=(
        "Agentic birth time rectification using multi-technique scoring: "
        "primary directions, secondary progressions, solar arc, annual profections, "
        "outer-planet transits — across 4 house systems with Bayesian update."
    ),
    version="1.0.0",
)


# --- Request / Response schemas ---

class LifeEventRequest(BaseModel):
    description: str
    event_type: EventType = EventType.OTHER
    date: date
    date_certainty_days: int = Field(default=1, ge=1)
    weight: EventWeight = EventWeight.ANCHOR
    held_out: bool = False


class RectifyRequest(BaseModel):
    birth_data: BirthData
    events: list[LifeEventRequest] = Field(default_factory=list)
    use_llm: bool = False
    verbose: bool = False


class ExtractEventsRequest(BaseModel):
    biography: str
    held_out_fraction: float = Field(default=0.2, ge=0.0, le=0.5)


class MorinFilterRequest(BaseModel):
    biography: str
    physical_description: str = ""
    use_llm: bool = False


# --- Helper ---

def _parse_events(raw: list[LifeEventRequest]) -> list[LifeEvent]:
    return [
        LifeEvent(
            description=r.description,
            event_type=r.event_type,
            date=r.date,
            date_certainty_days=r.date_certainty_days,
            weight=r.weight,
            held_out=r.held_out,
        )
        for r in raw
    ]


# --- Endpoints ---

@app.get("/health")
def health_check():
    """Service health check."""
    return {"status": "ok", "service": "astrology-rectifier"}


@app.post("/morin-filter", response_model=RisingSignPrior)
def morin_filter(request: MorinFilterRequest) -> RisingSignPrior:
    """
    Compute the Morin structural pre-filter.
    Returns probability distribution over 12 rising signs.
    """
    llm_prior = None
    if request.use_llm:
        llm_prior = build_llm_morin_prior(
            request.biography, request.physical_description
        )

    prior = build_morin_prior(
        request.biography,
        request.physical_description,
        llm_prior=llm_prior,
    )
    return prior


@app.post("/extract-events")
def extract_events(request: ExtractEventsRequest) -> list[dict]:
    """
    Extract hard-dated life events from biography text using the LLM.
    Falls back to empty list if no API key is configured.
    """
    events = extract_events_llm(request.biography, request.held_out_fraction)
    return [e.model_dump() for e in events]


@app.post("/rectify", response_model=RectificationResult)
def rectify(request: RectifyRequest) -> RectificationResult:
    """
    Run full 4-loop birth time rectification.

    Loops:
      0 — Morin pre-filter (reduces candidate pool)
      1 — Broad 15-min scoring pass with Bayesian update
      2 — Narrow 1-min pass ±30min around top candidate
      3 — House system consensus check (Placidus/Koch/Whole Sign/Regiomontanus)
    """
    # Validate birth data
    if not request.birth_data.birth_date:
        raise HTTPException(status_code=400, detail="birth_date is required")

    # Parse events
    events = _parse_events(request.events)

    # Optionally extract additional events from biography via LLM
    if request.use_llm and request.birth_data.biography:
        llm_events = extract_events_llm(request.birth_data.biography)
        # Merge, avoiding duplicates by date+type
        existing_keys = {(e.date, e.event_type) for e in events}
        for e in llm_events:
            if (e.date, e.event_type) not in existing_keys:
                events.append(e)

    # Build Morin prior
    llm_prior = None
    if request.use_llm:
        llm_prior = build_llm_morin_prior(
            request.birth_data.biography,
            request.birth_data.physical_description,
        )
    prior = build_morin_prior(
        request.birth_data.biography,
        request.birth_data.physical_description,
        llm_prior=llm_prior,
    )

    # Run rectification
    rectifier = Rectifier(
        birth_data=request.birth_data,
        events=events,
        morin_prior=prior,
        verbose=request.verbose,
    )

    return rectifier.rectify()

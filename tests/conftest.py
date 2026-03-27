"""Shared fixtures for the test suite."""

import pytest
from datetime import date

from src.models import BirthData, LifeEvent, EventType, EventWeight


@pytest.fixture
def sample_birth_data():
    """A celebrity born in London (no known birth time)."""
    return BirthData(
        name="Test Celebrity",
        birth_date=date(1964, 5, 23),
        birth_city="London",
        latitude=51.5074,
        longitude=-0.1278,
        timezone_offset=1.0,  # BST (UTC+1)
        biography=(
            "A famous rock musician and actor who became internationally known "
            "for their music career peak in 1990. Known for political activism "
            "and humanitarian causes. Married twice; first marriage in 1988, "
            "second in 2001. Suffered a serious accident in 1995."
        ),
        physical_description="Tall, lean, with angular facial features.",
    )


@pytest.fixture
def sample_events():
    """Hard-dated life events for the test celebrity."""
    return [
        LifeEvent(
            description="First marriage",
            event_type=EventType.MARRIAGE,
            date=date(1988, 7, 14),
            date_certainty_days=1,
            weight=EventWeight.ANCHOR,
            held_out=False,
        ),
        LifeEvent(
            description="Career peak — first major album release",
            event_type=EventType.CAREER_PEAK,
            date=date(1990, 3, 21),
            date_certainty_days=1,
            weight=EventWeight.ANCHOR,
            held_out=False,
        ),
        LifeEvent(
            description="Serious motorcycle accident",
            event_type=EventType.ACCIDENT,
            date=date(1995, 9, 3),
            date_certainty_days=3,
            weight=EventWeight.ANCHOR,
            held_out=False,
        ),
        LifeEvent(
            description="Second marriage",
            event_type=EventType.MARRIAGE,
            date=date(2001, 6, 10),
            date_certainty_days=1,
            weight=EventWeight.ANCHOR,
            held_out=True,  # reserved for cross-validation
        ),
    ]

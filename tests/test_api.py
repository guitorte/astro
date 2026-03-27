"""Tests for the FastAPI service layer."""

import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


class TestHealthCheck:
    def test_health_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestMorinFilter:
    def test_returns_twelve_signs(self):
        response = client.post("/morin-filter", json={
            "biography": "A famous politician and president.",
            "use_llm": False,
        })
        assert response.status_code == 200
        data = response.json()
        assert "probabilities" in data
        assert len(data["probabilities"]) == 12

    def test_probabilities_sum_to_one(self):
        response = client.post("/morin-filter", json={
            "biography": "An academic philosopher and writer.",
            "use_llm": False,
        })
        probs = response.json()["probabilities"]
        total = sum(float(v) for v in probs.values())
        assert abs(total - 1.0) < 1e-5

    def test_with_physical_description(self):
        response = client.post("/morin-filter", json={
            "biography": "A rock musician.",
            "physical_description": "Tall and athletic.",
            "use_llm": False,
        })
        assert response.status_code == 200


class TestExtractEvents:
    def test_returns_list_without_api_key(self):
        # Without ANTHROPIC_API_KEY, should return empty list gracefully
        response = client.post("/extract-events", json={
            "biography": "Born in 1970. Married in 1995. Had a car accident in 2000.",
            "held_out_fraction": 0.2,
        })
        assert response.status_code == 200
        assert isinstance(response.json(), list)


class TestRectify:
    def _minimal_request(self, events=None):
        return {
            "birth_data": {
                "name": "Test Person",
                "birth_date": "1964-05-23",
                "birth_city": "London",
                "latitude": 51.5074,
                "longitude": -0.1278,
                "timezone_offset": 1.0,
                "biography": "A famous rock musician.",
            },
            "events": events or [],
            "use_llm": False,
            "verbose": False,
        }

    def test_minimal_request_succeeds(self):
        response = client.post("/rectify", json=self._minimal_request())
        assert response.status_code == 200

    def test_response_has_rectified_time(self):
        response = client.post("/rectify", json=self._minimal_request())
        data = response.json()
        assert "rectified_time_minutes" in data
        assert 0 <= data["rectified_time_minutes"] <= 1439

    def test_response_has_confidence(self):
        response = client.post("/rectify", json=self._minimal_request())
        data = response.json()
        assert "confidence_score" in data
        assert 0.0 <= data["confidence_score"] <= 1.0

    def test_response_has_house_consensus(self):
        response = client.post("/rectify", json=self._minimal_request())
        data = response.json()
        assert "house_system_consensus" in data
        assert 0 <= data["house_system_consensus"] <= 4

    def test_with_events(self):
        events = [
            {
                "description": "First marriage",
                "event_type": "marriage",
                "date": "1988-07-14",
                "date_certainty_days": 1,
                "weight": "anchor",
                "held_out": False,
            },
            {
                "description": "Career peak",
                "event_type": "career_peak",
                "date": "1990-03-21",
                "date_certainty_days": 1,
                "weight": "anchor",
                "held_out": False,
            },
        ]
        response = client.post("/rectify", json=self._minimal_request(events=events))
        assert response.status_code == 200
        data = response.json()
        assert 0 <= data["rectified_time_minutes"] <= 1439

    def test_provisional_flag_present(self):
        response = client.post("/rectify", json=self._minimal_request())
        data = response.json()
        assert "is_provisional" in data

    def test_empty_events_returns_provisional(self):
        response = client.post("/rectify", json=self._minimal_request(events=[]))
        data = response.json()
        assert data["is_provisional"] is True

    def test_evidence_ledger_is_list(self):
        response = client.post("/rectify", json=self._minimal_request())
        data = response.json()
        assert isinstance(data["evidence_ledger"], list)

    def test_invalid_latitude_rejected(self):
        req = self._minimal_request()
        req["birth_data"]["latitude"] = 999.0  # invalid
        response = client.post("/rectify", json=req)
        assert response.status_code == 422

    def test_invalid_event_type_rejected(self):
        events = [{"description": "X", "event_type": "NOT_VALID", "date": "1990-01-01"}]
        response = client.post("/rectify", json=self._minimal_request(events=events))
        assert response.status_code == 422

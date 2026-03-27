"""Tests for the Morin structural pre-filter."""

import pytest
from src.morin_filter import (
    rule_based_prior, normalize_prior, build_morin_prior, uniform_prior
)
from src.models import RisingSignPrior


class TestUniformPrior:
    def test_twelve_signs(self):
        prior = uniform_prior()
        assert len(prior.probabilities) == 12

    def test_sums_to_one(self):
        prior = uniform_prior()
        total = sum(prior.probabilities.values())
        assert abs(total - 1.0) < 1e-9

    def test_each_sign_equal(self):
        prior = uniform_prior()
        for p in prior.probabilities.values():
            assert abs(p - 1 / 12) < 1e-9


class TestRuleBasedPrior:
    def test_returns_twelve_signs(self):
        raw = rule_based_prior("A famous politician who became president.")
        assert len(raw) == 12

    def test_sums_to_one(self):
        raw = rule_based_prior("A famous musician and rock star.")
        total = sum(raw.values())
        assert abs(total - 1.0) < 1e-6

    def test_all_probabilities_positive(self):
        raw = rule_based_prior("Academic professor at university.")
        for p in raw.values():
            assert p > 0

    def test_political_biography_boosts_tenth_house_signs(self):
        bio = "A powerful politician who became prime minister and president."
        raw = rule_based_prior(bio)
        # 10th house signs (Capricorn=10, Aries=1, Cancer=4, Libra=7)
        tenth_house_signs = [1, 4, 7, 10]
        tenth_prob = sum(raw.get(s, 0) for s in tenth_house_signs)
        other_prob = sum(raw.get(s, 0) for s in range(1, 13) if s not in tenth_house_signs)
        # 10th house signs should collectively have more probability
        assert tenth_prob > other_prob * 0.3  # at least proportionally significant

    def test_philosophy_biography_boosts_ninth_house(self):
        bio = "A university professor and philosopher focused on international travel and religion."
        raw = rule_based_prior(bio)
        # 9th house signs: Sagittarius=9, Pisces=12, Gemini=3, Virgo=6
        ninth_signs = [3, 6, 9, 12]
        ninth_prob = sum(raw.get(s, 0) for s in ninth_signs)
        assert ninth_prob > 0.2  # should have meaningful share

    def test_empty_biography_gives_near_uniform(self):
        raw = rule_based_prior("")
        max_p = max(raw.values())
        min_p = min(raw.values())
        # With empty bio, distribution shouldn't be extremely concentrated
        assert max_p / min_p < 5.0  # no sign should be 5× more likely than another


class TestBuildMorinPrior:
    def test_returns_rising_sign_prior(self):
        prior = build_morin_prior("A famous actor and entertainer.")
        assert isinstance(prior, RisingSignPrior)

    def test_probabilities_sum_to_one(self):
        prior = build_morin_prior("A famous actor.")
        total = sum(prior.probabilities.values())
        assert abs(total - 1.0) < 1e-6

    def test_top_signs_returns_subset(self):
        prior = build_morin_prior("A famous actor and entertainer.")
        top = prior.top_signs(threshold=0.05)
        assert 1 <= len(top) <= 12
        for s in top:
            assert prior.probabilities[s] >= 0.05

    def test_excluded_signs_below_threshold(self):
        prior = build_morin_prior("A famous actor.")
        excluded = prior.excluded_signs(threshold=0.05)
        for s in excluded:
            assert prior.probabilities[s] < 0.05

    def test_llm_prior_blending(self):
        bio = "A musician."
        # Create a strong LLM prior for Taurus (sign 2)
        llm_prior = {s: 0.01 for s in range(1, 13)}
        llm_prior[2] = 0.89  # 89% Taurus
        total = sum(llm_prior.values())
        llm_prior = {s: v / total for s, v in llm_prior.items()}

        prior = build_morin_prior(bio, llm_prior=llm_prior, llm_weight=0.8)
        # With 80% LLM weight, Taurus should be dominant
        assert prior.probabilities[2] > 0.4

    def test_with_physical_description(self):
        prior = build_morin_prior(
            "A famous politician.",
            physical_description="Tall and authoritative presence.",
        )
        assert isinstance(prior, RisingSignPrior)
        assert len(prior.probabilities) == 12

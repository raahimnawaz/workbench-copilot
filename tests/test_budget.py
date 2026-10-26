import json
from types import SimpleNamespace

import pytest

from wbc.agent.budget import Budget, BudgetExceeded

PRICES = {
    "spark": {"in": 1.25, "cached_in": 0.15, "out": 4.25},
    "ears": {"per_hour": 0.18},
    "sam": {"per_1k_images": 2.50},
}


def test_token_cost_with_cached_input():
    b = Budget(10, PRICES)
    usage = SimpleNamespace(input_tokens=1_000_000, output_tokens=100_000,
                            input_tokens_details=SimpleNamespace(cached_tokens=400_000))
    cost = b.add_tokens("spark", usage)
    assert cost == pytest.approx(0.6 * 1.25 + 0.4 * 0.15 + 0.1 * 4.25)


def test_chat_completions_usage_shape():
    b = Budget(10, PRICES)
    b.add_tokens("spark", {"prompt_tokens": 2_000_000, "completion_tokens": 0})
    assert b.spent == pytest.approx(2.5)


def test_audio_and_images():
    b = Budget(10, PRICES)
    b.add_audio("ears", 1800)
    b.add_images("sam", 400)
    assert b.spent == pytest.approx(0.09 + 1.0)


def test_check_stops_at_limit():
    b = Budget(1.0, PRICES)
    b.check()
    b.add_images("sam", 400)
    with pytest.raises(BudgetExceeded):
        b.check()


def test_spend_persists_across_restarts(tmp_path):
    state = tmp_path / "spend.json"
    Budget(10, PRICES, state).add_images("sam", 1000)
    assert json.loads(state.read_text())["spent_usd"] == pytest.approx(2.5)
    assert Budget(10, PRICES, state).spent == pytest.approx(2.5)


def test_unknown_model_names_the_config_file():
    with pytest.raises(KeyError, match="models.yaml"):
        Budget(10, PRICES).add_tokens("mystery", {})


def test_missing_usage_costs_nothing():
    b = Budget(10, PRICES)
    assert b.add_tokens("spark", None) == 0

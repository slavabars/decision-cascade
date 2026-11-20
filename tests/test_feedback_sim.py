"""Смещение оценки q при фидбэке по эскалациям: направление, величина, IPS."""
from __future__ import annotations

import pytest

from eval.feedback.core import SCENARIOS, report, run_policy, simulate
from src.backends.benefit import EXPLORE_WEIGHT_CAP, explore_weight, shrink_accuracy

SEEDS = range(3)
ROUNDS = 20
ROWS = 500


def _run(name: str, estimator: str, explore: float):
    return simulate(
        SCENARIOS[name],
        estimator=estimator,
        explore=explore,
        seeds=SEEDS,
        rounds=ROUNDS,
        rows=ROWS,
    )


def test_explore_weight_is_inverse_propensity_with_cap():
    assert explore_weight(1.0) == pytest.approx(1.0)
    assert explore_weight(0.5) == pytest.approx(2.0)
    assert explore_weight(0.01) == pytest.approx(EXPLORE_WEIGHT_CAP)
    with pytest.raises(ValueError, match="propensity"):
        explore_weight(0.0)
    with pytest.raises(ValueError, match="propensity"):
        explore_weight(float("nan"))


def test_shrink_accuracy_matches_plain_counts():
    # без исследования вес каждого наблюдения 1: прежняя формула
    assert shrink_accuracy(0.9, 50.0, 10.0, 7.0) == pytest.approx(
        (0.9 * 50 + 7) / 60
    )
    assert shrink_accuracy(0.9, 50.0, 0.0, 0.0) == pytest.approx(0.9)


def test_correlated_difficulty_biases_q_down():
    result = _run("correlated", "naive", 0.0)

    assert result["bias_mean"] < -0.1
    # цикл самоусиливается: заниженная q опускает порог, эскалаций меньше
    assert result["policy_rate"] < 0.5


def test_random_slice_without_weights_is_still_biased():
    """Одной случайной подвыборки мало: без весов оценка остаётся смещённой."""
    naive = _run("correlated", "naive", 0.05)

    assert naive["bias_mean"] < -0.1
    assert naive["calls_rate"] > naive["policy_rate"]


def test_ips_with_exploration_removes_bias():
    naive = _run("correlated", "naive", 0.0)
    ips = _run("correlated", "ips", 0.05)

    assert abs(ips["bias_mean"]) < 0.03
    assert abs(ips["bias_mean"]) < abs(naive["bias_mean"]) / 3
    # цена исправления — исследовательские вызовы, но потеря полезности меньше
    assert ips["calls_rate"] > naive["calls_rate"]
    assert ips["regret_rate"] < naive["regret_rate"]


def test_anti_correlated_difficulty_biases_q_up():
    naive = _run("anti", "naive", 0.0)
    ips = _run("anti", "ips", 0.05)

    assert naive["bias_mean"] > 0.02
    assert abs(ips["bias_mean"]) < 0.03
    assert ips["regret_rate"] < naive["regret_rate"]


def test_without_exploration_ips_equals_naive():
    naive = _run("correlated", "naive", 0.0)
    ips = _run("correlated", "ips", 0.0)

    assert ips["q_hat_mean"] == pytest.approx(naive["q_hat_mean"])
    assert ips["calls_rate"] == pytest.approx(naive["policy_rate"])


def test_simulation_is_deterministic():
    first = _run("correlated", "ips", 0.05)
    second = _run("correlated", "ips", 0.05)

    assert first["q_hat_mean"] == second["q_hat_mean"]
    assert first["bias_mean"] == second["bias_mean"]


def test_run_policy_validates_inputs():
    with pytest.raises(ValueError, match="estimator"):
        run_policy([], estimator="oracle", explore=0.0, q_true=0.8)
    with pytest.raises(ValueError, match="explore"):
        run_policy([], estimator="naive", explore=1.5, q_true=0.8)


def test_report_covers_scenario_estimator_and_explore_grid():
    rows = report(
        scenarios=["correlated"],
        estimators=["naive", "ips"],
        explores=[0.0, 0.05],
        seeds=range(1),
        rounds=2,
        rows=50,
    )

    assert len(rows) == 4
    assert {row["estimator"] for row in rows} == {"naive", "ips"}
    assert {row["explore"] for row in rows} == {0.0, 0.05}

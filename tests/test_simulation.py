import numpy as np
import pandas as pd

from src.simulation import run_monte_carlo, simulate_observational_sample


def test_simulation_is_deterministic_and_returns_the_known_sample_ate():
    first, first_ate = simulate_observational_sample(200, 0.75, 11)
    second, second_ate = simulate_observational_sample(200, 0.75, 11)

    pd.testing.assert_frame_equal(first, second)
    assert first_ate == second_ate
    assert np.isfinite(first_ate)
    assert first["propensity"].between(0.05, 0.95).all()
    assert np.allclose(first["p1"] - first["p0"], first["true_effect"])
    assert first_ate == np.mean(first["true_effect"])


def test_monte_carlo_has_three_estimators_and_declared_strengths():
    results = run_monte_carlo(strengths=(0.0, 1.5), n_runs=2, n=400, seed=8)

    assert set(results["estimator"]) == {"naive", "adjusted_dml", "omitted_confounder_dml"}
    assert set(results["confounding_strength"]) == {0.0, 1.5}
    assert (results["n_runs"] == 2).all()
    assert results["bias"].notna().all()
    assert results["rmse"].notna().all()


def test_omitting_the_confounder_is_worse_under_strong_selection():
    results = run_monte_carlo(strengths=(1.5,), n_runs=5, n=800, seed=8)
    omitted = results.loc[results["estimator"] == "omitted_confounder_dml", "bias"].abs().iloc[0]
    adjusted = results.loc[results["estimator"] == "adjusted_dml", "bias"].abs().iloc[0]

    assert omitted > adjusted

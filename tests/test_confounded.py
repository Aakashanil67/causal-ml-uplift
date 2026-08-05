import pytest

from src.config import RANDOM_SEED, TREATMENT_COL
from src.confounded import (
    dml_estimate,
    identify_confounded,
    make_confounded_sample,
    naive_diff,
    ols_estimate,
)
from src.naive import covariate_balance


@pytest.fixture(scope="module")
def df():
    from src.data_loader import load_hillstrom

    return load_hillstrom()


@pytest.fixture(scope="module")
def confounded_v1(df):
    return make_confounded_sample(
        df, ["recency", "history"], {"recency": -1, "history": 1}, strength=1.2, seed=RANDOM_SEED
    )


def test_confounded_sample_actually_is_confounded(df, confounded_v1):
    # the whole benchmark is meaningless if this doesn't hold: without it, there's nothing for
    # OLS/DML to correct, and passing "variant 1" would prove nothing.
    balance = covariate_balance(confounded_v1, TREATMENT_COL)
    assert balance.loc["recency", "standardised_diff"] < -0.1
    assert balance.loc["history", "standardised_diff"] > 0.1
    # the real, unconfounded RCT this was built from should not show this pattern
    real_balance = covariate_balance(df, TREATMENT_COL)
    assert abs(real_balance.loc["recency", "standardised_diff"]) < 0.1


def test_naive_estimate_overstates_the_benchmark_on_confounded_sample(confounded_v1):
    naive = naive_diff(confounded_v1, "visit")
    # benchmark (full-RCT DML ATE on visit) is ~0.060 (reports/05_dml_ate.md); the confound is
    # constructed to inflate the naive estimate well past it.
    assert naive > 0.08


def test_ols_and_dml_recover_the_benchmark_when_confounder_is_observed(confounded_v1):
    benchmark = 0.0601
    ols = ols_estimate(confounded_v1, "visit")
    dml = dml_estimate(confounded_v1, "visit")
    assert abs(ols - benchmark) < 0.02
    assert dml["ci_low"] <= benchmark <= dml["ci_high"]


def test_dml_fails_when_confounder_is_withheld(df):
    sub = make_confounded_sample(df, ["newbie"], {"newbie": -1}, strength=1.5, seed=RANDOM_SEED)
    benchmark = 0.0601
    dml_withheld = dml_estimate(sub, "visit", drop_cols=["newbie"])
    # this is the point of variant 2: the benchmark should NOT be inside the CI
    assert not (dml_withheld["ci_low"] <= benchmark <= dml_withheld["ci_high"])
    ols_withheld = ols_estimate(sub, "visit", drop_cols=["newbie"])
    assert abs(ols_withheld - benchmark) > 0.02


def test_confounded_dag_backdoor_set_matches_confounders(confounded_v1):
    identified = identify_confounded(confounded_v1, "visit", ["recency", "history"])
    assert set(identified.get_backdoor_variables()) == {"recency", "history"}

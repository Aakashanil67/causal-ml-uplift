import numpy as np
import pandas as pd
import pytest

from src.naive import (
    _standardised_diff,
    covariate_balance,
    diff_in_means,
    naive_estimates,
    write_naive_report,
)


@pytest.fixture(scope="module")
def df():
    from src.data_loader import load_hillstrom

    return load_hillstrom()


def test_diff_in_means_matches_known_arm_rates(df):
    row = diff_in_means(df, "treatment", "visit")
    # from reports/data_dictionary.md: No E-Mail 10.617%, both email arms pooled
    assert row["control_mean"] == pytest.approx(0.106167, abs=1e-4)
    assert row["n_control"] == 21306
    assert row["n_treated"] == 42694
    assert row["diff"] == pytest.approx(row["treated_mean"] - row["control_mean"])
    assert row["ci_low"] < row["diff"] < row["ci_high"]


def test_naive_estimates_covers_all_outcomes(df):
    est = naive_estimates(df, "treatment")
    assert set(est.index) == {"visit", "conversion", "spend"}
    # the email arms visit more, on average, than control — true in this data by construction
    assert est.loc["visit", "diff"] > 0


def test_standardised_diff_zero_for_identical_groups():
    same = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert _standardised_diff(same, same) == 0.0


def test_standardised_diff_zero_variance_no_divide_by_zero():
    constant = pd.Series([5.0, 5.0, 5.0])
    assert _standardised_diff(constant, constant) == 0.0


def test_standardised_diff_flags_separated_constant_groups():
    treated = pd.Series([1.0, 1.0, 1.0])
    control = pd.Series([0.0, 0.0, 0.0])

    assert np.isinf(_standardised_diff(treated, control))


def test_naive_report_flags_undefined_infinite_balance_values(tmp_path):
    estimates = pd.DataFrame(
        [{"treated_mean": 0.5, "control_mean": 0.1, "diff": 0.4, "ci_low": 0.2, "ci_high": 0.6}],
        index=pd.Index(["visit"], name="outcome"),
    )
    balance = pd.DataFrame(
        [{"treated_mean": 1.0, "control_mean": 0.0, "standardised_diff": np.inf}],
        index=pd.Index(["separated"], name="covariate"),
    )
    path = tmp_path / "naive.md"

    write_naive_report(estimates, balance, path)

    report = path.read_text(encoding="utf-8")
    assert "undefined (zero within-group variance) ⚠" in report


def test_standardised_diff_known_value():
    treated = pd.Series([2.0, 4.0, 6.0])  # mean 4, var 4
    control = pd.Series([1.0, 3.0, 5.0])  # mean 3, var 4
    # pooled sd = sqrt((4+4)/2) = 2, diff = 1 -> standardised diff = 0.5
    assert _standardised_diff(treated, control) == pytest.approx(0.5)


def test_covariate_balance_well_balanced_on_real_rct(df):
    # this is the actual sanity check the project's identification claim rests on: on a genuine
    # RCT, every covariate's standardised difference should be small.
    balance = covariate_balance(df, "treatment")
    assert balance["standardised_diff"].abs().max() < 0.1


def test_covariate_balance_flags_a_constructed_imbalance():
    n = 2000
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "treatment": [1] * n + [0] * n,
            "recency": np.concatenate([rng.normal(10, 1, n), rng.normal(2, 1, n)]),
            "history": rng.normal(200, 50, 2 * n),
            "mens": rng.integers(0, 2, 2 * n),
            "womens": rng.integers(0, 2, 2 * n),
            "newbie": rng.integers(0, 2, 2 * n),
            "zip_code": rng.choice(["Urban", "Rural", "Surburban"], 2 * n),
            "channel": rng.choice(["Web", "Phone", "Multichannel"], 2 * n),
        }
    )
    balance = covariate_balance(df, "treatment")
    assert balance.loc["recency", "standardised_diff"] > 1.0

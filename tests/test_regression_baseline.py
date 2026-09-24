import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.config import TREATMENT_COL
from src.regression_baseline import (
    average_discrete_change,
    average_marginal_effects,
    build_design_matrix,
    fit_logit,
    fit_ols,
)


@pytest.fixture(scope="module")
def df():
    from src.data_loader import load_hillstrom

    return load_hillstrom()


def test_design_matrix_has_constant_and_treatment(df):
    X = build_design_matrix(df)
    assert "const" in X.columns
    assert TREATMENT_COL in X.columns
    assert (X["const"] == 1.0).all()


def test_logit_visit_treatment_ame_matches_naive_direction_and_magnitude(df):
    model = fit_logit(df, "visit")
    ame = average_marginal_effects(model, df)
    treatment_ame = ame.loc[TREATMENT_COL, "effect"]
    # naive diff-in-means on visit is +0.0609 (reports/02_naive_estimate.md); on an RCT the
    # covariate-adjusted AME should sit close to it, not just share a sign.
    assert 0.05 < treatment_ame < 0.08


def test_logit_conversion_treatment_ame_positive_and_small(df):
    model = fit_logit(df, "conversion")
    ame = average_marginal_effects(model, df)
    treatment_ame = ame.loc[TREATMENT_COL, "effect"]
    assert 0.001 < treatment_ame < 0.02  # conversion rate is under 1%, effect must be small too


def test_reported_treatment_contrast_matches_independent_counterfactual_designs(df):
    model = fit_logit(df, "visit")
    untreated = df.copy()
    treated = df.copy()
    untreated[TREATMENT_COL] = 0
    treated[TREATMENT_COL] = 1

    contrast = average_discrete_change(
        model, build_design_matrix(untreated), build_design_matrix(treated)
    )
    independent = (
        model.predict(build_design_matrix(treated)) - model.predict(build_design_matrix(untreated))
    ).mean()

    assert contrast["effect"] == pytest.approx(independent)
    assert contrast["ci_low"] < contrast["effect"] < contrast["ci_high"]
    assert contrast["se"] > 0


def test_binary_logit_change_is_not_reported_as_a_continuous_derivative():
    rng = np.random.default_rng(19)
    n = 12_000
    df = pd.DataFrame({"const": 1.0, TREATMENT_COL: rng.binomial(1, 0.5, n)})
    x = rng.normal(size=n)
    design = pd.DataFrame({"const": 1.0, TREATMENT_COL: df[TREATMENT_COL], "x": x})
    probability = 1 / (1 + np.exp(-(-2.0 + 2.8 * df[TREATMENT_COL] + x)))
    outcome = rng.binomial(1, probability)
    model = sm.Logit(outcome, design).fit(disp=0, cov_type="HC1")
    design_zero = design.copy()
    design_one = design.copy()
    design_zero[TREATMENT_COL] = 0
    design_one[TREATMENT_COL] = 1

    discrete = average_discrete_change(model, design_zero, design_one)["effect"]
    derivative = model.get_margeff(at="overall").summary_frame().loc[TREATMENT_COL, "dy/dx"]

    assert abs(discrete - derivative) > 0.05


def test_ols_spend_treatment_coefficient_matches_naive(df):
    model = fit_ols(df, "spend")
    coef = model.params[TREATMENT_COL]
    # naive diff-in-means on spend is +0.5968 (reports/02_naive_estimate.md)
    assert 0.4 < coef < 0.8


def test_robust_cov_type_is_hc1(df):
    model = fit_logit(df, "visit")
    assert model.cov_type == "HC1"
    ols_model = fit_ols(df, "spend")
    assert ols_model.cov_type == "HC1"

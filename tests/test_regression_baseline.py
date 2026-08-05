import pytest

from src.config import TREATMENT_COL
from src.regression_baseline import (
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
    ame = average_marginal_effects(model)
    treatment_ame = ame.loc[TREATMENT_COL, "dy/dx"]
    # naive diff-in-means on visit is +0.0609 (reports/02_naive_estimate.md); on an RCT the
    # covariate-adjusted AME should sit close to it, not just share a sign.
    assert 0.05 < treatment_ame < 0.08


def test_logit_conversion_treatment_ame_positive_and_small(df):
    model = fit_logit(df, "conversion")
    ame = average_marginal_effects(model)
    treatment_ame = ame.loc[TREATMENT_COL, "dy/dx"]
    assert 0.001 < treatment_ame < 0.02  # conversion rate is under 1%, effect must be small too


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

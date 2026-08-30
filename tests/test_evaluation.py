import numpy as np
import pandas as pd

from src.config import ARM_COL, ARMS, COVARIATE_COLS, NOMINAL_PROPENSITIES
from src.evaluation import crossfit_arm_outcomes, evaluate_policies


def test_crossfit_arm_outcomes_returns_aligned_probabilities():
    from src.data_loader import load_hillstrom

    df = load_hillstrom().groupby(ARM_COL, group_keys=False).head(1200).reset_index(drop=True)
    predictions = crossfit_arm_outcomes(df, n_splits=3, seed=19, n_estimators=10)

    assert predictions.index.equals(df.index)
    assert predictions.columns.tolist() == ARMS
    assert predictions.notna().all().all()
    assert predictions.ge(0).all().all()
    assert predictions.le(1).all().all()


def test_evaluate_policies_reports_marginal_and_paired_intervals():
    df = pd.DataFrame(
        {
            ARM_COL: ARMS * 4,
            "visit": [0.0, 1.0, 0.0] * 4,
        }
    )
    predictions = pd.DataFrame(
        {arm: np.full(len(df), value) for arm, value in zip(ARMS, [0.1, 0.8, 0.2], strict=True)}
    )
    policies = {
        "learned": np.full(len(df), "Mens E-Mail"),
        "no email": np.full(len(df), "No E-Mail"),
    }

    values, comparisons = evaluate_policies(
        df,
        policies,
        predictions,
        propensities=NOMINAL_PROPENSITIES,
        reference="learned",
        n_boot=200,
        seed=4,
    )

    assert values.columns.tolist() == ["value", "ci_low", "ci_high"]
    assert values.index.tolist() == ["learned", "no email"]
    assert comparisons.index.tolist() == ["learned - no email"]
    assert comparisons.loc["learned - no email", "difference"] > 0
    assert values.loc["learned", "ci_low"] <= values.loc["learned", "value"]
    assert values.loc["learned", "value"] <= values.loc["learned", "ci_high"]


def test_crossfit_contract_uses_the_project_covariates():
    assert COVARIATE_COLS == [
        "recency",
        "history",
        "mens",
        "womens",
        "newbie",
        "zip_code",
        "channel",
    ]

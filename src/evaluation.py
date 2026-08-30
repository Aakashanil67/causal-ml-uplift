"""Held-out nuisance predictions and doubly robust policy evaluation."""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

from src.config import ARM_COL, ARMS, NOMINAL_PROPENSITIES, RANDOM_SEED
from src.data_loader import build_covariate_matrix
from src.policy import dr_policy_contributions, stratified_bootstrap_indices


def crossfit_arm_outcomes(
    df: pd.DataFrame,
    outcome_col: str = "visit",
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
    n_estimators: int = 100,
) -> pd.DataFrame:
    """Cross-fitted estimates of E[Y | X, arm] for every held-out row and action."""
    X = build_covariate_matrix(df).to_numpy()
    strat_key = df[ARM_COL].astype(str) + "_" + df[outcome_col].astype(str)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    predictions = pd.DataFrame(index=df.index, columns=ARMS, dtype=float)

    for train_idx, test_idx in folds.split(X, strat_key):
        train = df.iloc[train_idx]
        for arm in ARMS:
            arm_mask = train[ARM_COL].to_numpy() == arm
            model = LGBMClassifier(
                n_estimators=n_estimators,
                verbosity=-1,
                random_state=seed,
                n_jobs=1,
            )
            model.fit(X[train_idx][arm_mask], train.loc[arm_mask, outcome_col])
            predictions.loc[df.index[test_idx], arm] = model.predict_proba(X[test_idx])[:, 1]
    return predictions


def evaluate_policies(
    df: pd.DataFrame,
    policies: dict[str, np.ndarray],
    outcome_predictions: pd.DataFrame,
    propensities: dict[str, float] | None = None,
    reference: str = "learned (DRPolicyForest)",
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DR policy values and paired differences, bootstrapped within randomized arms."""
    propensity = NOMINAL_PROPENSITIES if propensities is None else propensities
    contributions = {
        name: dr_policy_contributions(df, rec, "visit", propensity, outcome_predictions)
        for name, rec in policies.items()
    }
    draws = stratified_bootstrap_indices(df.reset_index(drop=True), n_boot, seed)

    value_rows = []
    for name, values in contributions.items():
        estimates = values[draws].mean(axis=1)
        low, high = np.percentile(estimates, [2.5, 97.5])
        value_rows.append({"policy": name, "value": values.mean(), "ci_low": low, "ci_high": high})
    policy_values = pd.DataFrame(value_rows).set_index("policy")

    comparison_rows = []
    reference_values = contributions[reference]
    for name, values in contributions.items():
        if name == reference:
            continue
        differences = reference_values - values
        estimates = differences[draws].mean(axis=1)
        low, high = np.percentile(estimates, [2.5, 97.5])
        comparison_rows.append(
            {
                "comparison": f"{reference} - {name}",
                "difference": differences.mean(),
                "ci_low": low,
                "ci_high": high,
            }
        )
    comparisons = pd.DataFrame(comparison_rows).set_index("comparison")
    return policy_values, comparisons

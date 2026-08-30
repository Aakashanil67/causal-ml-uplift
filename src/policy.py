"""Three-action policy learning: for each customer, which of {no email, mens email, womens email}
maximises expected `visit`, versus what an obvious purchase-history heuristic would do.

The published policy comparison is built in `src.build_artifacts` and evaluated by
`src.evaluation` with cross-fitted outcome models and a doubly robust estimator. This module owns
the policy learner, recommendation rules, and small inverse-propensity utilities used for
diagnostics and tests. The pipeline, rather than this module, is the sole publisher of results.
"""

import numpy as np
import pandas as pd
from econml.policy import DRPolicyForest
from lightgbm import LGBMClassifier, LGBMRegressor

from src.config import ARM_COL, ARMS, CONTROL_ARM, RANDOM_SEED
from src.data_loader import build_covariate_matrix


def fit_policy_forest(train: pd.DataFrame, seed: int = RANDOM_SEED) -> DRPolicyForest:
    X = build_covariate_matrix(train).to_numpy()
    T = train[ARM_COL].to_numpy()
    Y = train["visit"].to_numpy(dtype=float)
    pf = DRPolicyForest(
        model_regression=LGBMRegressor(n_estimators=100, verbose=-1, random_state=seed),
        model_propensity=LGBMClassifier(n_estimators=100, verbose=-1, random_state=seed),
        n_estimators=200,
        min_samples_leaf=50,
        max_depth=5,
        random_state=seed,
        cv=3,
        categories=[CONTROL_ARM] + [a for a in ARMS if a != CONTROL_ARM],
    )
    pf.fit(Y, T, X=X)
    return pf


def _treatment_name_to_arm(name: str, control: str = CONTROL_ARM) -> str:
    return control if name == "None" else name.split("T0_", 1)[1]


def policy_forest_recommendations(pf: DRPolicyForest, df: pd.DataFrame) -> np.ndarray:
    X = build_covariate_matrix(df).to_numpy()
    codes = pf.predict(X)
    names = [_treatment_name_to_arm(n) for n in pf.policy_treatment_names()]
    return np.array([names[c] for c in codes])


def heuristic_recommendations(df: pd.DataFrame) -> np.ndarray:
    """The obvious rule a marketer would try without any modelling: match the creative to the
    customer's own purchase history, defaulting to the womens creative when a customer has bought
    both (the CATE work in reports/07_uplift_policy.md shows that segment responds slightly better
    to the pooled treatment than mens-only customers do)."""
    womens_first = np.where(df["womens"] == 1, "Womens E-Mail", "")
    mens_fallback = np.where(df["mens"] == 1, "Mens E-Mail", CONTROL_ARM)
    return np.where(womens_first != "", womens_first, mens_fallback)


def ipw_policy_value(
    df: pd.DataFrame, recommended_arm: np.ndarray, outcome_col: str, propensities: dict
) -> float:
    return float(policy_value_contributions(df, recommended_arm, outcome_col, propensities).mean())


def policy_value_contributions(
    df: pd.DataFrame, recommended_arm: np.ndarray, outcome_col: str, propensities: dict
) -> np.ndarray:
    """Return one Horvitz--Thompson contribution per row for a fixed policy."""
    actual_arm = df[ARM_COL].to_numpy()
    Y = df[outcome_col].to_numpy(dtype=float)
    matched = actual_arm == recommended_arm
    p = np.array([propensities[a] for a in actual_arm])
    return matched * Y / p


def dr_policy_contributions(
    df: pd.DataFrame,
    recommended_arm: np.ndarray,
    outcome_col: str,
    propensities: dict,
    outcome_predictions: pd.DataFrame,
) -> np.ndarray:
    """One augmented-IPW contribution per row for a fixed policy."""
    actual_arm = df[ARM_COL].to_numpy()
    rec = np.asarray(recommended_arm)
    if len(rec) != len(df) or len(outcome_predictions) != len(df):
        raise ValueError("Recommendations and outcome predictions must align with df.")
    missing = set(np.unique(np.concatenate([actual_arm, rec]))) - set(outcome_predictions.columns)
    if missing:
        raise ValueError(f"Outcome predictions missing arms: {sorted(missing)}")

    row = np.arange(len(df))
    mu_rec = outcome_predictions.to_numpy()[row, outcome_predictions.columns.get_indexer(rec)]
    mu_actual = outcome_predictions.to_numpy()[
        row, outcome_predictions.columns.get_indexer(actual_arm)
    ]
    matched = actual_arm == rec
    p = np.array([propensities[arm] for arm in actual_arm])
    residual = df[outcome_col].to_numpy(dtype=float) - mu_actual
    return mu_rec + matched * residual / p


def dr_policy_value(
    df: pd.DataFrame,
    recommended_arm: np.ndarray,
    outcome_col: str,
    propensities: dict,
    outcome_predictions: pd.DataFrame,
) -> float:
    return float(
        dr_policy_contributions(
            df, recommended_arm, outcome_col, propensities, outcome_predictions
        ).mean()
    )


def stratified_bootstrap_indices(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """Bootstrap rows within randomized arms, preserving each arm's sample size."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(df[ARM_COL].to_numpy() == arm) for arm in df[ARM_COL].unique()]
    return np.array(
        [
            np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
            for _ in range(n_boot)
        ]
    )


def bootstrap_policy_value_ci(
    df: pd.DataFrame,
    recommended_arm: np.ndarray,
    outcome_col: str,
    propensities: dict,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    df_arr = df.reset_index(drop=True)
    estimates = np.empty(n_boot)
    for b, idx in enumerate(stratified_bootstrap_indices(df_arr, n_boot, seed)):
        estimates[b] = ipw_policy_value(
            df_arr.iloc[idx], recommended_arm[idx], outcome_col, propensities
        )
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def bootstrap_policy_difference_ci(
    df: pd.DataFrame,
    recommended_arm_a: np.ndarray,
    recommended_arm_b: np.ndarray,
    outcome_col: str,
    propensities: dict,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap interval for policy A minus policy B.

    Resampling row-level contribution differences preserves their covariance. Marginal confidence
    intervals, whether they overlap or not, cannot answer this comparison.
    """
    a = policy_value_contributions(df, recommended_arm_a, outcome_col, propensities)
    b = policy_value_contributions(df, recommended_arm_b, outcome_col, propensities)
    differences = a - b
    indices = stratified_bootstrap_indices(df.reset_index(drop=True), n_boot, seed)
    estimates = differences[indices].mean(axis=1)
    return (
        float(differences.mean()),
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


def main() -> None:
    raise SystemExit("Run `python -m src.pipeline` to rebuild the published policy outputs.")


if __name__ == "__main__":
    main()

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


def policy_comparison_conclusion(ci_low: float, ci_high: float) -> str:
    """Summarize which policy has higher expected visit value from its paired interval."""
    if ci_low > ci_high:
        raise ValueError("policy comparison interval lower endpoint exceeds upper endpoint")
    if ci_low <= 0 <= ci_high:
        return (
            "Held-out evidence does not establish which policy has higher expected visit value. "
            "Blanket mens emailing is the simpler action for this experiment; validate any "
            "learned policy on another campaign before broader deployment."
        )
    if ci_low > 0:
        return (
            "Held-out evidence supports higher expected visit value for the learned policy than "
            "blanket mens emailing in this evaluation. Validate this campaign-specific result "
            "before broader deployment."
        )
    return (
        "Held-out evidence supports higher expected visit value for blanket mens emailing than "
        "for the learned policy in this evaluation."
    )


def heuristic_recommendations(df: pd.DataFrame) -> np.ndarray:
    """Match exclusive history and use the stronger overall mens creative for dual buyers."""
    mens = df["mens"].to_numpy() == 1
    womens = df["womens"].to_numpy() == 1
    return np.select(
        [mens, womens],
        ["Mens E-Mail", "Womens E-Mail"],
        default=CONTROL_ARM,
    )


def incremental_net_value(
    policy_spend: float,
    no_email_spend: float,
    contact_rate: float,
    gross_margin: float,
    email_cost: float,
) -> float:
    """Return incremental contribution per customer under an explicit margin assumption."""
    if not 0 <= contact_rate <= 1:
        raise ValueError("contact_rate must be between 0 and 1")
    if not 0 <= gross_margin <= 1:
        raise ValueError("gross_margin must be between 0 and 1")
    if email_cost < 0:
        raise ValueError("email_cost must be non-negative")
    return (policy_spend - no_email_spend) * gross_margin - contact_rate * email_cost


def incremental_net_value_interval(
    spend_difference_low: float,
    spend_difference_high: float,
    contact_rate_difference: float,
    gross_margin: float,
    email_cost: float,
) -> tuple[float, float]:
    """Transform a paired reported-spend interval under fixed margin and contact-cost inputs."""
    if not 0 <= gross_margin <= 1:
        raise ValueError("gross_margin must be between 0 and 1")
    if email_cost < 0:
        raise ValueError("email_cost must be non-negative")
    if spend_difference_low > spend_difference_high:
        raise ValueError("spend interval lower endpoint must not exceed its upper endpoint")
    contact_cost = contact_rate_difference * email_cost
    return (
        spend_difference_low * gross_margin - contact_cost,
        spend_difference_high * gross_margin - contact_cost,
    )


def incremental_net_value_with_interval(
    spend_difference: float,
    spend_difference_low: float,
    spend_difference_high: float,
    contact_rate_difference: float,
    gross_margin: float,
    email_cost: float,
) -> dict[str, float]:
    """Return point and paired conditional interval under fixed economic assumptions."""
    ci_low, ci_high = incremental_net_value_interval(
        spend_difference_low,
        spend_difference_high,
        contact_rate_difference,
        gross_margin,
        email_cost,
    )
    return {
        "value": spend_difference * gross_margin - contact_rate_difference * email_cost,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def break_even_margin(
    policy_spend: float,
    no_email_spend: float,
    contact_rate: float,
    email_cost: float,
) -> float | None:
    """Return the gross margin at which incremental contribution reaches zero."""
    if not 0 <= contact_rate <= 1:
        raise ValueError("contact_rate must be between 0 and 1")
    if email_cost < 0:
        raise ValueError("email_cost must be non-negative")
    delta_spend = policy_spend - no_email_spend
    if delta_spend == 0:
        return 0.0 if contact_rate == 0 else None
    if delta_spend < 0:
        return None
    return contact_rate * email_cost / delta_spend


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
    if not df.index.is_unique or not outcome_predictions.index.is_unique:
        raise ValueError("df and outcome-prediction indices must be unique.")
    if not outcome_predictions.index.equals(df.index):
        if set(outcome_predictions.index) != set(df.index):
            raise ValueError("df and outcome-prediction indices must match.")
        outcome_predictions = outcome_predictions.reindex(df.index)

    actions = set(np.unique(np.concatenate([actual_arm, rec])))
    unsupported = actions - set(ARMS)
    if unsupported:
        raise ValueError(
            f"Recommendations or observed treatment contain unsupported action: {sorted(unsupported)}"
        )
    missing = actions - set(outcome_predictions.columns)
    if missing:
        raise ValueError(f"Outcome predictions missing arms: {sorted(missing)}")
    if outcome_col not in df:
        raise ValueError(f"Outcome column {outcome_col!r} is missing from df.")
    outcomes = df[outcome_col].to_numpy(dtype=float)
    if not np.isfinite(outcomes).all():
        raise ValueError("Observed outcomes must contain only finite values.")
    try:
        probabilities = np.array([propensities[arm] for arm in actual_arm], dtype=float)
    except KeyError as error:
        raise ValueError(f"Propensities are missing observed action {error.args[0]!r}.") from error
    if not np.isfinite(probabilities).all() or (probabilities <= 0).any():
        raise ValueError("Propensities for observed actions must be finite and strictly positive.")
    if (probabilities > 1).any():
        raise ValueError("Propensities must not exceed one.")
    if not np.isfinite(outcome_predictions.to_numpy(dtype=float)).all():
        raise ValueError("Outcome predictions must contain only finite values.")

    row = np.arange(len(df))
    rec_idx = outcome_predictions.columns.get_indexer(rec)
    actual_idx = outcome_predictions.columns.get_indexer(actual_arm)
    if (rec_idx < 0).any() or (actual_idx < 0).any():
        raise ValueError("Outcome predictions are missing an action required for evaluation.")
    mu_rec = outcome_predictions.to_numpy()[row, rec_idx]
    mu_actual = outcome_predictions.to_numpy()[row, actual_idx]
    matched = actual_arm == rec
    residual = outcomes - mu_actual
    return mu_rec + matched * residual / probabilities


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

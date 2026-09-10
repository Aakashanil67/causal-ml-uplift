"""Bounded semi-synthetic checks for estimator bias under known confounding."""

import numpy as np
import pandas as pd
from econml.dml import LinearDML
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.special import expit, logit


def simulate_observational_sample(
    n: int,
    confounding_strength: float,
    seed: int,
) -> tuple[pd.DataFrame, float]:
    """Generate an observational binary-outcome sample and its exact sample ATE."""
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.5, size=n)
    x3 = rng.normal(size=n)
    propensity = np.clip(expit(confounding_strength * (0.8 * x1 - 0.6 * x2)), 0.05, 0.95)
    p0 = expit(-1.5 + 0.7 * x1 - 0.3 * x2 + 0.2 * x3)
    p1 = expit(logit(p0) + 0.4 + 0.2 * x2)
    treatment = rng.binomial(1, propensity)
    probability = np.where(treatment == 1, p1, p0)
    outcome = rng.binomial(1, probability)
    data = pd.DataFrame(
        {
            "X1": x1,
            "X2": x2,
            "X3": x3,
            "propensity": propensity,
            "treatment": treatment,
            "outcome": outcome,
            "p0": p0,
            "p1": p1,
            "true_effect": p1 - p0,
        }
    )
    return data, float(np.mean(p1 - p0))


def _fit_dml(data: pd.DataFrame, covariates: list[str], seed: int) -> tuple[float, float, float]:
    estimator = LinearDML(
        model_y=LGBMRegressor(n_estimators=50, verbosity=-1, random_state=seed, n_jobs=1),
        model_t=LGBMClassifier(n_estimators=50, verbosity=-1, random_state=seed, n_jobs=1),
        discrete_treatment=True,
        cv=2,
        random_state=seed,
    )
    X = data[covariates].to_numpy()
    estimator.fit(data["outcome"].to_numpy(), data["treatment"].to_numpy(), X=X)
    interval = estimator.ate_interval(X)
    return float(estimator.ate(X)), float(interval[0]), float(interval[1])


def run_monte_carlo(
    strengths: tuple[float, ...] = (0.0, 0.75, 1.5),
    n_runs: int = 10,
    n: int = 1500,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare naive, adjusted and omitted-confounder estimates over known DGPs."""
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    rows = []
    for strength in strengths:
        for run in range(n_runs):
            data, true_ate = simulate_observational_sample(n, strength, seed + run)
            naive = (
                data.loc[data["treatment"] == 1, "outcome"].mean()
                - data.loc[data["treatment"] == 0, "outcome"].mean()
            )
            estimates = {
                "naive": (float(naive), np.nan, np.nan),
                "adjusted_dml": _fit_dml(data, ["X1", "X2", "X3"], seed + run),
                "omitted_confounder_dml": _fit_dml(data, ["X3"], seed + run),
            }
            for estimator, (estimate, ci_low, ci_high) in estimates.items():
                rows.append(
                    {
                        "confounding_strength": strength,
                        "run": run,
                        "estimator": estimator,
                        "estimate": estimate,
                        "true_ate": true_ate,
                        "bias": estimate - true_ate,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "covered": (
                            bool(ci_low <= true_ate <= ci_high) if np.isfinite(ci_low) else np.nan
                        ),
                    }
                )
    raw = pd.DataFrame(rows)
    summary = raw.groupby(["confounding_strength", "estimator"], as_index=False).agg(
        mean_estimate=("estimate", "mean"),
        bias=("bias", "mean"),
        rmse=("bias", lambda values: float(np.sqrt(np.mean(np.square(values))))),
        coverage=("covered", "mean"),
        n_runs=("run", "nunique"),
    )
    return summary

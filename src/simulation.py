"""Fully synthetic checks for estimator bias under known confounding."""

import argparse

import numpy as np
import pandas as pd
from econml.dml import LinearDML
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


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
    """Small DML smoke run comparing estimates to known synthetic sample ATEs."""
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    rows = []
    for strength in strengths:
        for run in range(n_runs):
            data, true_ate = simulate_observational_sample(n, strength, seed + run)
            treated = data.loc[data["treatment"] == 1, "outcome"]
            control = data.loc[data["treatment"] == 0, "outcome"]
            naive = float(treated.mean() - control.mean())
            naive_se = float(
                np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
            )
            estimates = {
                "naive": (naive, naive - 1.96 * naive_se, naive + 1.96 * naive_se),
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
    return _summarize_runs(pd.DataFrame(rows))


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Coverage interval requires at least one finite confidence interval.")
    z = 1.96
    rate = successes / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    half_width = z * np.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def _summarize_runs(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strength, estimator), group in raw.groupby(["confounding_strength", "estimator"]):
        biases = group["bias"].to_numpy(dtype=float)
        n_runs = int(group["run"].nunique())
        bias = float(biases.mean())
        bias_mc_se = float(biases.std(ddof=1) / np.sqrt(n_runs)) if n_runs > 1 else 0.0
        covered = group["covered"].dropna().astype(bool)
        if len(covered):
            coverage = float(covered.mean())
            coverage_ci_low, coverage_ci_high = _wilson_interval(int(covered.sum()), len(covered))
        else:
            coverage = coverage_ci_low = coverage_ci_high = None
        rows.append(
            {
                "confounding_strength": float(strength),
                "estimator": str(estimator),
                "mean_estimate": float(group["estimate"].mean()),
                "bias": bias,
                "bias_mc_se": bias_mc_se,
                "bias_mc_ci_low": bias - 1.96 * bias_mc_se,
                "bias_mc_ci_high": bias + 1.96 * bias_mc_se,
                "rmse": float(np.sqrt(np.mean(np.square(biases)))),
                "coverage": coverage,
                "coverage_ci_low": coverage_ci_low,
                "coverage_ci_high": coverage_ci_high,
                "n_runs": n_runs,
            }
        )
    return pd.DataFrame(rows)


def _aipw_estimate(
    data: pd.DataFrame,
    seed: int,
    covariates: list[str] | None = None,
    oracle_nuisance: bool = False,
) -> tuple[float, float, float]:
    """Cross-fitted AIPW ATE and influence-function interval for binary outcomes."""
    treatment = data["treatment"].to_numpy(dtype=int)
    outcome = data["outcome"].to_numpy(dtype=float)
    if oracle_nuisance:
        propensity = data["propensity"].to_numpy(dtype=float)
        mu0 = data["p0"].to_numpy(dtype=float)
        mu1 = data["p1"].to_numpy(dtype=float)
    else:
        covariates = covariates or ["X1", "X2", "X3"]
        X = data[covariates].to_numpy(dtype=float)
        propensity = np.empty(len(data))
        mu0 = np.empty(len(data))
        mu1 = np.empty(len(data))
        folds = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
        for train_idx, test_idx in folds.split(X, treatment):
            propensity_model = LogisticRegression(C=1e4, max_iter=500, random_state=seed)
            propensity_model.fit(X[train_idx], treatment[train_idx])
            propensity[test_idx] = propensity_model.predict_proba(X[test_idx])[:, 1]
            for arm, target in ((0, mu0), (1, mu1)):
                arm_train = train_idx[treatment[train_idx] == arm]
                outcome_model = LogisticRegression(C=1e4, max_iter=500, random_state=seed)
                outcome_model.fit(X[arm_train], outcome[arm_train])
                target[test_idx] = outcome_model.predict_proba(X[test_idx])[:, 1]
        propensity = np.clip(propensity, 0.025, 0.975)

    score = (
        mu1
        - mu0
        + treatment * (outcome - mu1) / propensity
        - (1 - treatment) * (outcome - mu0) / (1 - propensity)
    )
    estimate = float(score.mean())
    se = float(score.std(ddof=1) / np.sqrt(len(score)))
    return estimate, estimate - 1.96 * se, estimate + 1.96 * se


def run_extended_monte_carlo(
    strengths: tuple[float, ...] = (0.0, 0.75, 1.5),
    n_runs: int = 500,
    n: int = 1500,
    seed: int = 42,
) -> pd.DataFrame:
    """Run a larger known-truth simulation with AIPW and oracle-nuisance diagnostics.

    The target is the sample average of the generated probability differences, `mean(p1 - p0)`.
    Bias intervals quantify Monte Carlo error across repetitions; coverage intervals use Wilson
    intervals for the fraction of run-specific 95% intervals containing that target.
    """
    if n_runs < 2:
        raise ValueError("n_runs must be at least 2 to estimate Monte Carlo uncertainty.")
    raw_rows = []
    for strength in strengths:
        for run in range(n_runs):
            run_seed = seed + run
            data, true_ate = simulate_observational_sample(n, strength, run_seed)
            treated = data.loc[data["treatment"] == 1, "outcome"]
            control = data.loc[data["treatment"] == 0, "outcome"]
            naive = float(treated.mean() - control.mean())
            naive_se = float(
                np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
            )
            estimates = {
                "naive": (naive, naive - 1.96 * naive_se, naive + 1.96 * naive_se),
                "adjusted_aipw": _aipw_estimate(data, run_seed),
                "omitted_confounder_aipw": _aipw_estimate(data, run_seed, covariates=["X3"]),
                "oracle_aipw": _aipw_estimate(data, run_seed, oracle_nuisance=True),
            }
            for estimator, (estimate, ci_low, ci_high) in estimates.items():
                raw_rows.append(
                    {
                        "confounding_strength": strength,
                        "run": run,
                        "estimator": estimator,
                        "estimate": estimate,
                        "true_ate": true_ate,
                        "bias": estimate - true_ate,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "covered": bool(ci_low <= true_ate <= ci_high),
                    }
                )
    return _summarize_runs(pd.DataFrame(raw_rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the optional fully synthetic AIPW benchmark.")
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--sample-size", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = run_extended_monte_carlo(n_runs=args.repetitions, n=args.sample_size, seed=args.seed)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

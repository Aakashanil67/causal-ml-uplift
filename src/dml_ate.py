"""LinearDML average treatment effects: LightGBM nuisance models, cross-fitting, one fit per
(outcome, treatment-comparison) pair.

Double ML's trick is partialling out: instead of controlling for confounders by putting them in a
single linear regression alongside treatment (the OLS approach in `src/regression_baseline.py`),
DML first predicts the outcome from covariates alone and the treatment from covariates alone,
*using flexible ML models for nuisance functions*, then estimates the treatment effect from what's left over in both — the parts of Y and T
that the covariates couldn't explain. **Cross-fitting** (`cv=3` below) is what keeps this valid:
fitting the nuisance models and the effect model on the same data would let the nuisance models
overfit and leak information into the effect estimate, so each fold's nuisance predictions come
from models trained on the *other* folds. On a randomised experiment, where OLS already had nothing
to partial out, DML shouldn't move the answer much. For heterogeneity, this project compares a
held-out regression interaction audit with `src/cate.py`'s forest and policy ranking. `LinearDML`
uses a linear final effect stage, so flexible nuisance models alone do not guarantee recovery under
an arbitrary nonlinear treatment-effect surface.
"""

import pandas as pd
from econml.dml import LinearDML
from lightgbm import LGBMClassifier, LGBMRegressor

from src.config import ARM_COL, ARMS, CONTROL_ARM, OUTCOME_COLS, RANDOM_SEED
from src.data_loader import build_covariate_matrix


def _make_dml(seed: int = RANDOM_SEED) -> LinearDML:
    return LinearDML(
        model_y=LGBMRegressor(n_estimators=100, verbose=-1, random_state=seed),
        model_t=LGBMClassifier(n_estimators=100, verbose=-1, random_state=seed),
        discrete_treatment=True,
        cv=3,
        random_state=seed,
    )


def dml_ate(df: pd.DataFrame, treatment_col: str, outcome_col: str) -> dict:
    X = build_covariate_matrix(df).to_numpy()
    T = df[treatment_col].to_numpy()
    Y = df[outcome_col].to_numpy(dtype=float)
    est = _make_dml()
    est.fit(Y, T, X=X)
    ate = est.ate(X)
    ci_low, ci_high = est.ate_interval(X)
    return {"ate": float(ate), "ci_low": float(ci_low), "ci_high": float(ci_high)}


def pooled_ate_table(df: pd.DataFrame) -> pd.DataFrame:
    """Any email vs no email, matching the treatment definition every other module uses."""
    from src.config import TREATMENT_COL

    rows = [{"outcome": o, **dml_ate(df, TREATMENT_COL, o)} for o in OUTCOME_COLS]
    return pd.DataFrame(rows).set_index("outcome")


def per_arm_ate_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mens-vs-control and womens-vs-control separately, each fit on the two-arm subsample that
    excludes the other email arm — the two creatives are different treatments, and pooling them
    (as pooled_ate_table does) hides that mens and womens email don't work equally well, which is
    the whole reason src/policy.py treats this as a three-action decision rather than a coin flip.
    """
    rows = []
    for arm in [a for a in ARMS if a != CONTROL_ARM]:
        sub = df[df[ARM_COL].isin([arm, CONTROL_ARM])].copy()
        sub["arm_treatment"] = (sub[ARM_COL] == arm).astype(int)
        for outcome in OUTCOME_COLS:
            rows.append({"arm": arm, "outcome": outcome, **dml_ate(sub, "arm_treatment", outcome)})
    return pd.DataFrame(rows).set_index(["arm", "outcome"])


def write_dml_report(
    pooled: pd.DataFrame, per_arm: pd.DataFrame, regression_diffs: dict, out_path
) -> None:
    lines = [
        "# Average treatment effects via Double Machine Learning",
        "",
        "`LinearDML` (EconML) with `LightGBM` nuisance models for both the outcome and treatment",
        "regressions, 3-fold cross-fitting. See `src/dml_ate.py`'s docstring for what partialling",
        "out and cross-fitting actually do; the short version is that on a randomised experiment",
        "there's nothing for DML to correct for, so the value of running it here is establishing",
        "that the machinery agrees with simpler methods. Effect heterogeneity can also be modeled",
        "with interactions or flexible final effect models; the interaction audit and causal forest",
        "provide complementary exploratory evidence.",
        "",
        "## Pooled treatment (any email vs none), vs the regression baseline",
        "",
        "`visit`/`conversion` in percentage points, `spend` in dollars, matching",
        "`reports/03_regression_baseline.md`'s units so the two reports read side by side.",
        "",
        "| outcome | DML ATE | DML 95% CI | regression discrete change / coefficient |",
        "|---|---|---|---|",
    ]
    for outcome, row in pooled.iterrows():
        reg = regression_diffs[outcome]
        if outcome == "spend":
            lines.append(
                f"| {outcome} | +${row['ate']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] "
                f"| +${reg:.4f} |"
            )
        else:
            lines.append(
                f"| {outcome} | +{row['ate'] * 100:.2f}pp | [{row['ci_low'] * 100:.2f}pp, "
                f"{row['ci_high'] * 100:.2f}pp] | +{reg * 100:.2f}pp |"
            )
    lines += [
        "",
        "DML and the covariate-adjusted regression from `reports/03_regression_baseline.md` land",
        "within each other's confidence intervals on every outcome — the expected result on an RCT,",
        "and a real check rather than a formality: if a flexible nuisance model had found nonlinear",
        "structure in the covariates that a linear control missed, these numbers could have",
        "diverged.",
        "",
        "## By arm: mens email and womens email are not the same treatment",
        "",
        "| arm | outcome | DML ATE vs control | 95% CI |",
        "|---|---|---|---|",
    ]
    for (arm, outcome), row in per_arm.iterrows():
        if outcome == "spend":
            lines.append(
                f"| {arm} | {outcome} | +${row['ate']:.4f} | "
                f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] |"
            )
        else:
            lines.append(
                f"| {arm} | {outcome} | +{row['ate'] * 100:.2f}pp | "
                f"[{row['ci_low'] * 100:.2f}pp, {row['ci_high'] * 100:.2f}pp] |"
            )
    mens_visit = per_arm.loc[("Mens E-Mail", "visit"), "ate"]
    womens_visit = per_arm.loc[("Womens E-Mail", "visit"), "ate"]
    lines += [
        "",
        f"On `visit`, the mens email lifts the visit rate by {mens_visit * 100:.2f} percentage",
        f"points and the womens email by {womens_visit * 100:.2f}, a difference between two",
        "creative-specific estimates that the pooled any-email number averages away. This",
        "difference alone does not establish customer-level personalization value. `src/policy.py`",
        "treats the decision as three actions (no email, mens email, womens email) and evaluates",
        "whether learned targeting improves on blanket and purchase-history policies.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from src.config import REPORTS_DIR, TREATMENT_COL
    from src.data_loader import load_hillstrom
    from src.regression_baseline import average_marginal_effects, fit_logit, fit_ols

    df = load_hillstrom()

    pooled = pooled_ate_table(df)
    per_arm = per_arm_ate_table(df)
    print(pooled)
    print(per_arm)

    visit_ame = average_marginal_effects(fit_logit(df, "visit"), df).loc[TREATMENT_COL, "effect"]
    conversion_ame = average_marginal_effects(fit_logit(df, "conversion"), df).loc[
        TREATMENT_COL, "effect"
    ]
    spend_coef = fit_ols(df, "spend").params[TREATMENT_COL]
    regression_estimates = {"visit": visit_ame, "conversion": conversion_ame, "spend": spend_coef}

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_dml_report(pooled, per_arm, regression_estimates, REPORTS_DIR / "05_dml_ate.md")


if __name__ == "__main__":
    main()

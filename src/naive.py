"""The naive estimate: difference in mean outcomes between the treated (any email) and control
group, plus a covariate balance table.

On a randomised experiment this comparison is unbiased, so it is a legitimate estimate here, not
just a strawman. The balance table is the empirical check of *why*: if treatment assignment really
didn't depend on any covariate, treated and control should look statistically indistinguishable on
every covariate before the campaign ever ran. `reports/06_confounding_benchmark.md` runs the same
difference-in-means on a version of this data where that check fails on purpose, so the reader sees
what a biased naive estimate actually looks like, not just an assertion that this one isn't.
"""

import numpy as np
import pandas as pd

from src.config import BINARY_COVARIATES, CATEGORICAL_COVARIATES, NUMERIC_COVARIATES, OUTCOME_COLS


def diff_in_means(df: pd.DataFrame, treatment_col: str, outcome_col: str) -> dict:
    treated = df.loc[df[treatment_col] == 1, outcome_col]
    control = df.loc[df[treatment_col] == 0, outcome_col]
    diff = treated.mean() - control.mean()
    se = np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
    return {
        "outcome": outcome_col,
        "treated_mean": treated.mean(),
        "control_mean": control.mean(),
        "diff": diff,
        "se": se,
        "ci_low": diff - 1.96 * se,
        "ci_high": diff + 1.96 * se,
        "n_treated": len(treated),
        "n_control": len(control),
    }


def naive_estimates(df: pd.DataFrame, treatment_col: str) -> pd.DataFrame:
    return pd.DataFrame([diff_in_means(df, treatment_col, o) for o in OUTCOME_COLS]).set_index(
        "outcome"
    )


def _standardised_diff(treated: pd.Series, control: pd.Series) -> float:
    """(mean_t - mean_c) / pooled SD — the standard covariate-balance metric. |d| < 0.1 is the
    usual "well balanced" rule of thumb (Austin, 2009); values above that flag a covariate that
    plausibly differs between arms even before treatment could have had any effect."""
    pooled_sd = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
    if pooled_sd == 0:
        return 0.0
    return (treated.mean() - control.mean()) / pooled_sd


def covariate_balance(df: pd.DataFrame, treatment_col: str) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_COVARIATES + BINARY_COVARIATES:
        treated = df.loc[df[treatment_col] == 1, col]
        control = df.loc[df[treatment_col] == 0, col]
        rows.append(
            {
                "covariate": col,
                "treated_mean": treated.mean(),
                "control_mean": control.mean(),
                "standardised_diff": _standardised_diff(treated, control),
            }
        )
    for col in CATEGORICAL_COVARIATES:
        for level in sorted(df[col].unique()):
            indicator = (df[col] == level).astype(float)
            treated = indicator[df[treatment_col] == 1]
            control = indicator[df[treatment_col] == 0]
            rows.append(
                {
                    "covariate": f"{col}={level}",
                    "treated_mean": treated.mean(),
                    "control_mean": control.mean(),
                    "standardised_diff": _standardised_diff(treated, control),
                }
            )
    return pd.DataFrame(rows).set_index("covariate")


def write_naive_report(estimates: pd.DataFrame, balance: pd.DataFrame, out_path) -> None:
    lines = [
        "# Naive estimate: difference in means",
        "",
        "Treated = received either email (Mens or Womens), control = No E-Mail. On an RCT this",
        "comparison is unbiased by design, so unlike the usual textbook framing, this is a real",
        "estimate here, not just a demonstration of what goes wrong. Whether the design assumption",
        "actually holds in this file is checked below, not just asserted.",
        "",
        "## Outcome differences",
        "",
        "| outcome | treated mean | control mean | diff | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for outcome, row in estimates.iterrows():
        lines.append(
            f"| {outcome} | {row['treated_mean']:.5f} | {row['control_mean']:.5f} | "
            f"{row['diff']:+.5f} | [{row['ci_low']:.5f}, {row['ci_high']:.5f}] |"
        )
    lines += [
        "",
        "## Covariate balance",
        "",
        "Standardised difference = (treated mean − control mean) / pooled SD. Values inside ±0.1",
        "are the usual threshold for calling a covariate balanced (Austin, 2009); nothing here is a",
        "real pre-treatment difference if the randomisation worked as intended.",
        "",
        "| covariate | treated mean | control mean | standardised diff |",
        "|---|---|---|---|",
    ]
    max_abs_sdiff = balance["standardised_diff"].abs().max()
    for cov, row in balance.iterrows():
        flag = " ⚠" if abs(row["standardised_diff"]) >= 0.1 else ""
        lines.append(
            f"| {cov} | {row['treated_mean']:.4f} | {row['control_mean']:.4f} | "
            f"{row['standardised_diff']:+.4f}{flag} |"
        )
    n_flagged = (balance["standardised_diff"].abs() >= 0.1).sum()
    lines += [
        "",
        f"Largest standardised difference: {max_abs_sdiff:.4f}. {n_flagged} of {len(balance)}",
        "covariates (across numeric, binary and one row per categorical level) exceed the 0.1",
        "threshold. This is the sanity check the randomisation claim in",
        "`reports/01_causal_question.md` rests on: it is a testable statement about this file, not",
        "an assumption. `reports/06_confounding_benchmark.md` shows what this table looks like when",
        "that assumption is deliberately violated.",
        "",
        "**Why this comparison would be biased without randomisation.** If treatment had been",
        "chosen by a marketer rather than a coin flip, a nonzero standardised difference on",
        "`history` or `recency` above would mean the treated and control groups differed in ways",
        "that independently predict the outcome, and the diff-in-means table above would then be",
        "mixing the true effect of the email with the effect of already being a different kind of",
        "customer. That confound is exactly what `src/confounded.py` reconstructs on purpose,",
        "using this same dataset, to make the size of that bias visible.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from src.config import REPORTS_DIR, TREATMENT_COL
    from src.data_loader import load_hillstrom

    df = load_hillstrom()
    estimates = naive_estimates(df, TREATMENT_COL)
    balance = covariate_balance(df, TREATMENT_COL)
    print(estimates)
    print()
    print(balance)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_naive_report(estimates, balance, REPORTS_DIR / "02_naive_estimate.md")


if __name__ == "__main__":
    main()

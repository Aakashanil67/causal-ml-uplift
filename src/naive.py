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
        difference = treated.mean() - control.mean()
        return 0.0 if difference == 0 else float(np.copysign(np.inf, difference))
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
        "Treated = received either email (Mens or Womens), control = No E-Mail. Under randomized",
        "assignment, the difference in means estimates the average effect of assignment to the",
        "email mixture. The balance table is a descriptive check on this file, not proof that the",
        "randomization worked or that unmeasured causes are absent.",
        "",
        "## Outcome differences",
        "",
        "| outcome | treated mean | control mean | diff | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for outcome, row in estimates.iterrows():
        if outcome in ("visit", "conversion"):
            treated_mean, control_mean = f"{row['treated_mean']:.2%}", f"{row['control_mean']:.2%}"
            difference = f"{row['diff'] * 100:+.2f}pp"
            interval = f"[{row['ci_low'] * 100:+.2f}pp, {row['ci_high'] * 100:+.2f}pp]"
        else:
            treated_mean, control_mean = (
                f"${row['treated_mean']:.3f}",
                f"${row['control_mean']:.3f}",
            )
            difference = f"${row['diff']:+.3f}"
            interval = f"[${row['ci_low']:.3f}, ${row['ci_high']:.3f}]"
        lines.append(f"| {outcome} | {treated_mean} | {control_mean} | {difference} | {interval} |")
    lines += [
        "",
        "## Covariate balance",
        "",
        "Standardised difference = (treated mean − control mean) / pooled SD. Values near zero",
        "are consistent with balance on that measured covariate. Balance cannot test for",
        "unmeasured differences or certify that randomization worked.",
        "",
        "| covariate | treated mean | control mean | standardised diff |",
        "|---|---|---|---|",
    ]
    defined_sdiff = balance["standardised_diff"][
        np.isfinite(balance["standardised_diff"].to_numpy(dtype=float))
    ]
    max_abs_sdiff = defined_sdiff.abs().max() if len(defined_sdiff) else np.nan
    for cov, row in balance.iterrows():
        value = row["standardised_diff"]
        flag = " ⚠" if not np.isfinite(value) or abs(value) >= 0.1 else ""
        rendered_value = (
            f"{value:+.4f}" if np.isfinite(value) else "undefined (zero within-group variance)"
        )
        lines.append(
            f"| {cov} | {row['treated_mean']:.4f} | {row['control_mean']:.4f} | "
            f"{rendered_value}{flag} |"
        )
    n_flagged = (defined_sdiff.abs() >= 0.1).sum()
    lines += [
        "",
        f"Largest defined absolute standardised difference: {max_abs_sdiff:.4f}. {n_flagged} of {len(defined_sdiff)} defined",
        "covariates (across numeric, binary and one row per categorical level) exceed the 0.1",
        "threshold. This is a descriptive balance check on measured covariates, not a test of the",
        "assignment mechanism. `reports/06_confounding_benchmark.md` shows estimates under",
        "constructed selection mechanisms; the full-RCT estimate is only a reference for those",
        "changed populations.",
        "",
        "**Why this comparison would be biased without randomisation.** If treatment had been",
        "chosen by a marketer rather than a coin flip, a nonzero standardised difference on",
        "`history` or `recency` above would mean the treated and control groups differed in ways",
        "that independently predict the outcome, and the diff-in-means table above would then be",
        "mixing the true effect of the email with the effect of already being a different kind of",
        "customer. `src/confounded.py` constructs specified selection mechanisms on the same rows;",
        "differences from the full-RCT estimate are descriptive because selection changes the target population.",
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

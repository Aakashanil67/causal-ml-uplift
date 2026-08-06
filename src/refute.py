"""Placebo treatment, random common cause, and data-subset refutation, run directly against
`src/dml_ate.py`'s own `dml_ate()` rather than through DoWhy's `refute_estimate()` wrapper.

That wrapper's econml integration passes effect-modifier columns straight to econml's numeric API
without dummy-encoding them (`dowhy/causal_estimators/econml.py:effect()`), so it fails on this
project's categorical covariates (`channel`, `zip_code`) with a `KeyError` before any refutation
logic runs. Implementing the three refuters by hand keeps the actual estimator this project reports
under test — the same `LinearDML` + `build_covariate_matrix()` pipeline every other report uses —
rather than switching to a different, less-integrated estimator just to satisfy the wrapper.

None of these three refuters test what `src/confounded.py`'s benchmark tests. They test whether the
*estimation procedure* is well-behaved (insensitive to label noise, insensitive to irrelevant
covariates, stable across subsamples) — not whether the *identification assumption* holds. An
estimate biased by an omitted confounder can still pass all three: this module runs them on
`confounded.py`'s already-known-biased estimate (variant 2, `newbie` withheld) to show that
directly, rather than asserting it.
"""

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, TREATMENT_COL
from src.data_loader import build_covariate_matrix
from src.dml_ate import _make_dml


def placebo_treatment_refuter(
    df: pd.DataFrame, outcome_col: str, drop_cols: list[str] | None = None, seed: int = RANDOM_SEED
) -> dict:
    """Replaces the real treatment column with a random permutation of itself, breaking any link
    to potential outcomes while keeping the marginal treated/control split identical. A real effect
    should vanish; what's left over is whatever the estimator finds when there is nothing to find.

    `drop_cols` must match whatever was withheld from the estimate under test — otherwise this
    silently refutes a *different*, better-specified model than the one it's meant to be checking
    (caught during development: running this on `src/confounded.py`'s newbie-withheld estimate
    without also dropping `newbie` here gave a refuter result for the wrong, unbiased model)."""
    rng = np.random.default_rng(seed)
    placebo = df.copy()
    placebo[TREATMENT_COL] = rng.permutation(placebo[TREATMENT_COL].to_numpy())
    X = build_covariate_matrix(placebo)
    if drop_cols:
        X = X.drop(columns=drop_cols)
    X = X.to_numpy()
    T = placebo[TREATMENT_COL].to_numpy()
    Y = placebo[outcome_col].to_numpy(dtype=float)
    est = _make_dml(seed)
    est.fit(Y, T, X=X)
    ate = float(est.ate(X))
    ci_low, ci_high = est.ate_interval(X)
    return {"ate": ate, "ci_low": float(ci_low), "ci_high": float(ci_high)}


def random_common_cause_refuter(
    df: pd.DataFrame, outcome_col: str, drop_cols: list[str] | None = None, seed: int = RANDOM_SEED
) -> dict:
    """Adds a column of pure random noise, independent of treatment, outcome and every real
    covariate, to the covariate matrix and refits. A well-behaved estimator should barely move,
    since there is nothing in the new column for it to (mis)use. See `placebo_treatment_refuter`
    for why `drop_cols` has to match the estimate under test."""
    rng = np.random.default_rng(seed)
    X = build_covariate_matrix(df)
    if drop_cols:
        X = X.drop(columns=drop_cols)
    X["_random_noise"] = rng.normal(size=len(df))
    T = df[TREATMENT_COL].to_numpy()
    Y = df[outcome_col].to_numpy(dtype=float)
    est = _make_dml(seed)
    est.fit(Y, T, X=X.to_numpy())
    ate = float(est.ate(X.to_numpy()))
    ci_low, ci_high = est.ate_interval(X.to_numpy())
    return {"ate": ate, "ci_low": float(ci_low), "ci_high": float(ci_high)}


def data_subset_refuter(
    df: pd.DataFrame,
    outcome_col: str,
    drop_cols: list[str] | None = None,
    n_runs: int = 5,
    subset_frac: float = 0.8,
    seed: int = RANDOM_SEED,
) -> list[float]:
    """Refits on several independent 80% subsamples. A stable estimator gives similar answers each
    time; wide swings mean the headline number depends heavily on which rows happened to be
    sampled, not on a real, generalisable effect. See `placebo_treatment_refuter` for why
    `drop_cols` has to match the estimate under test."""
    rng = np.random.default_rng(seed)
    ates = []
    for i in range(n_runs):
        idx = rng.choice(len(df), size=int(len(df) * subset_frac), replace=False)
        sub = df.iloc[idx]
        X = build_covariate_matrix(sub)
        if drop_cols:
            X = X.drop(columns=drop_cols)
        X = X.to_numpy()
        T = sub[TREATMENT_COL].to_numpy()
        Y = sub[outcome_col].to_numpy(dtype=float)
        est = _make_dml(seed + i)
        est.fit(Y, T, X=X)
        ates.append(float(est.ate(X)))
    return ates


def _ci_contains_zero(r: dict) -> bool:
    return r["ci_low"] <= 0 <= r["ci_high"]


def _close_to(value: float, target: float, tol: float = 0.01) -> bool:
    return abs(value - target) < tol


def _refuter_table_rows(
    results: dict, reference_ate: float, pass_word: str, fail_word: str
) -> list[str]:
    placebo_pass = _ci_contains_zero(results["placebo"])
    cause_pass = _close_to(results["random_cause"]["ate"], reference_ate)
    subset_pass = np.std(results["subset"]) < 0.01
    return [
        "| refuter | result | read |",
        "|---|---|---|",
        f"| placebo treatment | {results['placebo']['ate']:+.4f} "
        f"[{results['placebo']['ci_low']:.4f}, {results['placebo']['ci_high']:.4f}] | "
        f"{pass_word if placebo_pass else fail_word}, CI "
        f"{'contains' if placebo_pass else 'excludes'} 0 |",
        f"| random common cause | {results['random_cause']['ate']:+.4f} "
        f"[{results['random_cause']['ci_low']:.4f}, {results['random_cause']['ci_high']:.4f}] | "
        f"{pass_word if cause_pass else fail_word}, "
        f"{'close to' if cause_pass else 'moved materially from'} the original |",
        f"| data subset (5 runs) | mean {np.mean(results['subset']):+.4f}, "
        f"std {np.std(results['subset']):.4f} | "
        f"{pass_word if subset_pass else fail_word}, "
        f"{'stable' if subset_pass else 'unstable'} across subsamples |",
    ], (placebo_pass, cause_pass, subset_pass)


def write_refutation_report(
    original_ate: float,
    rct_results: dict,
    confounded_ate_known_biased: float,
    confounded_benchmark: float,
    confounded_results: dict,
    out_path,
) -> None:
    rct_rows, rct_passes = _refuter_table_rows(rct_results, original_ate, "PASS", "FAIL")
    conf_rows, conf_passes = _refuter_table_rows(
        confounded_results, confounded_ate_known_biased, '"PASSES"', "flags something"
    )
    rct_all_pass = all(rct_passes)
    conf_all_pass = all(conf_passes)

    lines = [
        "# Refutation tests: what passing does and does not prove",
        "",
        "Placebo treatment, random common cause, and data-subset refutation, run against this",
        "project's own `LinearDML` pipeline (`src/dml_ate.py`) rather than through DoWhy's",
        "`refute_estimate()` wrapper, which fails on this project's categorical covariates before",
        "any refutation logic runs (see `src/refute.py`'s docstring). Run twice: once on the real,",
        "unconfounded RCT, and once on `src/confounded.py`'s already-known-biased estimate, to show",
        "directly what these tests can and cannot catch, rather than asserting it.",
        "",
        "## On the real RCT",
        "",
        f"Original pooled DML ATE on `visit`: **{original_ate:+.4f}** (`reports/05_dml_ate.md`).",
        "",
        *rct_rows,
        "",
    ]
    if rct_all_pass:
        lines += [
            "All three pass, and here is exactly what that does and does not mean. The placebo",
            "test shows the estimator does not manufacture an effect out of nothing when there is",
            "genuinely nothing there. The random-common-cause test shows adding an irrelevant",
            "column does not move the number, which it shouldn't. The subset test shows the",
            "estimate does not depend on which 80% of customers happened to be sampled.",
        ]
    else:
        failed = [
            name
            for name, ok in zip(
                ["placebo", "random common cause", "subset"], rct_passes, strict=True
            )
            if not ok
        ]
        lines += [
            f"Not a clean sweep: {', '.join(failed)} did not pass on this run and would need",
            "investigating before trusting the headline estimate further, rather than being",
            "explained away.",
        ]
    lines += [
        "None of these three tests the one assumption this project's identification claim",
        "actually rests on: that treatment assignment had no unobserved cause in common with the",
        "outcome. That assumption is not refuted here; it holds by design, because Hillstrom is",
        "randomised (`reports/01_causal_question.md`, `reports/04_identification.md`), and no",
        "refutation test run on the data after the fact can substitute for that design fact.",
        "",
        "## On a known-biased estimate, to show what these tests miss",
        "",
        "`src/confounded.py`'s variant 2 (`newbie` withheld from the estimator) already showed a",
        f"DML estimate of **{confounded_ate_known_biased:+.4f}**, confidence interval excluding the",
        f"true benchmark of **{confounded_benchmark:+.4f}** (`reports/06_confounding_benchmark.md`).",
        "This is a real, demonstrated bias. Running the same three refuters against it, with the",
        "same `newbie` column withheld so this is testing the actual biased model and not a",
        "different, better-specified one:",
        "",
        *conf_rows,
        "",
    ]
    if conf_all_pass:
        lines += [
            "The quotation marks are deliberate. These refuters test properties an estimator can",
            "hold regardless of whether it is right, so a confounded, biased estimate sails",
            "through all three exactly as cleanly as a correct one does, which is what the table",
            "above shows happening.",
        ]
    else:
        passed = [
            name
            for name, ok in zip(
                ["placebo", "random common cause", "subset"], conf_passes, strict=True
            )
            if ok
        ]
        lines += [
            f"Not every refuter passed here ({', '.join(passed) if passed else 'none did'}), but",
            "that undersells the point rather than making it: a refuter catching this specific,",
            "engineered bias is not evidence these tests would catch a real, unknown confounder in",
            "an actual observational study, where nothing is engineered to be findable.",
        ]
    lines += [
        "",
        "The only thing in this project that actually tested the identification assumption",
        "itself was `reports/06_confounding_benchmark.md`'s comparison against a known",
        "experimental benchmark, because that is the one place a ground truth existed to check",
        "against. Most real observational studies do not have that luxury, which is the honest",
        "limitation worth sitting with: these three refutation tests are a standard part of the",
        "DoWhy workflow and worth running, but passing them is not evidence against omitted-",
        "variable bias, and no one should present it as such.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from src.config import REPORTS_DIR
    from src.confounded import dml_estimate, make_confounded_sample
    from src.data_loader import load_hillstrom
    from src.dml_ate import dml_ate

    df = load_hillstrom()
    print("running refuters on the real RCT...")
    original = dml_ate(df, TREATMENT_COL, "visit")
    rct_results = {
        "placebo": placebo_treatment_refuter(df, "visit"),
        "random_cause": random_common_cause_refuter(df, "visit"),
        "subset": data_subset_refuter(df, "visit"),
    }
    print("original ATE:", original["ate"])
    print(rct_results)

    print("running refuters on the known-biased confounded estimate...")
    confounded_sample = make_confounded_sample(df, ["newbie"], {"newbie": -1}, strength=1.5)
    confounded_biased = dml_estimate(confounded_sample, "visit", drop_cols=["newbie"])
    confounded_results = {
        "placebo": placebo_treatment_refuter(confounded_sample, "visit", drop_cols=["newbie"]),
        "random_cause": random_common_cause_refuter(
            confounded_sample, "visit", drop_cols=["newbie"]
        ),
        "subset": data_subset_refuter(confounded_sample, "visit", drop_cols=["newbie"]),
    }
    print("confounded (biased) ATE:", confounded_biased["ate"])
    print(confounded_results)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_refutation_report(
        original["ate"],
        rct_results,
        confounded_biased["ate"],
        original["ate"],
        confounded_results,
        REPORTS_DIR / "08_refutations.md",
    )


if __name__ == "__main__":
    main()

# causal-ml-uplift

[![CI](https://github.com/Aakashanil67/causal-ml-uplift/actions/workflows/ci.yml/badge.svg)](https://github.com/Aakashanil67/causal-ml-uplift/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)

Can causal ML find deployable treatment-effect heterogeneity, or only a reliable average effect? This project tests that distinction on Hillstrom's randomised email experiment.

**[Full report (PDF)](reports/causal_report.pdf)** · **[live simulator](https://causal-ml-uplift.streamlit.app/)** · Hillstrom e-mail experiment, 64,000 customers, three arms.

![causal graph](reports/figures/dag.png)

## The problem

"Customers who got the email visited more" is not the same claim as "the email caused them to visit," and most portfolio projects that use an email dataset don't distinguish the two: they report a correlation and call it insight. Hillstrom's data lets the distinction actually be tested rather than argued, because treatment was randomly assigned before the campaign ran (`reports/01_causal_question.md`), and the covariate balance check that would expose a violation of that design is run and reported, not skipped (`reports/02_naive_estimate.md`: largest standardised difference across 11 covariates is 0.0088, an order of magnitude under the usual 0.1 threshold).

The harder, more interesting problem the rest of this project is built around: does Double ML actually correct for confounding, or does it just agree with everything else because there was nothing to correct? `reports/06_confounding_benchmark.md` contains a constructed selection stress test, while the manifest also reports a semi-synthetic Monte Carlo benchmark with known effects.

## Results

Pooled `LinearDML` effects for assignment to either email rather than no email:

| outcome | DML ATE | 95% CI |
|---|---:|---:|
| visit | +6.01% | [5.48%, 6.55%] |
| conversion | +0.50% | [0.36%, 0.64%] |
| spend | +$0.613 | [$0.389, $0.837] |

The forest finds stable variation in predicted CATEs, but its ranking is weak. Normalized Qini is 0.0111 [-0.0114, 0.0331]; across five honest splits it ranges from 0.0111 to 0.0360. The interval includes zero, so the repo does not claim that individual uplift ordering is deployment-ready.

Three-action policies are evaluated with cross-fitted doubly robust scores:

| policy | expected visit rate | 95% CI |
|---|---:|---:|
| learned (DRPolicyForest) | 0.1823 | [0.1730, 0.1916] |
| email everyone (mens creative) | 0.1810 | [0.1719, 0.1902] |
| email everyone (womens creative) | 0.1520 | [0.1433, 0.1615] |
| purchase-history heuristic | 0.1848 | [0.1751, 0.1940] |
| email nobody | 0.1070 | [0.0994, 0.1151] |

Learned minus blanket mens is +0.0013 [-0.0014, +0.0040]. That is not a policy win. Blanket mens emailing is the simpler action supported by this experiment; personalised deployment needs new evidence.

## Methods ladder

| step | file | what it establishes |
|---|---|---|
| 1. Naive diff-in-means | `src/naive.py` | unbiased here because it's an RCT, and that claim is checked, not assumed |
| 2. Regression baseline | `src/regression_baseline.py` | logit AME / OLS, agrees with #1 as it should on an RCT |
| 3. DoWhy identification | `src/identify.py` | formal confirmation: empty backdoor set for every outcome |
| 4. LinearDML | `src/dml_ate.py` | the headline ATE, pooled and per-arm |
| 5. Confounding stress test | `src/confounded.py` | illustrates adjustment with constructed selection scenarios |
| 6. CausalForestDML | `src/cate.py` | profile-level conditional average effects, not just the average |
| 7. Qini / uplift | `src/uplift.py` | is the ranking real, and what does targeting it pay for |
| 8. Three-arm policy | `src/policy.py` | learned policy vs heuristic vs blanket-email baselines |
| 9. Refutation | `src/refute.py` | what passing placebo/random-cause/subset tests does and doesn't prove |

## How to run it

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m src.pipeline
.venv\Scripts\python.exe -m streamlit run app/simulator.py
```

On macOS or Linux, use `.venv/bin/python` in place of the Windows path. The pipeline regenerates
the manifest, serving artifacts, Markdown reports, figures, production model and PDF in dependency
order. `scripts/verify.ps1` runs the pinned-environment quality gate.

## Design decisions and trade-offs

- **Placebo, random-common-cause and data-subset refuters are implemented by hand** (`src/refute.py`), not via DoWhy's `refute_estimate()` wrapper, because that wrapper's econml integration passes categorical effect-modifier columns straight to econml's numeric `.effect()` without dummy-encoding them and fails with a real `KeyError` on this project's `channel`/`zip_code` columns. Writing the refuters directly against `src/dml_ate.py`'s own estimator also means the *actual* model this project reports stays under test, not a substitute.
- **The confounding-benchmark's "withheld confounder" variant confounds on `newbie`, not on `recency`/`history` with one column dropped.** The first version tried the latter and it didn't produce a clean failure: `recency`, still observed, was correlated enough with the selection mechanism to partially self-correct (reproduced in `notebooks/01_exploration.ipynb`). `newbie` is close to orthogonal to the other covariates and has a standalone effect on `visit` comparable in size to the treatment effect itself, so withholding it produces a genuine, unambiguous failure worth reporting.
- **Qini's cumulative gain curve and raw area are computed locally** (`src/uplift.py`). Normalized Qini uses `sklift.metrics.qini_auc_score` and is verified on fixed fixtures.
- **`CausalForestDML`/`DRPolicyForest` heterogeneity work targets `visit` only.** `conversion` and `spend` have 578 non-zero events total across 64,000 rows, nowhere near enough to split across a forest's leaves without the CATE surface being mostly noise (`reports/data_dictionary.md`).
- **The production model artifact (`models/causal_forest.joblib`, 34.47MB) is committed to git**, refit on the full 64,000 rows rather than the 70% split used for honest evaluation, since heterogeneity was already validated on held-out data before this refit happens. Under this project's own 50MB threshold for needing a distilled LightGBM surrogate instead, so the real forest ships.
- **`starlette` is pinned to `0.52.1`** in `requirements.txt`, one release behind Streamlit 1.61.0's own declared range. Starlette shipped a breaking 1.0 release days before this build, and its new ASGI GZip middleware crashes Streamlit's server with a real `TypeError` on the version pip resolves without the pin.

## Status

The average treatment effect is well identified by the randomised design. Two narrower
claims are not established:

- The uplift ranking is weak: normalized Qini 0.0111
  [-0.0114, 0.0331] includes zero.
- The learned policy does not beat blanket mens emailing on the held-out sample.

Reported spend among the top-30% diagnostic is $0.7816
[$0.3733, $1.2743], but Hillstrom supplies neither gross margin nor
uncapped spend. It is therefore a gross-spend sensitivity, not a profit estimate. The experiment
covers one US retailer in March 2008; its effect sizes do not transfer to a 2026 South African bank
or telecoms campaign.

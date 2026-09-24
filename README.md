# causal-ml-uplift

[![CI](https://github.com/Aakashanil67/causal-ml-uplift/actions/workflows/ci.yml/badge.svg)](https://github.com/Aakashanil67/causal-ml-uplift/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)

Can causal ML find deployable treatment-effect heterogeneity, or only a reliable average effect? This project tests that distinction on Hillstrom's randomised email experiment.

**[Full report (PDF)](reports/causal_report.pdf)** · **[live simulator](https://causal-ml-uplift.streamlit.app/)** · Hillstrom e-mail experiment, 64,000 customers, three arms.

![causal graph](reports/figures/dag.png)

## The problem

"Customers who got the email visited more" is not the same claim as "the email caused them to visit." Hillstrom's data lets the assignment effect be estimated because treatment was randomly assigned before the campaign ran (`reports/01_causal_question.md`); the covariate balance check is also computed and reported (`reports/02_naive_estimate.md`).

The harder problem is whether estimated treatment effects can support a useful targeting rule. `reports/06_confounding_benchmark.md` uses constructed Hillstrom selection as a descriptive diagnostic against an estimated experimental reference. The fully synthetic Monte Carlo benchmark supplies known targets for bias and coverage claims.

## Results

Pooled `LinearDML` effects for assignment to either email rather than no email:

| outcome | DML ATE | 95% CI |
|---|---:|---:|
| visit | +6.01pp | [+5.48pp, +6.55pp] |
| conversion | +0.50pp | [+0.36pp, +0.64pp] |
| spend | +$0.613 | [$0.389, $0.837] |

The pooled visit effect corresponds to about 60 additional visits per 1,000 customers assigned email. The forest produces varying conditional-effect predictions. Normalized Qini is 0.0111 [-0.0114, 0.0331]; across five honest splits it ranges from 0.0111 to 0.0360. The interval crosses zero, so this analysis does not establish a positive ranking advantage. The repository does not claim that individual uplift ordering is deployment-ready.

Three-action policies are evaluated with cross-fitted doubly robust scores:

| policy | expected visit rate | 95% CI |
|---|---:|---:|
| learned (DRPolicyForest) | 18.23% | [17.30%, 19.16%] |
| email everyone (mens creative) | 18.10% | [17.19%, 19.02%] |
| email everyone (womens creative) | 15.20% | [14.33%, 16.15%] |
| purchase-history heuristic | 18.48% | [17.51%, 19.40%] |
| email nobody | 10.70% | [9.94%, 11.51%] |

Learned minus blanket mens is +0.13pp [-0.14pp, +0.40pp]. Held-out evidence does not establish which policy has higher expected visit value. Blanket mens emailing is the simpler action for this experiment; validate any learned policy on another campaign before broader deployment.

## Methods ladder

| step | file | what it establishes |
|---|---|---|
| 1. Naive diff-in-means | `src/naive.py` | estimates the randomized assignment effect; measured balance is a descriptive check |
| 2. Regression baseline | `src/regression_baseline.py` | adjusted logit prediction contrasts / OLS with HC1 uncertainty |
| 3. DoWhy identification | `src/identify.py` | derives the empty backdoor set under the randomized-design graph |
| 4. LinearDML | `src/dml_ate.py` | pooled and creative-specific average effects with cross-fitted nuisances |
| 5. Selection diagnostics | `src/confounded.py` | compares selected-sample estimates with a full-RCT estimate for another population |
| 6. CausalForestDML | `src/cate.py` | exploratory profile-level conditional average effects |
| 7. Qini / uplift | `src/uplift.py` | evaluates whether the model ranking beats random targeting |
| 8. Three-arm policy | `src/policy.py` | evaluates learned, heuristic, and blanket actions |
| 9. Refutation | `src/refute.py` | diagnostics for responses to particular perturbations, not correctness certificates |

## How to run it

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app/simulator.py
```

On macOS or Linux, use `.venv/bin/python` in place of the Windows path. The committed manifest
and serving artifacts support the quick start. To rebuild all estimates, figures, reports, model
artifacts and PDF, run `.venv\Scripts\python.exe -m src.pipeline`. The optional 500-repetition
synthetic benchmark can be run separately with `-m src.simulation --repetitions 500`.
`scripts/verify.ps1` runs the pinned-environment quality gate.

## Design decisions and trade-offs

- **Placebo, random-common-cause and data-subset refuters are implemented by hand** (`src/refute.py`), not via DoWhy's `refute_estimate()` wrapper, because that wrapper's econml integration passes categorical effect-modifier columns straight to econml's numeric `.effect()` without dummy-encoding them and fails with a real `KeyError` on this project's `channel`/`zip_code` columns. Writing the refuters directly against `src/dml_ate.py`'s own estimator also means the *actual* model this project reports stays under test, not a substitute.
- **Constructed selection is reported as a descriptive diagnostic.** It uses an estimated full-RCT reference, while selection changes the population being analyzed. Differences between those estimates are not known bias for the selected population. Bias and coverage against known targets are evaluated in the fully synthetic Monte Carlo benchmark.
- **Qini's cumulative gain curve and raw area are computed locally** (`src/uplift.py`). Normalized Qini uses `sklift.metrics.qini_auc_score` and is verified on fixed fixtures.
- **`CausalForestDML`/`DRPolicyForest` heterogeneity work targets `visit` only.** `conversion` and `spend` have 578 non-zero events total across 64,000 rows, nowhere near enough to split across a forest's leaves without the CATE surface being mostly noise (`reports/data_dictionary.md`).
- **The production model artifact (`models/causal_forest.joblib`, 34.47MB) is committed to git**, refit on the full 64,000 rows rather than the 70% split used for honest evaluation, since heterogeneity was already validated on held-out data before this refit happens. Under this project's own 50MB threshold for needing a distilled LightGBM surrogate instead, so the real forest ships.
- **`starlette` is pinned to `0.52.1`** in `requirements.txt`, one release behind Streamlit 1.61.0's own declared range. Starlette shipped a breaking 1.0 release days before this build, and its new ASGI GZip middleware crashes Streamlit's server with a real `TypeError` on the version pip resolves without the pin.

## Status

The average treatment effect is well identified by the randomised design. Targeting
evidence from this held-out evaluation is summarised below:

- The normalized Qini ranking score is 0.0111
  [-0.0114, 0.0331]. The interval crosses zero, so this analysis does not establish a positive ranking advantage.
- Learned minus blanket mens emailing is +0.13pp
  [-0.14pp, +0.40pp]. Held-out evidence does not establish which policy has higher expected visit value. Blanket mens emailing is the simpler action for this experiment; validate any learned policy on another campaign before broader deployment.

Reported spend among the top-30% diagnostic is $0.7816
[$0.3733, $1.2743], but Hillstrom supplies neither gross margin nor
uncapped spend. It is therefore a gross-spend sensitivity, not a profit estimate. The experiment
covers one US retailer in March 2008; its effect sizes do not transfer to a 2026 South African bank
or telecoms campaign.

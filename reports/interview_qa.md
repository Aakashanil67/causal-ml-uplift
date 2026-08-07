# Interview Q&A

Fifteen questions an interviewer would actually ask about this project, answered from what was built, not from a textbook. Each answer names the specific file or number it comes from, so it can be checked, not just recited.

## 1. What's the difference between ATE, ATT, and CATE, and which one did you estimate?

ATE (average treatment effect) is the average effect across the whole population if everyone were treated versus if no one were. ATT (average treatment effect on the treated) is the same average but restricted to people who actually got treated. On a randomised experiment like Hillstrom's, treated and control are drawn from the same population by construction, so ATE and ATT coincide, which is part of why the naive diff-in-means, the regression AME, and the DML estimate all agree in `reports/05_dml_ate.md` (+6.01pp on visit). CATE (conditional average treatment effect) is the effect for a specific customer given their covariates, which is what `src/cate.py`'s `CausalForestDML` estimates: the mean CATE (+0.0588) is close to the pooled ATE, but the individual values range from about −0.02 to +0.13, and that spread is the entire point of estimating it.

## 2. What's the difference between a confounder, a mediator, and a collider, and why does it matter which one you control for?

A confounder causes both treatment and outcome; not controlling for it biases the estimate, which is exactly what `src/confounded.py` demonstrates on purpose. A mediator sits on the causal path between treatment and outcome (`recency`/`history` don't mediate anything here, but `visit` plausibly mediates `treatment → conversion`); controlling for a mediator blocks part of the real effect and understates it, which is why `reports/01_causal_question.md` explicitly does not condition on `visit` when estimating the effect on `conversion`. A collider is caused by both treatment and outcome (or by two things you're conditioning on); controlling for a collider *induces* a spurious association where none existed. This project's DAG has no colliders by design, but a subset-refuter or an analyst filtering the data on a post-treatment variable would be creating one without realising it.

## 3. Why does cross-fitting matter for DML, and what breaks without it?

`src/dml_ate.py`'s `LinearDML` splits the data into folds and predicts each fold's nuisance functions (the outcome and treatment models) using the *other* folds, never fitting a nuisance model and evaluating its own residual on the same rows. Without that split, a flexible nuisance model (LightGBM here) can overfit the training data closely enough that its residuals underestimate the true noise, which biases the treatment-effect estimate and makes the confidence intervals too narrow. On an RCT this project's estimates would likely still land close to correct without cross-fitting, since there's little for the nuisance models to overfit to in the first place; the real stakes show up in `src/confounded.py`'s benchmark, where a flexible, overfit nuisance model could plausibly "explain away" some of the deliberate confounding by accident, making DML look like it's correcting for it when it's actually just fitting noise.

## 4. Your confounding benchmark shows DML "passing." What would make it fail, and did you actually build a case where it fails?

Yes: `reports/06_confounding_benchmark.md`'s variant 2. DML corrects confounding by adjusting for the covariates it's given; if the true confounder is never in that covariate set, there's nothing for it to adjust for. Withholding `newbie` (real, standalone effect on `visit` of −6.45pp, comparable in size to the treatment effect) from every estimator gives a DML estimate of +9.49pp with a 95% CI of [8.72, 10.26], nowhere near the true +6.01pp benchmark. The confidence interval doesn't just miss the point estimate, it excludes it entirely. This is the answer to "doesn't DML fix confounding?": it fixes confounding it can see.

## 5. What does passing a refutation test actually prove, and what doesn't it prove?

`reports/08_refutations.md` runs placebo treatment, random common cause, and data-subset refutation against the real RCT, and all three pass: shuffling treatment kills the effect, adding random noise doesn't move the estimate, and refits on different subsamples agree. That proves the *estimation procedure* is well-behaved: it doesn't manufacture effects from nothing, isn't thrown off by irrelevant covariates, and isn't overfit to one particular sample. It does not prove the *identification assumption* holds, and this project demonstrates that gap directly rather than asserting it: running the same three refuters on the already-known-biased confounded estimate from question 4, all three still "pass," because none of them test for an omitted confounder. The only thing that actually tested identification was comparing against a known experimental benchmark, which most real observational studies don't have.

## 6. Why not just use DoWhy's built-in `refute_estimate()` instead of writing your own?

Tried it first. DoWhy's econml integration passes effect-modifier columns straight to econml's numeric `.effect()` method without dummy-encoding them, and this project's covariates include two categoricals (`channel`, `zip_code`). It fails with a real `KeyError` before any refutation logic runs. Rather than restructure the whole project's covariate encoding to fit the wrapper, `src/refute.py` implements the three refuters directly against `src/dml_ate.py`'s own `_make_dml()`, which has the added benefit that the *actual* estimator this project reports is what gets refuted, not a simplified stand-in.

## 7. Your learned three-arm policy doesn't beat the simplest baseline. Isn't that a failure?

No, and the report says why rather than hiding the result: `reports/07_uplift_policy.md` shows the learned `DRPolicyForest` policy scoring 0.1831 against 0.1851 for "email everyone the mens creative," with heavily overlapping confidence intervals. The mens creative has a positive effect for almost every segment in this data, including the weakest one (mens-only-purchase customers, +0.043 CATE, still clearly positive). When the simple action already works almost everywhere, there's little headroom for a smarter policy to improve on it. The learned policy still does real work: it correctly identifies that the "obvious" purchase-history-matching heuristic is worse than blanket mens-emailing, a genuinely counterintuitive finding a marketer wouldn't guess without the model.

## 8. Why is your heterogeneity analysis only on `visit` and not `conversion` or `spend`?

Power. `conversion` and `spend` have 578 non-zero events total across 64,000 rows (`reports/data_dictionary.md`). Splitting that across a causal forest's leaves, with `min_samples_leaf=50`, would produce a CATE surface that's mostly sampling noise dressed up as signal. `visit` has 9,394 events, enough to support the forest's splits. This is stated as a scope decision, not glossed over.

## 9. `spend` is capped at $499. How does that affect your results?

Confirmed directly, not assumed: 12 customers show `spend` at exactly $499.00, with a real gap to the next-highest value ($482.31), which rules out coincidence. Every spend-based number in this project (the OLS coefficient, the DML ATE, the break-even revenue estimate) is a **lower bound** on the true effect, since the true spend of those 12 customers is unknown but at least $499. `reports/causal_report.md`'s limitations section states this rather than treating $499 as a real ceiling on what these customers spent.

## 10. Why does your revenue-based targeting number have such a wide confidence interval?

Sample size. The break-even calculation in `reports/07_uplift_policy.md` uses `spend` as the outcome, and only 578 of 64,000 customers have non-zero spend. Bootstrapping the top-30% targeting fraction gives a point estimate of $0.1196 per email against an assumed $0.10 cost, but the 95% CI is [−$0.48, $0.63], crossing zero. The report leads with the visit-based numbers, which are precise, and states plainly that the revenue claim can't be statistically distinguished from a loss at this sample size, rather than reporting only the point estimate.

## 11. What's the Qini coefficient, and why compute it by hand instead of using an existing library?

The Qini gain at a targeting fraction *k* is the sum of outcomes among the top-*k* treated customers minus the top-*k* control customers' outcome sum, rescaled to the treated group's size in that slice (Radcliffe & Surry, 2011). The Qini coefficient is the area between that curve and the random-targeting diagonal; this project's value is 47.10, positive, meaning the CATE ranking beats emailing customers in a random order. It's computed directly in `src/uplift.py` rather than via `scikit-uplift` so every number can be defended line by line rather than trusted as a library black box, and cross-checked against `sklift.metrics.qini_auc_score` in one test for sign agreement (different normalisations mean the magnitudes aren't directly comparable).

## 12. How do you know your covariate balance table isn't just something you expected to see on an RCT?

It's computed on the real downloaded file, not simulated: `reports/02_naive_estimate.md`'s largest standardised difference across 11 covariates is 0.0088. The test suite (`tests/test_naive.py`) also checks the balance function can detect a *real* imbalance, not just report zeros: it constructs a synthetic dataset with a deliberately imbalanced `recency` distribution and confirms the standardised difference comes back large, before trusting the same function's small numbers on the real Hillstrom data.

## 13. Would this pipeline work on an observational dataset without a randomised experiment?

Most of it, with one piece missing. The naive estimate, regression baseline, DoWhy identification, DML, CausalForestDML, and Qini/policy work all run the same way on observational data; the difference is that DoWhy's identification step would return a non-empty backdoor adjustment set (assuming one exists and is measured), and the naive estimate would need those covariates controlled for. What doesn't transfer directly is `reports/06_confounding_benchmark.md`'s validation: it works because this project has a real experimental benchmark to check DML against. On genuinely observational data there's no such ground truth, which is precisely the limitation `reports/causal_report.md` states about this project's own confounding benchmark: it validates DML against confounding *this project engineered and can name*, not against the possibility of an unmeasured confounder in the wild.

## 14. What's the single most likely thing an econometrics professor would push back on?

The DAG treats `visit`, `conversion`, and `spend` as three parallel outcomes of treatment rather than the real mediation chain `treatment → visit → conversion → spend`, which the data itself supports (conversion is a strict subset of visit, checked and enforced in `src/data_loader.py`). This project estimates the *total* effect of treatment on each outcome, which is standard and matches what DML and CausalForestDML actually compute, but it doesn't decompose how much of the spend effect runs through visiting first. A mediation analysis would need its own identification argument, since conditioning on a post-treatment mediator to study a downstream outcome can introduce collider bias. This is named directly in `reports/causal_report.md`'s limitations section rather than left for a reviewer to catch.

## 15. If you had another month, what would you build next?

A real spend-CATE model once more non-zero-spend data accumulates, rather than reusing the visit-based ranking as a proxy for revenue targeting. A second confounding-benchmark variant using a continuous, non-orthogonal confounder to see how gracefully DML degrades as the omitted variable becomes more correlated with what's already observed, rather than the current all-or-nothing (fully observed vs fully withheld) comparison. And a mediation analysis of the `visit → conversion → spend` chain, since the data supports it and the current project explicitly scopes it out rather than attempting it.

# Interview Q&A

Fifteen questions an interviewer would actually ask about this project, answered from what was built, not from a textbook. Each answer names the specific file or number it comes from, so it can be checked, not just recited.

## 1. What's the difference between ATE, ATT, and CATE, and which one did you estimate?

ATE averages the effect over the target population; ATT averages it over treated customers; CATE conditions on a covariate profile. Random assignment means treated and control customers represent the same population in expectation, not that finite-sample ATE and ATT are algebraically identical. This project reports a pooled DML ATE of +0.0601 on `visit`. The held-out forest's mean profile CATE is +0.0603, with estimates from -0.0073 to +0.1292.

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

It is the decision result. Cross-fitted doubly robust evaluation gives the learned policy 0.1823 and blanket mens emailing 0.1810. Their paired difference is +0.0013 [-0.0014, +0.0040]. The interval includes zero. I would deploy the simpler blanket action on this evidence and treat policy learning as a design for the next experiment, not as a production win.

## 8. Why is your heterogeneity analysis only on `visit` and not `conversion` or `spend`?

Power. `conversion` and `spend` have 578 non-zero events total across 64,000 rows (`reports/data_dictionary.md`). Splitting that across a causal forest's leaves, with `min_samples_leaf=50`, would produce a CATE surface that's mostly sampling noise dressed up as signal. `visit` has 9,394 events, enough to support the forest's splits. This is stated as a scope decision, not glossed over.

## 9. `spend` is capped at $499. How does that affect your results?

Twelve customers have reported spend exactly $499.00, with a gap to the next-highest value ($482.31), which makes top-coding plausible. Spend estimates are effects on the reported capped outcome. The uncapped-spend effect is not identified without a censoring model, so calling it a lower bound would add an unsupported assumption.

## 10. Why does your revenue-based targeting number have such a wide confidence interval?

The corrected top-30% estimate is $0.7816 in reported gross spend per targeted customer [0.3733, 1.2743]. Its interval is positive, but profitability is still unidentified: the data contain no gross margin, and twelve observations sit at the $499 maximum. The report therefore presents a gross-spend sensitivity rather than comparing it directly with email cost.

## 11. What's the Qini coefficient, and why compute it by hand instead of using an existing library?

Raw Qini is the area between cumulative incremental gain under the model ranking and random targeting; here it is 17.84, which depends on sample size. The normalized score is 0.0111 [-0.0114, 0.0331]. Five honest splits range from 0.0111 to 0.0360. The primary interval includes zero, so a positive raw area is not enough to claim a deployment-quality ranking.

## 12. How do you know your covariate balance table isn't just something you expected to see on an RCT?

It's computed on the real downloaded file, not simulated: `reports/02_naive_estimate.md`'s largest standardised difference across 11 covariates is 0.0088. The test suite (`tests/test_naive.py`) also checks the balance function can detect a *real* imbalance, not just report zeros: it constructs a synthetic dataset with a deliberately imbalanced `recency` distribution and confirms the standardised difference comes back large, before trusting the same function's small numbers on the real Hillstrom data.

## 13. Would this pipeline work on an observational dataset without a randomised experiment?

Most of it, with one piece missing. The naive estimate, regression baseline, DoWhy identification, DML, CausalForestDML, and Qini/policy work all run the same way on observational data; the difference is that DoWhy's identification step would return a non-empty backdoor adjustment set (assuming one exists and is measured), and the naive estimate would need those covariates controlled for. What doesn't transfer directly is `reports/06_confounding_benchmark.md`'s validation: it works because this project has a real experimental benchmark to check DML against. On genuinely observational data there's no such ground truth, which is precisely the limitation `reports/causal_report.md` states about this project's own confounding benchmark: it validates DML against confounding *this project engineered and can name*, not against the possibility of an unmeasured confounder in the wild.

## 14. What's the single most likely thing an econometrics professor would push back on?

The DAG treats `visit`, `conversion`, and `spend` as three parallel outcomes of treatment rather than the real mediation chain `treatment → visit → conversion → spend`, which the data itself supports (conversion is a strict subset of visit, checked and enforced in `src/data_loader.py`). This project estimates the *total* effect of treatment on each outcome, which is standard and matches what DML and CausalForestDML actually compute, but it doesn't decompose how much of the spend effect runs through visiting first. A mediation analysis would need its own identification argument, since conditioning on a post-treatment mediator to study a downstream outcome can introduce collider bias. This is named directly in `reports/causal_report.md`'s limitations section rather than left for a reviewer to catch.

## 15. If you had another month, what would you build next?

I would run a new campaign designed around action-level trade-offs. Both creatives help almost every segment here, which leaves little policy headroom. I would also add a semi-synthetic Monte Carlo benchmark with known heterogeneous effects, bias and coverage across confounding strengths, then revisit spend CATEs only after collecting enough non-zero purchase outcomes to support honest leaves.

# Interview Q&A

Fifteen questions an interviewer would actually ask about this project, answered from what was built, not from a textbook. Each answer names the specific file or number it comes from, so it can be checked, not just recited.

## 1. What's the difference between ATE, ATT, and CATE, and which one did you estimate?

ATE averages the effect over the target population; ATT averages it over treated customers; CATE conditions on a covariate profile. Random assignment makes treated and control customers represent the same population in expectation, though finite-sample ATE and ATT need not be algebraically identical. The pooled DML ATE on visit is +6.01pp [+5.48pp, +6.55pp]. The held-out forest's mean profile CATE is +6.03pp, ranging from -0.73pp to +12.92pp. Those are model-based conditional averages, not individual treatment effects.

## 2. What's the difference between a confounder, a mediator, and a collider, and why does it matter which one you control for?

A confounder causes both treatment and outcome. `src/confounded.py` constructs selection using measured covariates, but its comparison with the full-RCT estimate is diagnostic because the selected population differs. A mediator lies on a causal path; `visit` could mediate some effect on `conversion`, but the outcome sequence alone does not establish the full mediation structure. Conditioning on a post-treatment mediator changes the estimand and can introduce bias; collider bias depends on the causal structure and should not be inferred automatically.

## 3. Why does cross-fitting matter for DML, and what breaks without it?

`src/dml_ate.py`'s `LinearDML` predicts each fold's nuisance functions using the other folds. That separation limits the bias and overly narrow uncertainty that can result when flexible nuisance models fit and evaluate residuals on the same rows. Randomisation breaks the treatment-confounder link; it does not prevent outcome-model overfitting. Cross-fitting remains useful here, and it does not establish that LinearDML's linear final effect stage is correctly specified.

## 4. Your confounding benchmark shows DML "passing." What would make it fail, and did you actually build a case where it fails?

Variant 2 withholds `newbie`, so the estimator lacks a variable used to construct selection. Its estimate differs from the full-RCT estimate, but that alone does not establish bias: selection changes the covariate distribution, and heterogeneous effects can change the selected-population ATE. I use the fully synthetic benchmark, where the probabilities and sample-average target are known, for claims about bias and coverage. Adjustment can only address measured confounding represented in the fitted model, subject to its assumptions and specification.

## 5. What does passing a refutation test actually prove, and what doesn't it prove?

`reports/08_refutations.md` records the results of placebo treatment, random common cause, and data-subset checks on this RCT and a constructed selected sample. Passing means only that each particular check met its stated criterion. It does not prove identification, rule out omitted confounding, or certify general estimator behavior. The selected sample targets a different covariate distribution from the RCT, so its difference from the experimental estimate is not a known-bias demonstration. The synthetic benchmark is where bias is checked against known truth.

## 6. Why not just use DoWhy's built-in `refute_estimate()` instead of writing your own?

Tried it first. DoWhy's econml integration passes effect-modifier columns straight to econml's numeric `.effect()` method without dummy-encoding them, and this project's covariates include two categoricals (`channel`, `zip_code`). It fails with a real `KeyError` before any refutation logic runs. Rather than restructure the whole project's covariate encoding to fit the wrapper, `src/refute.py` implements the three refuters directly against `src/dml_ate.py`'s own `_make_dml()`, which has the added benefit that the *actual* estimator this project reports is what gets refuted, not a simplified stand-in.

## 7. Your learned three-arm policy doesn't beat the simplest baseline. Isn't that a failure?

Cross-fitted doubly robust evaluation gives the learned policy 18.23% and blanket mens emailing 18.10%. Their paired difference is +0.13pp [-0.14pp, +0.40pp]. Held-out evidence does not establish which policy has higher expected visit value. Blanket mens emailing is the simpler action for this experiment; validate any learned policy on another campaign before broader deployment.

## 8. Why is your heterogeneity analysis only on `visit` and not `conversion` or `spend`?

Power. `conversion` and `spend` have 578 non-zero events total across 64,000 rows (`reports/data_dictionary.md`). Splitting that across a causal forest's leaves, with `min_samples_leaf=50`, would produce a CATE surface that's mostly sampling noise dressed up as signal. `visit` has 9,394 events, enough to support the forest's splits. This is stated as a scope decision, not glossed over.

## 9. The maximum reported `spend` is $499. Does that establish a cap?

No. Twelve observations at $499.00 and a gap below are consistent with top-coding, but they do not establish a cap or its mechanism. Different purchase histories do not rule out the same priced product, and the source does not confirm a privacy or outlier-suppression motive. Estimates describe reported spend; the uncapped-spend effect needs more information or a justified censoring model.

## 10. Why does your revenue-based targeting number have such a wide confidence interval?

The top-30% pooled-email diagnostic is +6.03pp [+4.06pp, +7.95pp] in visit-rate difference per emailed customer. Reported spend is a separate policy sensitivity: learned minus no email is $+0.726 [$+0.238, $+1.255], and learned minus blanket mens is $-0.056 [$-0.157, $+0.022]. Reported spend may be top-coded at $499; gross margin and uncapped spend are unavailable, so these are not profit estimates.

## 11. What's the Qini coefficient, and why compute it by hand instead of using an existing library?

Raw Qini is the area between cumulative incremental gain under the model ranking and random targeting; here it is 17.84, which depends on sample size. The normalized score is 0.0111 [-0.0114, 0.0331]. Five honest sample splits range from 0.0111 to 0.0360; these are sensitivity evidence, not independent replications. The interval is conditional on the fitted ranking. The interval crosses zero, so this analysis does not establish a positive ranking advantage. This conditional interval does not itself validate transport to a new campaign.

## 12. How do you know your covariate balance table isn't just something you expected to see on an RCT?

It's computed on the real downloaded file, not simulated: `reports/02_naive_estimate.md`'s largest standardised difference across 11 covariates is 0.0088. The test suite (`tests/test_naive.py`) also checks the balance function can detect a *real* imbalance, not just report zeros: it constructs a synthetic dataset with a deliberately imbalanced `recency` distribution and confirms the standardised difference comes back large, before trusting the same function's small numbers on the real Hillstrom data.

## 13. Would this pipeline work on an observational dataset without a randomised experiment?

It would need several changes before that use. Propensities must be estimated from the observational assignment process, overlap and support checked, and policy/ranking metrics evaluated with appropriate adjusted estimators and a justified target population. The one-third randomisation probabilities and raw subgroup treatment-control differences used here do not carry over unchanged. Without an experiment or known synthetic target, observational data alone cannot establish bias or coverage against ground truth.

## 14. What's the single most likely thing an econometrics professor would push back on?

The DAG uses parallel outcome arrows, while purchases in the data occur after visits. That sequence makes mediation plausible but does not establish the full causal structure. This project estimates total effects on each outcome and does not decompose mediation. A mediation analysis would need its own identification argument; conditioning on a post-treatment visit changes the estimand and may introduce bias depending on the causal graph. Collider bias is not automatic from the observed outcome sequence.

## 15. If you had another month, what would you build next?

I would run a new campaign designed around action-level trade-offs and validate the policy on another campaign. Positive average effects for both creatives would not rule out personalization: relative effects could cross across customers. This analysis shows exploratory segment differences. The interval crosses zero, so this analysis does not establish a positive ranking advantage. Held-out evidence does not establish which policy has higher expected visit value. Blanket mens emailing is the simpler action for this experiment; validate any learned policy on another campaign before broader deployment. I would strengthen the existing fully synthetic known-truth benchmark across confounding strengths and test transport across campaigns before revisiting sparse-outcome spend CATEs.

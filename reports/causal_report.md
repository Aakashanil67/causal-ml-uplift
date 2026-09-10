# Causal ML: what actually works?

## Can causal ML find deployable treatment-effect heterogeneity, or only a reliable average effect?

Hillstrom's 64,000-customer randomised experiment identifies a clear average result: assignment to either email raises the two-week visit rate by 6.01% [5.48%, 6.55%]. The individual-targeting result is weaker. Normalized Qini is 0.0111 [-0.0114, 0.0331], and the learned policy's advantage over blanket mens emailing is +0.0013 [-0.0014, +0.0040]. Neither interval supports personalised deployment. The decision supported by this campaign is blanket mens emailing, while the causal-ML work diagnoses what a better follow-up experiment must change.

## 1. The causal question and why identification is clean here

The graph this report works from (`reports/01_causal_question.md`, `reports/figures/dag.png`) has exactly one substantive property: nothing points into `treatment`. Seven covariates plausibly cause the outcomes (`recency`, `history`, `mens`, `womens`, `newbie`, `zip_code`, `channel`), but none of them caused which arm a customer landed in, because assignment was randomised. That is not a modelling assumption; it is a design fact about how Hillstrom ran the experiment, and Section 2 checks it against the data rather than taking it on faith. DoWhy's `identify_effect()` confirms the formal consequence: the backdoor adjustment set is empty for all three outcomes (`reports/04_identification.md`), so `E[Y|T=1] − E[Y|T=0]` is already the correct nonparametric estimand, no covariate adjustment required.

![causal graph: no arrows into treatment](figures/dag.png)

## 2. The naive estimate, and the check that it is allowed to be naive

Standardised covariate differences between treated and control, across 11 covariates (numeric, binary, and one row per categorical level), max out at 0.0088 (`reports/02_naive_estimate.md`), an order of magnitude under the usual 0.1 "balanced" threshold. That is the empirical version of Section 1's graph-level claim. On that basis, the naive difference-in-means is a legitimate estimate, not a strawman: +6.09pp on visit (95% CI [5.54, 6.63]), +0.50pp on conversion, +$0.597 on spend.

## 3. The methods ladder: naive, regression, and Double ML agree

| method | visit | conversion | spend |
|---|---|---|---|
| naive diff-in-means | +6.09pp [5.54, 6.63] | +0.50pp | +$0.597 |
| logit AME / OLS, covariate-adjusted | +6.49pp [5.87, 7.10] | +0.56pp [0.38, 0.75] | +$0.596 [0.375, 0.816] |
| LinearDML (LightGBM nuisances) | +6.01pp [5.48, 6.55] | +0.50pp [0.36, 0.64] | +$0.613 [0.389, 0.837] |

All three land within a few hundredths of a percentage point of each other on every outcome (`reports/03_regression_baseline.md`, `reports/05_dml_ate.md`). That agreement is not a coincidence and not a demonstration of DML's power. On a randomised experiment there is nothing for covariate adjustment or partialling-out to correct, so every unbiased method should converge on roughly the same number, and here they do. DML earns its place in the ladder for a different reason: the same partialling-out machinery is what Section 5 reuses to estimate an effect that varies by customer, which none of the simpler methods above can do at all.

Splitting the pooled treatment by creative rather than averaging it away shows the two emails are not interchangeable: mens email lifts visit by +7.47pp (DML, [6.82, 8.12]), womens by +4.49pp ([3.87, 5.12]), a real, non-overlapping gap that motivates Sections 6 and 7.

## 4. Constructed confounding stress test

Every method above agreed because there was nothing to disagree about. `reports/06_confounding_benchmark.md` manufactures real confounding from the same real rows, retaining customers by a probability that depends on both their arm and their covariates and never fabricating an outcome, then checks whether each estimator can see through it, against the actual DML benchmark of +6.01pp.

**Confound on `recency`/`history`, both observed by every estimator:** naive diff-in-means overstates the effect by two-thirds (+10.03pp), OLS and DML both correct back to +5.93pp and +5.99pp respectively, the latter's CI [4.79, 7.19] comfortably containing the true benchmark. DoWhy's identification on the equivalent confounded graph confirms this is not a coincidence: adding `recency`/`history → treatment` edges changes the backdoor set from empty to exactly `{recency, history}`, precisely what both estimators condition on.

**Confound on `newbie`, withheld from every estimator:** naive +9.56pp, OLS +10.09pp, DML +9.49pp with a CI of [8.72, 10.26], nowhere near the true benchmark. DML does not fix confounding it cannot see. This is the honest version of "does DML work" that this report can actually stand behind: it corrects for what it is given, and no method here, or anywhere, corrects for what it was never told about.

## 5. Heterogeneity: who the email actually helps

The causal forest estimates profile-level conditional average effects on `visit`. I did not treat its segments as evidence on their own. A post-hoc held-out HC1-robust audit estimates +0.0403 for mens-only, +0.0617 for womens-only and +0.1582 for both-category customers. The womens-only minus mens-only contrast has Holm p=0.113; the joint Wald p-value is 8.55e-07. This is exploratory evidence of segment differences, not a person-level causal effect.

## 6. Uplift ranking and targeting economics

The primary split's normalized Qini is 0.0111 [-0.0114, 0.0331]. Five repeated sample splits are all positive in this run, but range from 0.0111 to 0.0360; the primary fixed-model conditional bootstrap interval still includes zero. That is weak ranking evidence. The top-30% pooled-mixture diagnostic estimates +0.0603 visits per emailed customer, but it does not tell a marketer which creative to send. The three-action analysis answers that separate question.

## 7. Three actions, not two: an honest result

Cross-fitted doubly robust evaluation gives the learned policy a visit-rate value of 0.1823, against 0.1810 for blanket mens emailing. The paired difference is +0.0013 [-0.0014, +0.0040]. Personalisation does not earn its operational complexity here. The evidence-supported decision is the simpler one: use the stronger mens creative broadly, then test a new campaign designed to create genuine action-level trade-offs.

## 8. Refutation tests, and what they do not prove

DoWhy's own `refute_estimate()` wrapper fails outright on this project's categorical covariates (a real `KeyError`, confirmed before abandoning the approach), so placebo treatment, random common cause, and data-subset refutation were implemented by hand against the actual `LinearDML` pipeline this report uses, not a substitute (`reports/08_refutations.md`). On the real RCT, all three pass cleanly: shuffling treatment kills the effect (+0.0003, CI containing 0), adding pure random noise barely moves the estimate (+0.0605 vs +0.0601), and five refits on 80% subsamples stay tight (std 0.0015).

The more useful result comes from running the same three refuters on Section 4's already-known-biased confounded estimate (+0.0949, confirmed wrong). All three "pass" there too: placebo still shows a CI containing zero, random common cause still barely moves the (wrong) number, the subset refits are still stable, just stably wrong. None of these three standard refutation tests can detect omitted-variable bias; they test whether the estimation procedure is well-behaved, not whether the identification assumption holds. The only thing in this project that actually tested the identification assumption was Section 4's comparison against a known experimental benchmark, because that is the one place a ground truth existed to check against. Most real observational studies do not have that luxury.

## 9. Limitations, stated rather than buried

- **One campaign, one retailer, US, March 2008.** Nothing here says these exact numbers generalise to another retailer, another country, or another decade. What generalises is the method (the identify-estimate-refute-validate loop this report runs), not the +6pp.
- **`spend` may be top-coded at exactly $499.00** for 12 customers (`reports/data_dictionary.md`). Estimates are therefore effects on the reported capped outcome; the uncapped-spend effect is not identified without a censoring model and is not automatically a lower bound.
- **The DAG treats `visit`/`conversion`/`spend` as three parallel outcomes**, not the real mediation chain (`treatment → visit → conversion → spend`, which the data supports: conversion is a strict subset of visit). This report estimates the total effect of treatment on each outcome, which is standard and matches what DML/CausalForestDML compute; it does not decompose how much of the spend effect runs through visiting, which would need its own identification argument.
- **The confounding validation in Section 4 tests selection on named, engineered covariates.** It shows DML corrects for confounding it can see and fails honestly on confounding it cannot. It says nothing about whether an unobserved confounder exists in the real, unconfounded Hillstrom data; the entire reason this project uses a randomised experiment is that it does not need to answer that question.
- **Reported gross spend is not profit.** Section 6 does not infer break-even without a margin assumption, and possible spend top-coding leaves the uncapped-spend effect unidentified.
- **The learned three-arm policy's advantage over the simplest baseline is not statistically established** on this eval set (Section 7). A larger eval sample, or a setting with real segment-level harm from the default action, would be needed to see policy learning's value more clearly than this data can show it.

## 10. What would and would not transfer to a South African retention campaign

The mechanics transfer directly: a bank or telco running a retention-offer RCT could rerun this exact pipeline (naive check, DoWhy identification, DML ATE, CausalForestDML heterogeneity, Qini-based targeting, a constructed confounding benchmark if a historical observational dataset exists alongside the RCT, and the same three refutation tests with the same honest caveat about what they do not prove). The specific numbers would not transfer: US speciality-retail purchase behaviour in 2008 says nothing about SA retail-bank or telco customers in 2026, and a resend-the-stronger-offer-to-everyone baseline being hard to beat is a property of *this* campaign's effect distribution (positive almost everywhere), not a general law. A SA campaign with genuine segment-level harm (an offer that actively annoys a segment into churning, for instance) is exactly the setting where Section 7's targeting would show a measurable advantage over a blanket policy, rather than the theoretical one it shows here. Finding out which case a real SA campaign is in is itself the deliverable a decision-science team would be hired to produce.

## Closing

Using Double Machine Learning to estimate heterogeneous treatment effects, and validating that estimation against a constructed benchmark before trusting it on a question without one, is the general version of what Section 4 does with `recency`/`history`/`newbie`. That is the shape of a thesis-length question worth asking about a real emerging-market intervention: not "does the policy work on average" but "for whom does it work, how would we know if our method could tell the difference, and what happens to the recommended policy once the default action itself is worth being smarter than."


## References

- Hillstrom, K. (2008), [MineThatData E-Mail Analytics Challenge](https://blog.minethatdata.com/2008/05/best-answer-e-mail-analytics-challenge.html).
- Radcliffe, N. J. and Surry, P. D. (2011), [Real-World Uplift Modelling with Significance-Based Uplift Trees](https://stochasticsolutions.com/pdf/sig-based-up-trees.pdf).
- Sharma, A. and Kiciman, E. (2020), [DoWhy: An End-to-End Library for Causal Inference](https://arxiv.org/abs/2011.04216).
- Microsoft Research, [EconML 0.16 documentation](https://econml.azurewebsites.net/), including `CausalForestDML` and `DRPolicyForest`.
---

**Repository**: [github.com/Aakashanil67/causal-ml-uplift](https://github.com/Aakashanil67/causal-ml-uplift) · **Live simulator**: see README · **Full reports**: `reports/01` through `reports/08`, this document synthesises all of them; none of the numbers above are restated from memory, each is sourced to the report that first computed it.

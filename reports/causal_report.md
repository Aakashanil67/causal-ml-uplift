# Causal ML: what actually works?

## Can causal ML find deployable treatment-effect heterogeneity, or only a reliable average effect?

Hillstrom's 64,000-customer randomized experiment estimates an average visit-rate effect of +6.01pp [+5.48pp, +6.55pp], about 60 additional visits per 1,000 customers assigned email. The normalized Qini estimate is 0.0111 [-0.0114, 0.0331]. Learned minus blanket mens policy value is +0.13pp [-0.14pp, +0.40pp]. The evidence supports an average effect. The interval crosses zero, so this analysis does not establish a positive ranking advantage. Held-out evidence does not establish which policy has higher expected visit value. Blanket mens emailing is the simpler action for this experiment; validate any learned policy on another campaign before broader deployment.

## 1. The causal question and why identification is clean here

The assumed graph has no causes of treatment assignment other than the randomizer. Under the documented randomized design, the pooled email-versus-control comparison identifies an intention-to-treat effect. The graph states the design assumption; the measured balance table is a descriptive finite-sample check and cannot prove the absence of unmeasured causes. DoWhy derives an empty backdoor adjustment set under this graph.

## 2. The naive estimate, and the check that it is allowed to be naive

Under random assignment, the difference in means estimates the effect of assignment to the historical email mixture. The balance table is descriptive evidence about measured covariates, not proof of the assignment mechanism. The estimated effects are visit +6.09pp [+5.54pp, +6.63pp], conversion +0.50pp [+0.35pp, +0.64pp], and spend $+0.597 [$0.376, $0.817]. The largest defined absolute standardized difference is 0.0088.

## 3. The methods ladder: naive, regression, and Double ML agree

The treatment effects below are expressed in percentage points for binary outcomes and dollars for spend. Agreement is expected under randomized assignment; it does not establish that DML is superior. The adjusted logit treatment effect is an average discrete prediction change. Continuous covariates use average derivatives, binary covariates use 0-to-1 changes, and categorical covariates use valid profile contrasts. LinearDML allows effect modification through a final effect model, which is linear here. Interactions can also estimate heterogeneity. Randomization does not prevent nuisance-model overfitting, so cross-fitting remains useful.

| method | visit | conversion | spend |
|---|---:|---:|---:|
| naive difference in means | +6.09pp | +0.50pp | $+0.597 |
| adjusted logit/OLS | +6.07pp | +0.49pp | $+0.596 |
| LinearDML | +6.01pp | +0.50pp | $+0.613 |

## 4. Constructed confounding stress test

The full randomized sample's estimated DML visit effect is +6.01pp [+5.48pp, +6.55pp]. Selected-sample comparisons below are descriptive: selection changes the covariate distribution and can change the target-population effect. They are not known-bias estimates. The fully synthetic benchmark evaluates bias and coverage against known sample-average potential-outcome targets. LinearDML's flexible nuisance functions do not remove the restriction of its linear final effect stage.

| selection variant | estimator | estimate | difference from full-RCT reference |
|---|---|---:|---:|
| recency/history observed | naive | +10.03pp | +4.02pp |
| recency/history observed | OLS | +5.93pp | -0.09pp |
| recency/history observed | LinearDML | +5.99pp | -0.02pp |
| newbie withheld | naive | +9.56pp | +3.55pp |
| newbie withheld | OLS | +10.09pp | +4.08pp |
| newbie withheld | LinearDML | +9.49pp | +3.48pp |

## 5. Heterogeneity: who the email actually helps

A held-out HC1-robust interaction audit finds exploratory segment-level differences. These are average effects within observed segments, not individual effects.

| segment | visit effect | 95% CI |
|---|---:|---:|
| mens only | +4.03pp | [+2.61pp, +5.44pp] |
| womens only | +6.17pp | [+4.71pp, +7.64pp] |
| both | +15.82pp | [+11.98pp, +19.67pp] |

The womens-only minus mens-only contrast has Holm-adjusted p=0.113; the joint Wald p-value is 8.55e-07. These post-hoc checks do not show that one customer-level ranking is reliable.

## 6. Uplift ranking and targeting economics

Normalized Qini is 0.0111 [-0.0114, 0.0331]. Across honest splits it ranges from 0.0111 to 0.0360; repeated splits are sensitivity evidence, not independent replications. The interval crosses zero, so this analysis does not establish a positive ranking advantage.
The top-30% pooled-email visit difference is +6.03pp [+4.06pp, +7.95pp] per emailed customer. This diagnostic concerns the historical creative mixture and does not select between creatives.

Reported-spend policy sensitivity is learned minus no email: $+0.726 [$+0.238, $+1.255], and learned minus blanket mens: $-0.056 [$-0.157, $+0.022]. Reported spend may be top-coded at $499; no gross margin or uncapped outcome is available, so these are not profit estimates.

## 7. Three actions, not two: an honest result

Cross-fitted doubly robust evaluation estimates expected visit rates under three-action policies. Intervals are paired fixed-model conditional evaluation intervals.

| policy | expected visit rate | 95% CI |
|---|---:|---:|
| learned (DRPolicyForest) | 18.23% | [17.30%, 19.16%] |
| email everyone (mens creative) | 18.10% | [17.19%, 19.02%] |
| email everyone (womens creative) | 15.20% | [14.33%, 16.15%] |
| purchase-history heuristic | 18.48% | [17.51%, 19.40%] |
| email nobody | 10.70% | [9.94%, 11.51%] |

Learned minus blanket mens is +0.13pp [-0.14pp, +0.40pp]. Held-out evidence does not establish which policy has higher expected visit value. Blanket mens emailing is the simpler action for this experiment; validate any learned policy on another campaign before broader deployment.

## 8. Refutation tests, and what they do not prove

Placebo treatment, random common cause, and data-subset refits diagnose responses to specific perturbations. They do not test whether the causal identification assumptions hold and do not certify correctness.

### Randomized sample

| check | estimate / summary |
|---|---:|
| placebo treatment | +0.0003 [-0.0054, +0.0061] |
| random common cause | +0.0605 [+0.0551, +0.0658] |
| data subset | mean +0.0610, SD 0.0015 |

### Selected sample

| check | estimate / summary |
|---|---:|
| placebo treatment | +0.0022 [-0.0061, +0.0105] |
| random common cause | +0.0952 [+0.0875, +0.1030] |
| data subset | mean +0.0950, SD 0.0015 |

The selected-sample estimate is compared with an estimated full-RCT reference for a different population. That difference is descriptive, not an omitted-confounder bias demonstration. Claims about bias and coverage against known targets are limited to the fully synthetic benchmark.

## 9. Limitations, stated rather than buried

- The study is one US retailer campaign from March 2008. Its effect sizes do not establish what would happen in another country, retailer, or current campaign.
- Spend is reported and may be top-coded at $499. The maximum alone does not establish the cap mechanism or motive. The uncapped-spend effect and profitability are unidentified here.
- The observed order of visit, conversion, and spend is compatible with mediation, but sequence alone does not establish a causal graph. This analysis estimates total effects and does not decompose mediation.
- The constructed real-data selection examples show estimates under specified mechanisms. They do not establish bias under omitted confounding in an observational target population.
- Segment contrasts are exploratory. Qini evidence: The interval crosses zero, so this analysis does not establish a positive ranking advantage. Policy evidence: Held-out evidence does not establish which policy has higher expected visit value. Blanket mens emailing is the simpler action for this experiment; validate any learned policy on another campaign before broader deployment.

## 10. What would and would not transfer to a South African retention campaign

The estimation and validation workflow can inform a new randomized campaign, but the Hillstrom effect sizes and policy ranking do not transfer automatically. A local campaign should define its outcome, eligible actions, costs, target population, assignment probabilities, and follow-up window; estimate effects under its own design; and validate any learned policy on held-out or later campaign data. Observational use would additionally require estimated propensities, overlap checks, suitable adjusted estimators, and explicit assumptions. The current benchmark does not establish those assumptions for another dataset.

## Closing

The project estimates an average effect from a randomized experiment, examines exploratory treatment-effect variation, and evaluates a learned action policy. The average effect is supported by the design. The interval crosses zero, so this analysis does not establish a positive ranking advantage. Held-out evidence does not establish which policy has higher expected visit value. Blanket mens emailing is the simpler action for this experiment; validate any learned policy on another campaign before broader deployment. The synthetic benchmark checks estimator behavior against known generated targets, while the selected-sample examples and refutation diagnostics remain limited to descriptive stress tests.

## References

- Hillstrom, K. (2008), [MineThatData E-Mail Analytics Challenge](https://blog.minethatdata.com/2008/05/best-answer-e-mail-analytics-challenge.html).
- Radcliffe, N. J. and Surry, P. D. (2011), [Real-World Uplift Modelling with Significance-Based Uplift Trees](https://stochasticsolutions.com/pdf/sig-based-up-trees.pdf).
- Sharma, A. and Kiciman, E. (2020), [DoWhy: An End-to-End Library for Causal Inference](https://arxiv.org/abs/2011.04216).
- Microsoft Research, [EconML 0.16 documentation](https://econml.azurewebsites.net/), including `CausalForestDML` and `DRPolicyForest`.
---

**Repository**: [github.com/Aakashanil67/causal-ml-uplift](https://github.com/Aakashanil67/causal-ml-uplift) · **Live simulator**: see README · **Full reports**: `reports/01` through `reports/08`, this document synthesises all of them; none of the numbers above are restated from memory, each is sourced to the report that first computed it.

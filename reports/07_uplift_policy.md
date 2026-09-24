# Heterogeneous effects and targeting policy

## Interaction check

This post-hoc held-out interaction audit uses HC1-robust OLS with mens-only as the
observed reference segment. Direct segment effects are evaluated at mean recency and
history; the joint Wald p-value is 8.55e-07.

| segment | effect on visit | 95% CI |
|---|---:|---:|
| mens only | +4.03pp | [+2.61pp, +5.44pp] |
| womens only | +6.17pp | [+4.71pp, +7.64pp] |
| both | +15.82pp | [+11.98pp, +19.67pp] |

The effect-modification contrasts are reported below with Holm-adjusted p-values across
the four reported contrasts. This is a post-hoc exploratory audit defined after the primary analysis.

| contrast | estimate | 95% CI | Holm-adjusted p |
|---|---:|---:|---:|
| womens only - mens only | +2.15pp | [+0.12pp, +4.17pp] | 0.1128 |
| both - mens only | +11.80pp | [+7.64pp, +15.96pp] | 1.094e-07 |
| recency | -0.04pp | [-0.34pp, +0.25pp] | 1 |
| history | -0.00pp | [-0.01pp, +0.00pp] | 1 |


## Ranking evidence

Raw Qini area is 17.84. The comparable normalized score is 0.0111 [-0.0114, 0.0331]. Across five honest splits the score averages 0.0211, with a range from 0.0111 to 0.0360. The interval crosses zero, so this analysis does not establish a positive ranking advantage. The CATE surface may contain real segment-level structure while still failing to rank individual customers reliably enough for deployment.

| decile (0 = highest predicted uplift) | predicted CATE | observed uplift | n |
|---|---:|---:|---:|
| 0 | +10.09pp | +4.00pp | 1,920 |
| 1 | +8.63pp | +6.70pp | 1,920 |
| 2 | +7.66pp | +7.38pp | 1,920 |
| 3 | +6.84pp | +8.56pp | 1,920 |
| 4 | +6.17pp | +7.83pp | 1,920 |
| 5 | +5.61pp | +5.75pp | 1,920 |
| 6 | +5.06pp | +5.44pp | 1,920 |
| 7 | +4.48pp | +4.31pp | 1,920 |
| 8 | +3.69pp | +6.10pp | 1,920 |
| 9 | +2.07pp | +4.58pp | 1,920 |

## Pooled-email targeting diagnostic

The table estimates the treated-minus-control visit difference inside each selected
segment. It evaluates assignment to the historical mixture of two creatives, not a
creative-specific action.

| top-k targeted | visit-rate difference | 95% CI |
|---|---:|---:|
| 10% | +4.00pp | [+0.49pp, +7.75pp] |
| 20% | +5.36pp | [+2.98pp, +7.80pp] |
| 30% | +6.03pp | [+4.06pp, +7.95pp] |
| 50% | +6.91pp | [+5.42pp, +8.35pp] |
| 100% | +6.08pp | [+5.14pp, +7.04pp] |

At 30%, reported gross spend is $0.7816 [$0.3733, $1.2743] per targeted customer. Hillstrom does not provide margin, and twelve spend observations sit at the dataset maximum of $499. This cannot be read as profit or break-even evidence.

## Three-action policy

Values use cross-fitted outcome models, nominal one-third randomisation probabilities and
doubly robust scores. Bootstrap samples preserve the arm counts; intervals are fixed-model
conditional evaluation intervals.

| policy | visit-rate value | 95% CI |
|---|---:|---:|
| learned (DRPolicyForest) | 18.23% | [17.30%, 19.16%] |
| email everyone (mens creative) | 18.10% | [17.19%, 19.02%] |
| email everyone (womens creative) | 15.20% | [14.33%, 16.15%] |
| purchase-history heuristic | 18.48% | [17.51%, 19.40%] |
| email nobody | 10.70% | [9.94%, 11.51%] |

The learned policy was trained to maximise visits, not spend or profit. Reported-spend
sensitivity is shown separately because margin and email cost are not identified by this
experiment.

| gross margin assumption | incremental net value vs email nobody |
|---:|---:|
| 25% | $+0.0815 |
| 50% | $+0.2631 |
| 100% | $+0.6261 |

| paired comparison | difference | 95% CI |
|---|---:|---:|
| learned (DRPolicyForest) - email everyone (mens creative) | +0.13pp | [-0.14pp, +0.40pp] |
| learned (DRPolicyForest) - email everyone (womens creative) | +3.03pp | [+1.76pp, +4.27pp] |
| learned (DRPolicyForest) - purchase-history heuristic | -0.25pp | [-1.02pp, +0.53pp] |
| learned (DRPolicyForest) - email nobody | +7.53pp | [+6.34pp, +8.77pp] |

Held-out evidence does not establish which policy has higher expected visit value. Blanket mens emailing is the simpler action for this experiment; validate any learned policy on another campaign before broader deployment.

## Fully synthetic estimator benchmark

The original fast DML check uses ten repetitions per confounding level. Its target is the sample average of the generated probability differences, `mean(p1 - p0)`. The interval below reports Monte Carlo uncertainty in mean bias; coverage is the share of run-level 95% intervals covering that known target.

| confounding strength | estimator | mean bias | 95% Monte Carlo CI for bias | coverage | Wilson 95% CI |
|---:|---|---:|---:|---:|---:|
| 0.00 | adjusted_dml | -0.0021 | [-0.0109, +0.0067] | 100.0% | [72.2%, 100.0%] |
| 0.00 | naive | -0.0006 | [-0.0120, +0.0108] | 100.0% | [72.2%, 100.0%] |
| 0.00 | omitted_confounder_dml | -0.0012 | [-0.0139, +0.0114] | 100.0% | [72.2%, 100.0%] |
| 0.75 | adjusted_dml | +0.0022 | [-0.0083, +0.0127] | 100.0% | [72.2%, 100.0%] |
| 0.75 | naive | +0.0689 | [+0.0596, +0.0782] | 10.0% | [1.8%, 40.4%] |
| 0.75 | omitted_confounder_dml | +0.0649 | [+0.0543, +0.0754] | 0.0% | [0.0%, 27.8%] |
| 1.50 | adjusted_dml | -0.0178 | [-0.0288, -0.0068] | 100.0% | [72.2%, 100.0%] |
| 1.50 | naive | +0.1134 | [+0.1035, +0.1234] | 0.0% | [0.0%, 27.8%] |
| 1.50 | omitted_confounder_dml | +0.1093 | [+0.0969, +0.1218] | 0.0% | [0.0%, 27.8%] |

At the strongest simulated confounding level (1.50), the fast adjusted DML run has smaller absolute mean bias (-0.0178) than the naive run (+0.1134); its remaining error is still reported with Monte Carlo uncertainty. This is improvement in this finite simulation, not automatic recovery under arbitrary effect surfaces.

## Extended fully synthetic benchmark (500 repetitions)

The following 500-repetition Monte Carlo uses known potential-outcome probabilities. The target is each generated sample's average `p1 - p0`. Bias intervals show Monte Carlo uncertainty across repetitions; coverage intervals are Wilson intervals for the share of run-specific 95% intervals covering the target.

| confounding strength | estimator | mean estimate | bias | 95% Monte Carlo CI for bias | coverage | Wilson 95% CI | RMSE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.00 | adjusted_aipw | 0.0785 | +0.0003 | [-0.0014, +0.0021] | 95.8% | [93.7%, 97.2%] | 0.0198 |
| 0.00 | naive | 0.0789 | +0.0008 | [-0.0011, +0.0026] | 96.0% | [93.9%, 97.4%] | 0.0208 |
| 0.00 | omitted_confounder_aipw | 0.0787 | +0.0006 | [-0.0012, +0.0025] | 96.0% | [93.9%, 97.4%] | 0.0209 |
| 0.00 | oracle_aipw | 0.0785 | +0.0004 | [-0.0013, +0.0021] | 94.8% | [92.5%, 96.4%] | 0.0195 |
| 0.75 | adjusted_aipw | 0.0786 | +0.0005 | [-0.0015, +0.0024] | 95.2% | [93.0%, 96.8%] | 0.0219 |
| 0.75 | naive | 0.1433 | +0.0652 | [+0.0633, +0.0670] | 14.6% | [11.8%, 18.0%] | 0.0686 |
| 0.75 | omitted_confounder_aipw | 0.1433 | +0.0652 | [+0.0633, +0.0670] | 15.2% | [12.3%, 18.6%] | 0.0686 |
| 0.75 | oracle_aipw | 0.0786 | +0.0005 | [-0.0014, +0.0024] | 94.6% | [92.3%, 96.3%] | 0.0216 |
| 1.50 | adjusted_aipw | 0.0779 | -0.0002 | [-0.0026, +0.0022] | 94.0% | [91.6%, 95.8%] | 0.0274 |
| 1.50 | naive | 0.1875 | +0.1094 | [+0.1075, +0.1113] | 0.0% | [0.0%, 0.8%] | 0.1115 |
| 1.50 | omitted_confounder_aipw | 0.1874 | +0.1092 | [+0.1073, +0.1112] | 0.0% | [0.0%, 0.8%] | 0.1114 |
| 1.50 | oracle_aipw | 0.0782 | +0.0001 | [-0.0022, +0.0024] | 94.2% | [91.8%, 95.9%] | 0.0261 |

Naive is the unadjusted estimator; adjusted AIPW uses cross-fitted estimated propensity and outcome models; omitted-confounder AIPW intentionally excludes the confounder; oracle AIPW uses generating nuisance probabilities as a diagnostic. This is a synthetic benchmark, not evidence that a particular observational study is identified.

## Additional sensitivity checks

| check | configuration | result |
|---|---|---:|
| forest | smaller leaves: min leaf 25, max depth uncapped | normalized Qini 0.0057 |
| forest | primary: min leaf 50, max depth uncapped | normalized Qini 0.0111 |
| forest | larger leaves: min leaf 100, max depth uncapped | normalized Qini 0.0147 |
| forest | depth capped: min leaf 50, max depth 5 | normalized Qini 0.0194 |
| duplicate rows | 57,438 rows after exact-row deduplication (from 64,000) | visit difference +6.22pp; full +6.09pp |

| policy split seed | learned value | blanket mens | learned minus blanket | action shares: no email / mens / womens |
|---:|---:|---:|---:|---|
| 42 | 18.23% | 18.10% | +0.13pp | 0.0% / 96.5% / 3.5% |
| 7 | 18.15% | 18.43% | -0.28pp | 0.0% / 86.5% / 13.5% |
| 19 | 18.35% | 18.29% | +0.06pp | 0.0% / 95.6% / 4.4% |
| 73 | 18.04% | 18.25% | -0.21pp | 0.0% / 79.2% / 20.8% |
| 101 | 18.14% | 18.02% | +0.12pp | 0.0% / 98.1% / 1.9% |

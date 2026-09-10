# Heterogeneous effects and targeting policy

## Interaction check

This post-hoc held-out interaction audit uses HC1-robust OLS with mens-only as the
observed reference segment. Direct segment effects are evaluated at mean recency and
history; the joint Wald p-value is 8.55e-07.

| segment | effect on visit | 95% CI |
|---|---:|---:|
| mens only | +0.04025 | [+0.02611, +0.05439] |
| womens only | +0.06170 | [+0.04705, +0.07635] |
| both | +0.15822 | [+0.11976, +0.19668] |

The effect-modification contrasts are reported below with Holm-adjusted p-values across
the four reported contrasts. This is a post-hoc exploratory audit defined after the primary analysis.

| contrast | estimate | 95% CI | Holm-adjusted p |
|---|---:|---:|---:|
| womens only - mens only | +0.02145 | [+0.00123, +0.04167] | 0.1128 |
| both - mens only | +0.11797 | [+0.07636, +0.15957] | 1.094e-07 |
| recency | -0.00045 | [-0.00335, +0.00246] | 1 |
| history | -0.00001 | [-0.00006, +0.00003] | 1 |


## Ranking evidence

Raw Qini area is 17.84. The comparable normalized score is 0.0111 [-0.0114, 0.0331]. Across five honest splits the score averages 0.0211, with a range from 0.0111 to 0.0360. The primary fixed-model conditional bootstrap interval includes zero. The CATE surface may contain real segment-level structure while still failing to rank individual customers reliably enough for deployment.

| decile (0 = highest predicted uplift) | predicted CATE | observed uplift | n |
|---|---:|---:|---:|
| 0 | +0.1009 | +0.0400 | 1,920 |
| 1 | +0.0863 | +0.0670 | 1,920 |
| 2 | +0.0766 | +0.0738 | 1,920 |
| 3 | +0.0684 | +0.0856 | 1,920 |
| 4 | +0.0617 | +0.0783 | 1,920 |
| 5 | +0.0561 | +0.0575 | 1,920 |
| 6 | +0.0506 | +0.0544 | 1,920 |
| 7 | +0.0448 | +0.0431 | 1,920 |
| 8 | +0.0369 | +0.0610 | 1,920 |
| 9 | +0.0207 | +0.0458 | 1,920 |

## Pooled-email targeting diagnostic

The table estimates the treated-minus-control visit difference inside each selected
segment. It evaluates assignment to the historical mixture of two creatives, not a
creative-specific action.

| top-k targeted | visits per customer emailed | 95% CI |
|---|---:|---:|
| 10% | +0.0400 | [+0.0049, +0.0775] |
| 20% | +0.0536 | [+0.0298, +0.0780] |
| 30% | +0.0603 | [+0.0406, +0.0795] |
| 50% | +0.0691 | [+0.0542, +0.0835] |
| 100% | +0.0608 | [+0.0514, +0.0704] |

At 30%, reported gross spend is $0.7816 [$0.3733, $1.2743] per targeted customer. Hillstrom does not provide margin, and twelve spend observations sit at the dataset maximum of $499. This cannot be read as profit or break-even evidence.

## Three-action policy

Values use cross-fitted outcome models, nominal one-third randomisation probabilities and
doubly robust scores. Bootstrap samples preserve the arm counts; intervals are fixed-model
conditional evaluation intervals.

| policy | visit-rate value | 95% CI |
|---|---:|---:|
| learned (DRPolicyForest) | 0.1823 | [0.1730, 0.1916] |
| email everyone (mens creative) | 0.1810 | [0.1719, 0.1902] |
| email everyone (womens creative) | 0.1520 | [0.1433, 0.1615] |
| purchase-history heuristic | 0.1848 | [0.1751, 0.1940] |
| email nobody | 0.1070 | [0.0994, 0.1151] |

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
| learned (DRPolicyForest) - email everyone (mens creative) | +0.0013 | [-0.0014, +0.0040] |
| learned (DRPolicyForest) - email everyone (womens creative) | +0.0303 | [+0.0176, +0.0427] |
| learned (DRPolicyForest) - purchase-history heuristic | -0.0025 | [-0.0102, +0.0053] |
| learned (DRPolicyForest) - email nobody | +0.0753 | [+0.0634, +0.0877] |

Held-out evidence does not establish that the learned policy beats blanket mens emailing. Blanket mens is the simpler evidence-supported action; personalised policy deployment needs a new experiment or stronger cross-campaign evidence.

# Heterogeneous effects and targeting policy

## Interaction check

The segmentation claim is checked with one pre-specified HC1-robust OLS containing the
four treatment interactions below. Holm-adjusted p-values control the family-wise error
rate; the joint Wald p-value is 1.01e-10.

| interaction | estimate | 95% CI | raw p | Holm-adjusted p |
|---|---:|---:|---:|---:|
| treatment × `mens` | +0.03333 | [+0.00949, +0.05717] | 0.006146 | 0.01844 |
| treatment × `womens` | +0.06593 | [+0.04208, +0.08978] | 6.038e-08 | 2.415e-07 |
| treatment × `recency` | -0.00013 | [-0.00173, +0.00147] | 0.8771 | 1 |
| treatment × `history` | -0.00000 | [-0.00003, +0.00002] | 0.7143 | 1 |

Purchase-history interactions survive adjustment; recency and history do not. This is
the executable check behind the forest interpretation, not a p-value copied into prose.

## Ranking evidence

Raw Qini area is 17.84. The comparable normalized score is 0.0111 [-0.0114, 0.0331]. Across five honest splits the score averages 0.0211, with a range from 0.0111 to 0.0360. The primary bootstrap interval includes zero. The CATE surface may contain real segment-level structure while still failing to rank individual customers reliably enough for deployment.

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
doubly robust scores. Bootstrap samples preserve the arm counts.

| policy | visit-rate value | 95% CI |
|---|---:|---:|
| learned (DRPolicyForest) | 0.1823 | [0.1730, 0.1916] |
| email everyone (mens creative) | 0.1810 | [0.1719, 0.1902] |
| purchase-history heuristic | 0.1803 | [0.1706, 0.1900] |
| email nobody | 0.1070 | [0.0994, 0.1151] |

| paired comparison | difference | 95% CI |
|---|---:|---:|
| learned (DRPolicyForest) - email everyone (mens creative) | +0.0013 | [-0.0014, +0.0040] |
| learned (DRPolicyForest) - purchase-history heuristic | +0.0019 | [-0.0079, +0.0111] |
| learned (DRPolicyForest) - email nobody | +0.0753 | [+0.0634, +0.0877] |

Held-out evidence does not establish that the learned policy beats blanket mens emailing. Blanket mens is the simpler evidence-supported action; personalised policy deployment needs a new experiment or stronger cross-campaign evidence.

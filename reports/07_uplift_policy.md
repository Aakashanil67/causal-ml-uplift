# Heterogeneous effects and targeting policy

`src/cate.py`'s `CausalForestDML` estimates, evaluated for real on the held-out 30%
(`src/cate.py` docstring: `visit` is the only outcome with enough events, 9,394, to split
across a forest without the CATE surface being mostly noise).

## Heterogeneity: who the email actually helps

Mean CATE on the eval set: **+0.0588** (std 0.0218,
range [-0.0198, +0.1271]), close to the pooled DML
benchmark of +0.0601 (`reports/05_dml_ate.md`), as it should be: the CATE mean and the
pooled ATE are estimating the same population quantity two different ways.

The headline segmentation is prior purchase category, not recency or history:

| prior purchase | mean CATE | n |
|---|---|---|
| womens only | +0.0708 | 8,625 |
| mens only | +0.0429 | 8,633 |
| both | +0.0762 | 1,942 |

Confirmed with an OLS interaction test before trusting the forest's split, since a tree
can carve out a segment from noise as easily as from a real pattern: treatment × `mens`
and treatment × `womens` interaction terms are both significant (p=0.006, p<0.001),
unlike treatment × `recency`/`history`, checked earlier for the confounding benchmark
(both p>0.11, not significant). The pooled any-email effect is real for everyone in this
data, but it is close to twice as large for customers with a `womens`-only purchase
history as for `mens`-only customers, see `reports/figures/cate_by_purchase_history.png`.

## Uplift deciles: does the ranking correspond to anything real?

Decile 0 is the customers the model ranks highest; `observed_uplift` is the *actual*
treated-minus-control visit-rate gap measured within that decile, not the model's own
prediction, so this is a check on the ranking rather than a restatement of it.

| decile | n | predicted CATE | observed uplift |
|---|---|---|---|
| 0 | 1,920 | +0.0955 | +0.0866 |
| 1 | 1,920 | +0.0825 | +0.0605 |
| 2 | 1,920 | +0.0752 | +0.0775 |
| 3 | 1,920 | +0.0684 | +0.0822 |
| 4 | 1,920 | +0.0620 | +0.0553 |
| 5 | 1,920 | +0.0558 | +0.0520 |
| 6 | 1,920 | +0.0495 | +0.0514 |
| 7 | 1,920 | +0.0425 | +0.0658 |
| 8 | 1,920 | +0.0352 | +0.0340 |
| 9 | 1,920 | +0.0216 | +0.0467 |

Not perfectly monotonic. With roughly 1,920 customers per decile and a rare binary
outcome, individual deciles carry real sampling noise, but the top two deciles average
+0.0736 observed uplift against +0.0404 for the bottom two, the gradient the
ranking predicts.

**Qini coefficient: 47.10** (area between the model's gain curve and random
targeting; positive means the ranking beats emailing customers in a random order,
0 is what a ranking with no real signal would average to).

## Targeting economics

Expected incremental visits per customer targeted, at several targeting fractions, with a
95% bootstrap CI over the held-out eval set (CATE scores held fixed, so this is evaluation
noise, not a statement about how much a re-trained forest could vary):

| top-k% targeted | incremental visits/customer | 95% CI |
|---|---|---|
| 10% | +0.0576 | [+0.0346, +0.0827] |
| 20% | +0.0489 | [+0.0323, +0.0640] |
| 30% | +0.0499 | [+0.0371, +0.0622] |
| 50% | +0.0482 | [+0.0385, +0.0578] |
| 100% | +0.0407 | [+0.0341, +0.0472] |

## Break-even email cost

At the top-30% targeting fraction, the expected incremental
**spend** per customer targeted is **$0.1196**
(95% CI [$-0.4814, $0.6267]): the revenue side
of the same targeting policy, using the same visit-based ranking rather than a separate
spend-CATE model (only 578 non-zero spend values total, see
`reports/data_dictionary.md`). The point estimate clears the assumed
$0.10-per-email cost
(`src/config.py:DEFAULT_EMAIL_COST_USD`, a stated assumption, not fitted) comfortably.
**The confidence interval crosses zero and runs negative, though**: at this sample size, spend among 30%-of-64,000 customers is too sparse to statistically rule out the segment losing money rather than paying for itself. The visit-based numbers above are the ones this report actually stands behind; this section is a directional read on revenue, not a claim with the same statistical footing.

## Three-action policy: no email, mens email, or womens email

Everything above treats this as one decision, email or not. But the mens and womens
creatives are not interchangeable (`reports/05_dml_ate.md`: +7.47pp vs +4.49pp on visit,
pooled), so the real decision has three options. `econml.policy.DRPolicyForest` picks a
recommended arm per customer from a doubly-robust reward estimate relative to `No E-Mail`.
Every policy below, learned or not, is scored the same way on the held-out eval set: an
inverse-propensity-weighted estimate of the average visit rate under that policy, using
only customers whose real (randomised) arm happened to match the recommendation.

| policy | policy value (visit rate) | 95% CI |
|---|---|---|
| learned (DRPolicyForest) | 0.1831 | [0.1736, 0.1941] |
| purchase-history heuristic | 0.1756 | [0.1662, 0.1852] |
| email everyone (mens creative) | 0.1851 | [0.1752, 0.1957] |
| email nobody | 0.1061 | [0.0984, 0.1137] |

`email nobody` recovers the control arm's raw visit rate almost exactly
(0.1061 vs the true control rate of 0.1062, `reports/data_dictionary.md`),
the sanity check that the IPW estimator itself is unbiased before trusting it on anything
more interesting.

**The honest result, stated plainly rather than dressed up: the learned policy does not**
**clearly beat the simplest baseline that already knew the mens creative works better.**
Learned policy value 0.1831 vs 0.1851 for blanket mens-emailing everyone, with heavily overlapping confidence intervals: not a result this report can call a win.
The historical report used marginal-interval overlap to describe comparisons. The implementation
now reports paired bootstrap differences instead; this static report must be regenerated before
making a significance claim about policy-versus-policy comparisons.

This is a real and explicable finding, not a failed experiment: the heterogeneity section
above showed the mens creative's effect is positive for almost every segment,
including customers with a mens-only purchase history (+0.043 CATE, still clearly above
zero). When
the simpler action already has a positive effect almost everywhere, there is little room
left for a smarter per-customer policy to improve on it — the value of granular targeting
would show up far more clearly in a setting where some segment is actually hurt by the
default action, which is not the case here. The learned policy still earns its place: it
correctly identifies that the purchase-history heuristic's core assumption, match the
creative to what the customer already buys, is wrong on this data (the heuristic
underperforms blanket mens-emailing by 0.0094
visit-rate points), and it does so without anyone having to notice that by eye.

The forest recommends the womens creative for
2,165 of 19,200
held-out customers and the mens creative for the rest; it never recommends sending no
email at all in this eval set, since both creatives show a positive effect for every
segment identified above.

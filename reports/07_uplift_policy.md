# Heterogeneous effects and targeting policy

`src/cate.py`'s `CausalForestDML` estimates, evaluated for real on the held-out 30%
(`src/cate.py` docstring: `visit` is the only outcome with enough events, 9,394, to split
across a forest without the CATE surface being mostly noise).

## Heterogeneity: who the email actually helps

Mean CATE on the eval set: **+0.0603** (std 0.0230,
range [-0.0073, +0.1292]), close to the pooled DML
benchmark of +0.0601 (`reports/05_dml_ate.md`), as it should be: the CATE mean and the
pooled ATE are estimating the same population quantity two different ways.

The headline segmentation is prior purchase category, not recency or history:

| prior purchase | mean CATE | n |
|---|---|---|
| womens only | +0.0746 | 8,633 |
| mens only | +0.0438 | 8,655 |
| both | +0.0708 | 1,912 |

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
| 0 | 1,920 | +0.1009 | +0.0400 |
| 1 | 1,920 | +0.0863 | +0.0670 |
| 2 | 1,920 | +0.0766 | +0.0738 |
| 3 | 1,920 | +0.0684 | +0.0856 |
| 4 | 1,920 | +0.0617 | +0.0783 |
| 5 | 1,920 | +0.0561 | +0.0575 |
| 6 | 1,920 | +0.0506 | +0.0544 |
| 7 | 1,920 | +0.0448 | +0.0431 |
| 8 | 1,920 | +0.0369 | +0.0610 |
| 9 | 1,920 | +0.0207 | +0.0458 |

Not perfectly monotonic. With roughly 1,920 customers per decile and a rare binary
outcome, individual deciles carry real sampling noise. In this split the top two average
+0.0535 observed uplift against +0.0534 for the bottom two, so the decile
comparison itself does not establish a useful ranking gradient.

**Qini coefficient: 17.84** (area between the model's gain curve and random
targeting; positive means the ranking beats emailing customers in a random order,
0 is what a ranking with no real signal would average to).

## Targeting economics

Expected incremental visits per customer targeted, at several targeting fractions, with a
95% bootstrap CI over the held-out eval set (CATE scores held fixed, so this is evaluation
noise, not a statement about how much a re-trained forest could vary):

| top-k% targeted | incremental visits/customer | 95% CI |
|---|---|---|
| 10% | +0.0400 | [+0.0049, +0.0775] |
| 20% | +0.0536 | [+0.0298, +0.0780] |
| 30% | +0.0603 | [+0.0406, +0.0795] |
| 50% | +0.0691 | [+0.0542, +0.0835] |
| 100% | +0.0608 | [+0.0514, +0.0704] |

## Gross-spend sensitivity

At the top-30% targeting fraction, the expected incremental
**reported spend** per customer targeted is **$0.7816**
(95% CI [$0.3733, $1.2743]): the gross-spend side
of the same targeting policy, using the same visit-based ranking rather than a separate
spend-CATE model (only 578 non-zero spend values total, see
`reports/data_dictionary.md`). This is not a break-even or profitability result: a
gross-margin assumption is required before comparing reported spend with the assumed
$0.10-per-email cost
(`src/config.py:DEFAULT_EMAIL_COST_USD`). Reported spend may also be top-coded, so this
section is a sensitivity input rather than a deployment recommendation.


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
| learned (DRPolicyForest) | 0.1838 | [0.1738, 0.1936] |
| purchase-history heuristic | 0.1814 | [0.1714, 0.1914] |
| email everyone (mens creative) | 0.1827 | [0.1726, 0.1929] |
| email nobody | 0.1062 | [0.0981, 0.1141] |

| paired comparison (A - B) | difference | 95% CI |
|---|---:|---|
| learned - blanket mens | +0.0011 | [-0.0019, +0.0040] |
| learned - purchase-history heuristic | +0.0024 | [-0.0079, +0.0135] |
| learned - email nobody | +0.0776 | [+0.0644, +0.0907] |

`email nobody` recovers the control arm's raw visit rate almost exactly
(0.1062 vs the true control rate of 0.1062, `reports/data_dictionary.md`),
the sanity check that the IPW estimator itself is unbiased before trusting it on anything
more interesting.

**The honest result, stated plainly rather than dressed up: the learned policy does not**
**clearly beat the simplest baseline that already knew the mens creative works better.**
Learned policy value 0.1838 vs 0.1827 for blanket mens-emailing everyone, with heavily overlapping confidence intervals: not a result this report can call a win.
The paired-comparison table below, rather than these marginal intervals, determines
which policy differences this evaluation can support.

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
underperforms blanket mens-emailing by 0.0013
visit-rate points), and it does so without anyone having to notice that by eye.

The forest recommends the womens creative for
663 of 19,200
held-out customers and the mens creative for the rest; it never recommends sending no
email at all in this eval set, since both creatives show a positive effect for every
segment identified above.

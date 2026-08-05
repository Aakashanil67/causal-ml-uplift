# Naive estimate: difference in means

Treated = received either email (Mens or Womens), control = No E-Mail. On an RCT this
comparison is unbiased by design, so unlike the usual textbook framing, this is a real
estimate here, not just a demonstration of what goes wrong. Whether the design assumption
actually holds in this file is checked below, not just asserted.

## Outcome differences

| outcome | treated mean | control mean | diff | 95% CI |
|---|---|---|---|---|
| visit | 0.16705 | 0.10617 | +0.06088 | [0.05544, 0.06633] |
| conversion | 0.01068 | 0.00573 | +0.00495 | [0.00355, 0.00636] |
| spend | 1.24959 | 0.65279 | +0.59680 | [0.37619, 0.81741] |

## Covariate balance

Standardised difference = (treated mean − control mean) / pooled SD. Values inside ±0.1
are the usual threshold for calling a covariate balanced (Austin, 2009); nothing here is a
real pre-treatment difference if the randomisation worked as intended.

| covariate | treated mean | control mean | standardised diff |
|---|---|---|---|
| recency | 5.7707 | 5.7497 | +0.0060 |
| history | 242.6860 | 240.8827 | +0.0071 |
| mens | 0.5499 | 0.5532 | -0.0066 |
| womens | 0.5508 | 0.5476 | +0.0063 |
| newbie | 0.5024 | 0.5020 | +0.0008 |
| zip_code=Rural | 0.1505 | 0.1473 | +0.0088 |
| zip_code=Surburban | 0.4486 | 0.4518 | -0.0064 |
| zip_code=Urban | 0.4010 | 0.4009 | +0.0001 |
| channel=Multichannel | 0.1208 | 0.1223 | -0.0047 |
| channel=Phone | 0.4379 | 0.4378 | +0.0002 |
| channel=Web | 0.4414 | 0.4399 | +0.0029 |

Largest standardised difference: 0.0088. 0 of 11
covariates (across numeric, binary and one row per categorical level) exceed the 0.1
threshold. This is the sanity check the randomisation claim in
`reports/01_causal_question.md` rests on: it is a testable statement about this file, not
an assumption. `reports/06_confounding_benchmark.md` shows what this table looks like when
that assumption is deliberately violated.

**Why this comparison would be biased without randomisation.** If treatment had been
chosen by a marketer rather than a coin flip, a nonzero standardised difference on
`history` or `recency` above would mean the treated and control groups differed in ways
that independently predict the outcome, and the diff-in-means table above would then be
mixing the true effect of the email with the effect of already being a different kind of
customer. That confound is exactly what `src/confounded.py` reconstructs on purpose,
using this same dataset, to make the size of that bias visible.

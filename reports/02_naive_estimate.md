# Naive estimate: difference in means

Treated customers received either email (Mens or Womens); controls received No E-Mail. Under the randomized assignment design, the difference in means estimates the average effect of assignment to the email mixture. The balance table is a descriptive check on this file, not proof that randomization worked.

## Outcome differences

| outcome | treated mean | control mean | difference | 95% CI |
|---|---:|---:|---:|---:|
| visit | 16.70% | 10.62% | +6.09pp | [+5.54pp, +6.63pp] |
| conversion | 1.07% | 0.57% | +0.50pp | [+0.35pp, +0.64pp] |
| spend | $1.250 | $0.653 | $+0.597 | [$0.376, $0.817] |

## Covariate balance

Standardised difference is the treated-minus-control mean divided by the pooled standard deviation. A value near zero is consistent with balance on that measured covariate; balance cannot test unmeasured causes.

| covariate | treated mean | control mean | standardised difference |
|---|---:|---:|---:|
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

Largest defined absolute standardised difference: 0.0088; 0 of 11 defined covariates exceed 0.1. This is a descriptive randomization check. The constructed selection exercise in `reports/06_confounding_benchmark.md` shows how results move under specified selection mechanisms; differences from the full experiment are not known bias for a changed target population.

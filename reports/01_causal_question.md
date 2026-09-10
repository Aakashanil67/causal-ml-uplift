# The causal question

Does assignment to a marketing email cause a customer to visit the site, buy, and spend, or would
they have done the same thing anyway? Hillstrom's data lets this be answered without the usual
hedge, because the "would have done it anyway" group is not estimated: it was built. MineThatData
randomly split 64,000 customers who had purchased in the past twelve months into three near-equal
groups before the campaign ran (21,306 No E-Mail, 21,307 Mens E-Mail, 21,387 Womens E-Mail), so any
post-campaign difference between them is the intention-to-treat effect of assignment to the email, not a difference in who
the customers already were. `reports/02_naive_estimate.md` checks that claim directly rather than
assuming it.

![causal graph](figures/dag.png)

## The graph

`treatment` has no parents in this graph, and that is the graph's one substantive claim: nothing
caused a customer to be assigned to an arm except the randomiser. Every other node,
`recency`, `history`, `mens`, `womens`, `zip_code`, `newbie`, `channel`, points into all three
outcomes (`visit`, `conversion`, `spend`) as plausible common causes of general engagement, but not
into `treatment`, because none of them influenced which arm a customer landed in. That single
missing set of edges is what makes the naive difference-in-means unbiased here and is the reason
this dataset was chosen over, say, an observational churn file where the same graph would need
confounder-to-treatment edges and a much harder identification argument.

`src/dag.py` builds this as a `networkx.DiGraph` and exports it as a GML string for DoWhy, rather
than shelling out to `graphviz`'s `dot` binary, which is not installed on this machine.

## What the graph does not model

`conversion` is a strict subset of `visit` in the data (`reports/data_dictionary.md`): nobody buys
without visiting first, and every non-zero `spend` value corresponds to a `conversion=1` row. The
honest graph for that is a mediation chain, `treatment → visit → conversion → spend`, not three
parallel arrows from treatment. This project deliberately keeps the simpler parallel-arrows graph
and estimates the *total* effect of treatment on each outcome independently, which is the standard
approach for a method ladder like this one and matches what `LinearDML`/`CausalForestDML` actually
compute per outcome. A mediation analysis decomposing how much of the spend effect runs through
visiting is a real extension this project does not attempt; it would need its own identification
argument (visit is a post-treatment variable, so conditioning on it to study conversion would
introduce collider bias) and is flagged here rather than quietly assumed away.

## Why randomisation makes this easier than the general case, and what would break it

If treatment had been assigned by a marketer instead of a randomiser, at least three of the
covariates above would plausibly cause it as well as the outcome, which is exactly what makes a
variable a confounder rather than just a predictor. A marketer targeting high-value repeat
customers (median `history` in this file is $158.11, top decile above $561) with a promotional
email would create a `history → treatment` edge, and `history` already plausibly causes
`visit`/`spend` directly; the naive comparison would then conflate "the email worked" with "big
spenders spend more regardless." A win-back campaign aimed specifically at the 2,332 customers at
`recency=12`, the furthest band from a recent purchase, would create a `recency → treatment` edge
running the other way, biasing the naive estimate in the opposite direction. And a welcome-series
email targeted at the 32,144 `newbie` accounts would tie that variable to both assignment and to
how engaged a first-year customer already is, before any email is sent.

None of those edges exist in this graph, and the balance check in `reports/02_naive_estimate.md`
is a finite-sample sanity check consistent with the documented randomisation, rather than proof of
the assignment mechanism. The analysis assumes SUTVA/no interference, well-defined creative
versions and independent customer rows. Household identifiers, delivery, opens and clicks are
unavailable, so the estimand is assignment, not delivery, opening, reading or clicking. The check
of whether that claim actually holds in the data as delivered, rather than
just in the paragraph above. `src/confounded.py` builds a second, deliberately confounded version
of this same dataset by re-introducing exactly this kind of `history`/`recency` → treatment edge,
so that the difference the randomisation makes can be shown as a number rather than only argued in
prose.

## Scope

The three-action version of this question, "which of no-email, mens-email, or womens-email should
each customer get", is addressed separately in `src/policy.py`, once the binary-treatment ladder
(naive → OLS/logit → DoWhy → DML → CATE → Qini) has established the estimation machinery on the
simpler two-arm version first.

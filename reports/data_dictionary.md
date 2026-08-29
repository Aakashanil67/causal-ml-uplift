# Data dictionary: Hillstrom MineThatData e-mail experiment

Source: `http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv`,
downloaded 5 August 2026 (last-modified 21 March 2008 per the server, 3,964,977 bytes). A US
speciality-retail customer file: 64,000 customers who purchased in the previous twelve months,
randomised into one of three arms and tracked for two weeks after the email went out.

64,000 rows, 12 source columns, no nulls. There is no customer ID column, and 6,562 rows are exact
duplicates of another row (7,634 rows involved). That is not a data error: with `recency`
restricted to 1–12, `history` sitting at the $29.99 floor for a large share of low-value customers,
and every other covariate low-cardinality, dozens of genuinely different customers land on an
identical combination of recency/history/mens/womens/zip/newbie/channel/arm/outcomes by chance.
Nothing downstream keys on row identity, so this has no effect on any estimate in this project.

| column | type | description |
|---|---|---|
| `recency` | int, 1–12 | months since the customer's last purchase before the campaign |
| `history_segment` | categorical, 7 bands | `history` bucketed into dollar bands, e.g. `"1) $0 - $100"` |
| `history` | float, $29.99–$3,345.93 | dollars spent in the twelve months before the campaign (median $158.11) |
| `mens` | binary | bought a mens product in the past year |
| `womens` | binary | bought a womens product in the past year |
| `zip_code` | categorical: Surburban / Urban / Rural | customer's zip code type (the dataset's own spelling, kept verbatim rather than "corrected", since silently renaming a category is exactly the kind of change that would make this dictionary describe the wrong file) |
| `newbie` | binary | account opened in the last twelve months |
| `channel` | categorical: Web / Phone / Multichannel | channel(s) used to purchase in the past year |
| `segment` | categorical, 3 arms | which email arm the customer was randomised into (see below) |
| `visit` | binary | visited the site in the two weeks following the campaign |
| `conversion` | binary | purchased in the two weeks following the campaign |
| `spend` | float, $0–$499.00 | dollars spent in the two weeks following the campaign |

`treatment` is added by `src/data_loader.py`, not in the source file: 1 if `segment != "No E-Mail"`,
0 otherwise. It collapses the two email arms for the binary-treatment stages of the method ladder
(naive, regression, DML, CATE); the three-arm distinction is kept for `src/policy.py`.

## Arm sizes and outcome rates

| arm | n | visit | conversion | mean spend |
|---|---|---|---|---|
| No E-Mail | 21,306 | 10.617% | 0.573% | $0.6528 |
| Mens E-Mail | 21,307 | 18.276% | 1.253% | $1.4226 |
| Womens E-Mail | 21,387 | 15.140% | 0.884% | $1.0772 |

Arms are within 0.4% of exactly a three-way even split (21,306 / 21,307 / 21,387), consistent with
randomisation rather than a targeted campaign. The formal balance check on covariates is in
`reports/02_naive_estimate.md`.

## Two things checked, not assumed

**`conversion` is a strict subset of `visit`.** Zero rows have `conversion=1` and `visit=0`: every
buyer visited first, which is the sensible read of "visit" as "came to the site" and "conversion"
as "bought while there." `src/data_loader.load_hillstrom()` asserts this on every load, since two
downstream design choices depend on it: treating `visit` as the well-powered outcome for
heterogeneity work, and treating `conversion`/`spend` as the same underlying rare event.

**`spend` may be top-coded at exactly $499.00.** Twelve customers show `spend == 499.00` to the
cent, and no customer shows anything between $482.31 (the next-highest value) and $499.00. That is
a gap, not a coincidence. Those twelve span very different purchase histories ($29.99 to $1,515.82), so
this is not twelve customers independently buying the same $499 item; it reads as a hard cap
applied when MineThatData published the file, most likely to suppress outliers or protect
high-value customers from re-identification. Whatever the reason, every spend-based estimate in
this project (the OLS and DML results on observed `spend`) estimates the effect on the reported,
capped outcome. The uncapped-spend effect is not identified without an explicit model of the cap,
so it should not be called a lower bound.

## Statistical power, and why heterogeneity work targets `visit`

| outcome | events | rate | powered for CATE? |
|---|---|---|---|
| `visit` | 9,394 | 14.68% | yes |
| `conversion` | 578 | 0.90% | no |
| `spend` (non-zero) | 578 | 0.90% | no |

`conversion` and `spend` are the same 578 customers (spend is non-zero exactly when conversion is
1, checked directly, zero exceptions either direction). Splitting 578 events across a `CausalForestDML`
with dozens of leaves produces a CATE surface with almost no signal per leaf. Average treatment
effects with confidence intervals are reported for all three outcomes; profile-level conditional
average effects
(`src/cate.py` onward) are estimated on `visit` only, and the report states this rather than showing
a heterogeneity plot for conversion that would be mostly noise.

## A floor worth noting

136 customers show `spend` of exactly $29.99, the single most common non-zero value by a wide
margin (the next most frequent value appears twice). $29.99 also happens to be the minimum of
`history` across the whole file. Read together, that looks like the store's cheapest product price
point, not a data artefact. Noted here in case it surfaces again during modelling, though it
doesn't change any of the decisions above.

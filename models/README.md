# Model artifacts

`causal_forest.joblib` is a serialized `EconML CausalForestDML` fitted on all 64,000 Hillstrom
rows for interactive profile scoring. It is a serving artifact, not the source of the published
evaluation claim: the heterogeneity and ranking claims are evaluated on the held-out 30% split.

The artifact is accepted only when its embedded data checksum, source fingerprint, feature schema
and dependency versions match the current checkout. Run `scripts/verify.ps1` after installing the
pinned requirements before loading it.

The model estimates a profile-level conditional average effect on the `visit` outcome for email
assignment versus no email. It does not estimate a person-level causal effect, delivery, opening,
reading or clicking effect. Regenerate it with `python -m src.pipeline`; do not hand-edit or rename
the artifact without rebuilding the provenance metadata.

Current artifact checksums:

- `causal_forest.joblib`: `c7a19de291af4b38cf22069df9430dab6413308bee6c38d6ddbad06fb1e3cd6e`
- `evaluation_artifacts.joblib`: `78cc167d871e27cc3b9e632c51e906e45dedef3bf97bdf09c89591d53ae94118`

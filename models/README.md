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

- `causal_forest.joblib`: `85b68bb5e567ad7e53a157875746ffb3404a515aec4f2517a73c5eff4045d133`
- `evaluation_artifacts.joblib`: `568bc9f1207b1faa982a7dc6d44d7f58968337a2ad34f09372472f17f12ef312`

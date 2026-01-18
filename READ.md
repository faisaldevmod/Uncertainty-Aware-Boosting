# Uncertainty-Aware Boosting (UAB)

Gradient Boosting algorithms such as XGBoost and LightGBM learn by repeatedly
correcting residual errors. While powerful, this mechanism hides a subtle
failure mode:

> Boosting mistakes uncertainty for error.

Large residuals early in training are not always signals of meaningful mistakes.
They often arise from ambiguous, noisy, or inherently uncertain samples.
Standard boosting treats these residuals as equally urgent, leading to:

- early fixation on hard samples
- correlated trees
- belief collapse
- overconfident ensembles

This project reframes boosting as **iterative belief correction** and proposes
a principled fix: modulating correction strength using model confidence.

---

## Core Insight

Learning should distinguish between:
- **confident mistakes** → deserve strong correction
- **uncertain predictions** → deserve gentle updates

Standard boosting does not make this distinction.

---

## Contributions

- Diagnose early hard-sample fixation in gradient boosting
- Visualize belief and residual dynamics across boosting rounds
- Introduce Uncertainty-Aware Boosting (UAB)
- Improve calibration and robustness without sacrificing accuracy

---

## Philosophy

This work treats learning not as pure loss minimization,
but as **belief shaping under uncertainty**.

Gradients are interpreted as corrective forces whose strength
should depend on *certainty*, not error magnitude alone.

---

## Status

- [x] Baseline XGBoost behavior
- [x] Custom uncertainty-aware objective
- [x] Training scripts
- [ ] LightGBM extension
- [ ] Full empirical study

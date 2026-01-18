## Why Boosting Overreacts

In gradient boosting, each new model is fitted to the negative
gradient of the loss.

Early in training:
- models are poorly calibrated
- uncertainty is high
- residuals exaggerate ambiguity

Because the same loss drives all corrections,
trees repeatedly focus on the same uncertain samples.

This creates correlated belief trajectories and collapses
the very diversity ensembles rely on.

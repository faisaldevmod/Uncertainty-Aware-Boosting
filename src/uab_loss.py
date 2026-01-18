import numpy as np

def uncertainty_aware_logloss(preds, dtrain):
    """
    Uncertainty-Aware Boosting (UAB) objective.

    Intuition:
    ---------
    Standard log loss treats all errors equally.
    But not all errors are equal in meaning.

    Confident mistakes:
        - model was sure and wrong
        - deserve strong correction

    Uncertain predictions:
        - model admits ignorance
        - deserve gentle correction

    We encode this distinction by scaling gradients
    according to prediction confidence.

    This preserves ensemble diversity and prevents
    early hard-sample fixation.
    """

    labels = dtrain.get_label()

    # Convert raw scores (logits) to probabilities
    probs = 1.0 / (1.0 + np.exp(-preds))

    # Standard log loss gradient and hessian
    grad = probs - labels
    hess = probs * (1.0 - probs)

    # Confidence measure:
    # Distance from maximum uncertainty (0.5)
    confidence = np.abs(probs - 0.5) * 2.0  # scaled to [0, 1]

    # Core idea:
    # Let belief modulate correction strength
    grad_scaled = confidence * grad
    hess_scaled = confidence * hess

    return grad_scaled, hess_scaled

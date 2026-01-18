import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# Load data
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    flip_y=0.15,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dval = xgb.DMatrix(X_val, label=y_val)

# Load models
bst_baseline = xgb.Booster()
bst_baseline.load_model("experiments/baseline.model")

bst_uab = xgb.Booster()
bst_uab.load_model("experiments/uab.model")

def confidence_entropy(preds):
    confidence = np.mean(np.maximum(preds, 1 - preds))
    entropy = -np.mean(
        preds * np.log(preds + 1e-8) + (1 - preds) * np.log(1 - preds + 1e-8)
    )
    return confidence, entropy

def track(model):
    confs, ents = [], []
    for i in range(1, model.best_iteration + 1):
        preds = model.predict(dval, iteration_range=(0, i))
        c, e = confidence_entropy(preds)
        confs.append(c)
        ents.append(e)
    return confs, ents

conf_base, ent_base = track(bst_baseline)
conf_uab, ent_uab = track(bst_uab)

# Plot confidence
plt.figure(figsize=(8,5))
plt.plot(conf_base, label="Standard Boosting")
plt.plot(conf_uab, label="UAB")
plt.xlabel("Boosting Round")
plt.ylabel("Average Confidence")
plt.title("Confidence Evolution")
plt.legend()
plt.grid()
plt.show()

# Plot entropy
plt.figure(figsize=(8,5))
plt.plot(ent_base, label="Standard Boosting")
plt.plot(ent_uab, label="UAB")
plt.xlabel("Boosting Round")
plt.ylabel("Entropy")
plt.title("Uncertainty Evolution")
plt.legend()
plt.grid()
plt.show()

# Final metrics
pred_base = bst_baseline.predict(dval)
pred_uab = bst_uab.predict(dval)

print("Baseline Log Loss:", log_loss(y_val, pred_base))
print("UAB Log Loss:", log_loss(y_val, pred_uab))

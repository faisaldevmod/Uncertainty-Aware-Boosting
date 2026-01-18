import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from uab_loss import uncertainty_aware_logloss

# Synthetic dataset with noise to expose failure mode
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    flip_y=0.15,  # label noise
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    "max_depth": 4,
    "eta": 0.1,
    "objective": "binary:logistic",
    "eval_metric": "logloss"
}

# Train with Uncertainty-Aware Boosting
bst_uab = xgb.train(
    params,
    dtrain,
    num_boost_round=200,
    obj=uncertainty_aware_logloss,
    evals=[(dval, "validation")],
    verbose_eval=20
)

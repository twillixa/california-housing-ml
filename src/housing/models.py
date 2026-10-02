"""The report's seven regressors, plus the corrected variants used in the audit.

Port of ``Main.ipynb`` cells 17-47. The original notebook kept only the final hyperparameters
inside one-candidate ``RandomizedSearchCV`` objects; fitting the same pipeline directly gives
identical predictions (refit on the full training set, same seeds).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from .data import CAT_FEATURES, NUM_FEATURES, SEED


# Report Table 2 (test R²), for comparisons; local reruns match except GB (+3e-4, float noise)
REPORT_TEST_R2 = {"Linear Regression (OLS)": 0.6180, "Ridge": 0.6180, "Lasso": 0.6179,
                  "Random Forest": 0.7063, "Gradient Boosting": 0.7071, "SVR": 0.5994,
                  "Neural Network (MLP)": 0.6796}


def preprocessor(num_features=NUM_FEATURES, cat_features=CAT_FEATURES) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), list(num_features)),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(cat_features)),
    ])


def _pipe(model, num_features=NUM_FEATURES) -> Pipeline:
    return Pipeline([("preprocessor", preprocessor(num_features)), ("model", model)])


def _scaled_target(model):
    """Fit on a standardised target and map predictions back to dollars."""
    return TransformedTargetRegressor(regressor=model, transformer=StandardScaler())


def report_models(num_features=NUM_FEATURES) -> dict:
    """The seven models exactly as configured in the report's final notebook."""
    alphas = {"model__alpha": [0.1, 1.0, 10.0, 100.0]}
    return {
        "Linear Regression (OLS)": _pipe(LinearRegression(), num_features),
        # inner cv=5 is unshuffled, as in the original (it selects alpha=10 / 100)
        "Ridge": GridSearchCV(_pipe(Ridge(), num_features), alphas, cv=5, scoring="r2"),
        "Lasso": GridSearchCV(_pipe(Lasso(max_iter=5000), num_features), alphas, cv=5, scoring="r2"),
        "Random Forest": _pipe(RandomForestRegressor(
            n_estimators=300, max_depth=20, min_samples_split=5, max_features=0.5,
            random_state=SEED, n_jobs=-1), num_features),
        "Gradient Boosting": _pipe(GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8,
            random_state=SEED), num_features),
        # the report text says epsilon=0.1; the notebook that produced the numbers uses 0.5
        "SVR": _pipe(SVR(kernel="linear", C=100.0, epsilon=0.5), num_features),
        "Neural Network (MLP)": _pipe(MLPRegressor(
            hidden_layer_sizes=(64, 32), activation="relu", alpha=0.0001,
            learning_rate_init=0.01, max_iter=500, random_state=SEED), num_features),
    }


def final_gradient_boosting(num_features=NUM_FEATURES) -> Pipeline:
    """The report's 'Final Model' (learning rate 0.025, depth 7)."""
    return _pipe(GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.025, max_depth=7, subsample=0.8,
        random_state=SEED), num_features)


def corrected_models(num_features=NUM_FEATURES) -> dict:
    """SVR and MLP refitted on a standardised target, with light retuning.

    Both were fit on raw dollar prices (~$200,000). For an RBF-kernel SVR the binding limit
    is ``C``: each dual coefficient is capped at C=100 while the target spans hundreds of
    thousands, so the fit is heavily over-regularised (R² 0.41), which is why the report kept
    a linear kernel. Changes here, in order of effect (see ``svr_mlp_ablation``):

    * SVR: standardised target + RBF kernel, C=10, epsilon=0.1 (in standard-deviation units).
    * MLP: standardised target, learning rate 0.01 → 0.001, early stopping.
    """
    models = report_models(num_features)
    models["SVR"] = _pipe(_scaled_target(SVR(kernel="rbf", C=10.0, epsilon=0.1)), num_features)
    models["Neural Network (MLP)"] = _pipe(_scaled_target(MLPRegressor(
        hidden_layer_sizes=(64, 32), activation="relu", alpha=0.0001,
        learning_rate_init=0.001, max_iter=500, early_stopping=True,
        random_state=SEED)), num_features)
    return models


def svr_mlp_ablation(num_features=NUM_FEATURES) -> dict:
    """Separate the effect of target scaling from the other configuration changes."""
    mlp = dict(hidden_layer_sizes=(64, 32), activation="relu", alpha=0.0001, max_iter=500,
               random_state=SEED)
    return {
        "SVR · report (linear, raw $)": _pipe(SVR(kernel="linear", C=100.0, epsilon=0.5), num_features),
        "SVR · linear, scaled target": _pipe(_scaled_target(SVR(kernel="linear", C=1.0, epsilon=0.1)), num_features),
        "SVR · RBF, raw $": _pipe(SVR(kernel="rbf", C=100.0, epsilon=0.5), num_features),
        "SVR · RBF, scaled target (corrected)": _pipe(_scaled_target(SVR(kernel="rbf", C=10.0, epsilon=0.1)), num_features),
        "MLP · report (raw $)": _pipe(MLPRegressor(learning_rate_init=0.01, **mlp), num_features),
        "MLP · scaled target only": _pipe(_scaled_target(MLPRegressor(learning_rate_init=0.01, **mlp)), num_features),
        "MLP · retuned, raw $": _pipe(MLPRegressor(learning_rate_init=0.001, early_stopping=True, **mlp), num_features),
        "MLP · scaled + retuned (corrected)": _pipe(_scaled_target(MLPRegressor(
            learning_rate_init=0.001, early_stopping=True, **mlp)), num_features),
    }


def evaluate(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    start = time.time()
    model.fit(X_train, y_train)
    seconds = time.time() - start
    pred_train, pred_test = model.predict(X_train), model.predict(X_test)
    return {"Model": name,
            "R2 Train": r2_score(y_train, pred_train),
            "R2 Test": r2_score(y_test, pred_test),
            "RMSE Test": float(np.sqrt(mean_squared_error(y_test, pred_test))),
            "MAE Test": mean_absolute_error(y_test, pred_test),
            "Train Time (s)": seconds}


def compare(models: dict, split) -> pd.DataFrame:
    rows = [evaluate(name, m, split.X_train, split.y_train, split.X_test, split.y_test)
            for name, m in models.items()]
    table = pd.DataFrame(rows).sort_values("R2 Test", ascending=False).reset_index(drop=True)
    table["Gap"] = table["R2 Train"] - table["R2 Test"]
    return table


def feature_names(pipeline: Pipeline) -> list[str]:
    pre = pipeline.named_steps["preprocessor"]
    ohe = pre.named_transformers_["cat"]
    return list(pre.transformers_[0][2]) + ohe.get_feature_names_out(CAT_FEATURES).tolist()


def importances(pipeline: Pipeline, X_test=None, y_test=None) -> pd.Series:
    """Impurity importance for trees, |coef| for linear models, permutation otherwise.

    The three measures are on different scales; compare rankings, not values, across models.
    """
    if isinstance(pipeline, GridSearchCV):
        pipeline = pipeline.best_estimator_
    model = pipeline.named_steps["model"]
    names = feature_names(pipeline)
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    else:
        if X_test is None or y_test is None:
            raise ValueError("permutation importance needs X_test and y_test")
        X = pipeline.named_steps["preprocessor"].transform(X_test)
        values = permutation_importance(model, X, y_test, n_repeats=10, random_state=SEED,
                                        n_jobs=-1).importances_mean
    return pd.Series(values, index=names).sort_values(ascending=False)


def cross_validated(model, X, y, cv="random", groups=None, n_splits: int = 5) -> dict:
    """Shuffled K-fold, or GroupKFold on spatial cells when ``cv="spatial"``.

    The CSV is ordered geographically, so an unshuffled KFold would silently be a spatial split.
    """
    if cv == "random":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    elif cv == "spatial":
        splitter = GroupKFold(n_splits=n_splits)
    else:
        raise ValueError(f"cv must be 'random' or 'spatial', got {cv!r}")
    scores = cross_validate(model, X, y, cv=splitter, groups=groups, n_jobs=-1,
                            scoring=("r2", "neg_root_mean_squared_error"))
    r2 = scores["test_r2"]
    return {"R2 mean": r2.mean(), "R2 sd": r2.std(ddof=1),
            "RMSE mean": -scores["test_neg_root_mean_squared_error"].mean(),
            "folds": n_splits, "fold R2": r2}


def paired_difference(a: dict, b: dict) -> dict:
    """Compare two ``cross_validated`` results fold by fold (same splitter, so folds match).

    Fold difficulty is shared by both models, so the per-fold difference is what matters,
    not each model's own spread. Reports the mean difference and a paired t statistic.
    """
    diff = np.asarray(a["fold R2"]) - np.asarray(b["fold R2"])
    t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
    return {"mean diff": diff.mean(), "t": t, "folds better": int((diff > 0).sum()),
            "folds": len(diff)}


def spatial_cells(df: pd.DataFrame, size_deg: float = 1.0) -> pd.Series:
    """Grid-cell id for spatial cross-validation (whole regions held out together)."""
    lat = (df["latitude"] // size_deg).astype(int).astype(str)
    lon = (df["longitude"] // size_deg).astype(int).astype(str)
    return lat + "_" + lon

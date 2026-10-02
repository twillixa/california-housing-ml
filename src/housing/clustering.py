"""KMeans segmentation of census blocks (report section "Unsupervised Learning").

Port of ``Main.ipynb`` cells 49-53. ``clip_ratios=True`` caps the household-ratio features at
their 1st/99th percentiles before scaling. Without it, a handful of blocks with extreme
ratios (up to 1,243 people per household: group quarters such as prisons or dormitories,
whose residents aren't counted as households) sit so far from everything else that KMeans
spends a whole cluster on them.
"""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .data import SEED

ECONOMIC = ["median_income", "median_house_value", "housing_median_age",
            "rooms_per_household", "population_per_household"]
WITH_COORDS = ["latitude", "longitude", *ECONOMIC]
RATIO_COLS = ["rooms_per_household", "population_per_household"]


def _matrix(df: pd.DataFrame, features, clip_ratios: bool):
    X = df[features].copy()
    if clip_ratios:
        for col in set(RATIO_COLS) & set(features):
            X[col] = X[col].clip(X[col].quantile(0.01), X[col].quantile(0.99))
    return StandardScaler().fit_transform(X)


def elbow(df: pd.DataFrame, features=ECONOMIC, k_range=range(2, 11),
          clip_ratios: bool = False) -> pd.Series:
    """Total within-cluster sum of squares (inertia) for each k."""
    X = _matrix(df, features, clip_ratios)
    return pd.Series({k: KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X).inertia_
                      for k in k_range}, name="TWSS")


def kmeans(df: pd.DataFrame, k: int, features=ECONOMIC, clip_ratios: bool = False) -> pd.Series:
    X = _matrix(df, features, clip_ratios)
    labels = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(X)
    return pd.Series(labels, index=df.index, name="cluster")


def profile(df: pd.DataFrame, labels: pd.Series, features=ECONOMIC) -> pd.DataFrame:
    """Mean of each feature per cluster plus the cluster size (report Table 1)."""
    table = df[features].groupby(labels).mean()
    table["blocks"] = labels.value_counts().sort_index()
    return table.sort_values("median_house_value", ascending=False)

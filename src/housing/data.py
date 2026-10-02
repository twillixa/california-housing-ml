"""Load and clean the California housing data.

Port of the group's ``Main.ipynb`` (cells 4-7 and 17). With the default arguments the output is
identical to what the report was built on. ``leak_free=True`` fits the two data-dependent
cleaning steps (median imputation and the 99th-percentile caps) on the training rows only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "california_housing.csv"

TARGET = "median_house_value"
PRICE_CAP = 500_000  # the census topcodes values at $500,001
CAPPED_COLS = ["total_rooms", "total_bedrooms", "population", "households"]
RATIOS = ["rooms_per_household", "bedrooms_per_room", "population_per_household"]
NUM_FEATURES = ["housing_median_age", "total_rooms", "total_bedrooms", "population",
                "households", "median_income", *RATIOS]
CAT_FEATURES = ["ocean_proximity"]
COORDS = ["latitude", "longitude"]
SEED = 42


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV)


def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Row filters that don't depend on other rows: topcoded prices and the 5 ISLAND blocks."""
    df = df[df[TARGET] < PRICE_CAP]
    return df[df["ocean_proximity"] != "ISLAND"].reset_index(drop=True)


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    return df


@dataclass
class Cleaner:
    """Median imputation of ``total_bedrooms`` + 99th-percentile caps on block totals."""
    bedrooms_median: float = 0.0
    caps: dict = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "Cleaner":
        self.bedrooms_median = df["total_bedrooms"].median()
        # caps are computed after imputation, as in the original notebook
        imputed = df.assign(total_bedrooms=df["total_bedrooms"].fillna(self.bedrooms_median))
        self.caps = {c: imputed[c].quantile(0.99) for c in CAPPED_COLS}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.assign(total_bedrooms=df["total_bedrooms"].fillna(self.bedrooms_median))
        df = add_ratios(df)  # ratios use the uncapped totals, as in the original
        for col, cap in self.caps.items():
            df[col] = df[col].clip(upper=cap)
        return df


def clean(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """The report's cleaned dataset (19,643 rows × 13 columns)."""
    df = filter_rows(load_raw() if df is None else df)
    return Cleaner().fit(df).transform(df)


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train: pd.DataFrame  # full cleaned rows, for plots and clustering
    test: pd.DataFrame


def split(leak_free: bool = False, features=None, test_size: float = 0.2) -> Split:
    """80/20 split with the report's seed. Row order is the same in both modes."""
    features = features or NUM_FEATURES + CAT_FEATURES
    rows = filter_rows(load_raw())
    idx_train, idx_test = train_test_split(rows.index, test_size=test_size, random_state=SEED)
    if leak_free:
        cleaner = Cleaner().fit(rows.loc[idx_train])
    else:
        cleaner = Cleaner().fit(rows)
    cleaned = cleaner.transform(rows)
    train, test = cleaned.loc[idx_train], cleaned.loc[idx_test]
    return Split(train[features], test[features], train[TARGET], test[TARGET], train, test)

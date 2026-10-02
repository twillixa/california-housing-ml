"""README / notebook figures in one consistent style."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURES = Path(__file__).resolve().parents[2] / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT = "#86b6ef"
NEUTRAL = "#b8b7b1"
BACKGROUND_DOTS = "#dcdbd6"
BLUE_RAMP = mpl.colors.LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 28, "axes.axisbelow": True, "font.size": 10.5, "lines.linewidth": 2,
        "legend.frameon": False, "figure.dpi": 110,
    })


def _subtitle(ax, text: str) -> None:
    ax.text(0, 1.015, text, transform=ax.transAxes, color=INK_2, fontsize=10, va="bottom")


def _dollars(ax, axis="y"):
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def save(fig, name: str) -> Path:
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    return path


def _map_axes(ax):
    ax.set_aspect(1 / np.cos(np.radians(36)))  # equirectangular correction at California's latitude
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def price_map(df: pd.DataFrame):
    """Each census block at its coordinates, coloured by median house value."""
    d = df.sort_values("median_house_value")
    fig, ax = plt.subplots(figsize=(6.6, 7))
    pts = ax.scatter(d["longitude"], d["latitude"], c=d["median_house_value"], cmap=BLUE_RAMP,
                     s=3, linewidths=0)
    _map_axes(ax)
    cbar = fig.colorbar(pts, ax=ax, fraction=0.035, pad=0.01)
    cbar.outline.set_visible(False)
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k"))
    cbar.set_label("Median house value", color=INK_2)
    for name, (lon, lat) in {"San Francisco": (-122.42, 37.77), "Los Angeles": (-118.24, 34.05),
                             "San Diego": (-117.16, 32.72), "Sacramento": (-121.49, 38.58)}.items():
        ax.annotate(name, (lon, lat), xytext=(8, 0), textcoords="offset points", fontsize=9,
                    color=INK, va="center")
    ax.set_title("Prices follow the coast")
    _subtitle(ax, f"{len(df):,} census block groups, 1990 · darker = more expensive")
    return fig


def income_vs_value(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hexbin(df["median_income"], df["median_house_value"], gridsize=60, cmap=BLUE_RAMP,
              mincnt=1, bins="log", linewidths=0)
    r = df["median_income"].corr(df["median_house_value"])
    ax.set_xlabel("Median household income (tens of thousands of $)")
    ax.set_ylabel("Median house value")
    _dollars(ax)
    ax.set_title(f"Income is the strongest single signal (r = {r:.2f})")
    _subtitle(ax, "Density of census blocks · darker = more blocks")
    return fig


def ocean_proximity(df: pd.DataFrame):
    med = df.groupby("ocean_proximity")["median_house_value"].median().sort_values()
    counts = df["ocean_proximity"].value_counts()
    labels = {"<1H OCEAN": "< 1 hour from ocean", "INLAND": "Inland", "NEAR BAY": "Near bay",
              "NEAR OCEAN": "Near ocean"}
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    colors = [NEUTRAL if c == "INLAND" else BLUE for c in med.index]
    ax.barh([labels[c] for c in med.index], med.values, height=0.55, color=colors)
    for y, (cat, v) in enumerate(med.items()):
        ax.text(v + 4000, y, f"${v / 1000:,.0f}k  ·  {counts[cat]:,} blocks", va="center",
                fontsize=9, color=INK_2)
    ax.set_xlim(0, med.max() * 1.45)
    _dollars(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Median of block median house values")
    ax.set_title("Inland blocks sell for about half")
    _subtitle(ax, "The coastal categories barely differ from each other")
    return fig


def model_comparison(cv: pd.DataFrame, report: pd.Series):
    """Bars: corrected 5-fold CV R² (± sd). Ticks: the report's single-split test R²."""
    cv = cv.sort_values("R2 mean")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    y = np.arange(len(cv))
    flexible = cv["R2 mean"] > 0.65
    ax.barh(y, cv["R2 mean"], height=0.55, color=[BLUE if f else BLUE_LIGHT for f in flexible])
    ax.errorbar(cv["R2 mean"], y, xerr=cv["R2 sd"], fmt="none", ecolor=INK, elinewidth=1.2,
                capsize=3)
    ax.scatter(report.reindex(cv.index), y, marker="|", s=220, linewidths=2.4, color=ORANGE,
               zorder=3, label="report (single split)")
    for i, (name, row) in enumerate(cv.iterrows()):
        ax.text(row["R2 mean"] + row["R2 sd"] + 0.012, i, f"{row['R2 mean']:.3f}", va="center",
                fontsize=9, color=INK_2)
    ax.set_yticks(y, cv.index)
    ax.set_xlim(0.4, 0.8)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("R² (share of price variance explained)")
    ax.legend(loc="lower right")
    ax.set_title("Properly configured, the flexible models land near 0.70")
    _subtitle(ax, "Bars: 5-fold CV mean ± 1 sd, SVR/MLP corrected · ticks: the report's test R²")
    return fig


def importances(imp: pd.Series, top: int = 8, title="Income dominates, then inland vs coast"):
    labels = {"median_income": "Median income", "ocean_proximity_INLAND": "Inland",
              "population_per_household": "People per household",
              "housing_median_age": "Housing age", "bedrooms_per_room": "Bedrooms per room",
              "rooms_per_household": "Rooms per household", "latitude": "Latitude",
              "longitude": "Longitude", "total_bedrooms": "Total bedrooms",
              "total_rooms": "Total rooms", "households": "Households", "population": "Population"}
    imp = imp.head(top).sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    colors = [BLUE if i == len(imp) - 1 else BLUE_LIGHT for i in range(len(imp))]
    ax.barh([labels.get(c, c) for c in imp.index], imp.values, height=0.6, color=colors)
    for y, v in enumerate(imp.values):
        ax.text(v + imp.max() * 0.01, y, f"{v:.0%}", va="center", fontsize=9, color=INK_2)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Share of gradient-boosting feature importance")
    ax.set_title(title)
    _subtitle(ax, f"Final gradient boosting model · top {top} features")
    return fig


def location_gain(results: pd.DataFrame, designs: dict):
    """Grouped bars per validation design, with and without coordinates (± 1 sd)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    width = 0.34
    y = np.arange(len(designs))
    for offset, (label, color) in zip((width / 2, -width / 2),
                                      [("Without coordinates", BLUE_LIGHT),
                                       ("With latitude / longitude", BLUE)]):
        vals = results.loc[label, list(designs)].to_numpy(float)
        sds = results.loc[label, [f"{d}_sd" for d in designs]].to_numpy(float)
        ax.barh(y + offset, vals, height=width - 0.04, color=color, label=label,
                xerr=sds, error_kw={"ecolor": INK, "elinewidth": 1.1, "capsize": 3})
        for yy, v, sd in zip(y + offset, vals, sds):
            ax.text(v + sd + 0.01, yy, f"{v:.2f}", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(y, list(designs.values()))
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("R² (gradient boosting, mean ± 1 sd over 5 folds)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, reverse=True)
    ax.set_title("Coordinates help near known blocks, not in new regions")
    _subtitle(ax, "Spatial CV holds out whole grid cells, so most test blocks are far from any training block")
    return fig


def cluster_maps(df: pd.DataFrame, labels: pd.Series, names: dict):
    """Small multiples: each segment highlighted on the full map."""
    order = (df.groupby(labels)["median_house_value"].mean().sort_values(ascending=False).index)
    fig, axes = plt.subplots(1, len(order), figsize=(3.2 * len(order), 4.2))
    for ax, cl in zip(axes, order):
        mask = labels == cl
        ax.scatter(df["longitude"], df["latitude"], s=1.2, color=BACKGROUND_DOTS, linewidths=0)
        ax.scatter(df.loc[mask, "longitude"], df.loc[mask, "latitude"], s=1.6, color=BLUE,
                   linewidths=0)
        _map_axes(ax)
        sub = df[mask]
        ax.set_title(names[cl], fontsize=11, pad=18, loc="center")
        ax.text(0.5, 1.0, f"${sub['median_house_value'].mean() / 1000:,.0f}k · income "
                f"{sub['median_income'].mean():.1f} · {mask.sum():,} blocks",
                transform=ax.transAxes, ha="center", fontsize=8.5, color=INK_2)
    fig.suptitle("Four housing segments once outlier blocks are clipped", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig

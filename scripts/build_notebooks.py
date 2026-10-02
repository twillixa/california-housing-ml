"""Generate the project notebooks.

Run from the repo root, then execute them:
    python scripts/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks"
REPO = "twillixa/california-housing-ml"

SETUP = '''import sys, os
if "google.colab" in sys.modules:  # running on Colab: fetch the repo first
    !git clone -q https://github.com/twillixa/california-housing-ml
    %cd california-housing-ml/notebooks
sys.path.insert(0, os.path.abspath("../src"))

import warnings
os.environ["PYTHONWARNINGS"] = "ignore"  # parallel CV workers don't inherit warnings filters
warnings.filterwarnings("ignore", message=".*encountered in matmul")  # spurious on macOS Accelerate
warnings.filterwarnings("ignore", category=UserWarning)
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from housing import plots
plots.use_style()
pd.set_option("display.precision", 4)
pd.set_option("display.width", 160)'''


def badge(name):
    return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{name})")


def notebook(name, cells):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
                for kind, src in cells]
    nbf.write(nb, OUT / name)


md, code = "md", "code"

notebook("01_data_and_eda.ipynb", [
    (md, f"""# 1 · Data and exploratory analysis
{badge("01_data_and_eda.ipynb")}

The California Housing dataset: one row per **census block group** from the 1990 U.S. Census
(Pace & Barry, 1997; Kaggle copy with the `ocean_proximity` column). The target is the block's
`median_house_value`. These are 1990 prices, so the project benchmarks models and drivers, not today's market."""),
    (code, SETUP + "\nfrom housing.data import load_raw, filter_rows, Cleaner, clean, CAPPED_COLS, NUM_FEATURES"),
    (md, """## Cleaning (report section "Data Cleaning and Feature Engineering")
1. Drop blocks at the **$500,001 price ceiling**: the census topcodes them, so their true value is unknown.
2. Drop the 5 **ISLAND** blocks, too rare to learn from.
3. Impute the 207 missing `total_bedrooms` with the median.
4. Add three ratios: rooms per household, bedrooms per room, people per household.
5. Cap the four block totals at their 99th percentile."""),
    (code, '''raw = load_raw()
rows = filter_rows(raw)
cleaner = Cleaner().fit(rows)
df = clean()
print(f"raw {raw.shape} → removed {(raw.median_house_value >= 500_000).sum()} topcoded + "
      f"{(raw.ocean_proximity == 'ISLAND').sum()} ISLAND → cleaned {df.shape}")
print(f"bedrooms median {cleaner.bedrooms_median:.0f}; caps:", {c: round(v) for c, v in cleaner.caps.items()})
df.describe().T[["mean", "50%", "min", "max"]]'''),
    (md, "## Where the expensive blocks are"),
    (code, '''fig = plots.price_map(df); plots.save(fig, "price_map");'''),
    (code, '''fig = plots.ocean_proximity(df); plots.save(fig, "ocean_proximity")
df.groupby("ocean_proximity")["median_house_value"].describe()[["count", "50%", "mean"]]'''),
    (md, "## What moves with price"),
    (code, '''fig = plots.income_vs_value(df); plots.save(fig, "income_vs_value")
df[NUM_FEATURES + ["median_house_value"]].corr()["median_house_value"].drop("median_house_value").sort_values()'''),
    (md, """Income is the only feature with a strong linear correlation (r ≈ 0.65). Block totals (rooms, people,
households) mostly measure block *size* and are highly collinear with each other (r > 0.85), which is why the
ratio features were engineered. Location matters a lot (map, inland vs coast) but enters the report's models only
through the 4-level `ocean_proximity` category, not the coordinates. Notebook 3 tests that choice."""),
])

notebook("02_models_report.ipynb", [
    (md, f"""# 2 · The report's seven models, reproduced
{badge("02_models_report.ipynb")}

Same features (9 numeric + one-hot `ocean_proximity`), the same 80/20 split (seed 42), the same final hyperparameters.
Every pipeline standardises the numeric features."""),
    (code, SETUP + "\nfrom housing.data import split\nfrom housing import models as m\ns = split()\nprint(f'train {len(s.X_train):,} · test {len(s.X_test):,}')"),
    (code, '''table = m.compare(m.report_models(), s)
table.round(4).to_csv("../reports/model_comparison_report.csv", index=False)
table.round(4)'''),
    (md, """Report Table 2 for comparison: OLS 0.618 · Ridge 0.618 · Lasso 0.6179 · RF 0.7063 · GB 0.7071 · SVR 0.5994 ·
MLP 0.6796. Six models match to four decimals. Gradient boosting differs by 0.0003 (floating-point differences
between Colab and this machine)."""),
    (md, "## Final model (report section \"Final Model\")\nGradient boosting with learning rate 0.025 and depth 7."),
    (code, '''final = m.final_gradient_boosting()
result = m.evaluate("Final gradient boosting", final, s.X_train, s.y_train, s.X_test, s.y_test)
print({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()}, "(report: R² 0.7078)")
imp = m.importances(final)
fig = plots.importances(imp); plots.save(fig, "feature_importance")
imp.round(3)'''),
    (md, """Median income carries about half of the importance. The `INLAND` flag comes next, then people per household.
The finer coastal categories add almost nothing once inland vs coast is known."""),
])

notebook("03_audit_and_improvements.ipynb", [
    (md, f"""# 3 · Re-check and improvements
{badge("03_audit_and_improvements.ipynb")}

Four questions the report leaves open:
1. Were SVR and the neural network configured fairly?
2. Is gradient boosting really better than the random forest, or is that one lucky split?
3. Does the cleaning leak test information?
4. The maps show location drives price. What happens if the models get the coordinates?"""),
    (code, SETUP + '''
import numpy as np
from housing.data import split, clean, NUM_FEATURES, CAT_FEATURES, COORDS
from housing import models as m
s = split()
df = clean()
X, y = df[NUM_FEATURES + CAT_FEATURES], df["median_house_value"]'''),
    (md, """## 1 · SVR and the MLP
Both were fit on raw prices (≈ $200,000). For an RBF-kernel SVR the binding limit is `C`. Each support vector's
weight is capped at `C = 100`, tiny next to a target that spans hundreds of thousands of dollars, so the fit is
heavily over-regularised. That is why the report's search found "non-linear kernels R² < 0.50" and kept a linear kernel.
The ablation below separates target scaling from the other changes:"""),
    (code, '''ablation = m.compare(m.svr_mlp_ablation(), s)[["Model", "R2 Test", "R2 Train"]]
ablation.round(4)'''),
    (md, """- **SVR:** the RBF kernel only works on a standardised target (0.41 → 0.70). On the linear kernel, scaling adds only
  ≈ 0.01. So the report's real loss was being steered away from the non-linear kernel.
- **MLP:** scaling the target alone adds ≈ 0.012 (0.680 → 0.692). A smaller learning rate with early stopping adds the
  rest (→ 0.705). Retuning without scaling makes it *worse*, so the two changes work together."""),
    (code, '''report = m.compare(m.report_models(), s).set_index("Model")
corrected = m.compare(m.corrected_models(), s).set_index("Model")
pd.DataFrame({"report Table 2": pd.Series(m.REPORT_TEST_R2), "rerun (report config)": report["R2 Test"],
              "corrected": corrected["R2 Test"], "corrected train-test gap": corrected["Gap"]}
             ).sort_values("corrected", ascending=False).round(4)'''),
    (md, "## 2 · One split or five?\nThe report ranks models on a single 80/20 split. Five-fold cross-validation (shuffled; the CSV is ordered geographically) shows how much the ranking can move:"),
    (code, '''cv = {name: m.cross_validated(model, X, y) for name, model in m.corrected_models().items()}
table = pd.DataFrame({name: {"R2 mean": r["R2 mean"], "R2 sd": r["R2 sd"], "RMSE mean": r["RMSE mean"]}
                      for name, r in cv.items()}).T.astype(float)
out = table.assign(report_table2_R2=pd.Series(m.REPORT_TEST_R2)).sort_values("R2 mean", ascending=False)
out.round(4).to_csv("../reports/model_comparison.csv")
fig = plots.model_comparison(table, pd.Series(m.REPORT_TEST_R2)); plots.save(fig, "model_comparison")
out.round(4)'''),
    (md, "Models share the same folds, so the right comparison is **paired**: the per-fold difference, not each model's own spread."),
    (code, '''pairs = [("Random Forest", "Gradient Boosting"), ("Random Forest", "Neural Network (MLP)"),
         ("Gradient Boosting", "Neural Network (MLP)"), ("Random Forest", "SVR"), ("Gradient Boosting", "SVR")]
pd.DataFrame({f"{a} − {b}": m.paired_difference(cv[a], cv[b]) for a, b in pairs}).T.round(4)'''),
    (md, """- **Random forest, gradient boosting and the MLP are tied.** Their paired differences are ≤ 0.004 R² with t < 2, so the data
  does not support choosing gradient boosting over the random forest.
- **SVR is about 0.01 behind**, and that gap is real: it loses in all five folds (t ≈ 6–7).
- The linear models trail by about 0.09.

The report also preferred gradient boosting for its smaller train-test gap (0.06 vs 0.23). A gap measures how much a
model memorises. Generalisation is the test score itself, and that is the same for both.

*Note:* these CV runs use the report's cleaning, whose imputation median and caps see every row. Section 3 shows
that this leak doesn't matter here."""),
    (md, "## 3 · Cleaning fitted on the training rows only\nThe median imputation and the 99th-percentile caps used all rows, test rows included:"),
    (code, '''leak_free = split(leak_free=True)
gb = m.report_models()["Gradient Boosting"]
r = m.evaluate("GB", gb, leak_free.X_train, leak_free.y_train, leak_free.X_test, leak_free.y_test)
print(f"Gradient boosting test R²: {report.loc['Gradient Boosting', 'R2 Test']:.4f} (rerun, all-row cleaning) "
      f"vs {r['R2 Test']:.4f} (leak-free)")'''),
    (md, "A technically valid point, but practically irrelevant here."),
    (md, """## 4 · Giving the models the coordinates
Three ways to score it:
- **random CV**: test blocks are scattered among training blocks, so their neighbours are in the training set.
- **spatial CV, 1° cells**: whole 1° × 1° grid cells (≈ 110 km) are held out.
- **spatial CV, 2° cells**: larger regions are held out, so far fewer test blocks have a neighbour just across a cell border."""),
    (code, '''features_geo = NUM_FEATURES + COORDS
X_geo = df[features_geo + CAT_FEATURES]
designs = {"random": "Random 5-fold CV", "spatial1": "Spatial CV, 1° cells", "spatial2": "Spatial CV, 2° cells"}
runs, rows = {}, {}
for label, Xs, feats in [("Without coordinates", X, NUM_FEATURES), ("With latitude / longitude", X_geo, features_geo)]:
    runs[label] = {
        "random": m.cross_validated(m.final_gradient_boosting(feats), Xs, y),
        "spatial1": m.cross_validated(m.final_gradient_boosting(feats), Xs, y, cv="spatial", groups=m.spatial_cells(df, 1.0)),
        "spatial2": m.cross_validated(m.final_gradient_boosting(feats), Xs, y, cv="spatial", groups=m.spatial_cells(df, 2.0)),
    }
    rows[label] = {k: v for d, r in runs[label].items() for k, v in ((d, r["R2 mean"]), (f"{d}_sd", r["R2 sd"]))}
location = pd.DataFrame(rows).T
location.round(4).to_csv("../reports/location_gain.csv")
fig = plots.location_gain(location, designs); plots.save(fig, "location_gain")
gains = pd.DataFrame({designs[d]: runs["With latitude / longitude"][d]["fold R2"] - runs["Without coordinates"][d]["fold R2"]
                      for d in designs}, index=[f"fold {i + 1}" for i in range(5)])
print("Per-fold gain from adding coordinates:")
gains.round(3)'''),
    (code, '''s_geo = split(features=features_geo + CAT_FEATURES)
geo_model = m.final_gradient_boosting(features_geo)
geo = m.evaluate("GB + coordinates", geo_model, s_geo.X_train, s_geo.y_train, s_geo.X_test, s_geo.y_test)
print(f"Test R² {geo['R2 Test']:.4f} · RMSE ${geo['RMSE Test']:,.0f} · MAE ${geo['MAE Test']:,.0f}")
pred = s_geo.test[["longitude", "latitude", "ocean_proximity", "median_income", "median_house_value"]].copy()
pred["predicted_value"] = geo_model.predict(s_geo.X_test).round(0)
pred["error"] = pred["predicted_value"] - pred["median_house_value"]
pred.to_csv("../reports/test_predictions.csv", index=False)
pred.head()'''),
    (md, """With coordinates, the same gradient boosting model explains about 80% of price variance instead of 70% on blocks
*mixed in with* its training data. Its RMSE falls by about $9,600 (from $52,300 to $42,700). That is the realistic
setting for filling gaps in a known market.

The gain does **not** transfer reliably to new regions. Holding out 1° cells it averages +0.07, but varies from −0.01
to +0.16 across folds. Holding out 2° regions it is essentially zero. Most of what latitude and longitude add is
"nearby blocks cost about the same", i.e. interpolation, not a portable rule about California.

## Audit log

| Issue in the original | Effect | Fix |
|---|---|---|
| SVR and MLP fit on raw dollar prices | RBF-SVR R² 0.41, so a linear kernel was kept (0.60); MLP 0.68 | standardised target (+ RBF for SVR, smaller learning rate + early stopping for the MLP): SVR 0.70, MLP 0.705 |
| Model choice on one 80/20 split | GB "beats" RF by ΔR² 0.0008 | 5-fold paired CV: RF, GB and MLP tie; SVR ≈ 0.01 behind |
| Coordinates excluded from the models | location only via a 4-level category | + lat/lon: R² 0.70 → 0.80 near known blocks; little or no gain in unseen regions |
| Cleaning statistics fitted on all rows | minor train/test leakage | `split(leak_free=True)`: no measurable change |
| Ratio outliers (group-quarters blocks) not handled before KMeans | "cluster 3" is 3 blocks averaging 782 people per household | clip ratios: notebook 4 |
| Report text vs code | "six models" (seven listed), "stratified split" (random), SVR ε 0.1 (code: 0.5); Fig. 16 SVR bars ≈ 0.40 ≠ Table 2 | documented; code values used |"""),
])

notebook("04_clustering.ipynb", [
    (md, f"""# 4 · Housing segments (KMeans)
{badge("04_clustering.ipynb")}

The report clusters blocks on income, house value, housing age, rooms per household and people per household
(standardised), with k = 4, after a first attempt that included the coordinates (k = 6)."""),
    (code, SETUP + "\nfrom housing.data import clean\nfrom housing import clustering as c\ndf = clean()"),
    (md, "## The report's clusters"),
    (code, '''report_labels = c.kmeans(df, 4)
c.profile(df, report_labels).round(2)'''),
    (md, """Identical to report Table 1, including **"cluster 3": 3 blocks averaging 782 people per household**. These are real
group-quarters blocks (prisons, dormitories, barracks) whose residents aren't counted as households, so the ratio
explodes. Because the ratio features were never capped, KMeans spends a whole cluster on them, and the report is left
with three usable segments."""),
    (code, '''df[["population_per_household", "rooms_per_household"]].describe(percentiles=[.01, .5, .99]).round(2)'''),
    (md, "## Clipping the ratios at their 1st / 99th percentiles"),
    (code, '''pd.DataFrame({"report": c.elbow(df), "clipped": c.elbow(df, clip_ratios=True)}).round(0)'''),
    (code, '''labels = c.kmeans(df, 4, clip_ratios=True)
profile = c.profile(df, labels)
names = dict(zip(profile.index, ["Affluent newer suburbs", "Established older neighbourhoods",
                                 "Newer inland", "Crowded lower-income"]))
profile.insert(0, "segment", profile.index.map(names))
inland = df.groupby(labels)["ocean_proximity"].apply(lambda s: (s == "INLAND").mean())
profile["inland share"] = inland
profile.round(2).to_csv("../reports/cluster_profiles.csv")
fig = plots.cluster_maps(df, labels, names); plots.save(fig, "clusters");
profile.round(2)'''),
    (md, """Four segments of about 4,200 to 6,100 blocks each:
- **Affluent newer suburbs**: highest income and value, larger homes.
- **Established older neighbourhoods**: older stock, many near the bay or ocean, mid-high values at mid income.
- **Newer inland**: recent construction, mostly inland, low values.
- **Crowded lower-income**: lowest income, about 4.5 people per household.

The k = 4 choice is still a judgement call: the elbow curve has no sharp bend in either version."""),
])
print(f"wrote 4 notebooks to {OUT}")

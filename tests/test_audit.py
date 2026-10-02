"""Corrected / extended results reported in the README and notebook 03."""
import pytest
from sklearn.model_selection import GroupKFold

from housing import clustering as c
from housing import models as m
from housing.data import CAT_FEATURES, COORDS, NUM_FEATURES, Cleaner, clean, filter_rows, load_raw, split


@pytest.fixture(scope="module")
def s():
    return split()


@pytest.fixture(scope="module")
def df():
    return clean()


@pytest.fixture(scope="module")
def ablation(s):
    return m.compare(m.svr_mlp_ablation(), s).set_index("Model")["R2 Test"]


def test_rbf_svr_needs_a_scaled_target(ablation):
    assert ablation["SVR · RBF, raw $"] == pytest.approx(0.4079, abs=0.002)
    assert ablation["SVR · RBF, scaled target (corrected)"] == pytest.approx(0.6961, abs=0.002)
    # on the linear kernel, scaling alone adds only about 0.01
    gain = ablation["SVR · linear, scaled target"] - ablation["SVR · report (linear, raw $)"]
    assert 0 < gain < 0.02


def test_mlp_gain_splits_between_scaling_and_retuning(ablation):
    report = ablation["MLP · report (raw $)"]
    assert report == pytest.approx(0.6796, abs=0.0005)
    assert ablation["MLP · scaled target only"] == pytest.approx(0.6919, abs=0.002)
    assert ablation["MLP · retuned, raw $"] < report  # retuning alone hurts
    assert ablation["MLP · scaled + retuned (corrected)"] == pytest.approx(0.7050, abs=0.002)


def test_leak_free_cleaning_fits_on_train_only_and_changes_almost_nothing(s):
    rows = filter_rows(load_raw())
    all_rows, train_only = Cleaner().fit(rows), Cleaner().fit(rows.loc[s.X_train.index])
    assert all_rows.caps != train_only.caps
    sl = split(leak_free=True)
    assert (sl.X_test.index == s.X_test.index).all()
    gb = m.evaluate("gb", m.report_models()["Gradient Boosting"],
                    sl.X_train, sl.y_train, sl.X_test, sl.y_test)
    assert gb["R2 Test"] == pytest.approx(0.7071, abs=0.001)


def test_coordinates_add_ten_points_on_a_random_split(s):
    features = NUM_FEATURES + COORDS
    s2 = split(features=features + CAT_FEATURES)
    base = m.evaluate("gb", m.final_gradient_boosting(), s.X_train, s.y_train, s.X_test, s.y_test)
    geo = m.evaluate("gb+", m.final_gradient_boosting(features), s2.X_train, s2.y_train,
                     s2.X_test, s2.y_test)
    assert geo["R2 Test"] == pytest.approx(0.8053, abs=0.002)
    assert geo["RMSE Test"] == pytest.approx(42675, abs=50)
    assert geo["R2 Test"] - base["R2 Test"] > 0.09


@pytest.fixture(scope="module")
def cv(df):
    X, y = df[NUM_FEATURES + CAT_FEATURES], df["median_house_value"]
    models = m.corrected_models()
    return {name: m.cross_validated(models[name], X, y)
            for name in ("Random Forest", "Gradient Boosting", "Neural Network (MLP)", "SVR")}


def test_rf_gb_mlp_tie_under_paired_cv(cv):
    for a, b in [("Random Forest", "Gradient Boosting"), ("Random Forest", "Neural Network (MLP)")]:
        diff = m.paired_difference(cv[a], cv[b])
        assert abs(diff["mean diff"]) < 0.005 and abs(diff["t"]) < 2.776  # t(4) 5% critical value


def test_svr_is_genuinely_behind(cv):
    diff = m.paired_difference(cv["Random Forest"], cv["SVR"])
    assert diff["folds better"] == 5 and diff["t"] > 2.776
    assert diff["mean diff"] == pytest.approx(0.0105, abs=0.002)


def test_spatial_cv_never_splits_a_cell(df):
    cells = m.spatial_cells(df)
    assert cells.nunique() == 55
    for train, test in GroupKFold(n_splits=5).split(df, groups=cells):
        assert set(cells.iloc[train]).isdisjoint(cells.iloc[test])


def test_report_clusters_reproduced(df):
    sizes = c.profile(df, c.kmeans(df, 4))["blocks"].tolist()
    assert sizes == [4931, 3, 7901, 6808]  # report Table 1, sorted by house value


def test_clipping_ratios_removes_the_outlier_cluster(df):
    sizes = c.profile(df, c.kmeans(df, 4, clip_ratios=True))["blocks"]
    assert sizes.min() > 4000
    assert sizes.sum() == len(df)

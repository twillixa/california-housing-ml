"""The default pipeline must reproduce the report (Table 2 and the cleaning log)."""
import pytest

from housing import models as m
from housing.data import CAPPED_COLS, Cleaner, clean, filter_rows, load_raw, split


def test_raw_data():
    raw = load_raw()
    assert raw.shape == (20640, 10)
    assert raw["total_bedrooms"].isna().sum() == 207


def test_cleaning_log_matches_notebook():
    raw = load_raw()
    rows = filter_rows(raw)
    assert len(raw) - len(rows) == 992 + 5  # topcoded prices + ISLAND blocks
    cleaner = Cleaner().fit(rows)
    assert cleaner.bedrooms_median == 436.0
    assert {c: round(cleaner.caps[c]) for c in CAPPED_COLS} == {
        "total_rooms": 11211, "total_bedrooms": 2241, "population": 5866, "households": 1994}
    df = clean()
    assert df.shape == (19643, 13)
    assert df.isna().sum().sum() == 0


def test_split_matches_notebook():
    s = split()
    assert (len(s.X_train), len(s.X_test)) == (15714, 3929)


def test_leak_free_split_keeps_rows_and_changes_only_fitted_steps():
    a, b = split(), split(leak_free=True)
    assert (a.X_test.index == b.X_test.index).all()
    assert (a.y_test == b.y_test).all()


@pytest.fixture(scope="module")
def table():
    return m.compare(m.report_models(), split()).set_index("Model")


@pytest.mark.parametrize("model, r2, rmse", [
    ("Linear Regression (OLS)", 0.6180, 59772),
    ("Ridge", 0.6180, 59772),
    ("Lasso", 0.6179, 59784),
    ("Random Forest", 0.7063, 52411),
    ("SVR", 0.5994, 61209),
    ("Neural Network (MLP)", 0.6796, 54747),
])
def test_table2_exact(table, model, r2, rmse):
    assert table.loc[model, "R2 Test"] == pytest.approx(r2, abs=0.00005)
    assert table.loc[model, "RMSE Test"] == pytest.approx(rmse, abs=1)


def test_gradient_boosting_within_float_noise(table):
    # Colab vs local floating-point differences move GB by 3e-4 (0.7074 here, 0.7071 in the report)
    assert table.loc["Gradient Boosting", "R2 Test"] == pytest.approx(0.7071, abs=0.001)
    assert table.loc["Gradient Boosting", "Gap"] == pytest.approx(0.0643, abs=0.001)
    assert table.loc["Random Forest", "R2 Train"] == pytest.approx(0.9341, abs=0.00005)


def test_final_gradient_boosting():
    s = split()
    result = m.evaluate("final", m.final_gradient_boosting(), s.X_train, s.y_train, s.X_test, s.y_test)
    assert result["R2 Test"] == pytest.approx(0.7078, abs=0.001)

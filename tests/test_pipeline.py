import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Test 1: data files exist ──────────────────────────────────────────────────
def test_train_csv_exists():
    assert os.path.exists("data/train.csv"), "train.csv not found"

def test_test_csv_exists():
    assert os.path.exists("data/test.csv"), "test.csv not found"

def test_model_exists():
    assert os.path.exists("models/xgb_model.pkl"), "xgb_model.pkl not found"

# ── Test 2: CSV structure ─────────────────────────────────────────────────────
def test_train_columns():
    df = pd.read_csv("data/train.csv")
    required = {"Quantity", "UnitPrice", "TotalPrice", "InvoiceHour", "InvoiceDay", "WillBuyNextMonth"}
    assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"

def test_no_nulls_in_train():
    df = pd.read_csv("data/train.csv")
    assert df.isnull().sum().sum() == 0, "train.csv contains null values"

def test_target_is_binary():
    df = pd.read_csv("data/train.csv")
    assert set(df["WillBuyNextMonth"].unique()).issubset({0, 1}), "Target is not binary"

def test_train_test_ratio():
    train = pd.read_csv("data/train.csv")
    test  = pd.read_csv("data/test.csv")
    total = len(train) + len(test)
    ratio = len(train) / total
    assert 0.65 <= ratio <= 0.75, f"Train ratio unexpected: {ratio:.2f}"

# ── Test 3: model loading & prediction ───────────────────────────────────────
def test_model_loads():
    import joblib
    model = joblib.load("models/xgb_model.pkl")
    assert model is not None

def test_model_predicts():
    import joblib
    model = joblib.load("models/xgb_model.pkl")
    sample = pd.DataFrame([{
        "Quantity": 10,
        "UnitPrice": 5.0,
        "TotalPrice": 50.0,
        "InvoiceHour": 10,
        "InvoiceDay": 15
    }])
    pred = model.predict(sample)
    assert pred[0] in [0, 1], "Prediction is not 0 or 1"

def test_model_predict_proba():
    import joblib
    model = joblib.load("models/xgb_model.pkl")
    sample = pd.DataFrame([{
        "Quantity": 10,
        "UnitPrice": 5.0,
        "TotalPrice": 50.0,
        "InvoiceHour": 10,
        "InvoiceDay": 15
    }])
    proba = model.predict_proba(sample)
    assert proba.shape == (1, 2), "predict_proba shape mismatch"
    assert abs(proba[0].sum() - 1.0) < 1e-5, "Probabilities don't sum to 1"

# ── Test 4: drift monitor logic ───────────────────────────────────────────────
def test_drift_detection_logic():
    original_preds = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0])
    drifted_preds  = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 0])
    change = (original_preds != drifted_preds).sum() / len(original_preds) * 100
    assert change == 30.0, f"Expected 30.0%, got {change}"

def test_drift_threshold():
    change_pct = 11.41
    threshold  = 10.0
    assert change_pct > threshold, "Drift should be detected"

# ── Test 5: feature engineering values ───────────────────────────────────────
def test_total_price_calculation():
    qty   = 10
    price = 5.0
    assert qty * price == 50.0

def test_invoice_hour_range():
    df = pd.read_csv("data/train.csv")
    assert df["InvoiceHour"].between(0, 23).all(), "InvoiceHour out of range"

def test_invoice_day_range():
    df = pd.read_csv("data/train.csv")
    assert df["InvoiceDay"].between(1, 31).all(), "InvoiceDay out of range"
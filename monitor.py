"""
monitor.py - Drift simulation and detection for ML model monitoring (Level 2)
"""

import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime


FEATURE_COLS    = ["Quantity", "UnitPrice", "TotalPrice", "InvoiceHour", "InvoiceDay"]
TARGET_COL      = "WillBuyNextMonth"
DRIFT_THRESHOLD = 10.0   # percent


def load_artifacts(data_dir: str = "data", model_dir: str = "models"):
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    model   = joblib.load(os.path.join(model_dir, "xgb_model.pkl"))
    return test_df, model


def simulate_drift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate data drift by modifying feature distributions.
    Quantity  ×1.2  (more items per transaction)
    UnitPrice ×0.8  (lower prices — e.g. sale season)
    TotalPrice is recomputed accordingly.
    """
    drifted = df.copy()
    drifted["Quantity"]   = drifted["Quantity"]   * 1.2
    drifted["UnitPrice"]  = drifted["UnitPrice"]  * 0.8
    drifted["TotalPrice"] = drifted["Quantity"]   * drifted["UnitPrice"]
    return drifted


def get_predictions(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict(df[FEATURE_COLS])


def compute_prediction_change(original_preds: np.ndarray,
                               drifted_preds:  np.ndarray) -> float:
    n_changed = np.sum(original_preds != drifted_preds)
    pct_change = (n_changed / len(original_preds)) * 100
    return round(pct_change, 2)


def feature_drift_summary(original_df: pd.DataFrame,
                           drifted_df:  pd.DataFrame) -> pd.DataFrame:
    """Statistical summary of feature-level changes."""
    records = []
    for col in FEATURE_COLS:
        records.append({
            "Feature"       : col,
            "Original Mean" : round(original_df[col].mean(), 4),
            "Drifted Mean"  : round(drifted_df[col].mean(),  4),
            "Mean Δ%"       : round(
                abs(drifted_df[col].mean() - original_df[col].mean())
                / (original_df[col].mean() + 1e-9) * 100, 2
            ),
        })
    return pd.DataFrame(records)


def run_drift_monitoring(data_dir:  str = "data",
                          model_dir: str = "models",
                          report_dir: str = "reports") -> dict:
    """Full drift monitoring run — returns result dict."""

    print("=" * 50)
    print("  🔍 DRIFT MONITORING REPORT")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    test_df, model = load_artifacts(data_dir, model_dir)

    # Baseline predictions
    original_preds = get_predictions(model, test_df)

    # Drift simulation
    drifted_df    = simulate_drift(test_df)
    drifted_preds = get_predictions(model, drifted_df)

    # Prediction change metric
    pct_change = compute_prediction_change(original_preds, drifted_preds)

    # Feature summary
    summary = feature_drift_summary(test_df, drifted_df)

    print("\n📋 Feature Distribution Changes:")
    print(summary.to_string(index=False))

    print(f"\n📈 Prediction Change: {pct_change}%")

    drift_detected = pct_change > DRIFT_THRESHOLD
    if drift_detected:
        print(f"⚠️  DRIFT DETECTED  (threshold: {DRIFT_THRESHOLD}%)")
        print("   → Consider retraining the model with fresh data.")
    else:
        print(f"✅  No significant drift detected  (threshold: {DRIFT_THRESHOLD}%)")

    # Save report
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "monitoring_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Drift Monitoring Report\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(summary.to_string(index=False))
        f.write(f"\n\nPrediction Change: {pct_change}%\n")
        f.write("DRIFT DETECTED\n" if drift_detected else "No drift detected\n")
    print(f"\n[INFO] Report saved → {report_path}")

    return {
        "prediction_change_pct": pct_change,
        "drift_detected"       : drift_detected,
        "feature_summary"      : summary,
    }


if __name__ == "__main__":
    run_drift_monitoring(
        data_dir="data",
        model_dir="models",
        report_dir="reports",
    )

"""
train.py - Model training script for UK Retail purchase prediction
"""

import os
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)


FEATURE_COLS = ["Quantity", "UnitPrice", "TotalPrice", "InvoiceHour", "InvoiceDay"]
TARGET_COL   = "WillBuyNextMonth"


def load_splits(data_dir: str = "data"):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train, test


def train_model(train_df: pd.DataFrame) -> XGBClassifier:
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]

    print("[INFO] Training XGBoost classifier...")
    model = XGBClassifier(
        n_estimators  = 100,
        max_depth     = 5,
        learning_rate = 0.1,
        random_state  = 42,
        eval_metric   = "logloss",
    )
    model.fit(X_train, y_train)
    print("[INFO] Training complete ✅")
    return model


def evaluate_model(model: XGBClassifier, test_df: pd.DataFrame) -> dict:
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy" : round(accuracy_score(y_test, y_pred),  4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "F1-Score" : round(f1_score(y_test, y_pred,        zero_division=0), 4),
    }

    print("\n📊 Evaluation Metrics:")
    print("-" * 35)
    for k, v in metrics.items():
        print(f"  {k:<12}: {v}")
    print("-" * 35)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


def save_model(model: XGBClassifier, model_dir: str = "models") -> str:
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "xgb_model.pkl")
    joblib.dump(model, path)
    print(f"[INFO] Model saved → {path}")
    return path


def training_pipeline(data_dir: str = "data", model_dir: str = "models") -> dict:
    train_df, test_df = load_splits(data_dir)
    model   = train_model(train_df)
    metrics = evaluate_model(model, test_df)
    save_model(model, model_dir)
    return metrics


if __name__ == "__main__":
    training_pipeline()

"""
preprocess.py - Data preprocessing pipeline for Online Retail II dataset
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data."""
    print(f"[INFO] Loading data from: {filepath}")
    df = pd.read_csv(filepath, encoding="ISO-8859-1")
    print(f"[INFO] Raw shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and clean the dataset."""
    print("[INFO] Filtering for United Kingdom...")
    df = df[df["Country"] == "United Kingdom"].copy()

    print("[INFO] Dropping nulls and invalid rows...")
    df.dropna(subset=["Customer ID"], inplace=True)
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]

    print(f"[INFO] Cleaned shape: {df.shape}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features."""
    print("[INFO] Engineering features...")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["TotalPrice"]   = df["Quantity"] * df["Price"]
    df["InvoiceHour"]  = df["InvoiceDate"].dt.hour
    df["InvoiceDay"]   = df["InvoiceDate"].dt.day
    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M")

    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target: 1 if customer purchased next month, 0 otherwise.
    Strategy: for each customer-month pair, check if they appear in the next month.
    """
    print("[INFO] Creating target variable...")

    # Aggregate at customer-month level
    customer_months = (
        df.groupby(["Customer ID", "InvoiceMonth"])
        .agg(
            Quantity   = ("Quantity",   "sum"),
            UnitPrice  = ("Price",      "mean"),
            TotalPrice = ("TotalPrice", "sum"),
            InvoiceHour= ("InvoiceHour","mean"),
            InvoiceDay = ("InvoiceDay", "mean"),
        )
        .reset_index()
    )

    customer_months["InvoiceMonth"] = customer_months["InvoiceMonth"].astype(str)
    customer_months.sort_values(["Customer ID", "InvoiceMonth"], inplace=True)

    # For each customer, find all months they appear in
    active_set = set(
        zip(customer_months["Customer ID"], customer_months["InvoiceMonth"])
    )

    def next_month_str(period_str):
        p = pd.Period(period_str, freq="M") + 1
        return str(p)

    customer_months["NextMonth"] = customer_months["InvoiceMonth"].apply(next_month_str)

    customer_months["WillBuyNextMonth"] = customer_months.apply(
        lambda row: 1 if (row["Customer ID"], row["NextMonth"]) in active_set else 0,
        axis=1
    )

    print(f"[INFO] Target distribution:\n{customer_months['WillBuyNextMonth'].value_counts()}")
    return customer_months


def preprocess_pipeline(raw_filepath: str, output_dir: str = "data") -> None:
    """Full preprocessing pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(raw_filepath)
    df = clean_data(df)
    df = engineer_features(df)
    df = create_target(df)

    # Select model features
    feature_cols = ["Quantity", "UnitPrice", "TotalPrice", "InvoiceHour", "InvoiceDay"]
    target_col   = "WillBuyNextMonth"

    model_df = df[feature_cols + [target_col]].dropna()

    X = model_df[feature_cols]
    y = model_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    train_df = X_train.copy(); train_df[target_col] = y_train.values
    test_df  = X_test.copy();  test_df[target_col]  = y_test.values

    train_path = os.path.join(output_dir, "train.csv")
    test_path  = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)

    print(f"[INFO] Saved train.csv ({train_df.shape}) → {train_path}")
    print(f"[INFO] Saved test.csv  ({test_df.shape})  → {test_path}")
    print("[INFO] Preprocessing complete ✅")


if __name__ == "__main__":
    import sys
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw.csv"
    preprocess_pipeline(raw_path, output_dir="data")

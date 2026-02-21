"""
download_data.py - Downloads the Online Retail II dataset from Kaggle
and saves it to data/raw.csv.

Usage:
    python download_data.py
"""

import os
import glob
import pandas as pd
import kagglehub

DATA_DIR = "data"
OUT_PATH = os.path.join(DATA_DIR, "raw.csv")
DATASET  = "mashlyn/online-retail-ii-uci"


def download_raw(dataset: str = DATASET, out_path: str = OUT_PATH) -> str:
    """
    Download the Kaggle dataset and persist as a single CSV.
    Returns the path to the saved file.
    """
    print(f"[INFO] Downloading dataset: {dataset}")
    cache_path = kagglehub.dataset_download(dataset)
    print(f"[INFO] Cached at: {cache_path}")

    # Collect all tabular files
    patterns = ["*.csv", "*.xls", "*.xlsx"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(cache_path, "**", pat), recursive=True))

    if not files:
        raise FileNotFoundError(f"No CSV/Excel files found in: {cache_path}")

    frames = []
    for f in sorted(files):
        print(f"[INFO] Reading: {os.path.basename(f)}")
        if f.endswith((".xls", ".xlsx")):
            frames.append(pd.read_excel(f, dtype=str))
        else:
            frames.append(pd.read_csv(f, dtype=str, encoding="ISO-8859-1", low_memory=False))

    df = pd.concat(frames, ignore_index=True)
    print(f"[INFO] Combined shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[INFO] Saved → {out_path}  ({len(df):,} rows)")
    return out_path


if __name__ == "__main__":
    download_raw()

import pandas as pd
import os

def read_csv(csv_path):
    """Read CSV file with auto encoding detection"""
    try:
        return pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(csv_path, encoding="shift_jis")

def save_csv(df, output_path):
    """Save DataFrame to CSV with UTF-8 encoding"""
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
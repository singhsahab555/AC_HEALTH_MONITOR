"""
STEP 1 — Ingest & Clean CSV Data
===================================
Reads CSV1 (power/temp readings) and CSV2 (ambient temp/humidity),
filters to state=valid only (Ac_status == 1 or 'on' or 'Valid'),
computes 5-min averages, merges on timestamp, and outputs a clean Parquet.

Usage:
    python 01_ingest_clean.py \
        --csv1 path/to/csv1.csv \
        --csv2 path/to/csv2.csv \
        --out  output/clean_data.parquet
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path


# ── Column aliases ────────────────────────────────────────────────────────────
# Edit these if your CSV headers differ slightly

CSV1_COL_MAP = {
    "time":            "timestamp",
    "Total Current":   "total_current",
    "Fan Current":     "fan_current",
    "Inlet Temp":      "inlet_temp",
    "Outlet Temp":     "outlet_temp",
    "Ac_status":       "ac_status",
    "device_id":       "ac_id",  # <-- CRITICAL: Map the new column here
}


CSV2_COL_MAP = {
    "rec_ts":          "timestamp",
    "temperature":     "ambient_temp",
    "humidity":        "humidity",
    "device_id":        "ac_id",
}

# ── Valid-state detection ─────────────────────────────────────────────────────
VALID_FLAGS = {"1", "on", "valid", "true", "running", "1.0"}


def is_valid(series: pd.Series) -> pd.Series:
    """Return boolean mask where AC is in a valid/on state."""
    return series.astype(str).str.strip().str.lower().isin(VALID_FLAGS)


# ── Load helpers ──────────────────────────────────────────────────────────────
def load_csv1(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={k: v for k, v in CSV1_COL_MAP.items() if k in df.columns})

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    if "ac_status" in df.columns:
        df = df[is_valid(df["ac_status"])]

    for col in ["total_current", "fan_current", "inlet_temp", "outlet_temp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -> Added "ac_id" to the returned dataframe <-
    keep_cols = ["timestamp", "total_current", "fan_current", "inlet_temp", "outlet_temp", "ac_id"]
    return df[[c for c in keep_cols if c in df.columns]].copy()

def load_csv2(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={k: v for k, v in CSV2_COL_MAP.items() if k in df.columns})

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    for col in ["ambient_temp", "humidity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["timestamp"] + [c for c in ["ambient_temp", "humidity"]
               if c in df.columns]].copy()


# ── Current → kW conversion (edit VOLTAGE if your system differs) ─────────────
VOLTAGE = 230          # volts (single-phase); change to 415 for 3-phase
POWER_FACTOR = 0.85    # typical for AC compressor loads


def current_to_kw(current_a: pd.Series) -> pd.Series:
    return (current_a * VOLTAGE * POWER_FACTOR) / 1000.0


# ── 5-minute resampler ────────────────────────────────────────────────────────
def resample_5min(df: pd.DataFrame, value_cols: list) -> pd.DataFrame:
    if df.empty:
        return df
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    if "ac_id" in df.columns:
        # Group by machine ID first, then resample to keep units separate
        g = df.set_index("timestamp")
        agg = {c: "mean" for c in value_cols if c in g.columns}
        return g.groupby("ac_id").resample("5min").agg(agg).reset_index()
    else:
        # Fallback for environmental data (CSV2) which has no ac_id
        g = df.set_index("timestamp")
        agg = {c: "mean" for c in value_cols if c in g.columns}
        return g.resample("5min").agg(agg).reset_index()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ingest & clean AC CSV data")
    parser.add_argument("--csv1", required=True, help="Path to CSV1 (power + temp readings)")
    parser.add_argument("--csv2", required=True, help="Path to CSV2 (ambient + humidity)")
    parser.add_argument("--out",  default="output/clean_data.parquet")
    parser.add_argument("--ac_id", default="AC-01",
                        help="AC unit identifier (default: AC-01)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n── Loading CSV1 ──────────────────────────────────────")
    df1 = load_csv1(args.csv1)
    print(f"  Rows loaded : {len(df1)}")
    print(f"  Time range  : {df1['timestamp'].min()} → {df1['timestamp'].max()}")

    print("\n── Loading CSV2 ──────────────────────────────────────")
    df2 = load_csv2(args.csv2)
    print(f"  Rows loaded : {len(df2)}")

    # Convert current → kW
    if "total_current" in df1.columns:
        df1["total_load_kw"] = current_to_kw(df1["total_current"])
    if "fan_current" in df1.columns:
        df1["fan_load_kw"] = current_to_kw(df1["fan_current"])

    print("\n── Resampling to 5-min averages ──────────────────────")
    df1_5m = resample_5min(df1, ["total_load_kw", "fan_load_kw", "inlet_temp", "outlet_temp"])
    df2_5m = resample_5min(df2, ["ambient_temp", "humidity"])

    print(f"  CSV1 5-min bins : {len(df1_5m)}")
    print(f"  CSV2 5-min bins : {len(df2_5m)}")

    print("\n── Merging on timestamp (nearest ±5 min) ────────────")
    merged = pd.merge_asof(
        df1_5m.sort_values("timestamp"),
        df2_5m.sort_values("timestamp"),
        on="timestamp",
        tolerance=pd.Timedelta("5min"),
        direction="nearest"
    )
    # Protect our dynamic ac_id values from being wiped out
    if "ac_id" not in merged.columns or merged["ac_id"].isnull().all():
        merged["ac_id"] = args.ac_id

    # Quality flag
    merged["data_quality"] = "valid"
    # Flag rows where key sensors have NaN
    key_cols = ["total_load_kw", "inlet_temp", "outlet_temp"]
    merged.loc[merged[key_cols].isnull().any(axis=1), "data_quality"] = "incomplete"

    print(f"  Merged rows : {len(merged)}")
    print(f"  Valid rows  : {(merged['data_quality']=='valid').sum()}")
    print(f"  Incomplete  : {(merged['data_quality']=='incomplete').sum()}")

    merged.to_parquet(out_path, index=False)
    print(f"\n✓ Saved clean data → {out_path}")
    print(merged.head(3).to_string())


if __name__ == "__main__":
    main()

"""
STEP 2 — AC Health Metrics Calculator
========================================
Reads the cleaned parquet from Step 1 and computes all 5 derived physics
metrics + 5 component health scores + anomaly detection + overall score +
classification.  Outputs a "health metrics" parquet ready for storage.

Usage:
    python 02_calculate_metrics.py \
        --input  output/clean_data.parquet \
        --out    output/health_metrics.parquet \
        --season summer
"""

import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

CSV1_COL_MAP = {
    "time":            "timestamp",
    "Total Current":   "total_current",
    "Fan Current":     "fan_current",
    "Inlet Temp":      "inlet_temp",
    "Outlet Temp":     "outlet_temp",
    "Ac_status":       "ac_status",
    "device_id":       "ac_id",  # <-- CRITICAL: Map the new column here
}


# ── Season thresholds ─────────────────────────────────────────────────────────
SEASON_THRESHOLDS = {
    "summer": {
        "delta_t_expected":       8.0,   # °C ideal drop
        "delta_t_critical":       6.0,   # °C — compressor is struggling
        "delta_t_warning":        8.0,   # °C — watch closely
        "cop_warning":            0.8,
        "efficiency_floor":       2.8,   # °C/kW minimum healthy
        "load_stress_lo":         0.10,  # fan_load/total_load band
        "load_stress_hi":         0.35,
        "hum_hi":                60.0,   # % relative humidity alarm
        "total_load_critical":    5.5,   # kW
        "total_load_warning":     4.5,   # kW
        "fan_baseline_kw":        0.55,  # kW clean-filter baseline fan draw
    },
    "winter": {
        "delta_t_expected":       4.0,
        "delta_t_critical":       4.0,
        "delta_t_warning":        6.0,
        "cop_warning":            0.6,
        "efficiency_floor":       2.0,
        "load_stress_lo":         0.12,
        "load_stress_hi":         0.40,
        "hum_hi":                70.0,
        "total_load_critical":    4.5,
        "total_load_warning":     3.5,
        "fan_baseline_kw":        0.50,
    },
}

# ── Derived metric formulas ───────────────────────────────────────────────────
def derive_physics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 5 core derived metrics from raw sensor data."""
    d = df.copy()

    # 1. Delta T (°C) — inlet minus outlet (positive = cooling)
    d["delta_t"] = d["inlet_temp"] - d["outlet_temp"]

    # 2. Compressor load (kW) — everything above fan
    d["compressor_load_kw"] = (d["total_load_kw"] - d["fan_load_kw"]).clip(lower=0)

    # 3. Efficiency ratio (°C / kW) — cooling per unit power
    d["efficiency_ratio"] = np.where(
        d["total_load_kw"] > 0,
        d["delta_t"] / d["total_load_kw"],
        np.nan
    )

    # 4. Load stress index — fan fraction of total power
    d["load_stress_index"] = np.where(
        d["total_load_kw"] > 0,
        d["fan_load_kw"] / d["total_load_kw"],
        np.nan
    )

    # 5. COP proxy — (Δt × 0.38) / compressor_load
    d["cop_proxy"] = np.where(
        d["compressor_load_kw"] > 0,
        (d["delta_t"] * 0.38) / d["compressor_load_kw"],
        np.nan    
    )
    
    if "ac_status" == 0:
     cop_proxy = 0

    # Bonus: EER proxy (dimensionless cooling efficiency index)
    d["eer_proxy"] = np.where(
        d["total_load_kw"] > 0,
        (d["delta_t"] * 3.517) / d["total_load_kw"],   # 3.517 = approx kW/ton at Δt basis
        np.nan
    )

    return d


# ── Component health scores (penalty model) ───────────────────────────────────
def score_compressor(row: pd.Series, th: dict) -> float:
    score = 100.0
    if row["delta_t"] < th["delta_t_critical"]:
        score -= 30
    elif row["delta_t"] < th["delta_t_warning"]:
        score -= 15
    if not np.isnan(row["cop_proxy"]) and row["cop_proxy"] < th["cop_warning"]:
        score -= 8
    if not np.isnan(row.get("humidity", np.nan)) and row["humidity"] > th["hum_hi"]:
        score -= 4
    if row["total_load_kw"] > th["total_load_critical"]:
        score -= 12
    elif row["total_load_kw"] > th["total_load_warning"]:
        score -= 6
    return max(0.0, score)


def score_refrigerant(row: pd.Series, th: dict) -> float:
    score = 100.0
    if row["delta_t"] < th["delta_t_critical"]:
        score -= 35   # strong signal: refrigerant leak / low charge
    elif row["delta_t"] < th["delta_t_warning"]:
        score -= 18
    # High load with low Δt = refrigerant not doing its job
    if row["total_load_kw"] > th["total_load_warning"] and row["delta_t"] < th["delta_t_warning"]:
        score -= 20
    if not np.isnan(row["efficiency_ratio"]) and row["efficiency_ratio"] < th["efficiency_floor"]:
        score -= 10
    return max(0.0, score)


def score_capacitor(row: pd.Series, th: dict) -> float:
    score = 100.0
    # Capacitor issues: fan load is disproportionately high or volatile
    lsi = row.get("load_stress_index", np.nan)
    if not np.isnan(lsi):
        if lsi > th["load_stress_hi"] * 1.3:
            score -= 25
        elif lsi > th["load_stress_hi"]:
            score -= 15
        if lsi < th["load_stress_lo"] * 0.5:
            score -= 10   # suspiciously low fan load = possible open capacitor
    return max(0.0, score)


def score_air_filter(row: pd.Series, th: dict) -> float:
    """Higher fan load vs baseline = dirtier filter = lower score."""
    score = 100.0
    fan_kw   = row.get("fan_load_kw", np.nan)
    baseline = th["fan_baseline_kw"]
    if not np.isnan(fan_kw) and baseline > 0:
        extra = max(0, fan_kw - baseline)
        penalty = (extra / baseline) * 100 * 0.6   # 0.6 scaling factor
        score -= min(penalty, 60)   # cap at 60-point penalty
    return max(0.0, score)


def score_fan_motor(row: pd.Series, th: dict) -> float:
    score = 100.0
    lsi = row.get("load_stress_index", np.nan)
    if not np.isnan(lsi) and lsi > th["load_stress_hi"]:
        score -= 20
    # Additional: very high fan_load_kw absolute value
    fan_kw = row.get("fan_load_kw", np.nan)
    if not np.isnan(fan_kw) and fan_kw > th["fan_baseline_kw"] * 2:
        score -= 15
    return max(0.0, score)


def compute_overall(row: pd.Series) -> float:
    weights = {
        "compressor_score": 0.35,
        "refrigerant_score": 0.25,
        "capacitor_score":   0.15,
        "air_filter_score":  0.15,
        "fan_motor_score":   0.10,
    }
    total = sum(row[k] * w for k, w in weights.items())
    return round(total, 2)


def classify(row: pd.Series) -> str:
    if row["anomaly_level"] == "CRITICAL" or row["overall_score"] < 45:
        return "CRITICAL"
    if row["anomaly_level"] == "WARNING"  or row["overall_score"] < 70:
        return "WARNING"
    return "HEALTHY"


# ── Anomaly detection ─────────────────────────────────────────────────────────
def detect_anomalies(df: pd.DataFrame, th: dict) -> pd.Series:
    """Rule-based anomaly detection (works without scikit-learn)."""
    level = pd.Series("OK", index=df.index)
    # Critical: very low delta_t under load
    crit_mask = (df["delta_t"] < th["delta_t_critical"]) & \
                (df["total_load_kw"] > th["total_load_warning"] * 0.5)
    level[crit_mask] = "CRITICAL"
    # Warning: degraded efficiency
    warn_mask = (df["delta_t"] < th["delta_t_warning"]) & (level == "OK")
    level[warn_mask] = "WARNING"
    return level


# ── Isolation Forest (optional, activates if scikit-learn present) ────────────
def try_isolation_forest(df: pd.DataFrame) -> pd.Series:
    try:
        from sklearn.ensemble import IsolationForest
        features = ["delta_t", "efficiency_ratio", "cop_proxy", "load_stress_index"]
        feat_df  = df[features].fillna(df[features].median())
        model    = IsolationForest(contamination=0.05, random_state=42)
        preds    = model.fit_predict(feat_df)
        return pd.Series(preds == -1, index=df.index)   # True = anomaly
    except ImportError:
        return pd.Series(False, index=df.index)


# ── Rolling trend metrics ─────────────────────────────────────────────────────
def add_rolling_metrics(df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Add rolling means over last `window` 5-min bins (~1 hr default)."""
    for col in ["delta_t", "overall_score", "efficiency_ratio", "cop_proxy"]:
        if col in df.columns:
            df[f"{col}_rolling_{window}"] = (
                df.groupby("ac_id")[col]
                  .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compute AC health metrics")
    parser.add_argument("--input",  default="output/clean_data.parquet")
    parser.add_argument("--out",    default="output/health_metrics.parquet")
    parser.add_argument("--season", default="summer",
                        choices=["summer", "winter"])
    args = parser.parse_args()

    th = SEASON_THRESHOLDS[args.season]

    print(f"\n── Loading clean data ({args.input}) ──────────────────")
    df = pd.read_parquet(args.input)
    df = df[df["data_quality"] == "valid"].copy()
    print(f"  Rows : {len(df)}")

    print("\n── Deriving physics metrics ──────────────────────────")
    df = derive_physics(df)

    print("\n── Scoring components ────────────────────────────────")
    df["compressor_score"] = df.apply(score_compressor, axis=1, th=th)
    df["refrigerant_score"] = df.apply(score_refrigerant, axis=1, th=th)
    df["capacitor_score"]   = df.apply(score_capacitor, axis=1, th=th)
    df["air_filter_score"]  = df.apply(score_air_filter, axis=1, th=th)
    df["fan_motor_score"]   = df.apply(score_fan_motor, axis=1, th=th)
    df["overall_score"]     = df.apply(compute_overall, axis=1)

    print("\n── Anomaly detection ─────────────────────────────────")
    df["anomaly_level"]    = detect_anomalies(df, th)
    df["ml_anomaly_flag"]  = try_isolation_forest(df)
    df["status"]           = df.apply(classify, axis=1)

    print("\n── Rolling trend metrics ─────────────────────────────")
    df = add_rolling_metrics(df, window=12)

    print("\n── Output summary ────────────────────────────────────")
    status_counts = df["status"].value_counts()
    print(status_counts.to_string())
    print(f"\n  Score range : {df['overall_score'].min():.1f} → {df['overall_score'].max():.1f}")
    print(f"  Mean score  : {df['overall_score'].mean():.1f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\n✓ Saved health metrics → {out_path}")

    # Also export a human-readable CSV summary (latest record per AC)
    latest = df.sort_values("timestamp").groupby("ac_id").last().reset_index()
    summary_cols = [
        "ac_id", "timestamp", "delta_t", "compressor_load_kw",
        "efficiency_ratio", "load_stress_index", "cop_proxy",
        "compressor_score", "refrigerant_score", "capacitor_score",
        "air_filter_score", "fan_motor_score", "overall_score",
        "anomaly_level", "status"
    ]
    avail = [c for c in summary_cols if c in latest.columns]
    csv_out = out_path.with_name("health_summary_latest.csv")
    latest[avail].to_csv(csv_out, index=False)
    print(f"✓ Saved human-readable summary → {csv_out}")


if __name__ == "__main__":
    main()

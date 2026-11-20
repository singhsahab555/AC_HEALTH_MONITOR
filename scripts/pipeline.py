"""
pipeline.py — AC Engine: Anomaly Detection & Forecasting
=========================================================
Reads directly from PostgreSQL:
  - runtime_cycles        → per-cycle data from Step 4
  - ac_health_metrics     → 5-min aggregated health scores from Step 3
  - ac_raw_sensor_data    → full-resolution sensor readings from Step 3b

Runs:
  - IsolationForest (unsupervised anomaly detection)
  - XGBoost classifier (supervised, uses anomaly_level labels)
  - Anomaly type classification (rule engine)
  - Prophet forecasting (optional)

Writes back to PostgreSQL:
  - ac_anomaly_results
  - ac_forecasts
  - ac_health_report
"""

import os
import sys
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, text
import pickle

# ── Path fix so db.py is always importable ───────────────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from db import CONN_STR, create_schema, write_results, write_forecasts, write_report

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    warnings.warn("xgboost not installed — XGBoost stage will be skipped.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ac_engine")

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IF_CONTAMINATION = 0.08
IF_N_ESTIMATORS  = 200

THRESHOLDS = {
    "delta_t_critical":     6.0,
    "delta_t_warning":      8.0,
    "duty_cycle_high":     80.0,
    "duty_cycle_warning":  60.0,
    "cop_low":              1.5,
    "load_stress_high":     0.25,
    "short_cycle_min_min":  5.0,
    "temp_impossible_min": -5.0,
    "temp_impossible_max": 50.0,
}

ANOMALY_TYPES = [
    "Poor Cooling", "Short Cycling", "Overload",
    "Compressor Stress", "Gas Leakage", "Fan Issue",
]

# Columns used for ML — subset that runtime_cycles actually has
FEATURE_COLS = [
    "delta_t_mean", "delta_t_start", "delta_t_end",
    "cooling_achieved_degc",
    "rate_of_cooling_degc_per_min",
    "duty_cycle_pct",
    "current_mean_a", "current_peak_a", "current_startup_a",
    "compressor_load_mean_kw", "fan_load_mean_kw", "total_load_mean_kw",
    "energy_consumed_kwh",
    "runtime_minutes", "off_time_minutes",
    "ambient_mean_temp", "humidity_mean_pct",
    "cycle_health_score",
    "anomaly_count",
    # engineered
    "efficiency_ratio", "cop_proxy", "load_stress_index",
    "current_peak_ratio",
    "delta_t_mean_roll3", "duty_cycle_pct_roll3",
    "compressor_load_mean_kw_roll3",
    "component_min_score", "anomaly_density",
]

# ── Column name aliases — maps DB column variants to standard names ──────────
COLUMN_ALIASES = {
    # runtime_cycles column variants
    "cycle_duty_cycle_pct":          "duty_cycle_pct",
    "current_mean_A":                "current_mean_a",
    "current_peak_A":                "current_peak_a",
    "current_startup_A":             "current_startup_a",
    "total_load_mean_kW":            "total_load_mean_kw",
    "fan_load_mean_kW":              "fan_load_mean_kw",
    "compressor_load_mean_kW":       "compressor_load_mean_kw",
    "energy_consumed_kWh":           "energy_consumed_kwh",
    "cooling_achieved_degC":         "cooling_achieved_degc",
    "rate_of_cooling_degC_per_min":  "rate_of_cooling_degc_per_min",
    "rate_of_warmup_degC_per_min":   "rate_of_warmup_degc_per_min",
    "humidity_mean_pct":             "humidity_mean_pct",
    "ambient_mean_temp":             "ambient_mean_temp",
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING FROM POSTGRESQL
# ══════════════════════════════════════════════════════════════════════════════
def _safe_read(engine, table: str, label: str) -> pd.DataFrame:
    """Read a table — return empty DataFrame (with warning) if it doesn't exist."""
    try:
        df = pd.read_sql(f"SELECT * FROM {table}", con=engine)
        log.info(f"  Loaded {len(df):,} rows from {table}")
        return df
    except Exception as e:
        log.warning(f"  Could not read {table}: {e}")
        return pd.DataFrame()


def load_input_data_from_postgres() -> pd.DataFrame:
    """
    Builds the master training matrix:
      1. Loads runtime_cycles (primary — one row per compressor cycle)
      2. Enriches with component health scores from ac_health_metrics
         (matched by ac_id + nearest timestamp within each cycle window)
      3. Falls back to 100 for all scores if ac_health_metrics is empty
    """
    log.info("Connecting to PostgreSQL…")
    engine = create_engine(CONN_STR)

    
    # ── Load ac_runtime_cycles ────────────────────────────────────────────────
    cycles = _safe_read(engine, "ac_runtime_cycles", "ac_runtime_cycles")
    if cycles.empty:
        raise RuntimeError(
            "Table 'ac_runtime_cycles' is empty or missing. "
            "Run Steps 0-4 of the health pipeline first."
        )

    # Normalise column names
    cycles.rename(columns={k: v for k, v in COLUMN_ALIASES.items()
                            if k in cycles.columns}, inplace=True)

    # Parse timestamps — strip tz for processing
    for col in ["cycle_start", "cycle_end"]:
        if col in cycles.columns:
            # In pipeline.py
         cycles[col] = pd.to_datetime(cycles[col], errors="coerce").dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    # ── Load ac_health_metrics (for component scores) ─────────────────────────
    metrics = _safe_read(engine, "ac_health_metrics", "ac_health_metrics")

    score_cols = ["compressor_score", "refrigerant_score", "capacitor_score",
                  "air_filter_score", "fan_motor_score", "overall_score"]

    if not metrics.empty and "timestamp" in metrics.columns:
        metrics["timestamp"] = pd.to_datetime(
            metrics["timestamp"], errors="coerce"
        ).dt.tz_localize(None)
        metrics_scores = metrics[
            ["ac_id", "timestamp"] + [c for c in score_cols if c in metrics.columns]
        ].dropna(subset=["timestamp"])

        # For each cycle row, find the avg score within [cycle_start, cycle_end]
        enriched_scores = []
        for _, cyc in cycles.iterrows():
            ac = cyc.get("ac_id", "")
            cs = cyc.get("cycle_start")
            ce = cyc.get("cycle_end")
            window = metrics_scores[
                (metrics_scores["ac_id"] == ac) &
                (metrics_scores["timestamp"] >= cs) &
                (metrics_scores["timestamp"] <= ce)
            ]
            row_scores = {}
            for sc in score_cols:
                if sc in window.columns and not window.empty:
                    row_scores[sc] = window[sc].mean()
                else:
                    row_scores[sc] = cyc.get(sc, 100.0)
            enriched_scores.append(row_scores)

        scores_df = pd.DataFrame(enriched_scores, index=cycles.index)
        for sc in score_cols:
            cycles[sc] = scores_df[sc].fillna(100.0)
        log.info("  Component scores enriched from ac_health_metrics")
    else:
        # No metrics table — use whatever is already in cycles or default 100
        for sc in score_cols:
            if sc not in cycles.columns:
                cycles[sc] = 100.0
        log.info("  Using default component scores (ac_health_metrics not available)")

    log.info(f"Master matrix: {cycles.shape[0]} cycles × {cycles.shape[1]} columns")
    return cycles


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data — fix flags, clamp temps, fill NaN per-AC."""
    df = df.copy()

    # Convert bool/string flags to int
    flag_cols = [c for c in df.columns if c.endswith("_flag")]
    for c in flag_cols:
        df[c] = df[c].map({True: 1, False: 0, "t": 1, "f": 0, 1: 1, 0: 0}).fillna(0).astype(int)

    # Clamp impossible temperatures
    temp_cols = [c for c in df.columns if "temp" in c.lower()]
    for col in temp_cols:
        if df[col].dtype in [np.float64, np.float32]:
            df[col] = df[col].clip(
                THRESHOLDS["temp_impossible_min"],
                THRESHOLDS["temp_impossible_max"]
            )

    # Fill NaN with per-AC median, then global median
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if "ac_id" in df.columns:
        df[num_cols] = df.groupby("ac_id")[num_cols].transform(
            lambda x: x.fillna(x.median())
        )
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for ML."""
    df = df.copy()

    # Safety defaults for required columns
    defaults = {
        "duty_cycle_pct": 0.0, "delta_t_mean": 0.0,
        "cooling_achieved_degc": 0.0, "total_load_mean_kw": 0.0,
        "compressor_load_mean_kw": 0.0, "fan_load_mean_kw": 0.0,
        "current_mean_a": 0.0, "current_peak_a": 0.0,
        "runtime_minutes": 1.0, "anomaly_count": 0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    # Derive delta_t_mean from temps if missing
    if df["delta_t_mean"].eq(0).all():
        if "inlet_start_temp" in df.columns and "outlet_start_temp" in df.columns:
            df["delta_t_mean"] = df["inlet_start_temp"] - df["outlet_start_temp"]

    safe_div = lambda a, b: (a / b.replace(0, np.nan)).fillna(0)

    # Only calculate these if they weren't already provided by Step 4 (ac_runtime_cycles)
    if "efficiency_ratio" not in df.columns:
        df["efficiency_ratio"] = safe_div(df["cooling_achieved_degc"].abs(), df["total_load_mean_kw"])
        
    if "cop_proxy" not in df.columns:
        df["cop_proxy"] = safe_div(df["cooling_achieved_degc"].abs() * 3.517, df["compressor_load_mean_kw"]).clip(0, 10)
        
    if "load_stress_index" not in df.columns:
        df["load_stress_index"] = safe_div(df["compressor_load_mean_kw"], df["total_load_mean_kw"])
        
    if "current_peak_ratio" not in df.columns:
        df["current_peak_ratio"] = safe_div(df["current_peak_a"], df["current_mean_a"]).clip(0, 5)

    # Time features
    if "cycle_start" in df.columns:
        df["hour"]        = df["cycle_start"].dt.hour
        df["day_of_week"] = df["cycle_start"].dt.dayofweek

    # Rolling + diff features per AC
    df = df.sort_values(["ac_id", "cycle_start"]) if "ac_id" in df.columns else df.sort_values("cycle_start")
    grp_col = "ac_id" if "ac_id" in df.columns else None

    for metric in ["delta_t_mean", "duty_cycle_pct", "compressor_load_mean_kw"]:
        if metric in df.columns:
            if grp_col:
                df[f"{metric}_roll3"] = df.groupby(grp_col)[metric].transform(
                    lambda x: x.rolling(3, min_periods=1).mean()
                )
                df[f"{metric}_diff1"] = df.groupby(grp_col)[metric].transform(
                    lambda x: x.diff()
                ).fillna(0)
            else:
                df[f"{metric}_roll3"] = df[metric].rolling(3, min_periods=1).mean()
                df[f"{metric}_diff1"] = df[metric].diff().fillna(0)

    # Component min score
    score_cols = [c for c in ["compressor_score", "refrigerant_score",
                               "capacitor_score", "air_filter_score", "fan_motor_score"]
                  if c in df.columns]
    df["component_min_score"] = df[score_cols].min(axis=1) if score_cols else 100.0

    # Anomaly density
    df["anomaly_density"] = safe_div(
        df["anomaly_count"].fillna(0),
        df["runtime_minutes"].clip(lower=0.01)
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# ML MODELS
# ══════════════════════════════════════════════════════════════════════════════
def train_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    avail = [c for c in FEATURE_COLS if c in df.columns]
    if len(avail) < 3:
        log.warning("Too few features for IsolationForest — skipping")
        df["if_score"]   = 0.0
        df["if_anomaly"] = 0
        return df

    X = df[avail].fillna(0)
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)
    iforest = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=42, n_jobs=-1
    )
    iforest.fit(X_sc)

    df["if_score"]   = iforest.decision_function(X_sc)
    df["if_anomaly"] = (iforest.predict(X_sc) == -1).astype(int)

    # Save model
    with open(MODEL_DIR / "isolation_forest.pkl", "wb") as f:
        pickle.dump({"model": iforest, "scaler": scaler, "features": avail}, f)

    n_anom = df["if_anomaly"].sum()
    log.info(f"  IsolationForest: {n_anom}/{len(df)} anomalies ({100*n_anom/len(df):.1f}%)")
    return df


def load_or_train_isolation_forest(df: pd.DataFrame, force_retrain: bool = False) -> pd.DataFrame:
    """Predict using saved IsolationForest if available; retrain if missing/forced/stale."""
    df = df.copy()
    model_path = MODEL_DIR / "isolation_forest.pkl"

    if model_path.exists() and not force_retrain:
        try:
            with open(model_path, "rb") as f:
                saved = pickle.load(f)
            iforest, scaler, saved_features = saved["model"], saved["scaler"], saved["features"]

            missing = [c for c in saved_features if c not in df.columns]
            if missing:
                log.warning(f"  IF: saved model expects missing columns {missing} — retraining")
                return train_isolation_forest(df)

            X = df[saved_features].fillna(0)
            X_sc = scaler.transform(X)  # transform only, never fit here
            df["if_score"]   = iforest.decision_function(X_sc)
            df["if_anomaly"] = (iforest.predict(X_sc) == -1).astype(int)

            n_anom = df["if_anomaly"].sum()
            log.info(f"  IsolationForest (loaded): {n_anom}/{len(df)} anomalies "
                      f"({100*n_anom/max(len(df),1):.1f}%)")
            return df
        except Exception as e:
            log.warning(f"  IF: failed to load saved model ({e}) — retraining")
            return train_isolation_forest(df)

    return train_isolation_forest(df)


def load_or_train_xgboost(df: pd.DataFrame, force_retrain: bool = False) -> pd.DataFrame:
    """Predict using saved XGBoost if available; retrain if missing/forced/stale."""
    if not XGB_AVAILABLE:
        df["xgb_anomaly_level"] = df.get("anomaly_level", "HEALTHY")
        df["xgb_prob_critical"] = 0.0
        return df

    df = df.copy()
    model_path = MODEL_DIR / "xgboost_classifier.pkl"

    if model_path.exists() and not force_retrain:
        try:
            with open(model_path, "rb") as f:
                saved = pickle.load(f)
            model, label_map, saved_features = saved["model"], saved["label_map"], saved["features"]
            inv_map = {idx: lbl for lbl, idx in label_map.items()}

            missing = [c for c in saved_features if c not in df.columns]
            if missing:
                log.warning(f"  XGB: saved model expects missing columns {missing} — retraining")
                return train_xgboost(df)

            X_all = df[saved_features].fillna(0)
            proba = model.predict_proba(X_all)
            preds = model.predict(X_all)
            df["xgb_anomaly_level"] = pd.Series(preds, index=df.index).map(inv_map)
            crit_idx = label_map.get("CRITICAL")
            df["xgb_prob_critical"] = proba[:, crit_idx] if crit_idx is not None else 0.0

            log.info(f"  XGBoost (loaded): scored {len(df)} rows using saved model")
            return df
        except Exception as e:
            log.warning(f"  XGB: failed to load saved model ({e}) — retraining")
            return train_xgboost(df)

    return train_xgboost(df)


def train_xgboost(df: pd.DataFrame) -> pd.DataFrame:
    if not XGB_AVAILABLE:
        df["xgb_anomaly_level"] = df.get("anomaly_level", "HEALTHY")
        df["xgb_prob_critical"]  = 0.0
        return df

    df = df.copy()
    SEVERITY_ORDER = ["OK", "HEALTHY", "WARNING", "CRITICAL"]

    # ── FIX: Dynamically generate target labels from IsolationForest if missing ──
    if "anomaly_level" not in df.columns or df["anomaly_level"].nunique() < 2:
        if "if_anomaly" in df.columns and df["if_anomaly"].nunique() >= 2:
            log.info("  anomaly_level missing/uniform; utilizing IsolationForest pseudo-labels.")
            df["anomaly_level"] = df["if_anomaly"].map({0: "HEALTHY", 1: "CRITICAL"})
        else:
            df["anomaly_level"] = "HEALTHY"

    present = [lbl for lbl in SEVERITY_ORDER if lbl in df["anomaly_level"].values]
    if len(present) < 2:
        log.info("  XGBoost: only one label class present — skipping training")
        df["xgb_anomaly_level"] = df["anomaly_level"]
        df["xgb_prob_critical"]  = 0.0
        return df

    label_map = {lbl: idx for idx, lbl in enumerate(present)}
    inv_map   = {idx: lbl for lbl, idx in label_map.items()}
    df["_xgb_label"] = df["anomaly_level"].map(label_map)

    avail = [c for c in FEATURE_COLS if c in df.columns]
    labeled = df.dropna(subset=["_xgb_label"])
    X = labeled[avail].fillna(0)
    y = labeled["_xgb_label"].astype(int)

    # Need at least 2 samples per class
    if len(y.unique()) < 2 or len(y) < 10:
        log.info("  XGBoost: insufficient training data — using rule-based labels")
        df["xgb_anomaly_level"] = df["anomaly_level"]
        df["xgb_prob_critical"]  = 0.0
        df.drop(columns=["_xgb_label"], inplace=True, errors="ignore")
        return df

    try:
        stratify = y if y.value_counts().min() >= 2 else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
        objective = "binary:logistic" if len(present) == 2 else "multi:softprob"
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            objective=objective, random_state=42, n_jobs=-1,
            eval_metric="logloss",
        )
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        X_all  = df[avail].fillna(0)
        proba  = model.predict_proba(X_all)
        preds  = model.predict(X_all)
        df["xgb_anomaly_level"] = pd.Series(preds, index=df.index).map(inv_map)
        crit_idx = label_map.get("CRITICAL")
        df["xgb_prob_critical"] = (
            proba[:, crit_idx] if crit_idx is not None else 0.0
        )

        with open(MODEL_DIR / "xgboost_classifier.pkl", "wb") as f:
            pickle.dump({"model": model, "label_map": label_map, "features": avail}, f)

        # FIX: Calculate accuracy properly using the test features
        acc = model.score(X_te, y_te) if len(X_te) > 0 else 0
        log.info(f"  XGBoost: test accuracy = {acc:.2%}")

    except Exception as e:
        log.warning(f"  XGBoost training failed: {e} — using rule-based labels")
        df["xgb_anomaly_level"] = df["anomaly_level"]
        df["xgb_prob_critical"]  = 0.0

    df.drop(columns=["_xgb_label"], inplace=True, errors="ignore")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# ANOMALY TYPE CLASSIFICATION (Rule Engine)
# ══════════════════════════════════════════════════════════════════════════════
def classify_anomaly_type(row: pd.Series) -> str:
    types = []
    dt = row.get("delta_t_mean", 99)
    if dt < THRESHOLDS["delta_t_critical"]:
        types.append("Poor Cooling")
    if row.get("runtime_minutes", 99) < THRESHOLDS["short_cycle_min_min"]:
        types.append("Short Cycling")
    if row.get("duty_cycle_pct", 0) > THRESHOLDS["duty_cycle_high"]:
        types.append("Overload")
    if row.get("load_stress_index", 0) > THRESHOLDS["load_stress_high"]:
        types.append("Compressor Stress")
    if (row.get("cop_proxy", 99) < THRESHOLDS["cop_low"] and
            row.get("delta_t_mean", 99) < THRESHOLDS["delta_t_warning"]):
        types.append("Gas Leakage")
    if row.get("fan_motor_score", 100) < 50:
        types.append("Fan Issue")
    return " | ".join(types) if types else "None"


# ══════════════════════════════════════════════════════════════════════════════
# FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
def forecast_metric(df: pd.DataFrame, ac_id: str, metric: str) -> pd.DataFrame:
    if not PROPHET_AVAILABLE:
        return pd.DataFrame()
    sub = (df[df["ac_id"] == ac_id][["cycle_start", metric]]
           .dropna()
           .rename(columns={"cycle_start": "ds", metric: "y"}))
    if len(sub) < 10:
        return pd.DataFrame()
    try:
        m = Prophet(daily_seasonality=True, weekly_seasonality=False).fit(sub)
        future   = m.make_future_dataframe(periods=24, freq="30min")
        forecast = m.predict(future)
        forecast["ac_id"]   = ac_id
        forecast["metric"]  = metric
        if metric == "delta_t_mean":
            forecast["predicted_anomaly"] = (forecast["yhat"] < THRESHOLDS["delta_t_critical"]).astype(int)
        elif metric == "duty_cycle_pct":
            forecast["predicted_anomaly"] = (forecast["yhat"] > THRESHOLDS["duty_cycle_high"]).astype(int)
        else:
            forecast["predicted_anomaly"] = 0
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper",
                          "ac_id", "metric", "predicted_anomaly"]]
    except Exception as e:
        log.warning(f"  Prophet failed for {ac_id}/{metric}: {e}")
        return pd.DataFrame()


def run_all_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    all_fc = []
    for ac in df["ac_id"].unique():
        for m in ["delta_t_mean", "duty_cycle_pct"]:
            fc = forecast_metric(df, ac, m)
            if not fc.empty:
                all_fc.append(fc)
    return pd.concat(all_fc, ignore_index=True) if all_fc else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════
def generate_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ac_id, grp in df.groupby("ac_id"):
        n = len(grp)
        # Calculate anomaly stats
        if_anom = int(grp["if_anomaly"].sum()) if "if_anomaly" in grp else 0
        anomaly_rate = 100 * if_anom / max(n, 1)
        
        # Determine status based on actual data
        if anomaly_rate > 25.0:
            status = "CRITICAL"
        elif anomaly_rate > 10.0:
            status = "WARNING"
        else:
            status = "HEALTHY"

        top_type = "None"
        if "anomaly_type" in grp.columns:
            types = (grp["anomaly_type"].str.split(" | ")
                     .explode().value_counts())
            types = types[types.index != "None"]
            top_type = types.index[0] if not types.empty else "None"

        rows.append({
            "ac_id":              ac_id,
            "total_cycles":       n,
            "if_anomalies":       if_anom,
            "anomaly_rate_pct":   round(anomaly_rate, 1),
            "overall_status":     status,  # Now dynamic!
            "top_anomaly_type":   top_type,
            "avg_delta_t":        round(grp["delta_t_mean"].mean(), 2),
            "avg_duty_cycle":     round(grp["duty_cycle_pct"].mean(), 1),
            "avg_cop_proxy":      round(grp.get("cop_proxy", pd.Series([0])).mean(), 2),
            "avg_overall_score":  round(grp["overall_score"].mean(), 1),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(force_retrain: bool = False) -> dict:
    log.info("=" * 70)
    log.info("  AC ENGINE — Anomaly Detection & Forecasting")
    log.info(f"  force_retrain={force_retrain}")
    log.info("=" * 70)

    # Ensure output tables exist
    create_schema()

    # ── Step 1: Load data from PostgreSQL ─────────────────────────────────────
    health = load_input_data_from_postgres()

    # ── Step 2: Feature engineering ───────────────────────────────────────────
    health = clean(health)
    health = engineer_features(health)

    # ── Step 3: IsolationForest (loads saved model unless force_retrain) ──────
    log.info("Running IsolationForest…")
    health = load_or_train_isolation_forest(health, force_retrain=force_retrain)

    # ── Step 4: XGBoost (loads saved model unless force_retrain) ──────────────
    log.info("Running XGBoost…")
    health = load_or_train_xgboost(health, force_retrain=force_retrain)

    # ── Step 5: Anomaly type classification ───────────────────────────────────
    health["anomaly_type"] = health.apply(classify_anomaly_type, axis=1)
    n_flagged = (health["anomaly_type"] != "None").sum()
    log.info(f"Anomaly types: {n_flagged}/{len(health)} cycles flagged")

    # ── Step 6: Write back to PostgreSQL ──────────────────────────────────────
    log.info("Writing ac_anomaly_results → PostgreSQL…")
    write_results(health, if_exists="append")  # now upserted, not wiped — see db.py

    # ── Step 7: Forecasting ───────────────────────────────────────────────────
    forecasts = run_all_forecasts(health)
    if not forecasts.empty:
        log.info(f"Writing {len(forecasts)} forecast rows…")
        write_forecasts(forecasts)
    else:
        log.info("No forecasts generated (need ≥10 cycles per AC or install prophet)")

    # ── Step 8: Health report ─────────────────────────────────────────────────
    report = generate_report(health)
    if not report.empty:
        log.info(f"Writing health report ({len(report)} AC units)…")
        write_report(report)

    log.info("=" * 70)
    log.info("  AC ENGINE COMPLETE — PostgreSQL updated")
    log.info("=" * 70)

    return {"health_df": health, "forecasts": forecasts, "report": report}


if __name__ == "__main__":
    # Pass --retrain on the command line to force a fresh fit + overwrite the .pkl files
    _force = "--retrain" in sys.argv
    run_pipeline(force_retrain=_force)

"""
STEP 4 — Compressor Runtime Cycle Detector
============================================
Reads raw CSV1 data (NOT the 5-min avg parquet) because cycle detection
needs the raw high-resolution timestamps to catch the exact moment the
green line (Total Current) jumps up or drops back to zero.

From your dashboard: Total Current jumps from ~0 A → 7–8 A when the
compressor/condenser starts, and falls back to ~0 when the setpoint is
reached and it cycles OFF.

What this script detects and calculates
────────────────────────────────────────
Per cycle:
  - cycle_start / cycle_end timestamps
  - runtime_minutes  (how long compressor ran)
  - off_time_minutes (gap until next cycle = rest / off period)
  - cycle_period_minutes (runtime + off_time = full ON→OFF→ON period)
  - duty_cycle_%  (runtime / cycle_period × 100)

Temperature metrics per cycle:
  - inlet_start_temp / inlet_end_temp
  - outlet_start_temp / outlet_end_temp
  - delta_t_start / delta_t_end / delta_t_mean  (inlet − outlet)
  - cooling_achieved_degC  (how much the outlet temp dropped during the cycle)
  - rate_of_cooling_degC_per_min  (outlet temp slope during ON period)
  - rate_of_warmup_degC_per_min   (outlet temp slope during OFF period)

Current / Power metrics per cycle:
  - current_mean_A / current_peak_A
  - total_load_mean_kW / fan_load_mean_kW / compressor_load_mean_kW
  - energy_consumed_kWh  (power × time)

Anomaly flags per cycle:
  - short_cycle_flag   (runtime < SHORT_CYCLE_MINS → possible thermostat/refrigerant issue)
  - long_cycle_flag    (runtime > LONG_CYCLE_MINS  → possible refrigerant / filter issue)
  - rapid_restart_flag (off_time < RAPID_RESTART_MINS → stresses compressor)
  - poor_cooling_flag  (cooling_achieved < COOLING_THRESHOLD)

Aggregate summary (across all cycles in the file):
  - total cycles, mean/min/max runtime, mean duty cycle
  - maintenance prediction score

Usage:
    python 04_runtime_cycles.py \
        --csv1  data/csv1.csv \
        --csv2  data/csv2.csv \
        --out   output/runtime_cycles.parquet \
        --ac_id FC012CD6E014 \
        --plot          # optional: saves a PNG chart

Dependencies:
    pip install pandas numpy scipy matplotlib --break-system-packages
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d   # gentle smoothing
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")
from pathlib import Path

# --- INSERT THIS HELPER FUNCTION HERE ---
def find_latest_files(data_dir: str = "data"):
    """Finds the most recent main_*.csv and temp_hum_*.csv files."""
    d = Path(data_dir)
    main_files = sorted(d.glob("main_*.csv"), reverse=True)
    temp_files = sorted(d.glob("temp_hum_*.csv"), reverse=True)
    
    if not main_files or not temp_files:
        return None, None
    return str(main_files[0]), str(temp_files[0])
# ----------------------------------------


# ═══════════════════════════════════════════════════════════════════════════
# TUNABLE THRESHOLDS  — adjust to match your hardware
# ═══════════════════════════════════════════════════════════════════════════
CURRENT_ON_THRESHOLD   = 2.0    # A  — above this = compressor ON
                                 #      (from your data: idle ~0.027 A, running 7–8 A)
CURRENT_OFF_THRESHOLD  = 1.0    # A  — below this = compressor OFF (hysteresis gap)
MIN_ON_DURATION_SECS   = 80     # ignore ON blips shorter than this (noise)
MIN_OFF_DURATION_SECS  = 60     # ignore OFF blips shorter than this

VOLTAGE                = 230    # V  (single-phase; change to 415 for 3-phase)
POWER_FACTOR           = 0.85

# Timezone the raw CSV timestamps are actually recorded in.
# Root cause of the "converted back to UTC" bug: timestamps were being
# parsed as tz-NAIVE, then psycopg2/Postgres silently assumed they were
# already UTC when writing into the TIMESTAMPTZ columns — shifting every
# value by the local UTC offset. Localizing here makes them tz-aware so
# Postgres (and pandas/matplotlib) handle them correctly with no implicit
# reinterpretation.
DATA_TZ                = "Asia/Kolkata"

SHORT_CYCLE_MINS       = 3.0    # flag if compressor ran < 3 min (short cycling)
LONG_CYCLE_MINS        = 45.0   # flag if compressor ran > 45 min (can't reach setpoint)
RAPID_RESTART_MINS     = 3.0    # flag if off time < 3 min (compressor stress)
COOLING_THRESHOLD_DEG  = 1.5    # flag if outlet dropped < 1.5°C during a run


# ═══════════════════════════════════════════════════════════════════════════
# COLUMN MAP — must match your CSV headers
# ═══════════════════════════════════════════════════════════════════════════
CSV1_MAP = {
    "time":           "timestamp",
    "Total Current":  "total_current",
    "Fan Current":    "fan_current",
    "Inlet Temp":     "inlet_temp",
    "Outlet Temp":    "outlet_temp",
    "Ac_status":      "ac_status",
    "device_id":        "ac_id",
}
CSV2_MAP = {
    "rec_ts":         "timestamp",
    "temperature":    "ambient_temp",
    "humidity":       "humidity",
    "device_id":        "ac_id",
}

# ═══════════════════════════════════════════════════════════════════
# PostgreSQL DDL
# ═══════════════════════════════════════════════════════════════════
PG_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS {schema}.{table} (
    id                          BIGSERIAL,
    ac_id                       TEXT          NOT NULL,
    season                      TEXT,
    cycle_number                INTEGER,
    still_running               BOOLEAN       DEFAULT FALSE,

    -- Timestamps
    cycle_start                 TIMESTAMPTZ   NOT NULL,
    cycle_end                   TIMESTAMPTZ,

    -- Time metrics
    runtime_minutes             FLOAT,
    off_time_minutes            FLOAT,
    cycle_period_minutes        FLOAT,
    duty_cycle_pct              FLOAT,

    -- Temperature
    inlet_start_temp            FLOAT,
    inlet_end_temp              FLOAT,
    outlet_start_temp           FLOAT,
    outlet_end_temp             FLOAT,
    delta_t_start               FLOAT,
    delta_t_end                 FLOAT,
    delta_t_mean                FLOAT,
    cooling_achieved_degC       FLOAT,
    rate_of_cooling_degC_per_min FLOAT,
    rate_of_warmup_degC_per_min  FLOAT,
    ambient_mean_temp           FLOAT,
    humidity_mean_pct           FLOAT,

    -- Current / Power
    current_mean_A              FLOAT,
    current_peak_A              FLOAT,
    current_startup_A           FLOAT,
    total_load_mean_kW          FLOAT,
    fan_load_mean_kW            FLOAT,
    compressor_load_mean_kW     FLOAT,
    energy_consumed_kWh         FLOAT,

    -- Physics
    efficiency_ratio            FLOAT,
    cop_proxy                   FLOAT,
    load_stress_index           FLOAT,

    -- Anomaly flags
    short_cycle_flag            BOOLEAN,
    long_cycle_flag             BOOLEAN,
    rapid_restart_flag          BOOLEAN,
    poor_cooling_flag           BOOLEAN,
    high_stress_flag            BOOLEAN,
    anomaly_count               INTEGER,
    cycle_health_score          FLOAT,

    -- Meta
    created_at                  TIMESTAMPTZ   DEFAULT NOW(),

    PRIMARY KEY (ac_id, cycle_start)
);

CREATE INDEX IF NOT EXISTS idx_{table_safe}_ac_ts
    ON {schema}.{table} (ac_id, cycle_start DESC);

CREATE INDEX IF NOT EXISTS idx_{table_safe}_status
    ON {schema}.{table} (still_running, anomaly_count);
"""


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def load_and_merge(csv1: str, csv2: str) -> pd.DataFrame:
    """Load raw CSVs, basic cleaning, merge ambient data."""
    df1 = pd.read_csv(csv1)
    df1 = df1.rename(columns={k: v for k, v in CSV1_MAP.items() if k in df1.columns})
    df1["timestamp"] = pd.to_datetime(df1["timestamp"], errors="coerce")
    df1 = df1.dropna(subset=["timestamp"]).sort_values("timestamp")
    # Localize to the timezone the data was actually recorded in (tz-naive
    # timestamps were previously being misread as UTC downstream).
    if df1["timestamp"].dt.tz is None:
        df1["timestamp"] = df1["timestamp"].dt.tz_localize(
            DATA_TZ, ambiguous="NaT", nonexistent="NaT"
        )
    else:
        df1["timestamp"] = df1["timestamp"].dt.tz_convert(DATA_TZ)
    df1 = df1.dropna(subset=["timestamp"])

    for col in ["total_current", "fan_current", "inlet_temp", "outlet_temp"]:
        if col in df1.columns:
            df1[col] = pd.to_numeric(df1[col], errors="coerce")

    df2 = pd.read_csv(csv2)
    df2 = df2.rename(columns={k: v for k, v in CSV2_MAP.items() if k in df2.columns})
    df2["timestamp"] = pd.to_datetime(df2["timestamp"], errors="coerce")
    df2 = df2.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df2["timestamp"].dt.tz is None:
        df2["timestamp"] = df2["timestamp"].dt.tz_localize(
            DATA_TZ, ambiguous="NaT", nonexistent="NaT"
        )
    else:
        df2["timestamp"] = df2["timestamp"].dt.tz_convert(DATA_TZ)
    df2 = df2.dropna(subset=["timestamp"])
    for col in ["ambient_temp", "humidity"]:
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors="coerce")

    merged = pd.merge_asof(df1, df2, on="timestamp",
                           tolerance=pd.Timedelta("10min"), direction="nearest")
    # Ensure proper ordering

    merged = merged.sort_values(
    "timestamp"
    ).reset_index(drop=True)

    return merged


def smooth_current(series: pd.Series, window: int = 3) -> np.ndarray:
    """Light smoothing to suppress single-point noise spikes."""
    arr = series.fillna(0).values.astype(float)
    return uniform_filter1d(arr, size=window)


def detect_cycles(df: pd.DataFrame) -> list[dict]:
    """
    State-machine cycle detector with hysteresis.
    Transitions:  OFF → ON  when smoothed current > ON_THRESHOLD  for MIN_ON_DURATION_SECS
                  ON  → OFF when smoothed current < OFF_THRESHOLD for MIN_OFF_DURATION_SECS
    Returns list of cycle dicts with start/end indices.
    """
    current = smooth_current(df["total_current"])
    timestamps = df["timestamp"]  # tz-aware pandas Series — keep as-is, do NOT use .values
    tz = timestamps.dt.tz

    state       = "OFF"
    on_start    = None
    candidate_off_start = None
    candidate_on_start  = None

    cycles = []

    for i in range(len(current)):
        c = current[i]
        ts = timestamps.iloc[i]

        # ─────────────────────────────────────────────
        # AC STATUS VALIDATION
        # ─────────────────────────────────────────────
        ac_status = str(df.iloc[i].get("ac_status", "0")).strip()
        fan_current = float(df.iloc[i].get("fan_current", 0) or 0)

        # TRUE AC RUNNING STATE
        is_actual_on = (
            ac_status == "1"
            and (c >= CURRENT_ON_THRESHOLD )
        )

        # ─────────────────────────────────────────────
        # OFF → ON TRANSITION
        # ─────────────────────────────────────────────
        if state == "OFF":
            if is_actual_on:
                if candidate_on_start is None:
                    candidate_on_start = (i, ts)
                elapsed = (ts - candidate_on_start[1]).total_seconds()
                if elapsed >= MIN_ON_DURATION_SECS:
                    state = "ON"
                    on_start = candidate_on_start
                    candidate_on_start = None
            else:
                candidate_on_start = None

        # ─────────────────────────────────────────────
        # ON → OFF TRANSITION
        # ─────────────────────────────────────────────
        elif state == "ON":
            is_actual_off = (ac_status != "1" or c < OFF_THRESHOLD)
            if is_actual_off:
                if candidate_off_start is None:
                    candidate_off_start = (i, ts)
                elapsed = (ts - candidate_off_start[1]).total_seconds()
                if elapsed >= MIN_OFF_DURATION_SECS:
                    cycles.append({
                        "on_start_idx": on_start[0],
                        "on_start_ts": on_start[1],
                        "on_end_idx": candidate_off_start[0],
                        "on_end_ts": candidate_off_start[1],
                        "off_start_idx": candidate_off_start[0],
                        "off_start_ts": candidate_off_start[1],
                    })
                    state = "OFF"
                    on_start = None
                    candidate_off_start = None
            else:
                candidate_off_start = None

    # ─────────────────────────────────────────────
    # OUTSIDE THE FOR LOOP (End of file handling)
    # ─────────────────────────────────────────────
    if state == "ON" and on_start is not None:
        last_i  = len(current) - 1
        last_ts = timestamps.iloc[last_i]
        cycles.append({
            "on_start_idx":  on_start[0],
            "on_start_ts":   on_start[1],
            "on_end_idx":    last_i,
            "on_end_ts":     last_ts,
            "off_start_idx": last_i,
            "off_start_ts":  last_ts,
            "still_running": True,
        })

    return cycles

# Grab threshold into module scope (needed in detect_cycles)
OFF_THRESHOLD = CURRENT_OFF_THRESHOLD


def slope_degC_per_min(temps: np.ndarray, times_sec: np.ndarray) -> float:
    """Linear regression slope in °C/min from a temperature array."""
    if len(temps) < 2:
        return np.nan
    try:
        times_min = times_sec / 60.0
        coef = np.polyfit(times_min, temps, 1)
        return round(float(coef[0]), 4)
    except Exception:
        return np.nan


def build_cycle_metrics(df: pd.DataFrame, cycles: list[dict],
                        ac_id: str, season: str) -> pd.DataFrame:
    """
    For each detected cycle, compute every metric and anomaly flag.
    Also fills OFF-period metrics between consecutive cycles.
    """
    records = []

    for ci, cyc in enumerate(cycles):
        i0 = cyc["on_start_idx"]
        i1 = cyc["on_end_idx"]

        on_slice = df.iloc[i0 : i1 + 1]

        if len(on_slice) == 0:
            continue

        # ── Time metrics ─────────────────────────────────────────────────
        t_start = cyc["on_start_ts"]
        t_end   = cyc["on_end_ts"]
        runtime_sec = (t_end - t_start).total_seconds()
        runtime_min = round(runtime_sec / 60.0, 2)

        # OFF time = gap to next cycle
        if ci + 1 < len(cycles):
            next_on = cycles[ci + 1]["on_start_ts"]
            off_time_sec = (next_on - t_end).total_seconds()
            off_time_min = round(off_time_sec / 60.0, 2)

            if off_time_min > 230:
                off_time_min = CURRENT_OFF_THRESHOLD
            # Temperature during OFF period (warmup)
            j0 = cyc["off_start_idx"]
            j1 = cycles[ci + 1]["on_start_idx"]
            off_slice  = df.iloc[j0 : j1 + 1]
            warmup_outlet = off_slice["outlet_temp"].dropna().values if "outlet_temp" in off_slice else np.array([])
            warmup_times  = np.arange(len(warmup_outlet)) * (off_time_sec / max(len(warmup_outlet)-1,1))
            rate_warmup   = slope_degC_per_min(warmup_outlet, warmup_times)

            cycle_period_min = round(runtime_min + off_time_min, 2)
            duty_cycle_pct   = round(runtime_min / cycle_period_min * 100, 1) if cycle_period_min > 0 else np.nan
        else:
            off_time_min     = np.nan
            rate_warmup      = np.nan
            cycle_period_min = np.nan
            duty_cycle_pct   = np.nan

        # ── Temperature metrics ───────────────────────────────────────────
        inlet_vals  = on_slice["inlet_temp"].dropna().values  if "inlet_temp"  in on_slice else np.array([])
        outlet_vals = on_slice["outlet_temp"].dropna().values if "outlet_temp" in on_slice else np.array([])

        inlet_start  = float(inlet_vals[0])   if len(inlet_vals)  > 0 else np.nan
        inlet_end    = float(inlet_vals[-1])   if len(inlet_vals)  > 0 else np.nan
        outlet_start = float(outlet_vals[0])   if len(outlet_vals) > 0 else np.nan
        outlet_end   = float(outlet_vals[-1])  if len(outlet_vals) > 0 else np.nan

        delta_t_start = round(inlet_start - outlet_start, 2) if not (np.isnan(inlet_start) or np.isnan(outlet_start)) else np.nan
        delta_t_end   = round(inlet_end   - outlet_end,   2) if not (np.isnan(inlet_end)   or np.isnan(outlet_end))   else np.nan
        delta_t_vals  = inlet_vals - outlet_vals[:len(inlet_vals)] if (len(inlet_vals) > 0 and len(outlet_vals) > 0) else np.array([])
        delta_t_mean  = round(float(np.nanmean(delta_t_vals)), 2) if len(delta_t_vals) > 0 else np.nan

        # How much did outlet temp fall during this run?
        cooling_achieved = round(outlet_start - outlet_end, 2) if not (np.isnan(outlet_start) or np.isnan(outlet_end)) else np.nan

        # Rate of cooling: outlet slope during ON period
        outlet_times = np.arange(len(outlet_vals)) * (runtime_sec / max(len(outlet_vals)-1, 1))
        rate_cooling = slope_degC_per_min(outlet_vals, outlet_times)

        # ── Power / current metrics ───────────────────────────────────────
        curr_vals = on_slice["total_current"].dropna().values
        fan_vals  = on_slice["fan_current"].dropna().values if "fan_current" in on_slice else np.array([])

        current_mean_A = round(float(np.nanmean(curr_vals)), 3) if len(curr_vals) > 0 else np.nan
        current_peak_A = round(float(np.nanmax(curr_vals)), 3) if len(curr_vals) > 0 else np.nan
        current_startup_A = round(float(curr_vals[0]), 3) if len(curr_vals) > 0 else np.nan  # inrush proxy

        def to_kw(amps): return round(amps * VOLTAGE * POWER_FACTOR / 1000.0, 3)

        total_kw      = to_kw(current_mean_A) if not np.isnan(current_mean_A) else np.nan
        fan_kw        = to_kw(float(np.nanmean(fan_vals))) if len(fan_vals) > 0 else np.nan
        comp_kw       = round(total_kw - fan_kw, 3) if not (np.isnan(total_kw) or np.isnan(fan_kw)) else np.nan
        energy_kwh    = round(total_kw * runtime_min / 60.0, 4) if not np.isnan(total_kw) else np.nan

        # ── Physics metrics (per-cycle) ───────────────────────────────────
        eff_ratio  = round(delta_t_mean / total_kw, 3) if (not np.isnan(delta_t_mean) and total_kw and total_kw > 0) else np.nan
        cop_proxy  = round((delta_t_mean * 0.38) / comp_kw, 3) if (comp_kw and comp_kw > 0 and not np.isnan(delta_t_mean)) else np.nan
        load_stress = round(fan_kw / total_kw, 3) if (fan_kw and total_kw and total_kw > 0) else np.nan
        if "ac_status" == 0:
         cop_proxy =eff_ratio=load_stress= 0
        # Ambient
        amb_vals = on_slice["ambient_temp"].dropna().values if "ambient_temp" in on_slice else np.array([])
        hum_vals = on_slice["humidity"].dropna().values     if "humidity"     in on_slice else np.array([])
        ambient_mean = round(float(np.nanmean(amb_vals)), 1) if len(amb_vals) > 0 else np.nan
        humidity_mean = round(float(np.nanmean(hum_vals)), 1) if len(hum_vals) > 0 else np.nan

        # ── Anomaly flags ─────────────────────────────────────────────────
        short_cycle_flag    = runtime_min < SHORT_CYCLE_MINS
        long_cycle_flag     = runtime_min > LONG_CYCLE_MINS
        rapid_restart_flag  = (not np.isnan(off_time_min)) and (off_time_min < RAPID_RESTART_MINS)
        poor_cooling_flag   = (not np.isnan(cooling_achieved)) and (cooling_achieved < COOLING_THRESHOLD_DEG)
        high_stress_flag    = (not np.isnan(load_stress)) and (load_stress > 0.35)

        anomaly_count = sum([short_cycle_flag, long_cycle_flag, rapid_restart_flag,
                             poor_cooling_flag, high_stress_flag])
        cycle_health  = max(0, 100 - anomaly_count * 15)

        records.append({
            # Identity
            "ac_id":                ac_id,
            "season":               season,
            "cycle_number":         ci + 1,
            "still_running":        cyc.get("still_running", False),

            # Timestamps
            "cycle_start":          t_start,
            "cycle_end":            t_end,

            # Time metrics
            "runtime_minutes":      runtime_min,
            "off_time_minutes":     off_time_min,
            "cycle_period_minutes": cycle_period_min,
            "duty_cycle_pct":       duty_cycle_pct,

            # Temperature
            "inlet_start_temp":     inlet_start,
            "inlet_end_temp":       inlet_end,
            "outlet_start_temp":    outlet_start,
            "outlet_end_temp":      outlet_end,
            "delta_t_start":        delta_t_start,
            "delta_t_end":          delta_t_end,
            "delta_t_mean":         delta_t_mean,
            "cooling_achieved_degC": cooling_achieved,
            "rate_of_cooling_degC_per_min": rate_cooling,
            "rate_of_warmup_degC_per_min":  rate_warmup,
            "ambient_mean_temp":    ambient_mean,
            "humidity_mean_pct":    humidity_mean,

            # Current / Power
            "current_mean_A":       current_mean_A,
            "current_peak_A":       current_peak_A,
            "current_startup_A":    current_startup_A,
            "total_load_mean_kW":   total_kw,
            "fan_load_mean_kW":     fan_kw,
            "compressor_load_mean_kW": comp_kw,
            "energy_consumed_kWh":  energy_kwh,

            # Physics
            "efficiency_ratio":     eff_ratio,
            "cop_proxy":            cop_proxy,
            "load_stress_index":    load_stress,

            # Anomaly flags
            "short_cycle_flag":     short_cycle_flag,
            "long_cycle_flag":      long_cycle_flag,
            "rapid_restart_flag":   rapid_restart_flag,
            "poor_cooling_flag":    poor_cooling_flag,
            "high_stress_flag":     high_stress_flag,
            "anomaly_count":        anomaly_count,
            "cycle_health_score":   cycle_health,
        })

    return pd.DataFrame(records)


def aggregate_summary(df_cycles: pd.DataFrame) -> dict:
    """Print + return aggregate stats across all cycles."""
    if df_cycles.empty:
        return {}

    n = len(df_cycles)
    rt = df_cycles["runtime_minutes"]
    dc = df_cycles["duty_cycle_pct"].dropna()
    rc = df_cycles["rate_of_cooling_degC_per_min"].dropna()
    rw = df_cycles["rate_of_warmup_degC_per_min"].dropna()
    en = df_cycles["energy_consumed_kWh"].dropna()

    anomaly_rate = df_cycles["anomaly_count"].gt(0).mean() * 100

    # Maintenance prediction: simple heuristic
    # High duty cycle + short cycles + poor cooling = high maintenance urgency
    flags = ["short_cycle_flag","long_cycle_flag","rapid_restart_flag","poor_cooling_flag","high_stress_flag"]
    flag_rate = df_cycles[flags].mean().mean()
    maintenance_score = round(min(100, flag_rate * 100 * 2.5), 1)

    summary = {
        "total_cycles":           n,
        "runtime_mean_min":       round(rt.mean(), 2),
        "runtime_min_min":        round(rt.min(), 2),
        "runtime_max_min":        round(rt.max(), 2),
        "duty_cycle_mean_pct":    round(dc.mean(), 1) if len(dc) > 0 else np.nan,
        "duty_cycle_max_pct":     round(dc.max(), 1)  if len(dc) > 0 else np.nan,
        "rate_cooling_mean_degC_per_min":  round(rc.mean(), 3) if len(rc) > 0 else np.nan,
        "rate_warmup_mean_degC_per_min":   round(rw.mean(), 3) if len(rw) > 0 else np.nan,
        "total_energy_kWh":       round(en.sum(), 4) if len(en) > 0 else np.nan,
        "anomaly_cycle_rate_pct": round(anomaly_rate, 1),
        "maintenance_urgency_score": maintenance_score,
        "short_cycles":           int(df_cycles["short_cycle_flag"].sum()),
        "long_cycles":            int(df_cycles["long_cycle_flag"].sum()),
        "rapid_restarts":         int(df_cycles["rapid_restart_flag"].sum()),
        "poor_cooling_cycles":    int(df_cycles["poor_cooling_flag"].sum()),
    }

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║           CYCLE DETECTION SUMMARY                   ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Total cycles detected       : {summary['total_cycles']}")
    print(f"  Runtime  mean/min/max (min) : {summary['runtime_mean_min']} / {summary['runtime_min_min']} / {summary['runtime_max_min']}")
    print(f"  Duty cycle mean/max (%):    : {summary['duty_cycle_mean_pct']} / {summary['duty_cycle_max_pct']}")
    print(f"  Rate of cooling  (°C/min)   : {summary['rate_cooling_mean_degC_per_min']}")
    print(f"  Rate of warmup   (°C/min)   : {summary['rate_warmup_mean_degC_per_min']}")
    print(f"  Total energy consumed (kWh) : {summary['total_energy_kWh']}")
    print(f"  Anomalous cycles            : {summary['anomaly_cycle_rate_pct']}%")
    print(f"  Maintenance urgency score   : {summary['maintenance_urgency_score']} / 100")
    print(f"  Short cycles (<{SHORT_CYCLE_MINS}min)       : {summary['short_cycles']}")
    print(f"  Long cycles (>{LONG_CYCLE_MINS}min)      : {summary['long_cycles']}")
    print(f"  Rapid restarts (<{RAPID_RESTART_MINS}min off): {summary['rapid_restarts']}")
    print(f"  Poor cooling cycles         : {summary['poor_cooling_cycles']}")

    return summary


def plot_cycles(df_raw: pd.DataFrame, df_cycles: pd.DataFrame,
                ac_id: str, out_path: str):
    """Save a 4-panel diagnostic chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), gridspec_kw={"height_ratios":[2,2,2,1.2]})
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#181c25")
        ax.tick_params(colors="#9b9eb5", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2d3e")

    def color_axes(ax):
        ax.xaxis.label.set_color("#9b9eb5")
        ax.yaxis.label.set_color("#9b9eb5")
        ax.title.set_color("#e8e9ef")

    ts = df_raw["timestamp"]

    # ── Panel 1: Total current + cycle shading ────────────────────────
    ax0 = axes[0]
    ax0.plot(ts, df_raw["total_current"], color="#3ecf8e", lw=1.2, label="Total Current (A)")
    if "fan_current" in df_raw:
        ax0.plot(ts, df_raw["fan_current"], color="#f5a623", lw=0.8, alpha=0.7, label="Fan Current (A)")
    ax0.axhline(CURRENT_ON_THRESHOLD,  color="#ff4b55", lw=0.7, ls="--", alpha=0.6, label=f"ON threshold ({CURRENT_ON_THRESHOLD}A)")
    ax0.axhline(CURRENT_OFF_THRESHOLD, color="#f5a623", lw=0.7, ls="--", alpha=0.6, label=f"OFF threshold ({CURRENT_OFF_THRESHOLD}A)")

    for i, row in df_cycles.iterrows():
        ax0.axvspan(row["cycle_start"], row["cycle_end"],
                    alpha=0.12, color="#4a9eff",
                    label="ON cycle" if i == 0 else "")
        mid = row["cycle_start"] + (row["cycle_end"] - row["cycle_start"]) / 2
        ax0.text(mid, ax0.get_ylim()[1] * 0.85 if ax0.get_ylim()[1] > 0 else 5,
                 f"C{int(row['cycle_number'])}", ha="center", va="top",
                 fontsize=8, color="#4a9eff", fontweight="bold")

    ax0.set_ylabel("Current (A)", color="#9b9eb5")
    ax0.set_title(f"AC Runtime Cycle Analysis — {ac_id}", color="#e8e9ef", fontsize=12)
    ax0.legend(loc="upper right", fontsize=8, facecolor="#1f2436", labelcolor="#9b9eb5", framealpha=0.8)
    color_axes(ax0)

    # ── Panel 2: Temperatures + cycle shading ────────────────────────
    ax1 = axes[1]
    if "inlet_temp" in df_raw:
        ax1.plot(ts, df_raw["inlet_temp"],  color="#4a9eff", lw=1.2, label="Inlet Temp (°C)")
    if "outlet_temp" in df_raw:
        ax1.plot(ts, df_raw["outlet_temp"], color="#a78bfa", lw=1.2, label="Outlet Temp (°C)")
    if "ambient_temp" in df_raw:
        ax1.plot(ts, df_raw["ambient_temp"], color="#f5a623", lw=0.8, ls="--", alpha=0.7, label="Ambient (°C)")

    for _, row in df_cycles.iterrows():
        ax1.axvspan(row["cycle_start"], row["cycle_end"], alpha=0.07, color="#4a9eff")
        # Annotate cooling rate
        if not np.isnan(row["rate_of_cooling_degC_per_min"]):
            mid = row["cycle_start"] + (row["cycle_end"] - row["cycle_start"]) / 2
            ax1.annotate(f'{row["rate_of_cooling_degC_per_min"]:+.2f}°/min',
                         xy=(mid, row["outlet_end_temp"] if not np.isnan(row["outlet_end_temp"]) else 20),
                         fontsize=7, color="#3ecf8e", ha="center",
                         xytext=(0, -14), textcoords="offset points")

    ax1.set_ylabel("Temperature (°C)", color="#9b9eb5")
    ax1.legend(loc="upper right", fontsize=8, facecolor="#1f2436", labelcolor="#9b9eb5", framealpha=0.8)
    color_axes(ax1)

    # ── Panel 3: Delta T per cycle (bar chart overlay) ───────────────
    ax2 = axes[2]
    ax2.plot(ts, df_raw["inlet_temp"] - df_raw["outlet_temp"]
             if ("inlet_temp" in df_raw and "outlet_temp" in df_raw)
             else [np.nan]*len(ts),
             color="#2dd4bf", lw=1.2, label="ΔT (inlet − outlet) °C")
    ax2.axhline(8,  color="#f5a623", lw=0.7, ls="--", alpha=0.7, label="Warning threshold (8°C)")
    ax2.axhline(6,  color="#ff4b55", lw=0.7, ls="--", alpha=0.7, label="Critical threshold (6°C)")
    ax2.axhline(13, color="#3ecf8e", lw=0.7, ls="--", alpha=0.5, label="Healthy target (13°C)")

    for _, row in df_cycles.iterrows():
        ax2.axvspan(row["cycle_start"], row["cycle_end"], alpha=0.07, color="#4a9eff")

    ax2.set_ylabel("ΔT (°C)", color="#9b9eb5")
    ax2.legend(loc="upper right", fontsize=8, facecolor="#1f2436", labelcolor="#9b9eb5", framealpha=0.8)
    color_axes(ax2)

    # ── Panel 4: Duty cycle & runtime bar per cycle ───────────────────
    ax3 = axes[3]
    valid = df_cycles.dropna(subset=["duty_cycle_pct"])
    bars  = ax3.bar(
        range(len(valid)), valid["duty_cycle_pct"],
        color=["#ff4b55" if x > 80 else "#f5a623" if x > 60 else "#3ecf8e"
               for x in valid["duty_cycle_pct"]],
        width=0.6, alpha=0.85
    )
    for bar, (_, row) in zip(bars, valid.iterrows()):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1,
                 f'{row["runtime_minutes"]:.1f}m',
                 ha="center", va="bottom", fontsize=8, color="#9b9eb5")

    ax3.set_xticks(range(len(valid)))
    ax3.set_xticklabels([f'C{int(r["cycle_number"])}' for _, r in valid.iterrows()],
                        fontsize=9, color="#9b9eb5")
    ax3.set_ylabel("Duty cycle (%)", color="#9b9eb5")
    ax3.set_xlabel("Cycle number (bars show runtime in minutes above)", color="#9b9eb5")
    ax3.axhline(60, color="#f5a623", lw=0.7, ls="--", alpha=0.7, label="60% duty cycle")
    ax3.axhline(80, color="#ff4b55", lw=0.7, ls="--", alpha=0.7, label="80% duty cycle")
    ax3.set_ylim(0, 105)
    ax3.set_xlim(-0.5, len(valid) - 0.5)
    ax3.legend(loc="upper right", fontsize=8, facecolor="#1f2436", labelcolor="#9b9eb5", framealpha=0.8)
    ax3.tick_params(axis='x', which='both', bottom=False)
    color_axes(ax3)
    # Override x-axis (shared) to not show raw timestamps here
    ax3.set_xlabel("", color="#9b9eb5")

    fig.autofmt_xdate(rotation=30, ha="right")
    plt.tight_layout(h_pad=0.4)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    print(f"\n✓ Chart saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════
# POSTGRESQL STORAGE  (new — replaces SQLite-only)
# ═══════════════════════════════════════════════════════════════════
def upsert_postgres(df: pd.DataFrame, conn_str: str,
                    schema: str, table: str):
    """
    Upsert runtime cycle records into PostgreSQL.
    Creates table + indexes if they don't exist.
    ON CONFLICT (ac_id, cycle_start) → updates all columns.
    """
    import psycopg2
    from psycopg2.extras import execute_values

    df = df.copy()

    # Convert bool columns to actual Python bool (not numpy bool_)
    bool_cols = ["short_cycle_flag", "long_cycle_flag", "rapid_restart_flag",
                 "poor_cooling_flag", "high_stress_flag", "still_running"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].astype(bool)

    # Replace NaN with None for psycopg2
    df = df.where(pd.notnull(df), other=None)

    # Columns in the exact order the DDL defines
    desired = [
        "ac_id", "season", "cycle_number", "still_running",
        "cycle_start", "cycle_end",
        "runtime_minutes", "off_time_minutes", "cycle_period_minutes", "duty_cycle_pct",
        "inlet_start_temp", "inlet_end_temp", "outlet_start_temp", "outlet_end_temp",
        "delta_t_start", "delta_t_end", "delta_t_mean",
        "cooling_achieved_degC", "rate_of_cooling_degC_per_min", "rate_of_warmup_degC_per_min",
        "ambient_mean_temp", "humidity_mean_pct",
        "current_mean_A", "current_peak_A", "current_startup_A",
        "total_load_mean_kW", "fan_load_mean_kW", "compressor_load_mean_kW",
        "energy_consumed_kWh",
        "efficiency_ratio", "cop_proxy", "load_stress_index",
        "short_cycle_flag", "long_cycle_flag", "rapid_restart_flag",
        "poor_cooling_flag", "high_stress_flag",
        "anomaly_count", "cycle_health_score",
    ]
    cols = [c for c in desired if c in df.columns]
    rows = [tuple(r) for r in df[cols].itertuples(index=False, name=None)]

    col_list   = ", ".join(cols)
    update_set = ", ".join(
        f"{c} = EXCLUDED.{c}"
        for c in cols if c not in ("ac_id", "cycle_start")
    )
    table_safe = table.replace(".", "_")

    sql_upsert = f"""
        INSERT INTO {schema}.{table} ({col_list})
        VALUES %s
        ON CONFLICT (ac_id, cycle_start) DO UPDATE SET {update_set}
    """

    ddl = PG_CREATE_TABLE.format(
        schema=schema, table=table, table_safe=table_safe
    )

    with psycopg2.connect(conn_str) as con:
        with con.cursor() as cur:
            cur.execute(ddl)
            execute_values(cur, sql_upsert, rows, page_size=200)
        con.commit()

    print(f"✓ PostgreSQL: upserted {len(rows)} cycle records "
          f"→ {schema}.{table}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global CURRENT_ON_THRESHOLD, CURRENT_OFF_THRESHOLD, OFF_THRESHOLD, DATA_TZ
    
    # 1. Setup Parser
    parser = argparse.ArgumentParser(description="Detect & analyse AC compressor runtime cycles")
    parser.add_argument("--csv1", required=True, help="CSV1 path")
    parser.add_argument("--csv2", required=True, help="CSV2 path")
    parser.add_argument("--out", default="output/runtime_cycles.parquet")
    parser.add_argument("--ac_id", default="AC-01")
    parser.add_argument("--season", default="summer")
    parser.add_argument("--sqlite", default="output/ac_health.db")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_out", default="output/cycle_chart.png")
    parser.add_argument("--on_thresh", type=float, default=CURRENT_ON_THRESHOLD)
    parser.add_argument("--off_thresh", type=float, default=CURRENT_OFF_THRESHOLD)
    parser.add_argument("--data_tz", default=DATA_TZ,
                         help="Timezone the raw CSV timestamps are recorded in (default: Asia/Kolkata)")
    
    # 2. PostgreSQL Arguments (MUST USE UNDERSCORES)
    parser.add_argument("--pg_host",   default="localhost")
    parser.add_argument("--pg_port",   default="5432")
    parser.add_argument("--pg_db",     default="postgres")
    parser.add_argument("--pg_user",   default="postgres")
    parser.add_argument("--pg_pass",   default=os.environ.get("AC_PG_PASSWORD", ""))
    parser.add_argument("--pg_schema", default="public")
    parser.add_argument("--pg_table",  default="ac_runtime_cycles")

    args = parser.parse_args()

    # 3. Update thresholds from args
    CURRENT_ON_THRESHOLD = args.on_thresh
    OFF_THRESHOLD        = args.off_thresh
    DATA_TZ              = args.data_tz

   # 4. Run your logic
    print(f"── Loading raw data: {args.csv1} + {args.csv2}")
    df_raw = load_and_merge(args.csv1, args.csv2)
    
    # Account for Pandas adding suffixes (_x) during merge collisions
    id_col = None
    for candidate in ["ac_id_x", "ac_id", "device_id"]:
        if candidate in df_raw.columns:
            id_col = candidate
            break

    if id_col is None:
        print(f"❌ Error: Could not find an AC ID column. Columns found: {list(df_raw.columns)}")
        return

    all_cycles_df_list = []

    # GROUP BY EACH AC UNIT to prevent data from mixing!
    for unit_id, group_df in df_raw.groupby(id_col):
        # Skip empty or missing IDs
        if pd.isna(unit_id) or str(unit_id).strip() == "":
            continue
            
        print(f"\n── Analyzing AC: {unit_id} ──")
        
        # Sort chronologically for this specific unit
        group_df = group_df.sort_values("timestamp").reset_index(drop=True)
        
        # Detect cycles for just this unit
        cycles = detect_cycles(group_df)
        
        if not cycles:
            print(f"  ⚠ No cycles detected for {unit_id}.")
            continue
            
        print(f"  ✓ Detected {len(cycles)} cycles for {unit_id}.")
        
        # Build metrics using the specific unit_id
        df_cycles_unit = build_cycle_metrics(group_df, cycles, unit_id, args.season)
        all_cycles_df_list.append(df_cycles_unit)

    # Combine all units together
    if not all_cycles_df_list:
        print("\n⚠ No cycles detected across any AC units. Exiting.")
        return

    # Master dataframe containing every cycle for every device
    df_cycles = pd.concat(all_cycles_df_list, ignore_index=True)
    
    # ── 5. Store to PostgreSQL ──────────────────────────────────────────
    print(f"\n── Storing Runtime Cycles to PostgreSQL ({args.pg_host}/{args.pg_db}) ─────")
    
    try:
        conn_str = f"host={args.pg_host} port={args.pg_port} dbname={args.pg_db} user={args.pg_user} password={args.pg_pass}"
        upsert_postgres(df_cycles, conn_str, args.pg_schema, args.pg_table)
        
    except Exception as e:
        print(f"❌ Failed to store runtime cycles to PostgreSQL: {e}")
    
    # Save output
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_cycles.to_parquet(out, index=False)
    print(f"✓ Saved to {out}")

    if args.plot:
        plot_cycles(df_raw, df_cycles, args.ac_id, args.plot_out)

if __name__ == "__main__":
    main()

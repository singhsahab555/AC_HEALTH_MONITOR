"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AC HEALTH MONITORING DASHBOARD                                             ║
║  Connects to PostgreSQL, reads all 7 tables, auto-refreshes every 30s      ║
║                                                                              ║
║  Tables used:                                                                ║
║    ac_health_report    → Fleet overview cards                                ║
║    ac_anomaly_results  → Anomaly details per cycle                          ║
║    ac_health_metrics   → 5-min rolling time-series                          ║
║    ac_runtime_cycles   → Cycle-level engineering data                        ║
║    ac_forecasts        → Prophet forecast (delta_t, duty_cycle)             ║
║    ac_alarms           → Real-time alarm log                                 ║
║    ac_raw_sensor_data  → Raw current/temperature readings                   ║
║                                                                              ║
║  Run:  streamlit run app.py                                                  ║
║  Deps: pip install streamlit plotly pandas psycopg2-binary sqlalchemy        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import time
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text, inspect
from urllib.parse import quote_plus
import google.generativeai as genai


warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE NAME MAPPING
# ─────────────────────────────────────────────────────────────────────────────
DEVICE_NAMES = {
    "30EDA0267B94": "AC-0666C(6KV S/W Room)",
    "FC012CD6E014": "AC-0665C(33KV S/W Room)",
    "FC012CD6E088": "AC-0668C(33KV S/W Room)",
    "30EDA0267970": "AC-1130C(6KV S/W Room)",
    "FC012CC7FD1C": "AC-1131C(6KV S/W Room)",
    "30EDA0267974": "AC-0664C(6KV S/W Room)",
    "FC012CD6E0A4": "AC-0669C(6KV S/W Room)",
    "FC012CD6E02C": "AC-0667C(6KV S/W Room)"
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATABASE SETUP & HELPER FUNCTIONS (Top of app.py)
# ─────────────────────────────────────────────────────────────────────────────
# (Your existing engine setup like: engine = create_engine(CONN_STR) should be here)

def get_ac_historical_profile(ac_id: str, engine, days: int = 7) -> str:
    """
    Pulls a massive 360-degree statistical summary across 5 IoT tables.
    """
    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
    with engine.begin() as conn:
        # Table 1: Thermodynamics + ML model outputs (IsolationForest / XGBoost)
        q_anom = text("""
            SELECT COUNT(cycle_start), AVG(delta_t_mean), MIN(delta_t_mean), 
                   AVG(cop_proxy), AVG(efficiency_ratio),
                   SUM(CASE WHEN anomaly_level = 'CRITICAL' THEN 1 ELSE 0 END),
                   SUM(if_anomaly),
                   AVG(if_score),
                   SUM(CASE WHEN xgb_anomaly_level = 'CRITICAL' THEN 1 ELSE 0 END),
                   AVG(xgb_prob_critical),
                   MODE() WITHIN GROUP (ORDER BY anomaly_type) FILTER (WHERE anomaly_type <> 'None')
            FROM ac_anomaly_results WHERE ac_id = :ac_id AND cycle_start >= :cutoff
        """)
        res_anom = conn.execute(q_anom, {"ac_id": ac_id, "cutoff": cutoff_date}).fetchone()
        
        if not res_anom or res_anom[0] == 0:
            return f"No historical telemetry found for AC {ac_id} in the last {days} days."

        # Table 2: Runtime
        q_run = text("""
            SELECT SUM(runtime_minutes) / 60.0, SUM(energy_consumed_kwh), AVG(duty_cycle_pct)
            FROM ac_runtime_cycles WHERE ac_id = :ac_id AND cycle_start >= :cutoff
        """)
        res_run = conn.execute(q_run, {"ac_id": ac_id, "cutoff": cutoff_date}).fetchone()

        # Table 3: Health Scores
        q_health = text("""
            SELECT AVG(overall_score), MIN(compressor_score), MIN(refrigerant_score)
            FROM ac_health_metrics WHERE ac_id = :ac_id AND timestamp >= :cutoff
        """)
        res_health = conn.execute(q_health, {"ac_id": ac_id, "cutoff": cutoff_date}).fetchone()

        # Table 4: Alarms
        q_alarms = text("SELECT COUNT(*) FROM ac_alarms WHERE ac_id = :ac_id AND alarm_time >= :cutoff")
        res_alarms = conn.execute(q_alarms, {"ac_id": ac_id, "cutoff": cutoff_date}).fetchone()

        # Table 5: Forecasts
        q_forecast = text("SELECT SUM(predicted_anomaly) FROM ac_forecasts WHERE ac_id = :ac_id AND ds >= NOW() AND ds <= NOW() + INTERVAL '3 days'")
        res_forecast = conn.execute(q_forecast, {"ac_id": ac_id}).fetchone()

    profile = f"""
    [360° Historical Profile for {ac_id} - Last {days} Days]
    1. WORKLOAD: Total Runtime: {(res_run[0] or 0):.1f}h | Energy: {(res_run[1] or 0):.2f} kWh | Duty: {(res_run[2] or 0):.1f}%
    2. PHYSICS: Cycles: {res_anom[0]} | Avg Delta T: {(res_anom[1] or 0):.2f}°C | Worst Delta T: {(res_anom[2] or 0):.2f}°C | Avg COP: {(res_anom[3] or 0):.2f}
    3. COMPONENTS: Avg Health: {(res_health[0] or 0):.1f}/100 | Worst Compressor: {(res_health[1] or 0):.1f}/100 | Worst Ref: {(res_health[2] or 0):.1f}/100
    4. RISKS: Critical Cycles (rule-based): {res_anom[5] or 0} | Alarms Fired: {res_alarms[0] or 0} | 3-Day Forecasted Anom: {res_forecast[0] or 0}
    5. ML MODEL OUTPUT: IsolationForest flagged {res_anom[6] or 0}/{res_anom[0]} cycles as anomalous (avg anomaly score: {(res_anom[7] or 0):.3f}) | XGBoost flagged {res_anom[8] or 0}/{res_anom[0]} cycles CRITICAL (avg critical probability: {(res_anom[9] or 0):.1%}) | Most common anomaly type: {res_anom[10] or 'None'}
    """
    return profile

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AC Health Monitor",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ───────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"]  { background: #F5F5F0; }
[data-testid="stSidebar"]           { background: #1C1C1E; }
[data-testid="stSidebar"] *         { color: #F5F5F0 !important; }
[data-testid="stHeader"]            { background: transparent; }

/* ── Metric cards ───────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background    : #FFFFFF;
    border        : 1px solid #E5E5EA;
    border-radius : 14px;
    padding       : 16px 18px 12px;
    box-shadow    : 0 1px 4px rgba(0,0,0,0.06);
}
div[data-testid="metric-container"] label {
    font-size     : 11px !important;
    font-weight   : 600  !important;
    letter-spacing: .07em !important;
    text-transform: uppercase !important;
    color         : #8E8E93 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size   : 28px !important;
    font-weight : 400  !important;
    color       : #1C1C1E !important;
}

/* ── Fleet device cards ─────────────────────────────────────── */
.device-card {
    background    : #FFFFFF;
    border-radius : 18px;
    padding       : 20px 18px;
    text-align    : center;
    box-shadow    : 0 2px 10px rgba(0,0,0,0.07);
    border        : 2px solid transparent;
    cursor        : pointer;
    transition    : transform .15s, box-shadow .15s;
    position      : relative;
    overflow      : hidden;
}
.device-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.12); }
.card-healthy  { border-color: #30D158; }
.card-warning  { border-color: #FF9F0A; }
.card-critical { border-color: #FF3B30; }
.card-off      { border-color: #C7C7CC; }
.card-selected { box-shadow: 0 0 0 3px #007AFF !important; }

/* Pulsing glow for CRITICAL */
@keyframes criticalPulse {
    0%   { box-shadow: 0 0 0 0   rgba(255,59,48,0.6); }
    70%  { box-shadow: 0 0 0 12px rgba(255,59,48,0); }
    100% { box-shadow: 0 0 0 0   rgba(255,59,48,0); }
}
.pulse-critical { animation: criticalPulse 1.8s infinite; }

/* Status dot */
.status-dot {
    width: 14px; height: 14px; border-radius: 50%;
    display: inline-block; margin-right: 6px; vertical-align: middle;
}
.dot-healthy  { background: #30D158; }
.dot-warning  { background: #FF9F0A; }
.dot-critical { background: #FF3B30; }
.dot-off      { background: #C7C7CC; }

/* ── Alert banners ──────────────────────────────────────────── */
@keyframes slideIn {
    from { transform: translateX(-20px); opacity: 0; }
    to   { transform: translateX(0);     opacity: 1; }
}
.alert-banner {
    border-radius : 12px;
    padding       : 14px 18px;
    margin-bottom : 10px;
    display       : flex;
    align-items   : center;
    gap           : 14px;
    animation     : slideIn 0.35s ease-out;
    border-left   : 5px solid;
}
.alert-critical {
    background    : #FFF1F0;
    border-color  : #FF3B30;
    color         : #C0392B;
}
.alert-warning {
    background    : #FFFBF0;
    border-color  : #FF9F0A;
    color         : #8B6914;
}
.alert-healthy {
    background    : #F0FFF5;
    border-color  : #30D158;
    color         : #1A6B35;
}

/* ── Section titles ─────────────────────────────────────────── */
.section-title {
    font-size     : 11px;
    font-weight   : 700;
    color         : #8E8E93;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom : 12px;
}

/* ── Health bars ─────────────────────────────────────────────── */
.hbar-row  { display:flex; align-items:center; gap:10px; margin-bottom:9px; }
.hbar-name { font-size:13px; color:#3A3A3C; min-width:96px; }
.hbar-bg   { flex:1; background:#F2F2F7; border-radius:999px; height:9px; overflow:hidden; }
.hbar-fill { height:9px; border-radius:999px; transition: width .4s; }
.hbar-pct  { font-size:12px; font-weight:600; min-width:38px; text-align:right; font-family:monospace; }

/* ── Cards ───────────────────────────────────────────────────── */
.white-card {
    background    : #FFFFFF;
    border        : 1px solid #E5E5EA;
    border-radius : 16px;
    padding       : 18px 20px;
    margin-bottom : 14px;
    box-shadow    : 0 1px 4px rgba(0,0,0,0.05);
}
.gray-card {
    background    : #F2F2F7;
    border        : 1px solid #E5E5EA;
    border-radius : 16px;
    padding       : 18px 20px;
    margin-bottom : 14px;
}

/* ── Tabs ─────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 6px; background: transparent;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 10px !important;
    padding: 8px 18px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    background: #F2F2F7 !important;
    color: #3A3A3C !important;
    border: none !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #007AFF !important;
    color: #FFFFFF !important;
}

/* ── Badge ───────────────────────────────────────────────────── */
.badge {
    display: inline-block; padding: 3px 10px;
    border-radius: 999px; font-size: 11px; font-weight: 600;
}
.badge-healthy  { background:#E8F8EE; color:#1A6B35; }
.badge-warning  { background:#FFF3DC; color:#8B6914; }
.badge-critical { background:#FFF0EF; color:#C0392B; }
.badge-ok       { background:#EEF4FF; color:#1A4DB1; }

/* ── Hide streamlit chrome ───────────────────────────────────── */
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR / ICON HELPERS
# ─────────────────────────────────────────────────────────────────────────────
STATUS_COLOR  = {"CRITICAL":"#FF3B30","WARNING":"#FF9F0A","HEALTHY":"#30D158","OFF":"#C7C7CC","OK":"#30D158"}
STATUS_ICON   = {"CRITICAL":"🔴","WARNING":"🟡","HEALTHY":"🟢","OFF":"⚫","OK":"🟢"}
STATUS_CARD   = {"CRITICAL":"card-critical pulse-critical","WARNING":"card-warning","HEALTHY":"card-healthy","OFF":"card-off","OK":"card-healthy"}
BAR_COLOR     = lambda v: "#30D158" if v>=80 else "#FF9F0A" if v>=60 else "#FF3B30"

def badge_html(s):
    cls = {"CRITICAL":"badge-critical","WARNING":"badge-warning","HEALTHY":"badge-healthy","OK":"badge-ok"}.get(str(s).upper(),"badge-ok")
    return f'<span class="badge {cls}">{s}</span>'

# ─────────────────────────────────────────────────────────────────────────────
# DB CONNECTION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_engine(host, port, dbname, user, password):
    url = f"postgresql+psycopg2://{user}:{quote_plus(password)}@{host}:{port}/{dbname}"
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 10})

def test_connection(engine):
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True, None
    except Exception as e:
        return False, str(e)

def get_tables(engine):
    try:
        insp = inspect(engine)
        return insp.get_table_names()
    except:
        return []

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (TTL=30 for auto-refresh)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_table(_engine, table: str, limit: int = 0) -> pd.DataFrame:
    try:
        q = f"SELECT * FROM {table}" + (f" LIMIT {limit}" if limit else "") + ";"
        df = pd.read_sql(q, con=_engine)
        # Parse only KNOWN datetime columns and force them to tz-naive
        # Asia/Kolkata wall-clock time, regardless of whether the source
        # PostgreSQL column was timestamptz (tz-aware) or timestamp (tz-naive).
        #
        # IMPORTANT: this must be an exact-name allowlist, not a substring
        # match like `"time" in col`. A substring match against "time"
        # also fires on numeric columns such as runtime_minutes,
        # off_time_minutes, runtime_hrs, etc. (they contain "time" as a
        # substring), silently corrupting them into NaT/garbage datetimes
        # and breaking any later .sum() on them. Exact names only.
        DATETIME_COLUMNS = {
            "timestamp", "created_at", "ds", "alarm_time",
            "cycle_start", "cycle_end", "report_timestamp",
            "ack_at", "first_seen", "last_seen", "alarm_time_utc",
        }
        for col in df.columns:
            if col in DATETIME_COLUMNS:
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.dt.tz is not None:
                        parsed = parsed.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
                    df[col] = parsed
                except Exception:
                    pass
        return df
    except Exception as e:
        return pd.DataFrame()

def load_all(_engine) -> dict:
    tables = ["ac_health_report","ac_anomaly_results","ac_health_metrics",
              "ac_runtime_cycles","ac_forecasts","ac_alarms","ac_raw_sensor_data"]
    return {t: load_table(_engine, t) for t in tables}

# ─────────────────────────────────────────────────────────────────────────────
# TIME RANGE FILTER  (Grafana-style — applies to every time-series column)
# ─────────────────────────────────────────────────────────────────────────────
TIME_COL = {
    "ac_health_report":   "report_timestamp",
    "ac_anomaly_results": "cycle_start",
    "ac_health_metrics":  "timestamp",
    "ac_runtime_cycles":  "cycle_start",
    "ac_forecasts":       "ds",
    "ac_alarms":          "alarm_time",
    "ac_raw_sensor_data": "timestamp",
}

def apply_time_range(data: dict, start: datetime, end: datetime) -> dict:
    """Filter every table's primary timestamp column to [start, end].
    
    IMPORTANT: ac_health_report is EXEMPTED from time-range filtering.
    It is a current-state summary table (1 row per device) — not a time-series.
    Filtering it by time range causes it to go empty whenever a narrow window
    like "Last 1 hour" is selected, which breaks ALL downstream dashboard logic
    (fleet cards, KPI strip, device selectors, AI chatbot context, etc.).
    """
    # These tables contain the CURRENT state of each device, not historical events.
    # Always use their full/latest data regardless of the selected time window.
    TIME_EXEMPT = {"ac_health_report"}

    out = {}
    for name, df in data.items():
        if name in TIME_EXEMPT:
            out[name] = df        # pass through untouched
            continue
        col = TIME_COL.get(name)
        if col and col in df.columns and len(df):
            mask = (df[col] >= start) & (df[col] <= end)
            out[name] = df.loc[mask].copy()
        else:
            out[name] = df
    return out

@st.cache_data(ttl=300)
def get_global_bounds(_engine) -> tuple:
    """Earliest and latest timestamp across all tables — used as the
    absolute slider bounds and to report the data collection start date.
    Every timestamp is normalised to tz-naive Asia/Kolkata wall-clock time
    BEFORE comparison, since PostgreSQL columns can be a mix of
    timestamptz and timestamp, which pandas/python cannot compare directly."""
    def _to_naive(ts) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("Asia/Kolkata").tz_localize(None)
        return ts

    bounds = []
    for table, col in TIME_COL.items():
        try:
            q = text(f"SELECT MIN({col}) AS lo, MAX({col}) AS hi FROM {table}")
            with _engine.connect() as c:
                r = c.execute(q).fetchone()
            if r and r[0] is not None:
                bounds.append((_to_naive(r[0]), _to_naive(r[1])))
        except Exception:
            continue
    if not bounds:
        now = pd.Timestamp.now()
        return now - pd.Timedelta(days=7), now
    lo = min(b[0] for b in bounds)
    hi = max(b[1] for b in bounds)
    return lo, hi

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — connection + navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ❄️ AC Monitor")
    st.markdown("---")

    with st.expander("🔌 Database connection", expanded=False):
        pg_host = st.text_input("Host",     value="localhost",     key="pg_host")
        pg_port = st.number_input("Port",   value=5432,           key="pg_port", step=1)
        pg_db   = st.text_input("Database", value="ac_automation", key="pg_db")
        pg_user = st.text_input("User",     value="postgres",      key="pg_user")
        pg_pass = st.text_input("Password", type="password",       value="Sumit@2006", key="pg_pass")

    engine = get_engine(pg_host, int(pg_port), pg_db, pg_user, pg_pass)
    ok, err = test_connection(engine)

    if ok:
        st.markdown('<p style="color:#30D158;font-weight:600;">● Connected</p>', unsafe_allow_html=True)
        avail_tables = get_tables(engine)
        st.caption(f"Tables found: {len(avail_tables)}")
    else:
        st.markdown(f'<p style="color:#FF3B30;font-weight:600;">● Not connected</p>', unsafe_allow_html=True)
        st.caption(f"Error: {err}")

    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠  Fleet Overview",
        "📊  AC Details",
        "⚠️  Anomalies & Forecast",
        "📈  Fleet Health Summary",
        "🔔  Alarms",
    ])

    st.markdown("---")
    auto_ref = st.toggle("Auto-refresh (300s)", value=True)
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()

    # ── ⏱ TIME RANGE EDITOR (Grafana-style) ────────────────────────────────
    st.markdown("---")
    st.markdown("### ⏱ Time range")

    DATA_START, DATA_END = get_global_bounds(engine) if ok else (
        pd.Timestamp.now() - pd.Timedelta(days=7), pd.Timestamp.now())

    QUICK_RANGES = {
        "Last 1 hour":    pd.Timedelta(hours=1),
        "Last 6 hours":   pd.Timedelta(hours=6),
        "Last 24 hours":  pd.Timedelta(hours=24),
        "Last 3 days":    pd.Timedelta(days=3),
        "Last 7 days":    pd.Timedelta(days=7),
        "All time":       None,
        "Custom":         "custom",
    }
    quick_choice = st.selectbox("Quick range", list(QUICK_RANGES.keys()), index=5, key="quick_range")

    if QUICK_RANGES[quick_choice] == "custom":
        c_start, c_end = st.columns(2)
        with c_start:
            range_start_date = st.date_input("Start date", value=DATA_START.date(), key="rs_d")
            range_start_time = st.time_input("Start time", value=DATA_START.time(), key="rs_t")
        with c_end:
            range_end_date   = st.date_input("End date",   value=DATA_END.date(),   key="re_d")
            range_end_time   = st.time_input("End time",   value=DATA_END.time(),   key="re_t")
        RANGE_START = pd.Timestamp.combine(range_start_date, range_start_time)
        RANGE_END   = pd.Timestamp.combine(range_end_date,   range_end_time)
    elif QUICK_RANGES[quick_choice] is None:   # All time
        RANGE_START, RANGE_END = DATA_START, DATA_END
    else:
        RANGE_END   = DATA_END
        RANGE_START = max(DATA_START, DATA_END - QUICK_RANGES[quick_choice])

    st.markdown(f"""<div style="background:#2C2C2E;border-radius:10px;padding:9px 12px;
        font-size:11px;color:#AEAEB2;margin-top:4px;">
        <b style="color:#5AC8FA;">{RANGE_START.strftime('%Y-%m-%d %H:%M')}</b>
        &nbsp;→&nbsp;
        <b style="color:#5AC8FA;">{RANGE_END.strftime('%Y-%m-%d %H:%M')}</b>
    </div>""", unsafe_allow_html=True)

    st.caption(f"📅 Data collection started: **{DATA_START.strftime('%Y-%m-%d %H:%M')}**")

    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
if not ok:
    st.error("Cannot connect to PostgreSQL. Check credentials in the sidebar.")
    st.stop()

DATA = load_all(engine)
DATA = apply_time_range(DATA, RANGE_START, RANGE_END)

hr   = DATA["ac_health_report"]      # 8 rows
ar   = DATA["ac_anomaly_results"]    # 889 rows
hm   = DATA["ac_health_metrics"]     # 2050 rows
rc   = DATA["ac_runtime_cycles"]     # 889 rows
fc   = DATA["ac_forecasts"]          # 1980 rows
alm  = DATA["ac_alarms"]             # 0 rows (empty)
raw  = DATA["ac_raw_sensor_data"]    # 22843 rows

ALL_DEVICES = sorted(hr["ac_id"].unique()) if len(hr) else sorted(ar["ac_id"].unique())

# ─────────────────────────────────────────────────────────────────────────────
# DERIVED ENGINE — Cumulative hours · Efficiency ratio · Power usage
# Pure additive computation — does not touch any existing health/anomaly logic.
# Uses the FULL (unfiltered) table for cumulative hours / start date so the
# lifetime counter never resets when you narrow the time-range picker, while
# power usage and efficiency respect the selected window.
# ─────────────────────────────────────────────────────────────────────────────
FULL = load_all(engine)  # unfiltered, full history — needed for true cumulative stats

@st.cache_data(ttl=60)
def compute_device_lifetime_stats(_rc_full: pd.DataFrame, _devices: list) -> pd.DataFrame:
    """Per-device: first_seen (start date), cumulative_hours since start,
    lifetime energy (kWh), lifetime avg efficiency ratio."""
    rows = []
    for dev in _devices:
        sub = _rc_full[_rc_full["ac_id"]==dev] if len(_rc_full) else pd.DataFrame()
        if len(sub) and "cycle_start" in sub.columns:
            first_seen   = sub["cycle_start"].min()
            last_seen    = sub["cycle_end"].max() if "cycle_end" in sub.columns else sub["cycle_start"].max()
            cum_minutes  = sub["runtime_minutes"].sum() if "runtime_minutes" in sub.columns else 0
            cum_hours    = cum_minutes / 60.0
            lifetime_kwh = sub["energy_consumed_kwh"].sum() if "energy_consumed_kwh" in sub.columns else 0
            avg_eff      = sub["efficiency_ratio"].mean()   if "efficiency_ratio"   in sub.columns else np.nan
            total_cycles = len(sub)
            days_active  = max(1, (datetime.now() - first_seen.to_pydatetime()).days)
        else:
            first_seen=last_seen=pd.NaT; cum_hours=lifetime_kwh=0; avg_eff=np.nan; total_cycles=0; days_active=1
        rows.append(dict(
            ac_id=dev, start_date=first_seen, last_seen=last_seen,
            cumulative_hours=round(cum_hours,2), lifetime_kwh=round(lifetime_kwh,3),
            avg_efficiency_ratio=round(avg_eff,3) if pd.notna(avg_eff) else 0.0,
            total_cycles=total_cycles, days_active=days_active,
            avg_hours_per_day=round(cum_hours/days_active,2) if days_active else 0,
        ))
    return pd.DataFrame(rows)

LIFETIME = compute_device_lifetime_stats(FULL.get("ac_runtime_cycles", pd.DataFrame()), ALL_DEVICES)
FLEET_START_DATE = LIFETIME["start_date"].min() if len(LIFETIME) and LIFETIME["start_date"].notna().any() else None

# ── Outlet end temperature per device ────────────────────────────────────────
@st.cache_data(ttl=30)
def compute_outlet_temps(_raw_full: pd.DataFrame, _devices: list) -> pd.DataFrame:
    """Latest outlet_end_temp (or nearest equivalent column) per device.
    Uses the FULL unfiltered raw sensor table so the reading is always
    available regardless of the selected time window."""
    # Auto-detect which column name the pipeline uses
    temp_col = None
    if len(_raw_full):
        for candidate in ["outlet_end_temp"]:
            if candidate in _raw_full.columns:
                temp_col = candidate
                break

    rows = []
    for dev in _devices:
        sub = _raw_full[_raw_full["ac_id"] == dev] if len(_raw_full) else pd.DataFrame()
        if len(sub) and temp_col:
            sorted_sub = sub.sort_values("timestamp") if "timestamp" in sub.columns else sub
            latest_val = round(float(sorted_sub.iloc[-1][temp_col]), 2)
            avg_val    = round(float(sub[temp_col].mean()), 2)
        else:
            latest_val = avg_val = None
        rows.append(dict(ac_id=dev, outlet_temp_latest=latest_val, outlet_temp_avg=avg_val,
                         temp_col_found=temp_col))
    return pd.DataFrame(rows)

OUTLET_TEMPS = compute_outlet_temps(FULL.get("ac_runtime_cycles", pd.DataFrame()), ALL_DEVICES)

VOLTAGE_ASSUMED = 230  # single-phase V, matches 01_ingest_clean.py / db.py defaults

def power_kw_from_current(current_a: pd.Series, power_factor: float = 0.85) -> pd.Series:
    """Mirrors current_to_kw() in 01_ingest_clean.py exactly — same formula,
    so dashboard power numbers always match what the pipeline computed."""
    return (current_a * VOLTAGE_ASSUMED * power_factor) / 1000.0

@st.cache_data(ttl=30)
def compute_power_usage(_rc: pd.DataFrame, _devices: list) -> pd.DataFrame:
    """Power usage per device for the CURRENT time-range selection
    (energy_consumed_kwh already exists in runtime_cycles — we sum it,
    plus derive average power draw in kW for the window)."""
    rows = []
    for dev in _devices:
        sub = _rc[_rc["ac_id"]==dev] if len(_rc) else pd.DataFrame()
        if len(sub):
            energy_kwh   = sub["energy_consumed_kwh"].sum() if "energy_consumed_kwh" in sub.columns else 0
            avg_kw       = sub["total_load_mean_kw"].mean() if "total_load_mean_kw" in sub.columns else 0
            peak_kw      = sub["total_load_mean_kw"].max()  if "total_load_mean_kw" in sub.columns else 0
            runtime_hrs  = (sub["runtime_minutes"].sum()/60.0) if "runtime_minutes" in sub.columns else 0
        else:
            energy_kwh=avg_kw=peak_kw=runtime_hrs=0
        rows.append(dict(ac_id=dev, energy_kwh=round(energy_kwh,3),
                          avg_power_kw=round(avg_kw,3), peak_power_kw=round(peak_kw,3),
                          runtime_hrs=round(runtime_hrs,2)))
    return pd.DataFrame(rows)

POWER = compute_power_usage(rc, ALL_DEVICES)   # `rc` defined just below — see note

# ── Fleet-wide aggregates (shown as a persistent strip on every page) ────────
FLEET_CUM_HOURS   = LIFETIME["cumulative_hours"].sum()
FLEET_ENERGY_KWH  = POWER["energy_kwh"].sum()
FLEET_AVG_KW      = POWER["avg_power_kw"].mean()
FLEET_PEAK_KW     = POWER["peak_power_kw"].max()
FLEET_AVG_EFF     = LIFETIME["avg_efficiency_ratio"].replace(0, np.nan).mean()

def render_global_strip():
    """Persistent top strip — start date, cumulative hours, efficiency, power.
    Called at the top of every page body."""
    start_str = FLEET_START_DATE.strftime("%d %b %Y") if FLEET_START_DATE is not None and pd.notna(FLEET_START_DATE) else "—"
    eff_str   = f"{FLEET_AVG_EFF:.2f}" if pd.notna(FLEET_AVG_EFF) else "—"
    st.markdown(f"""
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px;">
      <div style="flex:1;min-width:170px;background:linear-gradient(135deg,#1C1C1E,#2C2C2E);
           border-radius:14px;padding:14px 16px;color:#F5F5F0;">
        <div style="font-size:10px;color:#8E8E93;text-transform:uppercase;letter-spacing:.06em;">📅 Data collection start</div>
        <div style="font-size:19px;font-weight:500;margin-top:4px;">{start_str}</div>
      </div>
      <div style="flex:1;min-width:170px;background:linear-gradient(135deg,#007AFF,#0A5FCC);
           border-radius:14px;padding:14px 16px;color:#fff;">
        <div style="font-size:10px;color:#D6E8FF;text-transform:uppercase;letter-spacing:.06em;">⏱ Fleet cumulative hours</div>
        <div style="font-size:19px;font-weight:500;margin-top:4px;">{FLEET_CUM_HOURS:,.1f} h</div>
      </div>
      <div style="flex:1;min-width:170px;background:linear-gradient(135deg,#30D158,#1FA847);
           border-radius:14px;padding:14px 16px;color:#fff;">
        <div style="font-size:10px;color:#DFFAE6;text-transform:uppercase;letter-spacing:.06em;">⚡ Fleet efficiency ratio</div>
        <div style="font-size:19px;font-weight:500;margin-top:4px;">{eff_str}</div>
      </div>
      <div style="flex:1;min-width:170px;background:linear-gradient(135deg,#BF5AF2,#9A3FD1);
           border-radius:14px;padding:14px 16px;color:#fff;">
        <div style="font-size:10px;color:#F0DFFF;text-transform:uppercase;letter-spacing:.06em;">🔌 Total energy (window)</div>
        <div style="font-size:19px;font-weight:500;margin-top:4px;">{FLEET_ENERGY_KWH:,.2f} kWh</div>
      </div>
      <div style="flex:1;min-width:170px;background:linear-gradient(135deg,#FF9F0A,#E07F00);
           border-radius:14px;padding:14px 16px;color:#fff;">
        <div style="font-size:10px;color:#FFE9CC;text-transform:uppercase;letter-spacing:.06em;">📊 Avg fleet power draw</div>
        <div style="font-size:19px;font-weight:500;margin-top:4px;">{FLEET_AVG_KW:.2f} kW</div>
      </div>
    </div>
    """, unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████  PAGE 1 — FLEET OVERVIEW  ████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────
if "Fleet Overview" in page:

    st.markdown("# Fleet Overview")
    st.markdown(f"<p style='color:#8E8E93;font-size:13px;margin-top:-10px;'>"
                f"{len(ALL_DEVICES)} AC units · {datetime.now().strftime('%d %b %Y, %H:%M')}</p>",
                unsafe_allow_html=True)

    render_global_strip()


    # ── KPI strip ────────────────────────────────────────────────────────────
    if len(hr):
        n_crit = (hr["overall_status"]=="CRITICAL").sum()
        n_warn = (hr["overall_status"]=="WARNING").sum()
        n_ok   = (hr["overall_status"]=="HEALTHY").sum()
        avg_sc = hr["avg_overall_score"].mean()
        avg_ar = hr["anomaly_rate_pct"].mean()
        total_cyc = hr["total_cycles"].sum()
    else:
        n_crit=n_warn=n_ok=0; avg_sc=avg_ar=0; total_cyc=0

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    with k1: st.metric("Fleet health",  f"{avg_sc:.1f}/100")
    with k2: st.metric("🔴 Critical",   n_crit,  delta=f"{n_crit} urgent" if n_crit else None, delta_color="inverse")
    with k3: st.metric("🟡 Warning",    n_warn,  delta=f"{n_warn} advisory" if n_warn else None, delta_color="inverse")
    with k4: st.metric("🟢 Healthy",    n_ok)
    with k5: st.metric("Total cycles",  f"{total_cyc:,}")
    with k6: st.metric("Avg anomaly %", f"{avg_ar:.1f}%", delta_color="inverse")

    # ── Global animated alert banner ─────────────────────────────────────────
    if n_crit > 0:
        # Get raw IDs and translate them to names
        raw_crit = hr[hr["overall_status"]=="CRITICAL"]["ac_id"].tolist()
        crit_devices = [DEVICE_NAMES.get(x, x[-8:]) for x in raw_crit]
        
        st.markdown(f"""<div class="alert-banner alert-critical">
            <span style="font-size:22px;">🚨</span>
            <div>
                <div style="font-weight:700;font-size:15px;">CRITICAL ALERT — {n_crit} unit(s) need immediate attention</div>
                <div style="font-size:12px;margin-top:3px;">Affected: {', '.join(crit_devices)}</div>
            </div>
        </div>""", unsafe_allow_html=True)
        
    elif n_warn > 0:
        # Get raw IDs and translate them to names
        raw_warn = hr[hr["overall_status"]=="WARNING"]["ac_id"].tolist()
        warn_devices = [DEVICE_NAMES.get(x, x[-8:]) for x in raw_warn]
        
        st.markdown(f"""<div class="alert-banner alert-warning">
            <span style="font-size:22px;">⚠️</span>
            <div>
                <div style="font-weight:700;font-size:15px;">Warning — {n_warn} unit(s) require preventive attention</div>
                <div style="font-size:12px;margin-top:3px;">Affected: {', '.join(warn_devices)}</div>
            </div>
        </div>""", unsafe_allow_html=True)
        
    else:
        st.markdown("""<div class="alert-banner alert-healthy">
            <span style="font-size:22px;">✅</span>
            <div><div style="font-weight:700;font-size:15px;">All systems nominal — Fleet is healthy</div></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Device cards grid (4 per row) ────────────────────────────────────────
    st.markdown('<div class="section-title">AC units</div>', unsafe_allow_html=True)
    cols_per_row = 4
    rows = [ALL_DEVICES[i:i+cols_per_row] for i in range(0, len(ALL_DEVICES), cols_per_row)]

    for row_devs in rows:
        row_cols = st.columns(cols_per_row)
        for col, dev in zip(row_cols, row_devs):
            row = hr[hr["ac_id"]==dev].iloc[0] if len(hr[hr["ac_id"]==dev]) else None
            status  = row["overall_status"]  if row is not None else "OFF"
            score   = row["avg_overall_score"] if row is not None else 0
            anom_r  = row["anomaly_rate_pct"]  if row is not None else 0
            top_a   = str(row["top_anomaly_type"])[:22] if row is not None else "—"
            avg_cop = row["avg_cop_proxy"]     if row is not None else 0
            avg_dt  = row["avg_delta_t"]       if row is not None else 0
            cycles  = row["total_cycles"]      if row is not None else 0

            life_row  = LIFETIME[LIFETIME["ac_id"]==dev]
            pow_row   = POWER[POWER["ac_id"]==dev]
            out_row   = OUTLET_TEMPS[OUTLET_TEMPS["ac_id"]==dev]
            cum_hrs   = float(life_row["cumulative_hours"].iloc[0])  if len(life_row) else 0.0
            eff_ratio = float(life_row["avg_efficiency_ratio"].iloc[0]) if len(life_row) else 0.0
            avg_kw    = float(pow_row["avg_power_kw"].iloc[0])       if len(pow_row)  else 0.0
            outlet_t  = out_row["outlet_temp_latest"].iloc[0]        if len(out_row) and out_row["outlet_temp_latest"].iloc[0] is not None else None

            card_cls = STATUS_CARD.get(status, "card-off")
            icon = STATUS_ICON.get(status,"⚫")
            sc_color = STATUS_COLOR.get(status,"#C7C7CC")
            # Lookup the name; fallback to short ID if not found
            dev_name = DEVICE_NAMES.get(dev, dev[-8:]) 

            with col:
                st.markdown(f"""<div class="device-card {card_cls}">
                    <div style="font-size:11px;font-weight:700;color:#8E8E93;letter-spacing:.03em;margin-bottom:6px;">{dev_name}</div>
                    <div style="font-size:32px;margin:4px 0;">{icon}</div>
                    <div style="font-size:13px;font-weight:700;color:{sc_color};margin-bottom:8px;">{status}</div>
                    <div style="font-size:28px;font-weight:300;color:#1C1C1E;margin-bottom:2px;">{score:.0f}
                        <span style="font-size:13px;color:#8E8E93;">/100</span></div>
                    <div style="font-size:11px;color:#8E8E93;margin-bottom:8px;">{cycles} cycles</div>
                    <div style="text-align:left;font-size:11px;color:#3A3A3C;background:#F8F8F8;border-radius:8px;padding:8px 10px;">
                        <div style="margin-bottom:3px;">📉 Δt: <b>{avg_dt:.2f}°C</b></div>
                        <div style="margin-bottom:3px;">⚡ COP: <b>{avg_cop:.2f}</b></div>
                        <div style="margin-bottom:3px;">🧮 Efficiency: <b>{eff_ratio:.2f}</b></div>
                        <div style="margin-bottom:3px;">⏱ Cum. hrs: <b>{cum_hrs:,.1f} h</b></div>
                        <div style="margin-bottom:3px;">🔋 Avg power: <b>{avg_kw:.2f} kW</b></div>
                        <div style="margin-bottom:3px;color:{'#FF3B30' if (outlet_t is not None and outlet_t > 30) else '#3A3A3C'};">🌡 Outlet temp: <b>{f"{outlet_t:.1f}°C" if outlet_t is not None else "—"}</b></div>
                        <div style="color:{'#FF3B30' if anom_r>20 else '#8E8E93'};">⚠ Anomaly: <b>{anom_r:.1f}%</b></div>
                    </div>
                    <div style="font-size:10px;color:#8E8E93;margin-top:8px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{top_a}</div>
                </div>""", unsafe_allow_html=True)
        # spacer for empty columns
        for _ in range(cols_per_row - len(row_devs)):
            with row_cols[len(row_devs) + _]:
                st.empty()

    # ── Fleet-wide score bar chart ────────────────────────────────────────────
    if len(hr):
        st.markdown("---")
        c1, c2 = st.columns([1.6, 1])
        with c1:
            st.markdown('<div class="section-title">Average overall score — all devices</div>', unsafe_allow_html=True)
            bar_df = hr.sort_values("avg_overall_score")
            colors = [STATUS_COLOR.get(s,"#C7C7CC") for s in bar_df["overall_status"]]
            fig = go.Figure(go.Bar(
                y=[DEVICE_NAMES.get(x, x[-8:]) for x in bar_df["ac_id"]],
                x=bar_df["avg_overall_score"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:.0f}" for v in bar_df["avg_overall_score"]],
                textposition="outside",
            ))
            fig.add_vline(x=80, line=dict(color="#30D158",dash="dot",width=1.5))
            fig.add_vline(x=60, line=dict(color="#FF9F0A",dash="dot",width=1.5))
            fig.update_layout(height=280, margin=dict(l=0,r=30,t=0,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False, xaxis=dict(range=[0,110],gridcolor="#F2F2F7",tickfont=dict(size=10)),
                yaxis=dict(tickfont=dict(size=11)))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

        with c2:
            st.markdown('<div class="section-title">Status distribution</div>', unsafe_allow_html=True)
            status_counts = hr["overall_status"].value_counts().reset_index()
            status_counts.columns = ["status","count"]
            pie_colors = [STATUS_COLOR.get(s,"#C7C7CC") for s in status_counts["status"]]
            fig2 = go.Figure(go.Pie(
                labels=status_counts["status"], values=status_counts["count"],
                marker_colors=pie_colors, hole=0.55,
                textinfo="label+value", textfont=dict(size=12),
            ))
            fig2.update_layout(height=240, margin=dict(l=0,r=0,t=0,b=0),
                paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████  PAGE 2 — AC DETAILS  ████████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────
elif "AC Details" in page:

    render_global_strip()

    dev = st.selectbox("Select AC unit", ALL_DEVICES,
                       format_func=lambda x: f"{DEVICE_NAMES.get(x, x[-8:])} ({x[-8:]})")

    # Latest status
    dev_hr  = hr[hr["ac_id"]==dev].iloc[0]  if len(hr[hr["ac_id"]==dev])  else None
    dev_hm  = hm[hm["ac_id"]==dev].sort_values("timestamp") if "timestamp" in hm.columns else pd.DataFrame()
    dev_rc  = rc[rc["ac_id"]==dev].sort_values("cycle_start") if "cycle_start" in rc.columns else pd.DataFrame()
    dev_ar  = ar[ar["ac_id"]==dev].sort_values("cycle_start") if "cycle_start" in ar.columns else pd.DataFrame()

    status  = dev_hr["overall_status"] if dev_hr is not None else "OFF"
    score   = dev_hr["avg_overall_score"] if dev_hr is not None else 0
    sc_col  = STATUS_COLOR.get(status,"#C7C7CC")

    # Status banner
    banner_bg    = {"CRITICAL":"#FFF1F0","WARNING":"#FFFBF0","HEALTHY":"#F0FFF5","OFF":"#F2F2F7"}
    banner_bord  = {"CRITICAL":"#FF3B30","WARNING":"#FF9F0A","HEALTHY":"#30D158","OFF":"#C7C7CC"}
    banner_title = {"CRITICAL":"🔴  Critical — Immediate attention required",
                    "WARNING" :"🟡  Warning — Preventive action advised",
                    "HEALTHY" :"🟢  Healthy — All systems nominal",
                    "OFF"     :"⚫  Unit is offline"}

    st.markdown(f"""<div style="background:{banner_bg.get(status,'#F2F2F7')};
        border:1.5px solid {banner_bord.get(status,'#C7C7CC')};border-radius:14px;
        padding:16px 22px;display:flex;align-items:center;gap:16px;margin-bottom:18px;">
      <div style="flex:1;">
        <div style="font-weight:700;font-size:16px;color:#1C1C1E;">{banner_title.get(status,'Unknown')}</div>
        <div style="font-size:12px;color:#8E8E93;margin-top:4px;">Unit: {dev} · Last report: {
            str(dev_hr["report_timestamp"])[:16] if dev_hr is not None and "report_timestamp" in dev_hr else "—"}</div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:36px;font-weight:300;color:{sc_col};">{score:.1f}
            <span style="font-size:14px;color:#8E8E93;">/100</span></div>
        <div style="font-size:10px;color:#8E8E93;">avg health score</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    if dev_hr is not None:
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        with c1: st.metric("Avg Δt",        f"{dev_hr['avg_delta_t']:.2f}°C")
        with c2: st.metric("Avg COP proxy", f"{dev_hr['avg_cop_proxy']:.2f}")
        with c3: st.metric("Avg duty cycle",f"{dev_hr['avg_duty_cycle']:.1f}%")
        with c4: st.metric("Total cycles",  int(dev_hr["total_cycles"]))
        with c5: st.metric("IF anomalies",  int(dev_hr["if_anomalies"]))
        with c6: st.metric("Anomaly rate",  f"{dev_hr['anomaly_rate_pct']:.1f}%", delta_color="inverse")

    # ── ⏱ Cumulative hours · 🧮 Efficiency · 🔋 Power usage (device level) ───
    dev_life = LIFETIME[LIFETIME["ac_id"]==dev].iloc[0] if len(LIFETIME[LIFETIME["ac_id"]==dev]) else None
    dev_pow  = POWER[POWER["ac_id"]==dev].iloc[0]       if len(POWER[POWER["ac_id"]==dev])       else None

    if dev_life is not None:
        sd_str = dev_life["start_date"].strftime("%d %b %Y, %H:%M") if pd.notna(dev_life["start_date"]) else "—"
        p1,p2,p3,p4,p5,p6 = st.columns(6)
        with p1: st.metric("📅 In service since", sd_str)
        with p2: st.metric("⏱ Cumulative hours",  f"{dev_life['cumulative_hours']:,.1f} h")
        with p3: st.metric("📆 Days active",       int(dev_life["days_active"]))
        with p4: st.metric("🧮 Avg efficiency",    f"{dev_life['avg_efficiency_ratio']:.2f}")
        with p5: st.metric("🔋 Lifetime energy",   f"{dev_life['lifetime_kwh']:,.2f} kWh")
        with p6: st.metric("⚡ Avg power (window)",f"{dev_pow['avg_power_kw']:.2f} kW" if dev_pow is not None else "—")

    st.markdown("---")

    # ── Charts + health bars ───────────────────────────────────────────────────
    chart_col, health_col = st.columns([1.7, 1])

    with chart_col:
        # --- Temperature trend (health_metrics rolling) ----------------------
        st.markdown('<div class="white-card"><div class="section-title">Temperature trend — delta_t rolling (ac_health_metrics)</div>', unsafe_allow_html=True)
        if len(dev_hm) > 0 and "delta_t" in dev_hm.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dev_hm["timestamp"], y=dev_hm["delta_t"],
                name="Δt raw", line=dict(color="#C7C7CC",width=1), opacity=0.5))
            if "delta_t_rolling_12" in dev_hm.columns:
                fig.add_trace(go.Scatter(x=dev_hm["timestamp"], y=dev_hm["delta_t_rolling_12"],
                    name="Δt rolling 12", line=dict(color="#FF9500",width=2)))
            fig.add_hline(y=8, line=dict(color="#FF9F0A",dash="dot",width=1), annotation_text="Warning 8°C",annotation_font_size=9)
            fig.add_hline(y=6, line=dict(color="#FF3B30",dash="dot",width=1), annotation_text="Critical 6°C",annotation_font_size=9)
            fig.update_layout(height=220, margin=dict(l=0,r=10,t=10,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(size=10),orientation="h",y=1.12),
                xaxis=dict(gridcolor="#F2F2F7",tickfont=dict(size=9)),
                yaxis=dict(gridcolor="#F2F2F7",tickfont=dict(size=9),ticksuffix="°"))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Duty cycle over runtime cycles ----------------------------------
        st.markdown('<div class="white-card"><div class="section-title">Duty cycle % — runtime cycles</div>', unsafe_allow_html=True)
        if len(dev_rc) > 0 and "duty_cycle_pct" in dev_rc.columns:
            colors_dc = ["#FF3B30" if v>85 else "#FF9F0A" if v>70 else "#30D158"
                         for v in dev_rc["duty_cycle_pct"]]
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=dev_rc["cycle_start"], y=dev_rc["duty_cycle_pct"],
                marker_color=colors_dc, name="Duty cycle %"))
            fig2.add_hline(y=80, line=dict(color="#FF9F0A",dash="dot",width=1))
            fig2.update_layout(height=180, margin=dict(l=0,r=10,t=10,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False, bargap=0.2,
                xaxis=dict(gridcolor="#F2F2F7",tickfont=dict(size=9)),
                yaxis=dict(gridcolor="#F2F2F7",tickfont=dict(size=9),range=[0,110],ticksuffix="%"))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

        # --- COP proxy trend -------------------------------------------------
        st.markdown('<div class="white-card"><div class="section-title">COP proxy over time (ac_anomaly_results)</div>', unsafe_allow_html=True)
        if len(dev_rc) > 0 and "cop_proxy" in dev_rc.columns:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=dev_rc["cycle_start"], y=dev_rc["cop_proxy"],
                name="COP proxy", line=dict(color="#5AC8FA",width=2),
                fill="tozeroy", fillcolor="rgba(90,200,250,0.07)"))
            fig3.update_layout(height=170, margin=dict(l=0,r=10,t=10,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis=dict(gridcolor="#F2F2F7",tickfont=dict(size=9)),
                yaxis=dict(gridcolor="#F2F2F7",tickfont=dict(size=9)))
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Efficiency ratio trend (NEW) -------------------------------------
        st.markdown('<div class="white-card"><div class="section-title">🧮 Efficiency ratio over time (Δt ÷ total load)</div>', unsafe_allow_html=True)
        if len(dev_rc) > 0 and "efficiency_ratio" in dev_rc.columns:
            fig_eff = go.Figure()
            fig_eff.add_trace(go.Scatter(x=dev_rc["cycle_start"], y=dev_rc["efficiency_ratio"],
                name="Efficiency ratio", line=dict(color="#34C759",width=2),
                fill="tozeroy", fillcolor="rgba(52,199,89,0.07)"))
            eff_mean = dev_rc["efficiency_ratio"].mean()
            fig_eff.add_hline(y=eff_mean, line=dict(color="#8E8E93",dash="dot",width=1),
                annotation_text=f"avg {eff_mean:.2f}", annotation_font_size=9)
            fig_eff.update_layout(height=170, margin=dict(l=0,r=10,t=10,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis=dict(gridcolor="#F2F2F7",tickfont=dict(size=9)),
                yaxis=dict(gridcolor="#F2F2F7",tickfont=dict(size=9)))
            st.plotly_chart(fig_eff, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Power usage trend (NEW) -------------------------------------------
        st.markdown('<div class="white-card"><div class="section-title">🔋 Power usage — total load (kW) per cycle</div>', unsafe_allow_html=True)
        if len(dev_rc) > 0 and "total_load_mean_kw" in dev_rc.columns:
            fig_pw = go.Figure()
            fig_pw.add_trace(go.Bar(x=dev_rc["cycle_start"], y=dev_rc["total_load_mean_kw"],
                name="Total load kW", marker_color="#FF9F0A", opacity=0.75))
            if "compressor_load_mean_kw" in dev_rc.columns:
                fig_pw.add_trace(go.Scatter(x=dev_rc["cycle_start"], y=dev_rc["compressor_load_mean_kw"],
                    name="Compressor kW", line=dict(color="#1C1C1E",width=1.4)))
            fig_pw.update_layout(height=190, margin=dict(l=0,r=10,t=10,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(size=10),orientation="h",y=1.15), bargap=0.2,
                xaxis=dict(gridcolor="#F2F2F7",tickfont=dict(size=9)),
                yaxis=dict(gridcolor="#F2F2F7",tickfont=dict(size=9),ticksuffix=" kW"))
            st.plotly_chart(fig_pw, use_container_width=True, config={"displayModeBar":False})
            if dev_pow is not None:
                st.markdown(f"""<div style="display:flex;gap:18px;font-size:12px;color:#3A3A3C;margin-top:6px;">
                    <span>Energy this window: <b>{dev_pow['energy_kwh']:.2f} kWh</b></span>
                    <span>Peak power: <b>{dev_pow['peak_power_kw']:.2f} kW</b></span>
                    <span>Runtime: <b>{dev_pow['runtime_hrs']:.1f} h</b></span>
                </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with health_col:
        # --- Health bars from latest health_metrics row ----------------------
        latest_hm = dev_hm.iloc[-1] if len(dev_hm) else None
        latest_ar = dev_ar.iloc[-1] if len(dev_ar) else None

        st.markdown('<div class="white-card"><div class="section-title">Component health scores</div>', unsafe_allow_html=True)
        components = [
            ("Compressor",  "compressor_score"),
            ("Refrigerant", "refrigerant_score"),
            ("Capacitor",   "capacitor_score"),
            ("Air filter",  "air_filter_score"),
            ("Fan motor",   "fan_motor_score"),
        ]
        for label, col_name in components:
            val = 0
            if latest_hm is not None and col_name in latest_hm.index:
                val = float(latest_hm[col_name] or 0)
            color = BAR_COLOR(val)
            st.markdown(f"""<div class="hbar-row">
                <span class="hbar-name">{label}</span>
                <div class="hbar-bg"><div class="hbar-fill" style="width:{val:.0f}%;background:{color};"></div></div>
                <span class="hbar-pct" style="color:{color};">{val:.0f}%</span>
            </div>""", unsafe_allow_html=True)

        # Overall rolling score sparkline
        if len(dev_hm) > 0 and "overall_score_rolling_12" in dev_hm.columns:
            st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)
            st.markdown('<div style="font-size:11px;color:#8E8E93;margin-bottom:4px;">ROLLING HEALTH SCORE</div>', unsafe_allow_html=True)
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(y=dev_hm["overall_score_rolling_12"].tail(60),
                line=dict(color="#007AFF",width=2), fill="tozeroy",
                fillcolor="rgba(0,122,255,0.07)"))
            fig_s.add_hline(y=80, line=dict(color="#30D158",dash="dot",width=1))
            fig_s.update_layout(height=100, margin=dict(l=0,r=0,t=0,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis=dict(visible=False),
                yaxis=dict(range=[40,105],tickfont=dict(size=8),gridcolor="#F2F2F7"))
            st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Derived metrics snapshot ----------------------------------------
        st.markdown('<div class="gray-card"><div class="section-title">Latest snapshot</div>', unsafe_allow_html=True)
        ref = latest_hm if latest_hm is not None else latest_ar
        snap = [
            ("Δt",               f"{ref.get('delta_t', ref.get('delta_t_mean', 0)):.2f}°C"),
            ("Efficiency ratio", f"{ref.get('efficiency_ratio',0):.3f}"),
            ("Load stress idx",  f"{ref.get('load_stress_index',0)*100:.1f}%"),
            ("COP proxy",        f"{ref.get('cop_proxy',0):.3f}"),
            ("Overall score",    f"{ref.get('overall_score',0):.1f}/100"),
        ] if ref is not None else []
        for lbl, val in snap:
            st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:5px 0;
                border-bottom:1px solid #E5E5EA;font-size:12px;">
                <span style="color:#8E8E93;">{lbl}</span>
                <span style="font-weight:600;font-family:monospace;color:#1C1C1E;">{val}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Raw current from ac_raw_sensor_data ───────────────────────────────────
    dev_raw = raw[raw["ac_id"]==dev].sort_values("timestamp") if len(raw) else pd.DataFrame()
    if len(dev_raw) > 0 and "total_current" in dev_raw.columns:
        st.markdown('<div class="white-card"><div class="section-title">Raw current draw — ac_raw_sensor_data (last 500 readings)</div>', unsafe_allow_html=True)
        dev_raw_tail = dev_raw.tail(500)
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=dev_raw_tail["timestamp"], y=dev_raw_tail["total_current"],
            name="Total current A", line=dict(color="#BF5AF2",width=1.2),
            fill="tozeroy", fillcolor="rgba(191,90,242,0.06)"))
        if "fan_current" in dev_raw_tail.columns:
            fig_r.add_trace(go.Scatter(x=dev_raw_tail["timestamp"], y=dev_raw_tail["fan_current"],
                name="Fan current A", line=dict(color="#5AC8FA",width=1)))
        fig_r.update_layout(height=200, margin=dict(l=0,r=10,t=10,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(size=10),orientation="h",y=1.12),
            xaxis=dict(gridcolor="#F2F2F7",tickfont=dict(size=9)),
            yaxis=dict(gridcolor="#F2F2F7",tickfont=dict(size=9),ticksuffix="A"))
        st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Cycle table ────────────────────────────────────────────────────────────
    if len(dev_rc) > 0:
        with st.expander("📋 Runtime cycles table"):
            show_cols = ["cycle_number","cycle_start","cycle_end","runtime_minutes",
                         "duty_cycle_pct","delta_t_mean","current_mean_a","cop_proxy",
                         "energy_consumed_kwh","cycle_health_score","anomaly_count"]
            avail = [c for c in show_cols if c in dev_rc.columns]
            st.dataframe(dev_rc[avail].sort_values("cycle_start",ascending=False),
                         use_container_width=True, hide_index=True, height=280)

# ─────────────────────────────────────────────────────────────────────────────
# ██████████████  PAGE 3 — ANOMALIES & FORECAST  ██████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────
elif "Anomalies" in page:

    render_global_strip()

    tab_a, tab_f = st.tabs(["⚠️  Anomaly results", "🔮  Forecast"])

    # ── ANOMALY RESULTS ───────────────────────────────────────────────────────
    with tab_a:
        st.markdown("## Anomaly detection results")
        st.caption("Source: ac_anomaly_results — XGBoost + Isolation Forest per cycle")

        # Filter controls
        f1, f2, f3 = st.columns(3)
        with f1:
            sel_dev_a = st.multiselect("Filter by device", ALL_DEVICES, default=ALL_DEVICES, key="sel_dev_a")
        with f2:
            lvl_opts = sorted(ar["anomaly_level"].dropna().astype(str).unique()) if "anomaly_level" in ar.columns else []
            sel_lvl  = st.multiselect("Anomaly level", lvl_opts, default=lvl_opts, key="sel_lvl")
        with f3:
            if_opts = [0,1] if "if_anomaly" in ar.columns else []
            sel_if  = st.multiselect("IF anomaly", if_opts, default=if_opts, key="sel_if")

        df_a = ar.copy()
        if sel_dev_a: df_a = df_a[df_a["ac_id"].isin(sel_dev_a)]
        if sel_lvl:   df_a = df_a[df_a["anomaly_level"].isin(sel_lvl)]

        # KPI row
        a1,a2,a3,a4,a5 = st.columns(5)
        with a1: st.metric("Total cycles",   len(df_a))
        with a2: st.metric("CRITICAL cycles", (df_a["anomaly_level"]=="CRITICAL").sum() if "anomaly_level" in df_a.columns else 0)
        with a3: st.metric("WARNING cycles",  (df_a["anomaly_level"]=="WARNING").sum()  if "anomaly_level" in df_a.columns else 0)
        with a4: st.metric("IF anomalies",    df_a["if_anomaly"].sum() if "if_anomaly" in df_a.columns else 0)
        with a5: st.metric("Avg COP proxy",   f"{df_a['cop_proxy'].mean():.2f}" if "cop_proxy" in df_a.columns else "—")

        c1, c2 = st.columns(2)

        # Anomaly level over time
        with c1:
            st.markdown('<div class="white-card"><div class="section-title">Anomaly level distribution over cycles</div>', unsafe_allow_html=True)
            if "anomaly_level" in df_a.columns:
                lvl_ct = df_a["anomaly_level"].value_counts().reset_index()
                lvl_ct.columns = ["level","count"]
                colors = [STATUS_COLOR.get(l,"#C7C7CC") for l in lvl_ct["level"]]
                fig = go.Figure(go.Bar(x=lvl_ct["level"],y=lvl_ct["count"],
                    marker_color=colors, text=lvl_ct["count"], textposition="outside"))
                fig.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False, bargap=0.3,
                    xaxis=dict(tickfont=dict(size=11)),
                    yaxis=dict(gridcolor="#F2F2F7",tickfont=dict(size=10)))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
            st.markdown('</div>', unsafe_allow_html=True)

        # IF score distribution
        with c2:
            st.markdown('<div class="white-card"><div class="section-title">Isolation Forest score distribution</div>', unsafe_allow_html=True)
            if "if_score" in df_a.columns:
                fig2 = go.Figure()
                fig2.add_trace(go.Histogram(x=df_a["if_score"], nbinsx=40,
                    marker_color="#BF5AF2", opacity=0.8))
                fig2.add_vline(x=0, line=dict(color="#FF3B30",dash="dot",width=1.5),
                    annotation_text="Anomaly threshold", annotation_font_size=9)
                fig2.update_layout(height=220, margin=dict(l=0,r=0,t=10,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False, bargap=0.05,
                    xaxis=dict(title="IF score",tickfont=dict(size=10)),
                    yaxis=dict(gridcolor="#F2F2F7",tickfont=dict(size=10)))
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
            st.markdown('</div>', unsafe_allow_html=True)

        # Per-device anomaly type breakdown
        st.markdown('<div class="white-card"><div class="section-title">Anomaly type frequency by device</div>', unsafe_allow_html=True)
        if "anomaly_type" in df_a.columns and "ac_id" in df_a.columns:
            atype_df = df_a[df_a["anomaly_type"]!="None"].copy()
            types_exp = atype_df.assign(atype=atype_df["anomaly_type"].str.split(" | ")).explode("atype")
            freq = types_exp.groupby(["ac_id","atype"]).size().reset_index(name="count")
            if len(freq):
                fig3 = px.bar(freq, x="ac_id", y="count", color="atype",
                    barmode="stack", color_discrete_sequence=px.colors.qualitative.Set2)
                fig3.update_layout(height=260, margin=dict(l=0,r=10,t=10,b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(font=dict(size=10),title=None),
                    xaxis=dict(tickfont=dict(size=10)),
                    yaxis=dict(gridcolor="#F2F2F7",tickfont=dict(size=10)))
                st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

        # Full table
        with st.expander("📋 Full anomaly results table"):
            show = ["ac_id","cycle_start","cycle_end","runtime_minutes","delta_t_mean",
                    "cop_proxy","efficiency_ratio","overall_score","anomaly_level",
                    "if_anomaly","if_score","xgb_anomaly_level","xgb_prob_critical","anomaly_type"]
            avail = [c for c in show if c in df_a.columns]
            st.dataframe(df_a[avail].sort_values("cycle_start",ascending=False).head(300),
                         use_container_width=True, hide_index=True, height=320)

    # ── FORECAST ──────────────────────────────────────────────────────────────
    with tab_f:
        st.markdown("## Prophet forecast")
        st.caption("Source: ac_forecasts — predicted values with confidence bands")

        if len(fc) == 0:
            st.info("No forecast data available yet.")
        else:
            fc_dev    = st.selectbox("Device", [d for d in ALL_DEVICES if d in fc["ac_id"].values], format_func=lambda x: f"{DEVICE_NAMES.get(x, x[-8:])}", key="fc_dev")
            fc_metric = st.selectbox("Metric", fc["metric"].unique(), key="fc_metric")

            sub_fc = fc[(fc["ac_id"]==fc_dev) & (fc["metric"]==fc_metric)].sort_values("ds")

            a1,a2,a3 = st.columns(3)
            n_anom = sub_fc["predicted_anomaly"].sum()
            first_anom = sub_fc[sub_fc["predicted_anomaly"]==1]["ds"].min()
            with a1: st.metric("Forecast points",   len(sub_fc))
            with a2: st.metric("Predicted anomalies", int(n_anom),
                               delta_color="inverse" if n_anom>0 else "off")
            with a3: st.metric("First anomaly at", str(first_anom)[:16] if pd.notna(first_anom) else "None")

            # Forecast chart
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(
                x=pd.concat([sub_fc["ds"], sub_fc["ds"][::-1]]),
                y=pd.concat([sub_fc["yhat_upper"], sub_fc["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(0,122,255,0.08)",
                line=dict(color="rgba(0,0,0,0)"), name="Confidence band"))
            fig_f.add_trace(go.Scatter(x=sub_fc["ds"], y=sub_fc["yhat"],
                name="Forecast", line=dict(color="#007AFF",width=2)))
            anom_pts = sub_fc[sub_fc["predicted_anomaly"]==1]
            if len(anom_pts):
                fig_f.add_trace(go.Scatter(x=anom_pts["ds"], y=anom_pts["yhat"],
                    mode="markers", name="Predicted anomaly",
                    marker=dict(color="#FF3B30",size=9,symbol="x-thin",line=dict(width=2,color="#FF3B30"))))
            if fc_metric == "delta_t_mean":
                fig_f.add_hline(y=8, line=dict(color="#FF9F0A",dash="dot",width=1),
                    annotation_text="Warning 8°C", annotation_font_size=9)
                fig_f.add_hline(y=6, line=dict(color="#FF3B30",dash="dot",width=1),
                    annotation_text="Critical 6°C", annotation_font_size=9)
            elif fc_metric == "duty_cycle_pct":
                fig_f.add_hline(y=80, line=dict(color="#FF9F0A",dash="dot",width=1),
                    annotation_text="Warning 80%", annotation_font_size=9)

            fig_f.update_layout(height=380, margin=dict(l=0,r=10,t=10,b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(size=10)),
                xaxis=dict(gridcolor="#F2F2F7",tickfont=dict(size=10)),
                yaxis=dict(gridcolor="#F2F2F7",tickfont=dict(size=10)))
            st.plotly_chart(fig_f, use_container_width=True, config={"displayModeBar":False})

            # All-device summary
            st.markdown('<div class="section-title" style="margin-top:10px;">All devices — forecast summary</div>', unsafe_allow_html=True)
            rows = []
            for ac in sorted(fc["ac_id"].unique()):
                for m in fc["metric"].unique():
                    s = fc[(fc["ac_id"]==ac)&(fc["metric"]==m)]
                    na = s["predicted_anomaly"].sum()
                    fa = s[s["predicted_anomaly"]==1]["ds"].min()
                    rows.append({"Device":ac,"Metric":m,"Points":len(s),
                                 "Anomalies":int(na),"Rate":f"{na/max(len(s),1)*100:.1f}%",
                                 "First anomaly":str(fa)[:16] if pd.notna(fa) else "—"})
            sum_df = pd.DataFrame(rows)
            st.dataframe(sum_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# ██████████████  PAGE 4 — FLEET HEALTH SUMMARY  ██████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────
elif "Fleet Health" in page:

    render_global_strip()

    st.markdown("## Fleet health summary")
    st.caption("Source: ac_health_report + ac_health_metrics")

    if len(hr) == 0:
        st.info("No health report data found.")
        st.stop()

    # ── Big KPIs ──────────────────────────────────────────────────────────────
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    with k1: st.metric("Avg fleet score",   f"{hr['avg_overall_score'].mean():.1f}/100")
    with k2: st.metric("Avg COP proxy",     f"{hr['avg_cop_proxy'].mean():.2f}")
    with k3: st.metric("Avg Δt",            f"{hr['avg_delta_t'].mean():.2f}°C")
    with k4: st.metric("Avg duty cycle",    f"{hr['avg_duty_cycle'].mean():.1f}%")
    with k5: st.metric("Total cycles",      int(hr["total_cycles"].sum()))
    with k6: st.metric("Fleet anomaly rate",f"{hr['anomaly_rate_pct'].mean():.1f}%", delta_color="inverse")

    st.markdown("---")

    c1, c2 = st.columns(2)

    # ── COP proxy gauge chart ─────────────────────────────────────────────────
    with c1:
        st.markdown('<div class="white-card"><div class="section-title">Average COP proxy — fleet gauge</div>', unsafe_allow_html=True)
        avg_cop = hr["avg_cop_proxy"].mean()
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(avg_cop, 2),
            delta={"reference": 3.0, "increasing": {"color":"#30D158"}, "decreasing": {"color":"#FF3B30"}},
            gauge={
                "axis"    : {"range":[0,12], "tickfont":{"size":10}},
                "bar"     : {"color":"#007AFF", "thickness":0.22},
                "bgcolor" : "#F2F2F7",
                "steps"   : [
                    {"range":[0,2],   "color":"#FFE5E5"},
                    {"range":[2,4],   "color":"#FFF5E0"},
                    {"range":[4,12],  "color":"#E8F8EE"},
                ],
                "threshold": {"line":{"color":"#FF3B30","width":3},"thickness":0.7,"value":2},
            },
            title={"text":"COP proxy", "font":{"size":14,"color":"#8E8E93"}},
        ))
        fig_g.update_layout(height=260, margin=dict(l=20,r=20,t=20,b=10),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#1C1C1E"))
        st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Status pie ────────────────────────────────────────────────────────────
    with c2:
        st.markdown('<div class="white-card"><div class="section-title">Fleet status distribution</div>', unsafe_allow_html=True)
        sc = hr["overall_status"].value_counts().reset_index()
        sc.columns = ["status","count"]
        fig_p = go.Figure(go.Pie(
            labels=sc["status"], values=sc["count"], hole=0.6,
            marker_colors=[STATUS_COLOR.get(s,"#C7C7CC") for s in sc["status"]],
            textinfo="label+percent", textfont=dict(size=12),
        ))
        fig_p.update_layout(height=240, margin=dict(l=0,r=0,t=20,b=0),
            paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Per-device comparison chart ───────────────────────────────────────────
    st.markdown('<div class="white-card"><div class="section-title">Per-device metrics comparison</div>', unsafe_allow_html=True)
    metrics_to_plot = ["avg_overall_score","avg_cop_proxy","avg_delta_t","avg_duty_cycle","anomaly_rate_pct"]
    avail_m = [m for m in metrics_to_plot if m in hr.columns]
    metric_choice = st.selectbox("Metric", avail_m, key="cmp_metric",
        format_func=lambda x: x.replace("_"," ").title())
    bar_colors = [STATUS_COLOR.get(s,"#C7C7CC") for s in hr["overall_status"]]
    fig_cmp = go.Figure(go.Bar(
        x=[DEVICE_NAMES.get(x, x[-8:]) for x in hr["ac_id"]], y=hr[metric_choice],
        marker_color=bar_colors,
        text=[f"{v:.1f}" for v in hr[metric_choice]],
        textposition="outside",
    ))
    fig_cmp.update_layout(height=260, margin=dict(l=0,r=10,t=10,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, bargap=0.3,
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#F2F2F7",tickfont=dict(size=10)))
    st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar":False})
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Rolling overall score trend (all devices) ─────────────────────────────
    if len(hm) > 0 and "overall_score_rolling_12" in hm.columns and "timestamp" in hm.columns:
        st.markdown('<div class="white-card"><div class="section-title">Overall score rolling trend — all devices (ac_health_metrics)</div>', unsafe_allow_html=True)
        fig_tr = go.Figure()
        palette = px.colors.qualitative.Set2
        for i, ac in enumerate(ALL_DEVICES):
            sub = hm[hm["ac_id"]==ac].sort_values("timestamp")
            if len(sub) > 0 and "overall_score_rolling_12" in sub.columns:
                fig_tr.add_trace(go.Scatter(x=sub["timestamp"], y=sub["overall_score_rolling_12"],
                    name=DEVICE_NAMES.get(ac, ac[-8:]), line=dict(width=1.5, color=palette[i % len(palette)])))
        fig_tr.add_hline(y=80, line=dict(color="#30D158",dash="dot",width=1))
        fig_tr.add_hline(y=60, line=dict(color="#FF9F0A",dash="dot",width=1))
        fig_tr.update_layout(height=280, margin=dict(l=0,r=10,t=10,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(size=10),orientation="h"),
            xaxis=dict(gridcolor="#F2F2F7",tickfont=dict(size=9)),
            yaxis=dict(gridcolor="#F2F2F7",tickfont=dict(size=9),range=[40,105]))
        st.plotly_chart(fig_tr, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── ⏱🧮🔋 Cumulative hours / Efficiency / Power — fleet table (NEW) ──────
    st.markdown('<div class="white-card"><div class="section-title">⏱ Cumulative hours · 🧮 Efficiency · 🔋 Power usage — per device</div>', unsafe_allow_html=True)
    merged = LIFETIME.merge(POWER, on="ac_id", how="left")
    merged["ac_id_short"] = merged["ac_id"].map(lambda x: DEVICE_NAMES.get(x, x[-8:]))
    disp_pw = merged[["ac_id_short","start_date","cumulative_hours","avg_hours_per_day",
                       "avg_efficiency_ratio","energy_kwh","avg_power_kw","peak_power_kw"]].copy()
    disp_pw.columns = ["Device","In service since","Cumulative hrs","Avg hrs/day",
                        "Avg efficiency","Energy (window) kWh","Avg power kW","Peak power kW"]
    st.dataframe(disp_pw, use_container_width=True, hide_index=True,
        column_config={
            "In service since": st.column_config.DatetimeColumn(format="DD MMM YYYY, HH:mm"),
            "Cumulative hrs":    st.column_config.NumberColumn(format="%.1f h"),
            "Avg hrs/day":       st.column_config.NumberColumn(format="%.2f h"),
            "Avg efficiency":    st.column_config.NumberColumn(format="%.2f"),
            "Energy (window) kWh": st.column_config.NumberColumn(format="%.2f"),
            "Avg power kW":      st.column_config.NumberColumn(format="%.2f"),
            "Peak power kW":     st.column_config.NumberColumn(format="%.2f"),
        })
    st.markdown('</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="white-card"><div class="section-title">🧮 Avg efficiency ratio — fleet gauge</div>', unsafe_allow_html=True)
        eff_val = LIFETIME["avg_efficiency_ratio"].replace(0, np.nan).mean()
        eff_val = float(eff_val) if pd.notna(eff_val) else 0.0
        fig_eg = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(eff_val, 2),
            gauge={
                "axis"   : {"range":[0,10], "tickfont":{"size":10}},
                "bar"    : {"color":"#34C759", "thickness":0.22},
                "bgcolor": "#F2F2F7",
                "steps"  : [{"range":[0,2],"color":"#FFE5E5"},
                            {"range":[2,5],"color":"#FFF5E0"},
                            {"range":[5,10],"color":"#E8F8EE"}],
            },
            title={"text":"Efficiency ratio (Δt/kW)", "font":{"size":13,"color":"#8E8E93"}},
        ))
        fig_eg.update_layout(height=230, margin=dict(l=20,r=20,t=20,b=10),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#1C1C1E"))
        st.plotly_chart(fig_eg, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="white-card"><div class="section-title">🔋 Energy usage by device (window)</div>', unsafe_allow_html=True)
        fig_e = go.Figure(go.Bar(
            x=[DEVICE_NAMES.get(x, x[-8:]) for x in POWER.sort_values("energy_kwh")["ac_id"]],
            y=POWER.sort_values("energy_kwh")["energy_kwh"],
            marker_color="#BF5AF2",
            text=[f"{v:.2f}" for v in POWER.sort_values("energy_kwh")["energy_kwh"]],
            textposition="outside",
        ))
        fig_e.update_layout(height=230, margin=dict(l=0,r=10,t=10,b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, bargap=0.3,
            xaxis=dict(tickfont=dict(size=10)),
            yaxis=dict(gridcolor="#F2F2F7",tickfont=dict(size=10),title="kWh"))
        st.plotly_chart(fig_e, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Full health report table ───────────────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-top:6px;">Full health report table</div>', unsafe_allow_html=True)
    disp = hr.copy()
    disp["ac_id"] = disp["ac_id"].map(lambda x: DEVICE_NAMES.get(x, x[-8:]))
    st.dataframe(disp, use_container_width=True, hide_index=True,
                 column_config={
                     "avg_overall_score": st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.1f"),
                     "anomaly_rate_pct":  st.column_config.NumberColumn(format="%.1f %%"),
                     "avg_cop_proxy":     st.column_config.NumberColumn(format="%.2f"),
                     "avg_delta_t":       st.column_config.NumberColumn(format="%.2f °C"),
                 })

# ─────────────────────────────────────────────────────────────────────────────
# ██████████████  PAGE 5 — ALARMS  ████████████████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────
elif "Alarms" in page:

    render_global_strip()

    st.markdown("## Alarm log")
    st.caption("Source: ac_alarms — real-time anomaly alerts with acknowledgement status")

    if len(alm) == 0:
        st.markdown("""<div class="alert-banner alert-healthy">
            <span style="font-size:22px;">✅</span>
            <div>
                <div style="font-weight:700;font-size:15px;">No alarms in the database</div>
                <div style="font-size:12px;margin-top:3px;">
                    Alarms are generated by db.py → insert_alarm() when CRITICAL anomalies are detected.
                    They will appear here automatically once the pipeline fires a critical event.
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Explain how alarms work
        st.markdown("---")
        st.markdown("#### How alarms are generated")
        st.markdown("""
The `ac_alarms` table is populated automatically by the monitor loop in `db.py`:

```python
# Runs inside the monitor_loop() every 5 minutes
for _, row in critical.iterrows():
    insert_alarm(
        ac_id       = row["ac_id"],
        severity    = "CRITICAL",
        anomaly_type= row["anomaly_type"],
        metric      = "delta_t_mean",
        value       = row["delta_t_mean"],
        threshold   = 6.0,   # Summer critical floor
    )
```

**To simulate an alarm:** Run this SQL directly in your PostgreSQL client:
```sql
INSERT INTO ac_alarms (ac_id, severity, anomaly_type, metric, metric_value, threshold)
VALUES ('30EDA0267974', 'CRITICAL', 'Poor Cooling', 'delta_t_mean', 4.2, 6.0);
```
Then click **Refresh now** in the sidebar.
        """)

    else:
        ack_filter = st.checkbox("Show only unacknowledged", value=True)
        df_al = alm[~alm["acknowledged"]] if ack_filter and "acknowledged" in alm.columns else alm

        a1,a2,a3 = st.columns(3)
        with a1: st.metric("Total alarms",    len(df_al))
        with a2: st.metric("CRITICAL",        (df_al["severity"]=="CRITICAL").sum() if "severity" in df_al.columns else 0, delta_color="inverse")
        with a3: st.metric("Unacknowledged",  (~df_al.get("acknowledged",pd.Series([True]*len(df_al)))).sum(), delta_color="inverse")

        # Animated alarm cards
        for _, row in df_al.sort_values("alarm_time", ascending=False).head(20).iterrows():
            sev = str(row.get("severity","WARNING")).upper()
            cls = "alert-critical" if sev=="CRITICAL" else "alert-warning"
            st.markdown(f"""<div class="alert-banner {cls}">
                <span style="font-size:22px;">{"🚨" if sev=="CRITICAL" else "⚠️"}</span>
                <div style="flex:1;">
                    <div style="font-weight:700;font-size:14px;">{sev} — {row.get('anomaly_type','Unknown')}</div>
                    <div style="font-size:12px;margin-top:3px;">
                        Device: <b>{DEVICE_NAMES.get(row.get('ac_id',''), str(row.get('ac_id',''))[-8:])}</b> ·
                        Metric: {row.get('metric','—')} =
                        <b>{row.get('metric_value','—')}</b>
                        (threshold: {row.get('threshold','—')}) ·
                        {str(row.get('alarm_time',''))[:16]}
                    </div>
                </div>
                <div>{badge_html("ACK" if row.get("acknowledged") else "OPEN")}</div>
            </div>""", unsafe_allow_html=True)

        with st.expander("📋 Full alarm table"):
            st.dataframe(df_al, use_container_width=True, hide_index=True, height=320)



# ─────────────────────────────────────────────────────────────────────────────
# AI FLEET ASSISTANT — Modern containerised dialog (NOT inline on main page)
# ─────────────────────────────────────────────────────────────────────────────
# The chatbot is surfaced via a floating sidebar button that opens a full
# Streamlit dialog modal — keeping the main dashboard clean while giving users
# a dedicated, scrollable chat panel with full conversation history.
# ─────────────────────────────────────────────────────────────────────────────

def _build_system_prompt(target_ac_id: str, fleet_context: str, historical_context: str, user_query: str) -> str:
    return f"""
You are an elite, highly intelligent HVAC Predictive Maintenance AI built for commercial enterprise asset tracking.
Your objective is to review multi-tiered relational IoT database metrics and answer the user's question with precise analytical depth, using friendly, accessible language.

LIVE SNAPSHOT (ac_health_report):
{fleet_context}

360° HISTORICAL DOSSIER FOR TARGET UNIT (Compiled from 5 Calculated Tables):
{historical_context}

CRITICAL EXECUTION RULES:
1. NO TECHNICAL JARGON: The user has absolutely no data science or engineering background.
   - Never say 'Delta T' → Say 'Cooling Effectiveness' or 'Temperature Drop'.
   - Never say 'COP Proxy' or 'Efficiency Ratio' → Say 'Energy Efficiency Score'.
   - Never say 'Duty Cycle %' → Say 'Workload Rate' or 'Operational Stress'.
   - Never say 'Prophet Forecast / Predicted Anomaly' → Say 'AI Future Risk Forecast'.
   - Never say 'IsolationForest' or 'anomaly score' → Say 'AI Pattern Detection'.
   - Never say 'XGBoost' or 'critical probability' → Say 'AI Risk Confidence Score'.

2. TRUST THE ML MODEL OUTPUT SECTION (item 5 in the dossier) as your primary anomaly signal —
   it reflects two independently trained models scoring the unit's actual sensor history, not
   just fixed thresholds. If IsolationForest and XGBoost both flag a high proportion of cycles,
   treat this as a stronger signal than the rule-based "Critical Cycles" count alone, and say so
   plainly (e.g. "our AI model has independently flagged X% of recent cycles as unusual").

3. CROSS-TABLE ANOMALY DETECTIVE:
   - Compare "Live Status" with "Historical Thermodynamics". If the current cooling drop is significantly lower than the 7-day average, explicitly warn them about rapid degradation.
   - Cross-reference "Workload & Energy" with "Component Health". If the unit is running high operational hours but compressor scores have hit a historical low, flag this hidden vulnerability.

4. ACTIONABLE RISK MITIGATION:
   - If the overall health score is under 70, or if future risks are detected, strongly recommend scheduling immediate preventative maintenance.

5. RESPONSE FORMATTING:
   - Keep answers structured, scannable, and clean using short bullet points and bold headers.

User's Question: {user_query}
"""


@st.dialog("🤖 AI Fleet Assistant", width="large")
def show_chatbot_dialog():
    """Full-featured chat panel rendered inside a Streamlit modal dialog.
    Opens when the user clicks the sidebar button — never blocks the main view."""

    # ── Device selector ───────────────────────────────────────────────────────
    if len(hr) > 0:
        ac_list = hr["ac_id"].unique().tolist()
        target_ac_id = st.selectbox(
            "🎯 Select AC unit to analyse",
            ac_list,
            format_func=lambda x: f"{DEVICE_NAMES.get(x, x[-8:])} ({x[-8:]})",
            key="chatbot_ac_select",
        )
    else:
        st.warning("No active AC units found in the database.")
        return

    # ── Gemini init ───────────────────────────────────────────────────────────
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("models/gemini-3.5-flash")
        ai_ready = True
    except Exception as exc:
        st.error(f"Gemini not configured — add GEMINI_API_KEY to .streamlit/secrets.toml  ({exc})")
        return

    # ── Session state for history ─────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Action bar: clear history + unit badge
    col_badge, col_clear = st.columns([3, 1])
    with col_badge:
        st.markdown(
            f'<div style="font-size:12px;color:#8E8E93;padding:4px 0;">'
            f'Discussing: <b style="color:#007AFF;">{target_ac_id[-8:]}</b> · '
            f'{len(st.session_state.chat_history)//2} message(s)</div>',
            unsafe_allow_html=True,
        )
    with col_clear:
        if st.button("🗑 Clear", key="chat_clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # ── Scrollable chat history container ────────────────────────────────────
    chat_box = st.container(height=420, border=False)
    with chat_box:
        if not st.session_state.chat_history:
            st.markdown(
                '<div style="text-align:center;padding:60px 20px;color:#C7C7CC;">'
                '<div style="font-size:36px;">💬</div>'
                '<div style="font-size:14px;margin-top:8px;">Ask anything about your AC fleet.</div>'
                '<div style="font-size:12px;margin-top:4px;color:#E5E5EA;">e.g. "Does this unit show signs of compressor failure?"</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ── Chat input ────────────────────────────────────────────────────────────
    user_query = st.chat_input(
        "Ask about cooling performance, anomalies, maintenance…",
        key="chatbot_dialog_input",
    )

    if user_query and ai_ready:
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.spinner("Compiling 360° metrics across 5 tables…"):
            historical_context = get_ac_historical_profile(target_ac_id, engine, days=7)
            fleet_context      = hr.to_json(orient="records")
            prompt             = _build_system_prompt(
                target_ac_id, fleet_context, historical_context, user_query
            )

        try:
            response = model.generate_content(prompt)
            st.session_state.chat_history.append({"role": "assistant", "content": response.text})
        except Exception as exc:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"⚠️ AI error: {exc}",
            })
        st.rerun()


# ── Sidebar launcher button ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown(
        '<div style="font-size:10px;color:#636366;text-transform:uppercase;'
        'letter-spacing:.07em;margin-bottom:6px;">AI Assistant</div>',
        unsafe_allow_html=True,
    )
    if st.button("💬  Open AI Fleet Chat", use_container_width=True, type="primary", key="open_chat_btn"):
        show_chatbot_dialog()
    st.caption("Powered by Gemini · Context: Micro System Foundation")

    # ─────────────────────────────────────────────────────────────────────────────
# AUTO-REFRESH
# ─────────────────────────────────────────────────────────────────────────────
if auto_ref:
    time.sleep(300)
    st.cache_data.clear()
    st.rerun()
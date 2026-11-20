"""
STEP 3 — Store Health Metrics
=================================
Loads the computed parquet and writes it to your chosen storage backend.

OPTION A — Same PostgreSQL DB, new table (recommended for production)
OPTION B — SQLite file (zero-config, works offline, great for dev/testing)

Both options create the table if it doesn't exist and UPSERT on (ac_id, timestamp)
so re-running never creates duplicate rows.

Usage:
    # Option A — PostgreSQL
    python 03_store.py \
        --input  output/health_metrics.parquet \
        --backend postgres \
        --pg-host localhost \
        --pg-port 5432 \
        --pg-db   your_db_name \
        --pg-user your_user \
        --pg-pass your_password \
        --pg-schema public \
        --pg-table  ac_health_metrics

    # Option B — SQLite
    python 03_store.py \
        --input   output/health_metrics.parquet \
        --backend sqlite \
        --sqlite-path output/ac_health.db

Dependencies:
    pip install pandas pyarrow psycopg2-binary sqlalchemy
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


# ── Column definitions for CREATE TABLE ──────────────────────────────────────
SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS {schema}.{table} (
    id                  BIGSERIAL,
    ac_id               TEXT         NOT NULL,
    timestamp           TIMESTAMPTZ  NOT NULL,
    season              TEXT,

    -- Raw sensor averages (5-min)
    inlet_temp          FLOAT,
    outlet_temp         FLOAT,
    total_load_kw       FLOAT,
    fan_load_kw         FLOAT,
    ambient_temp        FLOAT,
    humidity            FLOAT,

    -- Derived physics
    delta_t             FLOAT,
    compressor_load_kw  FLOAT,
    efficiency_ratio    FLOAT,
    load_stress_index   FLOAT,
    cop_proxy           FLOAT,
    eer_proxy           FLOAT,

    -- Component health scores (0–100)
    compressor_score    FLOAT,
    refrigerant_score   FLOAT,
    capacitor_score     FLOAT,
    air_filter_score    FLOAT,
    fan_motor_score     FLOAT,
    overall_score       FLOAT,

    -- Rolling metrics
    delta_t_rolling_12       FLOAT,
    overall_score_rolling_12 FLOAT,
    efficiency_ratio_rolling_12 FLOAT,
    cop_proxy_rolling_12     FLOAT,

    -- Anomaly & status
    anomaly_level       TEXT,
    ml_anomaly_flag     BOOLEAN,
    status              TEXT,

    -- Meta
    data_quality        TEXT,
    created_at          TIMESTAMPTZ  DEFAULT NOW(),

    PRIMARY KEY (ac_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_{table}_ac_ts
    ON {schema}.{table} (ac_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_{table}_status
    ON {schema}.{table} (status);
"""

SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS ac_health_metrics (
    ac_id               TEXT        NOT NULL,
    timestamp           TEXT        NOT NULL,
    season              TEXT,
    inlet_temp          REAL,
    outlet_temp         REAL,
    total_load_kw       REAL,
    fan_load_kw         REAL,
    ambient_temp        REAL,
    humidity            REAL,
    delta_t             REAL,
    compressor_load_kw  REAL,
    efficiency_ratio    REAL,
    load_stress_index   REAL,
    cop_proxy           REAL,
    eer_proxy           REAL,
    compressor_score    REAL,
    refrigerant_score   REAL,
    capacitor_score     REAL,
    air_filter_score    REAL,
    fan_motor_score     REAL,
    overall_score       REAL,
    delta_t_rolling_12       REAL,
    overall_score_rolling_12 REAL,
    efficiency_ratio_rolling_12 REAL,
    cop_proxy_rolling_12     REAL,
    anomaly_level       TEXT,
    ml_anomaly_flag     INTEGER,
    status              TEXT,
    data_quality        TEXT,
    created_at          TEXT        DEFAULT (datetime('now')),
    PRIMARY KEY (ac_id, timestamp)
);
"""


# ── Upsert helpers ────────────────────────────────────────────────────────────
def upsert_postgres(df: pd.DataFrame, conn_str: str, schema: str, table: str, season: str):
    """Upsert using psycopg2 COPY + ON CONFLICT."""
    import psycopg2
    from psycopg2.extras import execute_values

    df = df.copy()
    df["season"] = season
    df["ml_anomaly_flag"] = df["ml_anomaly_flag"].astype(bool)

    # Replace NaN with None for psycopg2
    df = df.where(pd.notnull(df), None)

    # Desired columns (intersection of what's in df and the schema)
    desired = [
        "ac_id", "timestamp", "season",
        "inlet_temp", "outlet_temp", "total_load_kw", "fan_load_kw",
        "ambient_temp", "humidity",
        "delta_t", "compressor_load_kw", "efficiency_ratio",
        "load_stress_index", "cop_proxy", "eer_proxy",
        "compressor_score", "refrigerant_score", "capacitor_score",
        "air_filter_score", "fan_motor_score", "overall_score",
        "delta_t_rolling_12", "overall_score_rolling_12",
        "efficiency_ratio_rolling_12", "cop_proxy_rolling_12",
        "anomaly_level", "ml_anomaly_flag", "status", "data_quality",
    ]
    cols = [c for c in desired if c in df.columns]
    rows = [tuple(r) for r in df[cols].itertuples(index=False, name=None)]

    col_list = ", ".join(cols)
    excluded = ", ".join(f"EXCLUDED.{c}" for c in cols if c not in ("ac_id", "timestamp"))
    update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in ("ac_id", "timestamp"))

    sql = f"""
    INSERT INTO {schema}.{table} ({col_list})
    VALUES %s
    ON CONFLICT (ac_id, timestamp) DO UPDATE SET {update_set}
    """

    with psycopg2.connect(conn_str) as con:
        with con.cursor() as cur:
            cur.execute(SCHEMA_POSTGRES.format(schema=schema, table=table))
            execute_values(cur, sql, rows, page_size=500)
        con.commit()
    print(f"  ✓ Upserted {len(rows)} rows into {schema}.{table}")


def upsert_sqlite(df: pd.DataFrame, db_path: str, season: str):
    """Upsert into SQLite using INSERT OR REPLACE."""
    import sqlite3

    df = df.copy()
    df["season"] = season
    df["ml_anomaly_flag"] = df["ml_anomaly_flag"].astype(int)
    df["timestamp"] = df["timestamp"].astype(str)

    desired = [
        "ac_id", "timestamp", "season",
        "inlet_temp", "outlet_temp", "total_load_kw", "fan_load_kw",
        "ambient_temp", "humidity",
        "delta_t", "compressor_load_kw", "efficiency_ratio",
        "load_stress_index", "cop_proxy", "eer_proxy",
        "compressor_score", "refrigerant_score", "capacitor_score",
        "air_filter_score", "fan_motor_score", "overall_score",
        "delta_t_rolling_12", "overall_score_rolling_12",
        "efficiency_ratio_rolling_12", "cop_proxy_rolling_12",
        "anomaly_level", "ml_anomaly_flag", "status", "data_quality",
    ]
    cols = [c for c in desired if c in df.columns]
    rows = [tuple(r) for r in df[cols].itertuples(index=False, name=None)]
    placeholders = ", ".join("?" * len(cols))
    col_list = ", ".join(cols)

    with sqlite3.connect(db_path) as con:
        con.execute(SCHEMA_SQLITE)
        con.executemany(
            f"INSERT OR REPLACE INTO ac_health_metrics ({col_list}) VALUES ({placeholders})",
            rows
        )
        con.commit()
    print(f"  ✓ Upserted {len(rows)} rows into {db_path} (ac_health_metrics)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Store health metrics to DB")
    parser.add_argument("--input",   default="output/health_metrics.parquet")
    parser.add_argument("--backend", choices=["postgres", "sqlite"], default="sqlite",
                        help="Storage backend (default: sqlite)")
    parser.add_argument("--season",  default="summer")

    # PostgreSQL options
    parser.add_argument("--pg-host",   default="localhost")
    parser.add_argument("--pg-port",   default="5432")
    parser.add_argument("--pg-db",     default="acdb")
    parser.add_argument("--pg-user",   default="postgres")
    parser.add_argument("--pg-pass",   default="")
    parser.add_argument("--pg-schema", default="public")
    parser.add_argument("--pg-table",  default="ac_health_metrics")

    # SQLite option
    parser.add_argument("--sqlite-path", default="output/ac_health.db")

    args = parser.parse_args()

    print(f"\n── Loading metrics parquet ({args.input}) ─────────────")
    df = pd.read_parquet(args.input)
    print(f"  Rows to store : {len(df)}")

    if args.backend == "postgres":
        conn_str = (
            f"host={args.pg_host} port={args.pg_port} "
            f"dbname={args.pg_db} user={args.pg_user} password={args.pg_pass}"
        )
        print(f"\n── Storing to PostgreSQL ({args.pg_host}/{args.pg_db}) ─────")
        upsert_postgres(df, conn_str, args.pg_schema, args.pg_table, args.season)

        print("\n── Sample query to verify ────────────────────────────")
        print(f"""
    SELECT ac_id, timestamp, overall_score, status
    FROM {args.pg_schema}.{args.pg_table}
    ORDER BY timestamp DESC
    LIMIT 10;
        """)

    else:  # sqlite
        print(f"\n── Storing to SQLite ({args.sqlite_path}) ────────────")
        upsert_sqlite(df, args.sqlite_path, args.season)

        print("\n── Sample query to verify ────────────────────────────")
        print(f"""
    sqlite3 {args.sqlite_path}
    SELECT ac_id, timestamp, overall_score, status
    FROM ac_health_metrics
    ORDER BY timestamp DESC
    LIMIT 10;
        """)


if __name__ == "__main__":
    main()

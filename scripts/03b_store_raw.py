"""
STEP 3b — Store Raw Sensor Data to PostgreSQL
================================================
Stores ALL raw rows from the API CSVs at full resolution into PostgreSQL.
This runs BEFORE resampling so every data point is preserved.

Tables created:
  ac_raw_sensor_data   — every reading from ac_data (CSV1)
  ac_raw_environment   — every reading from environment_data (CSV2)

Both use ON CONFLICT DO UPDATE so re-runs never duplicate rows.
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

VOLTAGE      = 230
POWER_FACTOR = 0.85

CSV1_MAP = {
    "time":          "timestamp",
    "Total Current": "total_current",
    "Fan Current":   "fan_current",
    "Inlet Temp":    "inlet_temp",
    "Outlet Temp":   "outlet_temp",
    "Ac_status":     "ac_status",
    "device_id":     "ac_id",
}
CSV2_MAP = {
    "rec_ts":      "timestamp",
    "time":        "timestamp",
    "temperature": "ambient_temp",
    "humidity":    "humidity",
    "device_id":   "ac_id",
}

DDL_SENSOR = """
CREATE TABLE IF NOT EXISTS {schema}.ac_raw_sensor_data (
    ac_id           TEXT          NOT NULL,
    device_id       TEXT,
    season          TEXT,
    timestamp       TIMESTAMPTZ   NOT NULL,
    total_current   FLOAT,
    fan_current     FLOAT,
    inlet_temp      FLOAT,
    outlet_temp     FLOAT,
    ac_status       TEXT,
    total_load_kw   FLOAT,
    fan_load_kw     FLOAT,
    created_at      TIMESTAMPTZ   DEFAULT NOW(),
    PRIMARY KEY (ac_id, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_raw_sensor_ts
    ON {schema}.ac_raw_sensor_data (ac_id, timestamp DESC);
"""

DDL_ENV = """
CREATE TABLE IF NOT EXISTS {schema}.ac_raw_environment (
    device_id     TEXT          NOT NULL,
    timestamp     TIMESTAMPTZ   NOT NULL,
    ambient_temp  FLOAT,
    humidity      FLOAT,
    created_at    TIMESTAMPTZ   DEFAULT NOW(),
    PRIMARY KEY (device_id, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_raw_env_ts
    ON {schema}.ac_raw_environment (device_id, timestamp DESC);
"""


def load_sensor_csv(path: str, ac_id: str, season: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8')
    df = df.rename(columns={k: v for k, v in CSV1_MAP.items() if k in df.columns})
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    
    for col in ["total_current", "fan_current", "inlet_temp", "outlet_temp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    if "total_current" in df.columns:
        df["total_load_kw"] = (df["total_current"] * VOLTAGE * POWER_FACTOR / 1000.0).round(4)
    if "fan_current" in df.columns:
        df["fan_load_kw"] = (df["fan_current"] * VOLTAGE * POWER_FACTOR / 1000.0).round(4)
    if "ac_status" in df.columns:
        df["ac_status"] = df["ac_status"].astype(str).str.strip()
        
    df["season"] = season
    
    if "ac_id" not in df.columns:
        df["ac_id"] = ac_id
        
    if "device_id" not in df.columns:
        df["device_id"] = ""
        
    # FIX: Drop duplicate timestamps for the same AC ID to prevent Postgres CardinalityViolation
    df = df.drop_duplicates(subset=["ac_id", "timestamp"], keep="last")
    
    return df.sort_values("timestamp").reset_index(drop=True)


def load_env_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8')
    df = df.rename(columns={k: v for k, v in CSV2_MAP.items() if k in df.columns})
    df = df.loc[:, ~df.columns.duplicated()]
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    
    for col in ["ambient_temp", "humidity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    if "device_id" not in df.columns:
        df["device_id"] = "UNKNOWN_ENV"
        
    # FIX: Drop duplicate timestamps to prevent Postgres CardinalityViolation
    df = df.drop_duplicates(subset=["device_id", "timestamp"], keep="last")
    
    return df.sort_values("timestamp").reset_index(drop=True)


def upsert(df, conn_str, schema, table, ddl, pk_cols, desired_cols):
    import psycopg2
    from psycopg2.extras import execute_values
    df = df.copy().where(pd.notnull(df), other=None)
    cols = [c for c in desired_cols if c in df.columns]
    rows = [tuple(r) for r in df[cols].itertuples(index=False, name=None)]
    
    if not rows:
        print(f"  ⚠  No rows for {schema}.{table}")
        return
        
    update_set = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c not in pk_cols)
    sql = (f"INSERT INTO {schema}.{table} ({', '.join(cols)}) VALUES %s "
           f"ON CONFLICT ({', '.join(pk_cols)}) DO UPDATE SET {update_set}")
           
    with psycopg2.connect(conn_str) as con:
        with con.cursor() as cur:
            cur.execute(ddl.format(schema=schema))
            execute_values(cur, sql, rows, page_size=500)
        con.commit()
    print(f"  ✓ {len(rows):,} rows → {schema}.{table}")


def main():
    parser = argparse.ArgumentParser(description="Store raw sensor data to PostgreSQL")
    parser.add_argument("--csv1",      required=True)
    parser.add_argument("--csv2",      required=True)
    parser.add_argument("--ac_id",     default="multiple")
    parser.add_argument("--season",    default="summer")
    parser.add_argument("--pg-host",   default="localhost")
    parser.add_argument("--pg-port",   default="5432")
    parser.add_argument("--pg-db",     default="ac_automation")
    parser.add_argument("--pg-user",   default="postgres")
    parser.add_argument("--pg-pass",   default="")
    parser.add_argument("--pg-schema", default="public")
    args = parser.parse_args()

    conn_str = (f"host={args.pg_host} port={args.pg_port} "
                f"dbname={args.pg_db} user={args.pg_user} password={args.pg_pass}")
    schema = args.pg_schema

    print(f"\n── Loading CSV1 ──────────────────────────────────────")
    df1 = load_sensor_csv(args.csv1, args.ac_id, args.season)
    print(f"  {len(df1):,} unique rows  |  {df1['timestamp'].min()} → {df1['timestamp'].max()}")
    if "ac_status" in df1.columns:
        print(f"  ac_status dist: {df1['ac_status'].value_counts().to_dict()}")

    print(f"\n── Loading CSV2 ──────────────────────────────────────")
    df2 = load_env_csv(args.csv2)
    print(f"  {len(df2):,} unique rows")

    print(f"\n── Storing to PostgreSQL ({args.pg_host}/{args.pg_db}) ──")
    upsert(df1, conn_str, schema, "ac_raw_sensor_data", DDL_SENSOR,
           ("ac_id", "timestamp"),
           ["ac_id", "device_id", "season", "timestamp", "total_current",
            "fan_current", "inlet_temp", "outlet_temp", "ac_status",
            "total_load_kw", "fan_load_kw"])

    upsert(df2, conn_str, schema, "ac_raw_environment", DDL_ENV,
           ("device_id", "timestamp"),
           ["device_id", "timestamp", "ambient_temp", "humidity"])

    print(f"\n✓ Raw data stored:")
    print(f"  {schema}.ac_raw_sensor_data  ← {len(df1):,} rows (full resolution)")
    print(f"  {schema}.ac_raw_environment  ← {len(df2):,} rows (full resolution)")


if __name__ == "__main__":
    main()
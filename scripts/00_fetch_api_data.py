"""
STEP 0 — Fetch AC Data from iotscada REST API
================================================
Fetches from http://iotscada.co.in:1880/ac/history and parses the
nested JSON structure:

  {
    "season": "Summer",
    "ac_data": [
      {
        "device_id": "30EDA0267970",
        "data": [
          { "time": "...", "Total Current": 6.98, "Fan Current": 0.349,
            "Inlet Temp": 27.9, "Outlet Temp": 19.2, "Ac_status": 1 },
          ...
        ]
      }
    ],
    "environment_data": [
      {
        "device_id": "30EDA0267B20",
        "data": [
          { "time": "...", "temperature": 26.7, "humidity": 52 },
          ...
        ]
      }
    ]
  }

Outputs two CSVs that 01_ingest_clean.py expects:
  data/main_YYYY-MM-DD.csv      (CSV1 — power + temp)
  data/temp_hum_YYYY-MM-DD.csv  (CSV2 — ambient + humidity)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import yaml
import time
import requests
import urllib3

# Mutes the InsecureRequestWarning that appears when verify=False is used
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



IST = ZoneInfo("Asia/Kolkata")
log = logging.getLogger("fetch_api")


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
def load_config(config_path):
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Date range helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_date_range(mode: str, override_date: str = None):
    """Returns (start_dt, end_dt) in IST."""
    if override_date:
        d = datetime.strptime(override_date, "%Y-%m-%d").date()
    elif mode == "yesterday":
        d = (datetime.now(IST) - timedelta(days=1)).date()
    else:  # today
        d = datetime.now(IST).date()

    start = datetime(d.year, d.month, d.day, 0,  0,  0,  tzinfo=IST)
    end   = datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=IST)
    return start, end


# ─────────────────────────────────────────────────────────────────────────────
# HTTP GET with retry
# ─────────────────────────────────────────────────────────────────────────────
def get_with_retry(url: str, params: dict, headers: dict,
                   timeout: int = 30, max_retries: int = 3) -> dict:
    for attempt in range(1, max_retries + 1):
        try:
            log.info(f"  GET {url}  params={params}  (attempt {attempt})")
            
            # -> Added verify=False here <-
            resp = requests.get(url, params=params, headers=headers, timeout=timeout, verify=False)
            
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries:
                log.error(f"  API call failed after {max_retries} attempts: {e}")
                raise
            wait = 2 ** attempt
            log.warning(f"  Attempt {attempt} failed ({e}). Retrying in {wait}s …")
            time.sleep(wait)

# ─────────────────────────────────────────────────────────────────────────────
# Parse iotscada nested JSON → flat DataFrames
# ─────────────────────────────────────────────────────────────────────────────
def parse_ac_data(body: dict, target_device_ids: list = None) -> pd.DataFrame:
    """
    Extracts CSV1 rows from body["ac_data"].
    Supports filtering against a list of target IDs. If empty or None,
    it automatically collects all devices.
    """
    ac_data_list = body.get("ac_data", [])
    if not ac_data_list:
        log.warning("  'ac_data' key missing or empty in API response")
        return pd.DataFrame()

    # Normalize target IDs to uppercase strings to prevent case-matching bugs
    if target_device_ids:
        target_device_ids = [str(uid).strip().upper() for uid in target_device_ids if uid]
    else:
        target_device_ids = []

    frames = []
    for device_block in ac_data_list:
        dev_id = str(device_block.get("device_id", "unknown")).strip().upper()

        # Skip only if a list is provided AND the device is not in that list
        if target_device_ids and dev_id not in target_device_ids:
            log.info(f"  Skipping ac_data device {dev_id}")
            continue

        rows = device_block.get("data", [])
        if not rows:
            log.warning(f"  No data rows for ac_data device {dev_id}")
            continue

        df = pd.DataFrame(rows)
        df["device_id"] = dev_id
        frames.append(df)
        log.info(f"  ac_data device {dev_id}: {len(df)} rows")

    if not frames:
        log.warning("  No ac_data frames collected (check device_id filter)")
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    log.info(f"  ac_data total rows: {len(merged)}, columns: {list(merged.columns)}")
    return merged
def parse_environment_data(body: dict, target_device_id: str = None) -> pd.DataFrame:
    """
    Extracts CSV2 rows from:
        body["environment_data"][i]["data"] → list of ambient readings

    01_ingest_clean.py expects columns: rec_ts, temperature, humidity
    The iotscada API returns: time, temperature, humidity
    So we rename 'time' → 'rec_ts'.
    """
    env_data_list = body.get("environment_data", [])
    if not env_data_list:
        log.warning("  'environment_data' key missing or empty in API response")
        return pd.DataFrame()

    frames = []
    for device_block in env_data_list:
        dev_id = device_block.get("device_id", "unknown")

        if target_device_id and dev_id != target_device_id:
            log.info(f"  Skipping env device {dev_id}")
            continue

        rows = device_block.get("data", [])
        if not rows:
            log.warning(f"  No data rows for env device {dev_id}")
            continue

        df = pd.DataFrame(rows)
        df["device_id"] = dev_id
        frames.append(df)
        log.info(f"  environment_data device {dev_id}: {len(df)} rows")

    if not frames:
        log.warning("  No environment_data frames collected")
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)

    # Rename 'time' → 'rec_ts' (what 01_ingest_clean.py's CSV2_COL_MAP expects)
    if "time" in merged.columns:
        merged = merged.rename(columns={"time": "rec_ts"})

    log.info(f"  environment_data total rows: {len(merged)}, columns: {list(merged.columns)}")
    return merged



# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fetch AC data from iotscada API")
    parser.add_argument("--config",   default="config/config.yaml")
    parser.add_argument("--date",     default=None,
                        help="Override fetch date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--out-dir",  default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    if not logging.root.handlers:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")

    api_cfg  = cfg["api"]
    base_url = api_cfg["base_url"].rstrip("/")
    endpoint = api_cfg["endpoints"].get("history", "/ac/history")
    url      = base_url + endpoint

    mode     = api_cfg.get("date_range_mode", "yesterday")
    start_dt, end_dt = get_date_range(mode, args.date)
    date_str = start_dt.strftime("%Y-%m-%d")

    # Device IDs from config (optional filter)
    ac_unit_ids  = api_cfg.get("ac_unit_ids") or []
    ac_device_id = ac_unit_ids[0] if ac_unit_ids else None
    env_device_ids = api_cfg.get("env_unit_ids") or []
    env_device_id  = env_device_ids[0] if env_device_ids else None

    out_dir = Path(args.out_dir or cfg["paths"]["data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build request headers
    headers = {"Accept": "application/json"}
    api_key = api_cfg.get("api_key", "")
    if api_key and api_key != "YOUR_API_KEY_HERE":
        headers["Authorization"] = f"Bearer {api_key}"
    extra = api_cfg.get("extra_headers") or {}
    headers.update(extra)

    # Build query params — send date range to API (it may filter server-side)
    params = {
        "start": start_dt.isoformat(),
        "end":   end_dt.isoformat(),
    }
    # Remove empty params
    params = {k: v for k, v in params.items() if v}

    log.info(f"Fetching {url} for date {date_str}")
    body = get_with_retry(url, params, headers,
                          timeout=api_cfg.get("timeout_seconds", 30))

    log.info(f"API response top-level keys: {list(body.keys())}")
    log.info(f"Season from API: {body.get('season', 'N/A')}")

    # ── Parse ac_data (CSV1) ──────────────────────────────────────────
    log.info("── Parsing ac_data (sensor readings) ──────────────────")
    # Pass the entire list instead of just index [0]
    df1 = parse_ac_data(body, target_device_ids=ac_unit_ids)
    

    if df1.empty:
        log.error("No ac_data rows after filtering. Aborting.")
        sys.exit(1)

    # -> Added "device_id" to the saved columns <-
    csv1_cols = ["time", "Total Current", "Fan Current",
                 "Inlet Temp", "Outlet Temp", "Ac_status", "device_id"]
    df1_out = df1[[c for c in csv1_cols if c in df1.columns]].copy()

    csv1_path = out_dir / f"main_{date_str}.csv"
    df1_out.to_csv(csv1_path, index=False)

    # Keep only the columns 01_ingest_clean.py needs
    csv1_cols = ["time", "Total Current", "Fan Current",
                 "Inlet Temp", "Outlet Temp", "Ac_status","device_id"]
    df1_out = df1[[c for c in csv1_cols if c in df1.columns]].copy()

    csv1_path = out_dir / f"main_{date_str}.csv"
    df1_out.to_csv(csv1_path, index=False)
    log.info(f"✓ CSV1 saved → {csv1_path}  ({len(df1_out)} rows)")

    # ── Parse environment_data (CSV2) ────────────────────────────────
    log.info("── Parsing environment_data (ambient readings) ─────────")
    df2 = parse_environment_data(body, target_device_id=env_device_id)
    

    csv2_path = out_dir / f"temp_hum_{date_str}.csv"
    if df2.empty:
        log.warning("No environment_data rows — saving empty CSV2")
        pd.DataFrame(columns=["rec_ts", "temperature", "humidity","device_id"]) \
          .to_csv(csv2_path, index=False)
    else:
        csv2_cols = ["rec_ts", "temperature", "humidity","device_id"]
        df2_out = df2[[c for c in csv2_cols if c in df2.columns]].copy()
        df2_out.to_csv(csv2_path, index=False)
        log.info(f"✓ CSV2 saved → {csv2_path}  ({len(df2_out)} rows)")

    # Print JSON result for run_pipeline.py to capture
    result = {"csv1": str(csv1_path), "csv2": str(csv2_path), "date": date_str}
    print(json.dumps(result))


if __name__ == "__main__":
    main()

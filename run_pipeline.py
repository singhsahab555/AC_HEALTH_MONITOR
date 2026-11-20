"""
run_pipeline.py — AC Health Metrics Pipeline Orchestrator
==========================================================
Runs all steps in sequence:
  Step 0  → Fetch data from REST API
  Step 1  → Ingest & Clean CSVs
  Step 2  → Calculate Health Metrics
  Step 3  → Store Aggregated Metrics to PostgreSQL
  Step 3b → Store Raw Full-Resolution Sensor Data to PostgreSQL
  Step 4  → Runtime Cycle Detection & Storage
  Step 5  → AC Engine (AI Anomaly Detection & Forecasting)
"""

import argparse
import json
import logging
import os
import smtplib
import subprocess
import sys
import traceback
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from zoneinfo import ZoneInfo
import yaml

IST = ZoneInfo("Asia/Kolkata")

ROOT    = Path(__file__).parent.resolve()
SCRIPTS = ROOT / "scripts"
CONFIG  = ROOT / "config" / "config.yaml"

FAIL_FLAG = ROOT / "logs" / "PIPELINE_FAILED.txt"

def load_config() -> dict:
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Password lives only in the AC_PG_PASSWORD env var, not in config/code.
    cfg["database"]["password"] = os.environ.get("AC_PG_PASSWORD") or cfg["database"].get("password") or ""
    return cfg

def alert_failure(run_date: str, error: str):
    """Write a FAILED flag (shown by the dashboard) and pop up a non-blocking Windows message box."""
    text = f"Run for {run_date} failed at {datetime.now(IST):%Y-%m-%d %H:%M IST}\n{error}\nLog: logs/pipeline_{run_date}.log"
    try:
        FAIL_FLAG.parent.mkdir(parents=True, exist_ok=True)
        FAIL_FLAG.write_text(text, encoding="utf-8")
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            code = ("import ctypes,sys; ctypes.windll.user32.MessageBoxW(0, sys.argv[1], "
                    "'AC Health Pipeline FAILED', 0x10 | 0x40000 | 0x10000)")
            subprocess.Popen([sys.executable, "-c", code, text],
                             creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
        except Exception:
            pass

def setup_logging(cfg: dict, run_date: str) -> logging.Logger:
    logs_dir = ROOT / cfg["paths"]["logs_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"pipeline_{run_date}.log"
    fmt = "%(asctime)s [%(levelname)-8s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt,
                        handlers=[logging.FileHandler(log_file, encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("pipeline")
    return log

def run_step(log: logging.Logger, label: str, cmd: list[str], cwd: Path = None) -> str:
    log.info(f"{'═'*60}")
    log.info(f"  {label}")
    log.info(f"{'═'*60}")
    
    # Hide passwords in logs
    display = []
    mask_next = False
    for tok in cmd:
        tok = str(tok)
        if mask_next:
            display.append("***"); mask_next = False
        elif tok in ("--pg-pass", "--pg_pass"):
            display.append(tok); mask_next = True
        else:
            display.append(tok)
    
    log.info(f"  CMD: {' '.join(display)}")

    proc = subprocess.Popen(
        [sys.executable, "-X", "utf8"] + [str(c) for c in cmd],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", cwd=str(cwd or ROOT)
    )

    lines = []
    for line in proc.stdout:
        line = line.rstrip()
        lines.append(line)
        log.info(f"    {line}")
    proc.wait()
    
    if proc.returncode != 0:
        raise RuntimeError(f"Step failed with exit code {proc.returncode}: {label}")
    for line in reversed(lines):
        if line.strip(): return line.strip()
    return ""

def send_email(cfg: dict, subject: str, body: str, log: logging.Logger):
    nc = cfg.get("notifications", {})
    if not nc.get("enabled"): return
    try:
        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = nc["smtp_user"], nc["notify_to"], subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP(nc["smtp_host"], nc["smtp_port"]) as s:
            s.starttls()
            s.login(nc["smtp_user"], nc["smtp_pass"])
            s.sendmail(nc["smtp_user"], nc["notify_to"], msg.as_string())
        log.info(f"📧 Email sent: {subject}")
    except Exception as e:
        log.warning(f"Email notification failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="AC Health Pipeline Orchestrator")
    parser.add_argument("--date", default=None)
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--csv1", default=None)
    parser.add_argument("--csv2", default=None)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-engine", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    run_date = args.date or (datetime.now(IST) - timedelta(days=1)).strftime("%Y-%m-%d")
    log = setup_logging(cfg, run_date)

    log.info(f"\n╔{'═'*58}╗\n║  AC Health Metrics Pipeline  —  {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}  ║\n║  Processing date: {run_date:<40}║\n╚{'═'*58}╝\n")

    data_dir, output_dir = ROOT / cfg["paths"]["data_dir"], ROOT / cfg["paths"]["output_dir"]
    data_dir.mkdir(parents=True, exist_ok=True); output_dir.mkdir(parents=True, exist_ok=True)
    
    ac_id, season, db_cfg = cfg["pipeline"]["ac_id"], cfg["pipeline"]["season"], cfg["database"]
    clean_parquet, metrics_parquet, cycles_parquet = output_dir / "clean_data.parquet", output_dir / "health_metrics.parquet", output_dir / "runtime_cycles.parquet"

    def pg_args():
        return ["--pg-host", db_cfg["host"], "--pg-port", str(db_cfg["port"]), "--pg-db", db_cfg["dbname"], "--pg-user", db_cfg["user"], "--pg-pass", db_cfg["password"], "--pg-schema", db_cfg["schema"]]

    pipeline_ok, error_summary, csv1_path, csv2_path = True, [], args.csv1, args.csv2

    try:
        # STEP 0: Fetch from API
        if args.skip_fetch or args.csv1 or args.step > 0:
            log.info("⏭  Skipping Step 0 (using existing CSV files)")
            if not csv1_path: csv1_path = str(sorted(data_dir.glob("main_*.csv"), reverse=True)[0]) if list(data_dir.glob("main_*.csv")) else str(data_dir / f"main_{run_date}.csv")
            if not csv2_path: csv2_path = str(sorted(data_dir.glob("temp_hum_*.csv"), reverse=True)[0]) if list(data_dir.glob("temp_hum_*.csv")) else str(data_dir / f"temp_hum_{run_date}.csv")
        else:
            last_line = run_step(log, "STEP 0 — Fetch Data from REST API", [SCRIPTS / "00_fetch_api_data.py", "--config", CONFIG, *(["--date", run_date] if args.date else [])])
            try:
                res = json.loads(last_line); csv1_path, csv2_path = res["csv1"], res["csv2"]
            except: 
                csv1_path, csv2_path = str(data_dir / f"main_{run_date}.csv"), str(data_dir / f"temp_hum_{run_date}.csv")

        # STEP 1: Ingest & Clean
        if args.step <= 1:
            run_step(log, "STEP 1 — Ingest & Clean CSV Data", [SCRIPTS / "01_ingest_clean.py", "--csv1", csv1_path, "--csv2", csv2_path, "--out", clean_parquet, "--ac_id", ac_id])

        # STEP 2: Calculate Metrics
        if args.step <= 2:
            run_step(log, "STEP 2 — Calculate Health Metrics", [SCRIPTS / "02_calculate_metrics.py", "--input", clean_parquet, "--out", metrics_parquet, "--season", season])

        # STEP 3: Store Aggregated Metrics
        if args.step <= 3:
            run_step(log, "STEP 3 — Store Aggregated Metrics to PostgreSQL", [SCRIPTS / "03_store.py", "--input", metrics_parquet, "--backend", "postgres", "--season", season] + pg_args() + ["--pg-table", db_cfg["tables"]["health_metrics"]])

        # STEP 3b: Store Raw Sensor Data
        if args.step <= 3:
            run_step(log, "STEP 3b — Store Raw Full-Resolution Sensor Data to PostgreSQL", [SCRIPTS / "03b_store_raw.py", "--csv1", csv1_path, "--csv2", csv2_path, "--ac_id", ac_id, "--season", season] + pg_args())

        # STEP 4: Runtime Cycle Detection
        if args.step <= 4:
            cmd = [
                SCRIPTS / "04_runtime_cycles.py", 
                "--csv1", csv1_path, 
                "--csv2", csv2_path, 
                "--ac_id", ac_id, 
                "--season", season, 
                "--pg_table", db_cfg["tables"].get("runtime_cycles", "ac_runtime_cycles"),
                "--pg_host", db_cfg["host"], 
                "--pg_port", str(db_cfg["port"]), 
                "--pg_db", db_cfg["dbname"], 
                "--pg_user", db_cfg["user"], 
                "--pg_pass", db_cfg["password"], 
                "--pg_schema", db_cfg["schema"]
            ]
            if not args.no_plot: 
                cmd.append("--plot")
            run_step(log, "STEP 4 — Runtime Cycle Detection", cmd)

        # STEP 5: AC Engine
        if args.step <= 5 and not args.no_engine:
            run_step(log, "STEP 5 — AC Engine (Anomaly Detection & Forecasting)", [SCRIPTS / "pipeline.py"], cwd=SCRIPTS)

    except Exception as e:
        pipeline_ok, error_msg = False, f"{type(e).__name__}: {e}"
        error_summary.append(error_msg)
        log.error(f"\n{'!'*60}\n  PIPELINE FAILED: {error_msg}\n{'!'*60}\n{traceback.format_exc()}")

    log.info("\n")
    if pipeline_ok:
        FAIL_FLAG.unlink(missing_ok=True)
        log.info(f"{'═'*60}\n  ✅ PIPELINE COMPLETED SUCCESSFULLY  [{run_date}]\n{'═'*60}")
    else:
        log.error(f"{'!'*60}\n  ❌ PIPELINE FAILED  [{run_date}]\n{'!'*60}")
        alert_failure(run_date, "; ".join(error_summary))
        sys.exit(1)

if __name__ == "__main__":
    main()
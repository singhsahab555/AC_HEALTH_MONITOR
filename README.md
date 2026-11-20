# AC Health Monitor

Daily IoT pipeline that pulls AC/HVAC sensor data (total & fan current, inlet/outlet temp, ambient temp/humidity), computes physics-based health metrics, detects compressor runtime cycles, runs ML anomaly detection, and serves a Streamlit dashboard with an AI assistant.

```
00 fetch API → 01 ingest & clean → 02 health metrics → 03/03b store to PostgreSQL
  → 04 runtime cycles → 05 ML engine (IsolationForest, XGBoost, Prophet) → dashboard
```

## Setup

```powershell
python -m pip install -r requirements.txt
setx AC_PG_PASSWORD "<your postgres password>"      # DB password is never stored in code/config
copy dashboard\.streamlit\secrets.toml.example dashboard\.streamlit\secrets.toml   # add your Gemini key
```

Edit `config/config.yaml` for API URL, device IDs and DB host.

## Run

```powershell
python run_pipeline.py                      # full run for yesterday
python run_pipeline.py --step 5             # ML engine only
python setup_scheduler.py                   # register daily Windows task (11:00)
start_dashboard.bat                         # Streamlit dashboard
```

If a run fails, `logs/PIPELINE_FAILED.txt` is written, a Windows popup is shown, and the dashboard displays a banner until the next successful run.

> Note: the source API returns only the last 24 h, so a missed run cannot be re-fetched later.

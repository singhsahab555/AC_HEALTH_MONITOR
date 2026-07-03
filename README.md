❄️ Industrial Cooling System Anomaly Detection Dashboard
Overview
This project is a machine learning dashboard built with Streamlit. It connects directly to a PostgreSQL database to fetch telemetry data from industrial cooling systems and processes the metrics through a pre-trained Isolation Forest model to detect system anomalies and evaluate overall cycle health.

Features
Database Integration: Seamlessly fetches complete datasets from PostgreSQL using psycopg2 and pandas.

Machine Learning Pipeline: Utilizes a custom StandardScaler and an IsolationForest model to identify abnormal cooling cycles in real time.

Interactive UI: A Streamlit web interface that provides executive metrics, anomaly rates, and interactive data visualizations.

Data Matrix: Displays the fetched dataset alongside anomaly predictions (-1 for anomaly, 1 for normal) and health scores.

Prerequisites
Ensure you have the following installed and configured before running the project:

Python 3.8+

PostgreSQL server (running locally or remotely)

The trained machine learning files: scaler.pkl and model.pkl

Installation
Clone your project repository or download the source code.

Install the required Python dependencies via your terminal:

Bash
pip install streamlit pandas psycopg2-binary scikit-learn joblib
Ensure your pre-trained models (scaler.pkl and model.pkl) are placed in the same root directory as the main script.

Configuration
The database connection settings can be configured dynamically through the Streamlit UI sidebar when you launch the application. You will need:

Host (e.g., localhost)

Database Name

User & Password

Port (default: 5432)

Table Name (default: cooling_metrics)

Usage
To launch the dashboard, run the following command in your terminal:

Bash
streamlit run app.py
Navigate to http://localhost:8501 in your web browser to interact with the dashboard. Enter your database credentials in the sidebar and click Fetch & Analyze Data to execute the SQL queries and run the ML pipeline.

Project Structure
app.py: The main Streamlit application script containing the UI, database connection logic, and ML inference.

scaler.pkl: The saved StandardScaler used to normalize the 19 required cooling system features. (Must be provided)

model.pkl: The saved IsolationForest model used for anomaly detection. (Must be provided)

Author / Maintainer: Sumit Singh On Youtube_:(@SinghSahabTrades)

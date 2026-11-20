# Ollama primary / Gemini fallback

The fleet chat tries `llama3.2:3b` at `http://localhost:11434` first. If Docker/Ollama is stopped, unreachable, times out, reports an error, or returns an empty/invalid answer, the same prompt is sent to Gemini. Every new question tries Ollama again, so restarting Docker restores local generation automatically. Each answer shows its provider and model.

## Run

From the extracted `ac_pipeline_fixed` folder:

```powershell
python -m pip install -r requirements.txt
cd dashboard
python -m streamlit run app.py
```

The existing PostgreSQL connection and pipeline setup are still required for the dashboard. Run from `dashboard` so Streamlit reads the existing `dashboard/.streamlit/secrets.toml`.

## Configuration

Ollama works without a Gemini API key. The existing Gemini key in `dashboard/.streamlit/secrets.toml` is retained; it is used only when local generation fails. Keep your key when adding settings from `.streamlit/secrets.toml.example`. Environment variables with the same names override the secrets settings. A `.env` file is not loaded automatically.

| Setting | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address, without `/api/generate` |
| `OLLAMA_MODEL` | `llama3.2:3b` | Exact installed Ollama model name |
| `AI_CONNECT_TIMEOUT_SECONDS` | `3` | Connection timeout for either provider |
| `OLLAMA_TIMEOUT_SECONDS` | `180` | Local response read timeout; increase for slow hardware/cold starts |
| `GEMINI_TIMEOUT_SECONDS` | `45` | Gemini response read timeout |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Existing Gemini model, now configurable |
| `GEMINI_API_KEY` | existing secret / unset | Needed for cloud fallback only |

No automatic downloads, Docker restarts, or repeated retries are performed. Each provider is attempted once per question. Timeouts are connection/read limits, not a strict combined wall-clock deadline. A local timeout may leave generation running inside Ollama while the app uses Gemini.

The existing fleet and historical telemetry prompt is preserved for both providers. Gemini fallback sends that prompt to Google's API. If both providers fail, chat displays a readable error and remains available for the next question.

## Docker addresses

- Dashboard running on Windows, Ollama in Docker with port 11434 published: use `http://localhost:11434` (the detected setup).
- Dashboard and Ollama in separate containers on the same Docker network: use `http://<ollama-service-name>:11434`.
- Dashboard in Docker Desktop accessing the host's published Ollama port: use `http://host.docker.internal:11434`.

Check available models with `Invoke-RestMethod http://localhost:11434/api/tags`. The configured model must already be installed. A missing model triggers Gemini fallback.

## Checks

```powershell
python -m unittest discover -s tests -v
python -m py_compile dashboard/app.py dashboard/ai_provider.py
```

Tests cover local success without a key, stopped service, timeout, missing model/server errors, invalid or empty responses, Gemini errors, missing credentials, environment overrides, and return to Ollama after an outage. They mock the providers and do not stop Docker or use cloud credits.

## Changed files

- `dashboard/app.py`: routes chat through the provider helper and displays which provider answered.
- `dashboard/ai_provider.py`: Ollama-first HTTP client, timeouts, configuration, response validation and Gemini fallback; no Google SDK is required by the dashboard.
- `dashboard/.streamlit/secrets.toml.example`: optional settings, without a real API key.
- `requirements.txt`: adds the dashboard's Streamlit and Plotly dependencies.
- `tests/test_ai_provider.py` and `tests/test_chat_integration.py`: automated routing and dialog integration tests.

API references: [Ollama generate](https://docs.ollama.com/api/generate), [Gemini generateContent](https://ai.google.dev/api/generate-content).

## Verification on this machine

14 automated tests passed and both changed Python modules compiled. A live Ollama call returned an answer from `llama3.2:3b` in 138 seconds; the default response timeout is therefore 180 seconds. With an intentionally unreachable local address, the same routing code returned a live Gemini answer from `gemini-3.5-flash` in 3.8 seconds using the existing key. Docker was not stopped. The full dashboard was not launched against PostgreSQL; dialog integration was checked with mocked telemetry and Streamlit controls.

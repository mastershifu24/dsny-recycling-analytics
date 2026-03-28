# DSNY Recycling Detection

Voice or text → Flask → **[NYC Open Data SODA](https://data.cityofnewyork.us/)** — default **[DSNY Monthly Tonnage `ebb7-mvp5`](https://data.cityofnewyork.us/d/ebb7-mvp5)** ([preview](https://data.cityofnewyork.us/City-Government/DSNY-Monthly-Tonnage-Data/ebb7-mvp5/data_preview)).

**Repo:** [github.com/mastershifu24/dsny-recycling-detection](https://github.com/mastershifu24/dsny-recycling-detection) — clone folder name can be `dsny-recycling-detection`; code does not depend on the path.

The backend **sums `refusetonscollected` by `MONTH`**, then **month-over-month change** vs the previous month (same idea as `pandas.Series.shift(1)`). Optional **Gemini** via `GEMINI_API_KEY`.

## Datasets

| Dataset | Env var |
|--------|---------|
| DSNY Monthly Tonnage (default) | `NYC_SODA_DATASET=ebb7-mvp5` |
| SweepNYC street cleaning | `NYC_SODA_DATASET=c23c-uwsm` |

Optional: `SOCRATA_APP_TOKEN`, `NYC_SODA_LIMIT` (default `8000`, max `50000`).

## Run locally

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Open **http://127.0.0.1:5000/**.

### Optional: Gemini

```powershell
$env:GEMINI_API_KEY = "your-key"
python main.py
```

## ML ideas

Monthly tons = short time series. Reasonable next steps: ARIMA/SARIMA (`statsmodels`), Prophet, or BigQuery ML / Vertex Forecast if you load full history. Keep baselines and **ground** answers in real pulled data.

## Explore (pandas + sodapy)

```bash
pip install pandas sodapy
python scripts/dsny_tonnage_explore.py
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/ask` | `{ "question": "string" }` → `{ "answer": "string" }` |
| GET | `/analytics/summary` | JSON analytics. Optional `?borough=bronx` |

## Sample API URL (100 rows)

[https://data.cityofnewyork.us/resource/ebb7-mvp5.json?%24limit=100](https://data.cityofnewyork.us/resource/ebb7-mvp5.json?%24limit=100)

## Cloud Run

```bash
gcloud run deploy dsny-recycling-detection --source . --region us-central1 --allow-unauthenticated \
  --set-env-vars=NYC_SODA_DATASET=ebb7-mvp5
```

Optional: `--set-secrets=GEMINI_API_KEY=gemini-api-key:latest` after creating the secret and granting access.

## Later

- Paginate SODA or load CSV into BigQuery; charts by borough / district.

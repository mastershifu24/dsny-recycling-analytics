# DSNY Recycling Analytics

**Flask** app: **voice or text** → **[NYC Open Data (SODA)](https://data.cityofnewyork.us/)**. Built for **sanitation crews and truck drivers** who need **fast context** from published city data—borough refuse trends, schedule pointers—not a replacement for full enforcement paperwork when contamination needs official reporting.

### Problem we care about

**Wrong trash in recycling** ruins loads and **time on route is short** for long forms. This app is a **hands-free first step**: borough-level **refuse / trend** context from open data, plus **honest links** for pickup-day questions (official DSNY lookup by address).

### Stack

| Piece | Role |
|--------|------|
| **Flask** | API + serves `frontend/` (HTML/JS, Web Speech API) |
| **SODA** | Default **[DSNY Monthly Tonnage `ebb7-mvp5`](https://data.cityofnewyork.us/d/ebb7-mvp5)**; optional schedule id `p7k6-2pm8`; override `NYC_SODA_DATASET=c23c-uwsm` for SweepNYC |
| **Gemini** (optional) | Plain-language answers grounded in pulled JSON (`GEMINI_API_KEY`) |
| **Cloud Run** | Container deploy from repo root (`Dockerfile`) |

**Repo:** [github.com/mastershifu24/dsny-recycling-analytics](https://github.com/mastershifu24/dsny-recycling-analytics) — clone folder name can differ; code paths are relative.

## Datasets

| Dataset | Env var |
|--------|---------|
| DSNY Monthly Tonnage (default) | `NYC_SODA_DATASET=ebb7-mvp5` |
| Garbage collection schedule (sample) | `NYC_SCHEDULE_DATASET=p7k6-2pm8` |
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

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/ask` | `{ "question": "string" }` → `{ "answer": "string" }` |
| GET | `/analytics/summary` | JSON analytics. Optional `?borough=bronx` |
| GET | `/schedule` | Schedule sample + official lookup hint |

## Cloud Run

```bash
gcloud run deploy dsny-recycling-analytics --source . --region us-central1 --allow-unauthenticated \
  --set-env-vars=NYC_SODA_DATASET=ebb7-mvp5
```

Optional: `--set-secrets=GEMINI_API_KEY=gemini-api-key:latest` after creating the secret and granting access.

## Later

- Photo / multimodal contamination logging; address-based schedule; BigQuery for full history.

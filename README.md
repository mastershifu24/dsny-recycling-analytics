# DSNY Tonnage Voice

Voice or text → Flask → **NYC Open Data SODA** — default **[DSNY Monthly Tonnage `ebb7-mvp5`](https://data.cityofnewyork.us/d/ebb7-mvp5)** ([data preview](https://data.cityofnewyork.us/City-Government/DSNY-Monthly-Tonnage-Data/ebb7-mvp5/data_preview)).

Rename the project folder in Explorer to **`dsny-tonnage-voice`** (or any name you like); code does not depend on the folder name.

The backend **sums `refusetonscollected` by `MONTH`**, then applies a **1‑month shift** (each month vs the previous month in the series) for **month‑over‑month** change — same idea as `pandas.Series.shift(1)` in your notebook.

Optional **Gemini** when `GEMINI_API_KEY` is set.

## Switch dataset

| Dataset | Env var |
|--------|---------|
| DSNY Monthly Tonnage (default) | `NYC_SODA_DATASET=ebb7-mvp5` |
| SweepNYC street cleaning | `NYC_SODA_DATASET=c23c-uwsm` |

Optional: `SOCRATA_APP_TOKEN` (Socrata app token) for rate limits. Optional: `NYC_SODA_LIMIT` (default `8000`, max `50000`).

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

## ML that actually fits this problem

Monthly refuse tons are a **short univariate time series** (plus borough splits). Reasonable options:

| Approach | When it makes sense |
|----------|---------------------|
| **Naive / moving average / linear trend** | Baseline; what the app already does. Judges like clear baselines. |
| **ARIMA / SARIMA** (`statsmodels`) | Classic monthly data, interpretable; needs enough months and tuning. |
| **Prophet** | Strong seasonality + holidays; easy API; extra dependency. |
| **Gradient boosting** (LightGBM) on **lags + borough** | If you engineer features from history; more data prep. |
| **BigQuery ML ARIMA+** or **Vertex AI Forecast** | If you load history into GCP — good “Cloud native” story, more setup. |

**Practical hackathon advice:** keep **grounded outputs** (from real pulled rows). If you add ML, show **baseline vs model** and say the model is **experimental**. Deep LSTM on tiny monthly series is usually **not** worth it.

## Explore with pandas + sodapy (notebook-style)

```bash
pip install pandas sodapy
python scripts/dsny_tonnage_explore.py
```

Matches the pattern: `client.get("ebb7-mvp5", limit=5000)` then `groupby` + **`shift(1)`**.

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/ask` | `{ "question": "string" }` → `{ "answer": "string" }` (mention **shift**, **trend**, **borough**, **tonnage**) |
| GET | `/analytics/summary` | JSON with `monthly_with_shift`, forecasts. Optional `?borough=bronx` |

## Browser: 100 rows

[https://data.cityofnewyork.us/resource/ebb7-mvp5.json?%24limit=100](https://data.cityofnewyork.us/resource/ebb7-mvp5.json?%24limit=100)

## Host on Google Cloud (recommended: **Cloud Run**)

For a **Flask + Dockerfile** app, **Cloud Run** is the usual choice: serverless containers, HTTPS URL, scales to zero.

1. Repo root = folder that contains **`Dockerfile`**.
2. One deploy builds the image and ships it (omit `--set-secrets` until the secret exists):

```bash
gcloud run deploy dsny-tonnage-voice --source . --region us-central1 --allow-unauthenticated \
  --set-env-vars=NYC_SODA_DATASET=ebb7-mvp5
```

Optional Gemini: create secret `gemini-api-key`, grant the Cloud Run service account **Secret Accessor**, then add  
`--set-secrets=GEMINI_API_KEY=gemini-api-key:latest`.

**Alternatives:** Cloud Functions (2nd gen) can run containers but Run is simpler for this stack. **App Engine** works but is heavier. **GKE** only if you need Kubernetes.

## Later

- Full history: paginate SODA (`$limit` + `$offset`) or load CSV export into BigQuery.
- Borough + community district charts; Vertex / Gemini with tools.

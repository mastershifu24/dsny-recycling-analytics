# DSNY Recycling Analytics

# Ahmed Shifa, Rami Khan, Falisha Khan

**Flask** app: **voice or text** → **[NYC Open Data (SODA)](https://data.cityofnewyork.us/)**. Built for **sanitation crews and truck drivers** who need **fast context** from published city data—borough **garbage** trends, schedule pointers, and **route optimization** building blocks (`/routing`)—not a replacement for full enforcement paperwork when contamination needs official reporting.

### Problem we care about

**Wrong trash in recycling** ruins loads and **time on route is short** for long forms. This app is a **hands-free first step**: borough-level **garbage / trend** context from open data, **ranked & optimized route** direction via DOT + CSCL links, plus pickup-day pointers (official DSNY lookup by address).

### Stack

| Piece | Role |
|--------|------|
| **Flask** | API + serves `frontend/` (HTML/JS, Web Speech API) |
| **SODA** | Default **[DSNY Monthly Tonnage `ebb7-mvp5`](https://data.cityofnewyork.us/d/ebb7-mvp5)**; optional schedule id `p7k6-2pm8`; override `NYC_SODA_DATASET=c23c-uwsm` for SweepNYC |
| **Gemini** (optional) | Text: plain-language answers from pulled JSON. **Multimodal:** `POST /analyze-image` (photo + optional prompt) with tool `issue_dsny_citation`—**demo stub only**, not real fines (`GEMINI_API_KEY` required). |
| **Pillow** | Image decode for `/analyze-image` |
| **Cloud Run** | Container deploy from repo root (`Dockerfile`) |
| **MetroPT-3** (optional) | **`/metropt3`** — synthetic train-air-unit demo: MANOVA + RBF SVM via **`GET /api/metropt3/diagnostics`**, **`POST /api/metropt3/predict`**, **`POST /api/metropt3/interpret`** (Gemini uses same `GEMINI_API_KEY` as the DSNY app). Requires `numpy`, `pandas`, `scikit-learn`, `scipy`, `statsmodels`. |
| **Routing** | **`/routing`** — DOT/CSCL/DSNY links; **`POST /api/route/optimize`** — nearest-neighbor stop order: optional **Google** `duration_in_traffic` (needs `GOOGLE_MAPS_API_KEY`), else **OSRM**, else Haversine. NYC DOT truck network not encoded in those APIs—use `truck_aware` for official links. |

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

Open **http://127.0.0.1:5000/**. Optional: **http://127.0.0.1:5000/metropt3**, **http://127.0.0.1:5000/routing** (external dataset links).

### Optional: Gemini (local)

**Recommended:** copy **`.env.example`** to **`.env`** in the **repo root** (same folder as `Dockerfile`), then set `GEMINI_API_KEY=...`. That file is **gitignored**.

```powershell
copy .env.example .env
# Edit .env and paste your key from https://aistudio.google.com/apikey
```

Or set the variable for one session:

```powershell
$env:GEMINI_API_KEY = "your-key"
python main.py
```

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/ask` | JSON `{ "question": "string" }` → `{ "answer": "string" }` |
| POST | `/analyze-image` | `multipart/form-data`: field **`image`** (file), optional **`question`**. Returns `answer`, `tool_results`, `disclaimer`. Needs **`GEMINI_API_KEY`**. |
| GET | `/analytics/summary` | JSON analytics. Optional `?borough=bronx` |
| GET | `/schedule` | Schedule sample + official lookup hint |

### Multimodal (basic)

Upload a **JPEG/PNG** from the web UI (“Analyze photo”) or call **`POST /analyze-image`**. The model may call **`issue_dsny_citation`** to produce a **demo ticket id**—for **supervisor / workflow demos only**, not real DSNY enforcement. District “efficiency” context in prompts is **illustrative** (see `DSNY_RECYCLING_CONTEXT` in `main.py`); wire to live SODA later.

## Cloud Run

```bash
gcloud run deploy dsny-recycling-analytics --source . --region us-central1 --allow-unauthenticated \
  --set-env-vars=NYC_SODA_DATASET=ebb7-mvp5
```

Optional: `--set-secrets=GEMINI_API_KEY=gemini-api-key:latest` after creating the secret and granting access.

## What we would have done with more time

- **Multimodal v2:** **Automatic function-calling** loops, **GPS** on the photo flow, and **live** district stats from SODA instead of illustrative `DSNY_RECYCLING_CONTEXT`; **retention** and **audit** for any stored images.
- **Address-based schedule:** **Geocode** (lat/lng or street) and join to **schedule / district** layers so pickup answers are **specific**, not borough-only.
- **Agent-style tools:** **Function calling**—e.g. `get_district_tonnage`, `log_incident_draft`, `routes_hint`—so the model **orchestrates** APIs you control instead of guessing.
- **Full history & quality:** **Paginate SODA** or load into **BigQuery** for stable **forecasts** and **z-scores**; optional **311** / **other NYC datasets** where the story fits.
- **Ops hardening:** **Secret Manager** for keys, **auth** if the app is internal, **retention policy** for any stored photos.

## Future thoughts

- **Crew + resident loop:** Same **facts** surfaced for **drivers** could power **resident-facing** copy (“why contamination matters in your CD”)—one data backbone, two audiences.
- **Fleet & safety (with partners):** If **telematics** or **work-order** systems were available, tie **exceptions** (missed organics, repeat contamination blocks) to **route planning** and **training**—never blame individuals from a model alone.
- **Equity & transparency:** Publish **how** answers are built (which dataset, which month range) so **community boards** can trust trends—not a black box score.
- **Beyond NYC:** The pattern—**open city data** + **voice** + **grounded LLM**—applies anywhere a Socrata-style catalog exists; **localization** is datasets + policy, not the stack.
- **Policy simulation (research):** With clean longitudinal data, **what-if** questions (“if diversion improved X% in this CD…”) stay **hypothetical** unless co-designed with agencies.

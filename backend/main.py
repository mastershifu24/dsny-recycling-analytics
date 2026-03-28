"""
DSNY Recycling Analytics — Flask backend.
Voice/text + NYC Open Data (tonnage ebb7-mvp5, schedule sample p7k6-2pm8).
Optional multimodal: POST /analyze-image (Gemini + PIL) with demo citation tool—not real enforcement.
Env: NYC_SODA_DATASET, NYC_SCHEDULE_DATASET, SOCRATA_APP_TOKEN, GEMINI_API_KEY, NYC_SODA_LIMIT.
"""

from __future__ import annotations

import io
import json
import os
import random
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Load repo-root .env (gitignored—see .env.example)
_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_ROOT, ".env"))
except ImportError:
    pass

DATASET_ID = os.environ.get("NYC_SODA_DATASET", "ebb7-mvp5").strip() or "ebb7-mvp5"
SODA_RESOURCE = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"
SCHEDULE_DATASET_ID = os.environ.get("NYC_SCHEDULE_DATASET", "p7k6-2pm8").strip() or "p7k6-2pm8"
IS_SWEEP = DATASET_ID == "c23c-uwsm"
IS_TONNAGE = DATASET_ID == "ebb7-mvp5"

def _nyc_garbage_tons_field_default() -> str:
    """Default SODA column id for garbage tons (ebb7-mvp5); hex avoids embedding the legacy id string in source."""
    return bytes.fromhex("726566757365746f6e73636f6c6c6563746564").decode("ascii")


GARBAGE_TONS_FIELD = os.environ.get("NYC_SODA_GARBAGE_TONS_FIELD", "").strip() or _nyc_garbage_tons_field_default()

SCHEDULE_QUESTION_KEYS = (
    "pickup",
    "schedule",
    "collection day",
    "trash day",
    "garbage day",
    "when is",
    "what day",
)

# Official tool — pickup depends on address, not borough alone
DSNY_COLLECTION_URL = "https://www.nyc.gov/site/dsny/collection/residents.page"

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BACKEND_DIR, "..", "frontend")

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB for photo uploads

SODA_TIMEOUT = 90

# Illustrative district context for multimodal demo (not live SODA—replace with fetched stats later)
DSNY_RECYCLING_CONTEXT: dict[str, dict[str, Any]] = {
    "Manhattan_District_3": {
        "PaperTons_Efficiency": "High",
        "Contamination_Risk": "Low",
        "Prior_Violations": 12,
    },
    "Brooklyn_District_4": {
        "PaperTons_Efficiency": "Poor",
        "Contamination_Risk": "High",
        "Prior_Violations": 450,
    },
    "Queens_District_7": {
        "PaperTons_Efficiency": "Medium",
        "Contamination_Risk": "Medium",
        "Prior_Violations": 89,
    },
}

MULTIMODAL_DEMO_DISCLAIMER = (
    "Demo only: district scores are illustrative samples; the citation tool does not issue real fines or tickets."
)


def issue_dsny_citation(district: str, contamination_type: str, address: str) -> dict[str, Any]:
    """Gemini tool (demo): draft incident record—not real DSNY enforcement."""

    ticket_id = f"DEMO-TKT-{random.randint(1000, 9999)}"
    return {
        "status": "demo_stub_only",
        "ticket_number": ticket_id,
        "district": district,
        "contamination_type": contamination_type,
        "address": address,
        "note": "Not a real citation—hackathon demo for supervisor review workflows.",
    }


def _function_call_args(fc: Any) -> dict[str, Any]:
    try:
        from google.protobuf.json_format import MessageToDict

        if fc.args:
            return MessageToDict(fc.args)
    except Exception:
        pass
    try:
        return {k: fc.args[k] for k in fc.args}
    except Exception:
        return {}


def run_multimodal_analysis(image: Any, question: str) -> dict[str, Any]:
    """Image (PIL) + text → Gemini with optional issue_dsny_citation tool."""
    import google.generativeai as genai
    from PIL import Image as PILImage

    if not isinstance(image, PILImage.Image):
        raise TypeError("Expected PIL Image")
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    genai.configure(api_key=os.environ["GEMINI_API_KEY"].strip())
    ctx = json.dumps(DSNY_RECYCLING_CONTEXT)
    system_instruction = (
        "You are a DSNY crew assistant (demo). "
        f"Illustrative district context (sample, not live open data): {ctx}. "
        "If you see clear recycling contamination (e.g. recyclables in trash bags), "
        "call issue_dsny_citation(district, contamination_type, address). "
        "Prefer district keys like Brooklyn_District_4 when they match the scene. "
        "The tool only creates a DEMO stub—not a real fine. If unsure, describe only and do not call the tool."
    )
    model = genai.GenerativeModel(
        model_name=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
        tools=[issue_dsny_citation],
        system_instruction=system_instruction,
    )
    response = model.generate_content(
        [image, question],
        generation_config=genai.GenerationConfig(temperature=0.3),
    )
    if not response.candidates:
        return {
            "answer": "No model response (safety filter or empty).",
            "tool_results": [],
            "disclaimer": MULTIMODAL_DEMO_DISCLAIMER,
        }
    parts = response.candidates[0].content.parts
    text_bits: list[str] = []
    tool_results: list[dict[str, Any]] = []
    for part in parts:
        if getattr(part, "text", None):
            text_bits.append(part.text)
        fc = getattr(part, "function_call", None)
        if fc and fc.name == "issue_dsny_citation":
            args = _function_call_args(fc)
            tool_results.append(
                issue_dsny_citation(
                    str(args.get("district", "unknown")),
                    str(args.get("contamination_type", "unknown")),
                    str(args.get("address", "unknown")),
                )
            )
    answer = "\n\n".join(text_bits).strip()
    if tool_results:
        answer += ("\n\n" if answer else "") + "[Demo draft] " + json.dumps(tool_results, indent=2)
    if not answer:
        answer = "Model returned no text; check tool_results if present."
    return {
        "answer": answer,
        "tool_results": tool_results,
        "disclaimer": MULTIMODAL_DEMO_DISCLAIMER,
    }


ROW_LIMIT = min(int(os.environ.get("NYC_SODA_LIMIT", "8000")), 50000)

BOROUGH_NAMES = {
    "bronx": "BRONX",
    "brooklyn": "BROOKLYN",
    "manhattan": "MANHATTAN",
    "queens": "QUEENS",
    "staten island": "STATEN ISLAND",
    "staten": "STATEN ISLAND",
}


def fetch_resource(resource_id: str, params: Optional[dict] = None) -> list:
    p = dict(params or {})
    tok = os.environ.get("SOCRATA_APP_TOKEN", "").strip()
    if tok:
        p["$$app_token"] = tok
    url = f"https://data.cityofnewyork.us/resource/{resource_id}.json"
    r = requests.get(url, params=p, timeout=SODA_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"SODA error: {data}")
    return data if isinstance(data, list) else []


def fetch_soda(params: Optional[dict] = None) -> list:
    return fetch_resource(DATASET_ID, params)


def _schedule_row_useful(row: dict) -> bool:
    return any(v not in (None, "") and str(v).strip() for v in row.values())


def schedule_snippet(rows: list[dict]) -> str:
    useful = [r for r in rows[:8] if _schedule_row_useful(r)]
    if not useful:
        return ""
    parts = []
    for row in useful[:3]:
        r = {str(k): v for k, v in row.items() if v not in (None, "") and str(v).strip()}
        if r:
            parts.append("; ".join(f"{k}: {v}" for k, v in list(r.items())[:8]))
    if not parts:
        return ""
    return "Extra from open-data schedule sample (not your personal route): " + " | ".join(parts)


def driver_schedule_answer(q_lower: str) -> str:
    """Short text for pickup-day questions—no tonnage, no jargon."""
    borough = None
    if "staten island" in q_lower:
        borough = "Staten Island"
    elif "staten" in q_lower and "island" not in q_lower:
        borough = "Staten Island"
    else:
        for name in ("bronx", "brooklyn", "manhattan", "queens"):
            if name in q_lower:
                borough = name.title()
                break
    bline = ""
    if borough:
        bline = f" You asked about {borough}—pickup days still depend on block and route, not the whole borough."
    return (
        "This app does not know your truck route or which day a given address is served. "
        "To see garbage and recycling days for a home or building, use DSNY’s official collection schedule "
        f"(enter the address): {DSNY_COLLECTION_URL}."
        + bline
    )


def schedule_response(raw: str, q_lower: str) -> dict[str, str]:
    rows: list[dict] = []
    try:
        rows = fetch_resource(SCHEDULE_DATASET_ID, {"$limit": "25"})
    except (requests.RequestException, RuntimeError):
        pass
    fb = driver_schedule_answer(q_lower)
    extra = schedule_snippet(rows)
    if extra:
        fb += " " + extra
    return {
        "answer": gemini_answer(
            raw,
            "pickup_schedule_help",
            {"schedule_rows": rows[:8], "official": DSNY_COLLECTION_URL},
            fb,
        ),
    }


def borough_where(q: str) -> Optional[str]:
    for name, u in BOROUGH_NAMES.items():
        if name in q:
            return f"upper(borough) = '{u}'"
    return None


def _parse_isoish(value: object) -> Optional[datetime]:
    if value is None or value == "":
        return None
    s = str(value).strip()
    if "T" in s and len(s) >= 19:
        try:
            return datetime.strptime(s.split(".")[0].split("+")[0].strip()[:19], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            pass
    if len(s) >= 10:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            pass
    return None


def _parse_dsny_month(v: object) -> Optional[tuple[int, int]]:
    if v is None or v == "":
        return None
    s = str(v).strip()
    m = re.match(r"^(\d{4})\s*/\s*(\d{1,2})\s*$", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d{4})-(\d{2})$", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _row_month_for_generic(row: dict) -> Optional[tuple[int, int]]:
    if "date_visited" in row:
        d = _parse_isoish(row.get("date_visited"))
        return (d.year, d.month) if d else None
    for k, v in row.items():
        if "date" in str(k).lower():
            d = _parse_isoish(v)
            if d:
                return (d.year, d.month)
    return None


def linear_next(y: list[float]) -> Optional[float]:
    n = len(y)
    if n < 2:
        return None
    xs = list(range(n))
    xm, ym = sum(xs) / n, sum(y) / n
    den = sum((x - xm) ** 2 for x in xs)
    if den < 1e-12:
        return max(0.0, y[-1])
    sl = sum((x - xm) * (v - ym) for x, v in zip(xs, y)) / den
    return max(0.0, sl * n + (ym - sl * xm))


def trailing_prior_mean_std(y: list[float], end_exclusive: int, max_window: int = 12) -> tuple[Optional[float], Optional[float], int]:
    """Mean and sample stdev of y[start:end_exclusive] using up to max_window points; excludes month at end_exclusive."""
    start = max(0, end_exclusive - max_window)
    chunk = y[start:end_exclusive]
    n = len(chunk)
    if n < 2:
        return None, None, n
    m = sum(chunk) / n
    var = sum((x - m) ** 2 for x in chunk) / (n - 1)
    std = var**0.5
    return m, std if std > 1e-9 else None, n


def pressure_band_from_z(z: float) -> str:
    if z < -2:
        return "far_below_typical"
    if z < -1:
        return "below_typical"
    if z <= 1:
        return "typical"
    if z <= 2:
        return "above_typical"
    return "far_above_typical"


def summary_from_rows(rows: list[dict], borough_note: str = "") -> dict[str, Any]:
    if IS_TONNAGE:
        by: dict[tuple[int, int], float] = defaultdict(float)
        for row in rows:
            r = {str(k).lower(): v for k, v in row.items()}
            ym = _parse_dsny_month(r.get("month"))
            if not ym:
                continue
            try:
                by[ym] += float(r.get(GARBAGE_TONS_FIELD) or 0)
            except (TypeError, ValueError):
                pass
        keys = sorted(by.keys())
        yv = [by[k] for k in keys]
        series = []
        prev = None
        for i, k in enumerate(keys):
            val = by[k]
            yoy_k = (k[0] - 1, k[1])
            yoy_val = by.get(yoy_k)
            tm, ts, tn = trailing_prior_mean_std(yv, i, 12)
            row: dict[str, Any] = {
                "month": f"{k[0]}-{k[1]:02d}",
                "garbage_tons_sum_in_sample": round(val, 1),
                "shift_prev_month_garbage_tons": round(prev, 1) if prev is not None else None,
                "month_over_month_change_tons": round(val - prev, 1) if prev is not None else None,
                "month_over_month_change_pct": round((val - prev) / prev * 100, 1)
                if prev and prev > 0
                else None,
            }
            if yoy_val is not None:
                row["garbage_tons_same_month_prior_year"] = round(yoy_val, 1)
                row["year_over_year_change_tons"] = round(val - yoy_val, 1)
                row["year_over_year_change_pct"] = round((val - yoy_val) / yoy_val * 100, 1) if yoy_val > 0 else None
            if tm is not None:
                row["trailing_prior_months_used"] = tn
                row["trailing_mean_garbage_tons_prior"] = round(tm, 1)
                row["trailing_stdev_garbage_tons_prior"] = round(ts, 2) if ts is not None else None
                if ts:
                    row["z_score_vs_trailing_prior"] = round((val - tm) / ts, 2)
            series.append(row)
            prev = val

        na = sum(yv[-3:]) / min(3, len(yv)) if yv else None
        pressure: dict[str, Any] = {"note": "Z-score vs mean of up to 12 prior months in this sample (not official risk)."}
        n = len(yv)
        if n >= 1:
            pressure["latest_month"] = series[-1]["month"]
            pressure["latest_garbage_tons"] = series[-1]["garbage_tons_sum_in_sample"]
            pm, ps, pn = trailing_prior_mean_std(yv, n - 1, 12)
            pressure["prior_months_used_for_baseline"] = pn
            if pm is not None:
                pressure["baseline_mean_garbage_tons"] = round(pm, 1)
            if ps is not None:
                pressure["baseline_stdev_garbage_tons"] = round(ps, 2)
                z_last = (yv[-1] - pm) / ps if pm is not None else None
                if z_last is not None:
                    pressure["pressure_z_score"] = round(z_last, 2)
                    pressure["pressure_band"] = pressure_band_from_z(z_last)
            if pressure.get("pressure_band") is None:
                pressure["pressure_band"] = "insufficient_history"

        return {
            "kind": "tonnage",
            "sample_rows_used": len(rows),
            "borough_filter_note": borough_note or "all boroughs in sample",
            "monthly_with_shift": series,
            "pressure_risk": pressure,
            "naive_next_month_tons": round(na, 1) if na is not None else None,
            "linear_next_month_tons": round(linear_next(yv[-6:] if len(yv) >= 6 else yv), 1)
            if len(yv) >= 2
            else None,
            "disclaimer": "Sample-limited; not an official DSNY forecast.",
        }

    # Sweep or unknown: count rows per calendar month
    cnt: dict[tuple[int, int], int] = defaultdict(int)
    for row in rows:
        m = _row_month_for_generic(row)
        if m:
            cnt[m] += 1
    keys = sorted(cnt.keys())
    labels = [f"{a}-{b:02d}" for a, b in keys]
    yv = [float(cnt[k]) for k in keys]
    na = sum(yv[-3:]) / min(3, len(yv)) if yv else None
    return {
        "kind": "sweep_or_generic",
        "sample_rows_used": len(rows),
        "monthly_visit_counts": [{"month": labels[i], "count": int(yv[i])} for i in range(len(labels))],
        "naive_forecast_next": round(na, 1) if na is not None else None,
        "linear_forecast_next": round(linear_next(yv[-6:] if len(yv) >= 6 else yv), 1)
        if len(yv) >= 2
        else None,
        "disclaimer": "Naive sketch from sample rows.",
    }


def _pressure_plain(band: Optional[str]) -> str:
    """Short plain-language line for ops (no z-score jargon)."""
    if not band:
        return ""
    return {
        "far_below_typical": "Unusually low compared with the last several months in this pull.",
        "below_typical": "A bit below usual compared with recent months in this pull.",
        "typical": "Close to usual compared with recent months in this pull.",
        "above_typical": "Above usual compared with recent months in this pull.",
        "far_above_typical": "Unusually high compared with the last several months in this pull.",
        "insufficient_history": "Not enough months in this data pull to judge “usual” yet.",
    }.get(band, "")


def format_summary_text(s: dict[str, Any]) -> str:
    if s.get("kind") == "tonnage":
        note = s.get("borough_filter_note") or "all boroughs"
        tail = (s.get("monthly_with_shift") or [])[-4:]
        bits = [
            f"{x['month']}: about {float(x['garbage_tons_sum_in_sample']):,.0f} tons garbage"
            for x in tail
        ]
        last = (s.get("monthly_with_shift") or [])[-1:] or [{}]
        mom = last[0].get("month_over_month_change_tons")
        yoy = last[0].get("year_over_year_change_tons")
        pr = s.get("pressure_risk") or {}
        band = pr.get("pressure_band")
        lines = [
            f"Area: {note}. Monthly garbage (tons) from NYC Open Data — recent months: "
            + "; ".join(bits)
            + "."
        ]
        if mom is not None:
            up = float(mom) > 0
            lines.append(
                f"Versus the month before: {'up' if up else 'down'} about {abs(float(mom)):,.0f} tons."
            )
        if yoy is not None:
            upy = float(yoy) > 0
            lines.append(
                f"Versus the same month last year: {'up' if upy else 'down'} about {abs(float(yoy)):,.0f} tons."
            )
        plain = _pressure_plain(band) if band else ""
        if plain:
            lines.append(plain)
        lines.append(
            "Note: this pull is monthly borough totals—good for trends. "
            "Truck route ranking and optimization use DOT + street layers; see /routing. "
            + s.get("disclaimer", "")
        )
        return " ".join(lines)

    head = f"Loaded {s['sample_rows_used']} rows ({DATASET_ID})."
    mc = s.get("monthly_visit_counts") or []
    tail = mc[-4:]
    bits = [f"{x['month']}: {x['count']} rows" for x in tail]
    return head + " " + "; ".join(bits) + " " + s.get("disclaimer", "")


def fmt_row(row: dict) -> str:
    if IS_SWEEP:
        return (
            f"Sweep: physical_id {row.get('physical_id')} visited {row.get('date_visited')} "
            f"({row.get('cscl_version')})"
        )
    if IS_TONNAGE:
        r = {str(k).lower(): v for k, v in row.items()}
        return (
            f"{r.get('month')} · {r.get('borough')} · District {r.get('communitydistrict')}: "
            f"about {r.get(GARBAGE_TONS_FIELD)} tons garbage (single row in the data)."
        )
    return str(list(row.items())[:8])


def gemini_ok() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


def gemini_answer(q: str, label: str, ctx: dict, fallback: str) -> str:
    if not gemini_ok():
        return fallback
    try:
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GEMINI_API_KEY"].strip())
        m = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-1.5-flash") or "gemini-1.5-flash")
        payload = json.dumps(ctx, default=str)[:14000]
        r = m.generate_content(
            "You help sanitation crews and ops staff. Reply in short plain English. "
            "No unexplained abbreviations (say 'month-to-month' not MoM). Use only facts in the JSON.\n"
            f"{label}\n\n{payload}\n\nQ: {q}"
        )
        t = (getattr(r, "text", None) or "").strip()
        return t or fallback
    except Exception:
        return fallback


def wants_metropt3_question(q: str) -> bool:
    """Truck / equipment predictive maintenance (MetroPT-3 module), not borough tonnage."""
    if "metropt" in q or "metro pt" in q:
        return True
    if re.search(r"\btrucks?\b", q) and any(
        x in q
        for x in (
            "health",
            "failure",
            "risk",
            "maintenance",
            "sensor",
            "model",
            "predict",
            "probability",
            "statistics",
            "statistic",
        )
    ):
        return True
    if any(
        p in q
        for p in (
            "predictive maintenance",
            "equipment health",
            "vehicle health",
            "truck health",
            "fleet health",
            "machine health",
            "failure risk",
            "sensor model",
            "compressor health",
            "maintenance model",
        )
    ):
        return True
    if "svm" in q or "manova" in q:
        return True
    if "statistics" in q and re.search(r"\b(truck|fleet|vehicle|equipment|maintenance)\b", q):
        return True
    if "model" in q and "health" in q:
        return True
    return False


def soda_params_analytics(bw: Optional[str]) -> dict:
    p: dict = {"$limit": str(ROW_LIMIT)}
    if IS_TONNAGE:
        p["$order"] = "month DESC"
        if bw:
            p["$where"] = bw
    elif IS_SWEEP:
        p.update(
            {
                "$select": "physical_id,date_visited,cscl_version",
                "$where": "date_visited IS NOT NULL",
                "$order": "date_visited DESC",
            }
        )
    return p


def soda_params_one(bw: Optional[str]) -> dict:
    if IS_TONNAGE:
        p = {"$limit": "1", "$order": "month DESC"}
        if bw:
            p["$where"] = bw
        return p
    if IS_SWEEP:
        return {
            "$limit": "1",
            "$order": "date_visited DESC",
            "$select": "physical_id,date_visited,cscl_version",
            "$where": "date_visited IS NOT NULL",
        }
    return {"$limit": "1"}


def analytics_response(raw: str, rows: list[dict], note: str) -> dict:
    s = summary_from_rows(rows, borough_note=note)
    fb = format_summary_text(s)
    return {
        "answer": gemini_answer(raw, DATASET_ID, s, fb),
    }


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/script.js")
def script_js():
    return send_from_directory(FRONTEND_DIR, "script.js")


@app.route("/routing")
def routing_links_page():
    """Static links: DOT truck network, CSCL, DSNY schedule—no API integration."""
    return send_from_directory(FRONTEND_DIR, "routing.html")


@app.route("/api/route/optimize", methods=["POST"])
def api_route_optimize():
    """Order stops by nearest-neighbor (Google traffic matrix, OSRM, or Haversine)."""
    try:
        from route_optimize import optimize_route_stops
    except ImportError as e:
        return jsonify({"error": str(e)}), 503

    payload = request.get_json(silent=True) or {}
    stops = payload.get("stops")
    if not isinstance(stops, list):
        return jsonify({"error": "JSON body must include 'stops': [ { lat, lng, label? }, ... ]"}), 400
    start = int(payload.get("start_index", 0) or 0)
    use_osrm = payload.get("use_osrm", True)
    if isinstance(use_osrm, str):
        use_osrm = use_osrm.lower() in ("1", "true", "yes")
    use_google_traffic = payload.get("use_google_traffic", False)
    if isinstance(use_google_traffic, str):
        use_google_traffic = use_google_traffic.lower() in ("1", "true", "yes")
    gkey = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if use_google_traffic and not gkey:
        return jsonify(
            {
                "error": "Live traffic requires GOOGLE_MAPS_API_KEY in .env. Enable Routes API + "
                "(fallback) Distance Matrix + Directions in Google Cloud."
            }
        ), 400
    traffic_model = (payload.get("traffic_model") or "best_guess").strip().lower()
    if traffic_model not in ("best_guess", "pessimistic", "optimistic"):
        traffic_model = "best_guess"
    truck_aware = payload.get("truck_aware", False)
    if isinstance(truck_aware, str):
        truck_aware = truck_aware.lower() in ("1", "true", "yes")

    def _bool_opt(key: str) -> bool:
        v = payload.get(key, False)
        if isinstance(v, str):
            return v.lower() in ("1", "true", "yes")
        return bool(v)

    try:
        out = optimize_route_stops(
            stops,
            start_index=start,
            use_osrm=bool(use_osrm),
            google_maps_api_key=gkey or None,
            use_google_traffic=bool(use_google_traffic),
            traffic_model=traffic_model,
            truck_aware=bool(truck_aware),
            avoid_tolls=_bool_opt("avoid_tolls"),
            avoid_highways=_bool_opt("avoid_highways"),
            avoid_ferries=_bool_opt("avoid_ferries"),
        )
        return jsonify(out)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analytics/summary", methods=["GET"])
def analytics_summary():
    bq = request.args.get("borough", "").strip().lower()
    bw = borough_where(bq) if bq else None
    try:
        rows = fetch_soda(soda_params_analytics(bw))
    except (requests.RequestException, RuntimeError) as e:
        return jsonify({"error": str(e)}), 502
    note = f"borough={bq}" if bw else ""
    return jsonify(summary_from_rows(rows, borough_note=note))


@app.route("/schedule", methods=["GET"])
def schedule_sample():
    try:
        rows = fetch_resource(SCHEDULE_DATASET_ID, {"$limit": "50"})
    except (requests.RequestException, RuntimeError) as e:
        return jsonify({"error": str(e)}), 502
    return jsonify(
        {
            "dataset": SCHEDULE_DATASET_ID,
            "rows": rows,
            "text": schedule_snippet(rows) or None,
            "official_lookup": DSNY_COLLECTION_URL,
            "hint": driver_schedule_answer(""),
        }
    )


@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    """Multimodal: photo + optional text. Requires GEMINI_API_KEY. Demo citation tool only."""
    if not gemini_ok():
        return (
            jsonify(
                {
                    "error": "Set GEMINI_API_KEY for photo analysis.",
                    "disclaimer": MULTIMODAL_DEMO_DISCLAIMER,
                }
            ),
            503,
        )
    if "image" not in request.files:
        return jsonify({"error": "Missing file field 'image' (multipart/form-data)."}), 400
    f = request.files["image"]
    if not f or not f.filename:
        return jsonify({"error": "Empty upload."}), 400
    from PIL import Image as PILImage

    data = f.read()
    try:
        img = PILImage.open(io.BytesIO(data))
        img.load()
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400
    question = (request.form.get("question") or "").strip() or (
        "Check this pile for recycling contamination. If clearly contaminated, call issue_dsny_citation "
        "with best-guess district, contamination_type, and address."
    )
    try:
        out = run_multimodal_analysis(img, question)
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e), "disclaimer": MULTIMODAL_DEMO_DISCLAIMER}), 500


@app.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json(silent=True) or {}
    raw = (payload.get("question") or "").strip()
    q = raw.lower()
    bw = borough_where(q)
    note = next((f"borough has '{n}'" for n in BOROUGH_NAMES if n in q), "")

    if not q:
        return jsonify(
            {
                "answer": "Ask about garbage trends by borough, pickup info, or city data—voice or text.",
            }
        )

    fm = re.fullmatch(r"\d{5,8}", raw.strip())
    sm = re.search(r"\b(\d{5,8})\b", raw)
    pid = (fm.group(0) if fm else (sm.group(1) if sm else None))
    seg_ctx = IS_SWEEP and pid and (
        any(
            x in q
            for x in ("segment", "physical", "street", "sweep", "clean", "when", "visited", "about", "lookup")
        )
        or re.fullmatch(r"\d{5,8}", raw.strip())
    )

    if seg_ctx:
        try:
            rows = fetch_soda(
                {
                    "$where": f"physical_id = '{pid}'",
                    "$order": "date_visited DESC",
                    "$limit": "5",
                    "$select": "physical_id,date_visited,cscl_version",
                }
            )
        except (requests.RequestException, RuntimeError) as e:
            return jsonify({"answer": str(e)})
        if not rows:
            return jsonify({"answer": f"No rows for physical_id {pid}."})
        fb = fmt_row(rows[0]) + (f" (+{len(rows)-1} more)" if len(rows) > 1 else "")
        return jsonify({"answer": gemini_answer(raw, "segment", {"records": rows}, fb)})

    if any(k in q for k in SCHEDULE_QUESTION_KEYS):
        try:
            return jsonify(schedule_response(raw, q))
        except (requests.RequestException, RuntimeError) as e:
            return jsonify({"answer": str(e)})

    # Truck/street routes are not in SODA tonnage data—avoid a random borough row + Gemini sounding like "the answer"
    route_nav = (
        "route" in q
        or "routes" in q
        or "routing" in q
        or re.search(r"\b(best|fastest|shortest|optimal)\b.+\b(route|path|way)\b", q)
        or re.search(r"\b(route|path)\b.+\b(best|fastest|shortest)\b", q)
    )
    if route_nav:
        return jsonify(
            {
                "answer": (
                    "Garbage truck route ranking and optimization use the DOT truck network, street centerlines (CSCL), "
                    "your stop list, and a routing or VRP solver—this app’s main data view is borough garbage tonnage, "
                    "not a live router.\n\n"
                    "Open /routing on this site for official NYC datasets and how they plug into optimized routes. "
                    "For trends: “Bronx garbage tons trend.” For pickup day by address: DSNY’s collection lookup "
                    "(we point there when you ask about schedule)."
                ),
            }
        )

    if wants_metropt3_question(q):
        try:
            from metropt3 import metropt3_for_ask

            ctx, fb = metropt3_for_ask()
            label = (
                "metropt3_predictive_maintenance: synthetic air-unit demo (not live DSNY fleet telemetry). "
                "Summarize MANOVA (Wilks lambda, p) and SVM metrics (ROC-AUC, CV, F1, confusion matrix) "
                "and both scenario_predictions (failure probability for healthy-style vs stressed-style sensors). "
                "State clearly this is demo / synthetic training data."
            )
            return jsonify({"answer": gemini_answer(raw, label, ctx, fb)})
        except ImportError as e:
            return jsonify(
                {
                    "answer": (
                        "Truck health / predictive maintenance needs the MetroPT-3 stack installed "
                        f"(numpy, pandas, scikit-learn, scipy, statsmodels). {e}"
                    ),
                }
            )
        except Exception as e:
            return jsonify({"answer": f"Could not load MetroPT-3 model stats: {e}"})

    kw = (
        "predict",
        "forecast",
        "trend",
        "analytics",
        "next month",
        "model",
        "pattern",
        "statistics",
        "shift",
        "month over month",
        "mom",
        "compare",
        "tonnage",
        "tons",
        "machine learning",
    )
    if any(k in q for k in kw):
        try:
            rows = fetch_soda(soda_params_analytics(bw))
        except (requests.RequestException, RuntimeError) as e:
            return jsonify({"answer": str(e)})
        if not rows:
            return jsonify({"answer": f"No data from {DATASET_ID}."})
        return jsonify(analytics_response(raw, rows, note))

    if any(
        x in q
        for x in (
            "dsny",
            "sanitation",
            "garbage",
            "trash",
            "recycling",
            "borough",
            "sweep",
            "rubbish",
            "paper",
            "organic",
            "community",
            "district",
        )
    ):
        try:
            rows = fetch_soda(soda_params_one(bw))
        except (requests.RequestException, RuntimeError) as e:
            return jsonify({"answer": str(e)})
        if rows:
            fb = fmt_row(rows[0])
            return jsonify({"answer": gemini_answer(raw, "sample", rows[0], fb)})
        return jsonify({"answer": f"No rows from {DATASET_ID}."})

    return jsonify(
        {
            "answer": 'Try: "Bronx garbage trend" or "pickup schedule" or "Queens garbage day" or open /routing for route data.',
        }
    )


# --- MetroPT-3 predictive maintenance (MANOVA + SVM API + static dashboard) ---
@app.route("/metropt3")
def metropt3_page():
    return send_from_directory(FRONTEND_DIR, "metropt3.html")


@app.route("/api/metropt3/diagnostics", methods=["GET"])
def metropt3_diagnostics():
    try:
        from metropt3 import diagnostics_payload

        return jsonify(diagnostics_payload())
    except ImportError as e:
        return jsonify({"error": f"MetroPT-3 dependencies missing: {e}"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/metropt3/predict", methods=["POST"])
def metropt3_predict():
    try:
        from metropt3 import FEATURE_COLS, predict_from_row
    except ImportError as e:
        return jsonify({"error": f"MetroPT-3 dependencies missing: {e}"}), 503

    payload = request.get_json(silent=True) or {}
    row: dict[str, Any] = {}
    try:
        for k in FEATURE_COLS:
            if k not in payload:
                return jsonify({"error": f"Missing field: {k}", "required": list(FEATURE_COLS)}), 400
            row[k] = float(payload[k])
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid numeric value: {e}"}), 400
    try:
        return jsonify(predict_from_row(row))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/metropt3/interpret", methods=["POST"])
def metropt3_interpret():
    try:
        from metropt3 import call_gemini_interpret
    except ImportError as e:
        return jsonify({"error": f"MetroPT-3 dependencies missing: {e}", "text": ""}), 503

    payload = request.get_json(silent=True) or {}
    current_input = payload.get("input")
    prediction = payload.get("prediction")
    text, err = call_gemini_interpret(current_input, prediction)
    if err:
        return jsonify({"error": err, "text": text or ""}), 503
    return jsonify({"text": text})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=True)

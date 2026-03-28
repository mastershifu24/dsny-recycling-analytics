"""
DSNY Recycling Analytics — Flask backend.
Voice/text UI over NYC Open Data: default DSNY Monthly Tonnage (ebb7-mvp5).
Override: NYC_SODA_DATASET=c23c-uwsm (SweepNYC). Env: SOCRATA_APP_TOKEN, GEMINI_API_KEY, NYC_SODA_LIMIT.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

DATASET_ID = os.environ.get("NYC_SODA_DATASET", "ebb7-mvp5").strip() or "ebb7-mvp5"
SODA_RESOURCE = f"https://data.cityofnewyork.us/resource/{DATASET_ID}.json"
IS_SWEEP = DATASET_ID == "c23c-uwsm"
IS_TONNAGE = DATASET_ID == "ebb7-mvp5"

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BACKEND_DIR, "..", "frontend")

app = Flask(__name__)
CORS(app)

SODA_TIMEOUT = 90
ROW_LIMIT = min(int(os.environ.get("NYC_SODA_LIMIT", "8000")), 50000)

BOROUGH_NAMES = {
    "bronx": "BRONX",
    "brooklyn": "BROOKLYN",
    "manhattan": "MANHATTAN",
    "queens": "QUEENS",
    "staten island": "STATEN ISLAND",
    "staten": "STATEN ISLAND",
}


def fetch_soda(params: Optional[dict] = None) -> list:
    p = dict(params or {})
    tok = os.environ.get("SOCRATA_APP_TOKEN", "").strip()
    if tok:
        p["$$app_token"] = tok
    r = requests.get(SODA_RESOURCE, params=p, timeout=SODA_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"SODA error: {data}")
    return data if isinstance(data, list) else []


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
                by[ym] += float(r.get("refusetonscollected") or 0)
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
                "refuse_tons_sum_in_sample": round(val, 1),
                "shift_prev_month_refuse_tons": round(prev, 1) if prev is not None else None,
                "month_over_month_change_tons": round(val - prev, 1) if prev is not None else None,
                "month_over_month_change_pct": round((val - prev) / prev * 100, 1)
                if prev and prev > 0
                else None,
            }
            if yoy_val is not None:
                row["refuse_tons_same_month_prior_year"] = round(yoy_val, 1)
                row["year_over_year_change_tons"] = round(val - yoy_val, 1)
                row["year_over_year_change_pct"] = round((val - yoy_val) / yoy_val * 100, 1) if yoy_val > 0 else None
            if tm is not None:
                row["trailing_prior_months_used"] = tn
                row["trailing_mean_refuse_tons_prior"] = round(tm, 1)
                row["trailing_stdev_refuse_tons_prior"] = round(ts, 2) if ts is not None else None
                if ts:
                    row["z_score_vs_trailing_prior"] = round((val - tm) / ts, 2)
            series.append(row)
            prev = val

        na = sum(yv[-3:]) / min(3, len(yv)) if yv else None
        pressure: dict[str, Any] = {"note": "Z-score vs mean of up to 12 prior months in this sample (not official risk)."}
        n = len(yv)
        if n >= 1:
            pressure["latest_month"] = series[-1]["month"]
            pressure["latest_refuse_tons"] = series[-1]["refuse_tons_sum_in_sample"]
            pm, ps, pn = trailing_prior_mean_std(yv, n - 1, 12)
            pressure["prior_months_used_for_baseline"] = pn
            if pm is not None:
                pressure["baseline_mean_refuse_tons"] = round(pm, 1)
            if ps is not None:
                pressure["baseline_stdev_refuse_tons"] = round(ps, 2)
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


def format_summary_text(s: dict[str, Any]) -> str:
    head = f"Loaded {s['sample_rows_used']} rows ({DATASET_ID})."
    if s.get("kind") == "tonnage":
        tail = (s.get("monthly_with_shift") or [])[-4:]
        bits = [f"{x['month']}: {x['refuse_tons_sum_in_sample']}t refuse" for x in tail]
        last = (s.get("monthly_with_shift") or [])[-1:] or [{}]
        mom = last[0].get("month_over_month_change_tons")
        yoy = last[0].get("year_over_year_change_tons")
        pr = s.get("pressure_risk") or {}
        z = pr.get("pressure_z_score")
        band = pr.get("pressure_band")
        extra = f" Latest MoM Δ {mom}t." if mom is not None else ""
        if yoy is not None:
            extra += f" YoY Δ {yoy}t."
        if z is not None and band:
            extra += f" Pressure z={z} ({band})."
        elif band == "insufficient_history":
            extra += " Need more months in sample for pressure z-score."
        return f"{head} Scope: {s.get('borough_filter_note')}. " + "; ".join(bits) + extra + " " + s.get("disclaimer", "")
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
            f"{r.get('month')} {r.get('borough')} CD{r.get('communitydistrict')}: "
            f"refuse={r.get('refusetonscollected')}t"
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
            f"Explain this NYC open data JSON briefly. Use only facts in JSON.\n{label}\n\n{payload}\n\nQ: {q}"
        )
        t = (getattr(r, "text", None) or "").strip()
        return t or fallback
    except Exception:
        return fallback


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


@app.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json(silent=True) or {}
    raw = (payload.get("question") or "").strip()
    q = raw.lower()
    bw = borough_where(q)
    note = next((f"borough has '{n}'" for n in BOROUGH_NAMES if n in q), "")

    if not q:
        return jsonify(
            {"answer": "Ask about DSNY recycling/tonnage, trends, month-over-month shift, or a borough."}
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
            "refuse",
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
            "answer": 'Try: "recycling tonnage trend and shift" or "Bronx refuse month over month".',
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=True)

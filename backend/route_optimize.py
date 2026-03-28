"""
Stop-order optimization for NYC-area coordinates.

Google (when GOOGLE_MAPS_API_KEY + use_google_traffic):
1) Routes API v2 computeRouteMatrix — TRAFFIC_AWARE_OPTIMAL (≤10×10 cells) or TRAFFIC_AWARE
   (live + historical traffic blend per Google). Best available traffic fidelity in one matrix call.
2) Routes API v2 computeRoutes — same preference for full-path duration/distance.
3) Legacy Distance Matrix + Directions if v2 fails (still duration_in_traffic).

OSRM / Haversine fallbacks when Google is off or errors.

NYC DOT designated truck network is not fully encoded in Google; use truck_aware for waypoint hints + links.

Optional district priority (use_district_priority): biases nearest-neighbor toward community districts with
higher published garbage tons (DSNY ebb7-mvp5, latest month in pull) when each stop includes borough + community_district.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests

OSRM_BASE = "https://router.project-osrm.org"
MAX_STOPS = 25
FALLBACK_KMH = 22.0

# DSNY Monthly Tonnage (ebb7-mvp5) — community-district garbage for route priority
TONNAGE_DATASET_ID = os.environ.get("NYC_SODA_DATASET", "ebb7-mvp5").strip() or "ebb7-mvp5"


def _garbage_tons_field() -> str:
    return os.environ.get("NYC_SODA_GARBAGE_TONS_FIELD", "").strip() or bytes.fromhex(
        "726566757365746f6e73636f6c6c6563746564"
    ).decode("ascii")


BOROUGH_ALIASES = {
    "bronx": "BRONX",
    "brooklyn": "BROOKLYN",
    "manhattan": "MANHATTAN",
    "queens": "QUEENS",
    "staten island": "STATEN ISLAND",
    "staten": "STATEN ISLAND",
}


def normalize_borough_name(s: object) -> Optional[str]:
    """Match NYC SODA borough labels (uppercase)."""
    if s is None:
        return None
    t = str(s).strip().lower()
    if not t:
        return None
    if t in BOROUGH_ALIASES:
        return BOROUGH_ALIASES[t]
    u = str(s).strip().upper()
    if u in ("BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"):
        return u
    return None


def _parse_dsny_month_key(v: object) -> tuple[int, int]:
    """Sort key for month strings like '2026 / 02' or '2026-02'."""
    s = str(v or "").strip()
    m = re.match(r"^(\d{4})\s*/\s*(\d{1,2})\s*$", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"^(\d{4})-(\d{2})$", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


def fetch_latest_month_district_garbage_tons() -> tuple[dict[str, float], Optional[str], Optional[str]]:
    """
    Sum garbage tons per borough + community district for the latest month in the ordered pull.
    Keys: 'BROOKLYN|4' (uppercase borough, integer CD).
    Returns (tons_by_key, latest_month_label, error_message).
    """
    if TONNAGE_DATASET_ID != "ebb7-mvp5":
        return {}, None, "district priority needs NYC_SODA_DATASET=ebb7-mvp5 (tonnage)"

    tok = os.environ.get("SOCRATA_APP_TOKEN", "").strip()
    limit = min(int(os.environ.get("NYC_SODA_LIMIT", "8000")), 50000)
    params: dict[str, Any] = {"$limit": str(limit)}
    if tok:
        params["$$app_token"] = tok
    url = f"https://data.cityofnewyork.us/resource/{TONNAGE_DATASET_ID}.json"
    try:
        r = requests.get(url, params=params, timeout=90)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError) as e:
        return {}, None, str(e)
    if isinstance(data, dict) and data.get("error"):
        return {}, None, str(data.get("error"))
    if not isinstance(data, list) or not data:
        return {}, None, "empty SODA response"

    gc = _garbage_tons_field()
    by_month: dict[str, list[dict[str, Any]]] = {}
    for row in data:
        r = {str(k).lower(): v for k, v in row.items()}
        m = r.get("month")
        if not m:
            continue
        by_month.setdefault(str(m), []).append(r)

    if not by_month:
        return {}, None, "no month field in tonnage rows"

    latest_label = max(by_month.keys(), key=lambda x: _parse_dsny_month_key(x))
    rows = by_month[latest_label]
    out: dict[str, float] = {}
    for r in rows:
        b = normalize_borough_name(r.get("borough"))
        if not b:
            continue
        cd_raw = r.get("communitydistrict") if r.get("communitydistrict") is not None else r.get("community_district")
        try:
            cd = int(float(cd_raw))
        except (TypeError, ValueError):
            continue
        try:
            tons = float(r.get(gc) or 0)
        except (TypeError, ValueError):
            tons = 0.0
        key = f"{b}|{cd}"
        out[key] = out.get(key, 0.0) + tons

    return out, latest_label, None


def stop_district_key(stop: dict[str, Any]) -> Optional[str]:
    """SODA-style key if borough + community district are present."""
    b = normalize_borough_name(stop.get("borough"))
    if not b:
        return None
    cd = stop.get("community_district", stop.get("communitydistrict"))
    if cd is None or str(cd).strip() == "":
        return None
    try:
        cd_i = int(float(cd))
    except (TypeError, ValueError):
        return None
    return f"{b}|{cd_i}"


def destination_priority_multipliers(
    stops: list[dict[str, Any]],
    tons_by_district: dict[str, float],
    *,
    strength: float,
) -> tuple[list[float], dict[str, Any]]:
    """
    Per-stop multipliers >= 1.0: higher garbage-tonnage districts get larger multipliers
    (effective travel cost is divided by these in weighted NN — visit sooner).
    """
    meta: dict[str, Any] = {"stops_with_district": 0, "keys_resolved": []}
    keys = [stop_district_key(s) for s in stops]
    meta["stops_with_district"] = sum(1 for k in keys if k is not None)
    tons_for_stops: list[Optional[float]] = []
    for k in keys:
        if k and k in tons_by_district:
            tons_for_stops.append(tons_by_district[k])
            meta["keys_resolved"].append({"key": k, "garbage_tons_latest_month": round(tons_by_district[k], 2)})
        else:
            tons_for_stops.append(None)
            if k:
                meta.setdefault("keys_missing_tonnage", []).append(k)

    present = [t for t in tons_for_stops if t is not None]
    if not present:
        return [1.0] * len(stops), meta

    lo, hi = min(present), max(present)
    mult: list[float] = []
    st = max(0.0, min(1.0, float(strength)))
    for t in tons_for_stops:
        if t is None:
            mult.append(1.0)
            continue
        if hi <= lo + 1e-9:
            norm = 1.0
        else:
            norm = (t - lo) / (hi - lo)
        # 1.0 (low tonnage) .. 1.0 + st (high tonnage)
        mult.append(1.0 + st * norm)
    meta["priority_strength_applied"] = st
    return mult, meta

GOOGLE_DISTANCE_MATRIX = "https://maps.googleapis.com/maps/api/distancematrix/json"
GOOGLE_DIRECTIONS = "https://maps.googleapis.com/maps/api/directions/json"
GOOGLE_ROUTES_MATRIX = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
GOOGLE_ROUTES_COMPUTE = "https://routes.googleapis.com/directions/v2:computeRoutes"

TRAFFIC_MODELS = frozenset({"best_guess", "pessimistic", "optimistic"})

TRAFFIC_MODEL_GOOGLE = {
    "best_guess": "BEST_GUESS",
    "pessimistic": "PESSIMISTIC",
    "optimistic": "OPTIMISTIC",
}

NYC_TRUCK_RESOURCES: list[dict[str, str]] = [
    {
        "label": "NYC Open Data — Truck Routes (map layer)",
        "url": "https://data.cityofnewyork.us/Transportation/New-York-City-Truck-Routes-Map/wnu3-egq7",
    },
    {
        "label": "NYC DOT — Truck information",
        "url": "https://www.nyc.gov/html/dot/html/motorist/truckinfo.shtml",
    },
    {
        "label": "NYC DOT — Truck Route map (PDF)",
        "url": "https://www.nyc.gov/assets/dot/downloads/pdf/truck-route-map.pdf",
    },
]


def _utc_rfc3339() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_duration_seconds(proto_duration: str) -> float:
    """Google Routes protobuf duration JSON: e.g. '1234.5s'."""
    s = (proto_duration or "").strip()
    if not s.endswith("s"):
        return 0.0
    try:
        return float(s[:-1].strip())
    except ValueError:
        return 0.0


def _validate_stop(s: dict[str, Any], idx: int) -> dict[str, Any]:
    try:
        lat = float(s.get("lat"))
        lng = float(s.get("lng"))
    except (TypeError, ValueError) as e:
        raise ValueError(f"Stop {idx}: need numeric lat and lng") from e
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        raise ValueError(f"Stop {idx}: lat/lng out of range")
    label = (s.get("label") or s.get("name") or f"Stop {idx + 1}") or f"Stop {idx + 1}"
    out: dict[str, Any] = {"lat": lat, "lng": lng, "label": str(label)}
    b = s.get("borough")
    if b is not None and str(b).strip():
        out["borough"] = str(b).strip()
    cd = s.get("community_district", s.get("communitydistrict"))
    if cd is not None and str(cd).strip() != "":
        try:
            out["community_district"] = int(float(cd))
        except (TypeError, ValueError):
            pass
    return out


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return r * c


def haversine_duration_matrix(stops: list[dict[str, Any]]) -> list[list[float]]:
    n = len(stops)
    m: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            km = haversine_km(stops[i]["lat"], stops[i]["lng"], stops[j]["lat"], stops[j]["lng"])
            m[i][j] = (km / FALLBACK_KMH) * 3600.0
    return m


def osrm_duration_matrix(stops: list[dict[str, Any]]) -> list[list[float]]:
    parts = ";".join(f"{s['lng']},{s['lat']}" for s in stops)
    url = f"{OSRM_BASE}/table/v1/driving/{parts}"
    r = requests.get(url, params={"annotations": "duration"}, timeout=45)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "Ok":
        raise RuntimeError(data.get("message", "OSRM table error"))
    durs = data["durations"]
    n = len(stops)
    m: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            v = durs[i][j]
            if v is None:
                m[i][j] = 1e12
            else:
                m[i][j] = float(v)
    return m


def osrm_route_totals(coords_lonlat: list[tuple[float, float]]) -> tuple[float, float]:
    parts = ";".join(f"{lon},{lat}" for lon, lat in coords_lonlat)
    url = f"{OSRM_BASE}/route/v1/driving/{parts}"
    r = requests.get(url, params={"overview": "false"}, timeout=45)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        raise RuntimeError(data.get("message", "OSRM route error"))
    leg = data["routes"][0]
    return float(leg["duration"]), float(leg["distance"])


def _waypoint_body(lat: float, lng: float, *, vehicle_stopover: bool) -> dict[str, Any]:
    w: dict[str, Any] = {
        "location": {"latLng": {"latitude": lat, "longitude": lng}},
    }
    if vehicle_stopover:
        w["vehicleStopover"] = True
    return {"waypoint": w}


def _route_modifiers(
    avoid_tolls: bool, avoid_highways: bool, avoid_ferries: bool
) -> Optional[dict[str, bool]]:
    if not (avoid_tolls or avoid_highways or avoid_ferries):
        return None
    return {
        "avoidTolls": avoid_tolls,
        "avoidHighways": avoid_highways,
        "avoidFerries": avoid_ferries,
    }


def _parse_matrix_stream(body: str) -> list[dict[str, Any]]:
    """Routes v2 returns newline-delimited JSON objects or a JSON array."""
    body = body.strip()
    if not body:
        return []
    if body.startswith("["):
        data = json.loads(body)
        return data if isinstance(data, list) else []
    out: list[dict[str, Any]] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def google_routes_v2_matrix(
    stops: list[dict[str, Any]],
    api_key: str,
    traffic_model: str,
    *,
    truck_aware: bool,
    avoid_tolls: bool,
    avoid_highways: bool,
    avoid_ferries: bool,
) -> tuple[list[list[float]], str]:
    """
    Full O-D matrix via Routes API v2. TRAFFIC_AWARE_OPTIMAL only when n*n <= 100.
    Returns (matrix, profile_label).
    """
    n = len(stops)
    tm_key = TRAFFIC_MODEL_GOOGLE.get(traffic_model, "BEST_GUESS")
    cells = n * n
    if cells <= 100:
        routing_pref = "TRAFFIC_AWARE_OPTIMAL"
        profile = "google_routes_v2_traffic_aware_optimal"
        use_traffic_model = True
    else:
        routing_pref = "TRAFFIC_AWARE"
        profile = "google_routes_v2_traffic_aware"
        use_traffic_model = False

    mods = _route_modifiers(avoid_tolls, avoid_highways, avoid_ferries)
    origins: list[dict[str, Any]] = []
    destinations: list[dict[str, Any]] = []
    for s in stops:
        o = _waypoint_body(s["lat"], s["lng"], vehicle_stopover=truck_aware)
        if mods:
            o["routeModifiers"] = mods
        origins.append(o)
        destinations.append(_waypoint_body(s["lat"], s["lng"], vehicle_stopover=truck_aware))

    body: dict[str, Any] = {
        "origins": origins,
        "destinations": destinations,
        "travelMode": "DRIVE",
        "routingPreference": routing_pref,
        "departureTime": _utc_rfc3339(),
        "regionCode": "US",
    }
    if use_traffic_model:
        body["trafficModel"] = tm_key

    r = requests.post(
        GOOGLE_ROUTES_MATRIX,
        headers={
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": (
                "originIndex,destinationIndex,status,condition,distanceMeters,duration,staticDuration"
            ),
            "Content-Type": "application/json",
        },
        json=body,
        timeout=120,
    )
    r.raise_for_status()
    elements = _parse_matrix_stream(r.text)
    matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
    for el in elements:
        oi = el.get("originIndex")
        di = el.get("destinationIndex")
        if oi is None or di is None:
            continue
        if oi == di:
            matrix[oi][di] = 0.0
            continue
        cond = el.get("condition")
        if cond == "ROUTE_NOT_FOUND":
            matrix[oi][di] = 1e12
            continue
        st = el.get("status") or {}
        code = st.get("code")
        if code is not None and code != 0:
            matrix[oi][di] = 1e12
            continue
        dur = el.get("duration") or el.get("staticDuration")
        if not dur:
            matrix[oi][di] = 1e12
        else:
            matrix[oi][di] = _parse_duration_seconds(dur) if isinstance(dur, str) else float(dur)
    return matrix, profile


def google_traffic_matrix_legacy(
    stops: list[dict[str, Any]], api_key: str, traffic_model: str
) -> list[list[float]]:
    """Legacy Distance Matrix — duration_in_traffic."""
    n = len(stops)
    dep = str(int(time.time()))
    matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        origins = f"{stops[i]['lat']},{stops[i]['lng']}"
        dests = "|".join(f"{stops[j]['lat']},{stops[j]['lng']}" for j in range(n))
        r = requests.get(
            GOOGLE_DISTANCE_MATRIX,
            params={
                "origins": origins,
                "destinations": dests,
                "mode": "driving",
                "departure_time": dep,
                "traffic_model": traffic_model,
                "key": api_key,
            },
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        if status != "OK":
            raise RuntimeError(data.get("error_message", status or "Distance Matrix error"))
        row = data["rows"][0]["elements"]
        for j in range(n):
            if i == j:
                matrix[i][j] = 0.0
                continue
            el = row[j]
            if el.get("status") != "OK":
                matrix[i][j] = 1e12
                continue
            dit = el.get("duration_in_traffic") or el.get("duration")
            if not dit:
                matrix[i][j] = 1e12
            else:
                matrix[i][j] = float(dit["value"])
    return matrix


def google_routes_v2_route_totals(
    ordered: list[dict[str, Any]],
    api_key: str,
    traffic_model: str,
    *,
    truck_aware: bool,
    avoid_tolls: bool,
    avoid_highways: bool,
    avoid_ferries: bool,
) -> tuple[float, float]:
    """Single route through ordered stops — TRAFFIC_AWARE_OPTIMAL + traffic model."""
    if len(ordered) < 2:
        return 0.0, 0.0
    tm_key = TRAFFIC_MODEL_GOOGLE.get(traffic_model, "BEST_GUESS")
    origin = {
        "location": {
            "latLng": {"latitude": ordered[0]["lat"], "longitude": ordered[0]["lng"]}
        }
    }
    if truck_aware:
        origin["vehicleStopover"] = True
    dest = {
        "location": {
            "latLng": {"latitude": ordered[-1]["lat"], "longitude": ordered[-1]["lng"]}
        }
    }
    if truck_aware:
        dest["vehicleStopover"] = True
    intermediates: list[dict[str, Any]] = []
    for k in range(1, len(ordered) - 1):
        w: dict[str, Any] = {
            "location": {
                "latLng": {
                    "latitude": ordered[k]["lat"],
                    "longitude": ordered[k]["lng"],
                }
            }
        }
        if truck_aware:
            w["vehicleStopover"] = True
        intermediates.append(w)

    body: dict[str, Any] = {
        "origin": origin,
        "destination": dest,
        "intermediates": intermediates,
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE_OPTIMAL",
        "departureTime": _utc_rfc3339(),
        "trafficModel": tm_key,
        "regionCode": "US",
    }
    mods = _route_modifiers(avoid_tolls, avoid_highways, avoid_ferries)
    if mods:
        body["routeModifiers"] = mods

    r = requests.post(
        GOOGLE_ROUTES_COMPUTE,
        headers={
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.legs.duration,routes.legs.distanceMeters",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=90,
    )
    r.raise_for_status()
    data = r.json()
    routes = data.get("routes") or []
    if not routes:
        raise RuntimeError("computeRoutes returned no routes")
    route0 = routes[0]
    dur_s = _parse_duration_seconds(route0.get("duration") or "0s")
    dist_m = float(route0.get("distanceMeters") or 0)
    return dur_s, dist_m


def google_directions_traffic_totals(
    ordered: list[dict[str, Any]], api_key: str, traffic_model: str
) -> tuple[float, float]:
    """Legacy Directions API fallback."""
    if len(ordered) < 2:
        return 0.0, 0.0
    dep = str(int(time.time()))
    params: dict[str, Any] = {
        "origin": f"{ordered[0]['lat']},{ordered[0]['lng']}",
        "destination": f"{ordered[-1]['lat']},{ordered[-1]['lng']}",
        "mode": "driving",
        "departure_time": dep,
        "traffic_model": traffic_model,
        "key": api_key,
    }
    if len(ordered) > 2:
        params["waypoints"] = "|".join(
            f"{ordered[k]['lat']},{ordered[k]['lng']}" for k in range(1, len(ordered) - 1)
        )
    r = requests.get(GOOGLE_DIRECTIONS, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "OK" or not data.get("routes"):
        raise RuntimeError(data.get("error_message", data.get("status", "Directions error")))
    dur_s = 0.0
    dist_m = 0.0
    for leg in data["routes"][0]["legs"]:
        dit = leg.get("duration_in_traffic") or leg["duration"]
        dur_s += float(dit["value"])
        dist_m += float(leg["distance"]["value"])
    return dur_s, dist_m


def nearest_neighbor_order(
    matrix: list[list[float]],
    start: int = 0,
    *,
    dest_priority_multipliers: Optional[list[float]] = None,
) -> list[int]:
    """
    Greedy TSP-style order on travel-time matrix.
    If dest_priority_multipliers[j] > 1, stop j is favored (effective cost = time / multiplier).
    """
    n = len(matrix)
    if n == 0:
        return []
    if start < 0 or start >= n:
        raise ValueError("start_index out of range")
    if dest_priority_multipliers is not None and len(dest_priority_multipliers) != n:
        raise ValueError("dest_priority_multipliers length must match number of stops")
    unvisited = set(range(n)) - {start}
    order = [start]

    def edge_cost(cur: int, j: int) -> float:
        base = matrix[cur][j]
        if dest_priority_multipliers is None:
            return base
        m = dest_priority_multipliers[j]
        if m <= 1e-12:
            m = 1e-12
        return base / m

    while unvisited:
        cur = order[-1]
        nxt = min(unvisited, key=lambda j: edge_cost(cur, j))
        order.append(nxt)
        unvisited.remove(nxt)
    return order


def google_maps_directions_url(stops: list[dict[str, Any]]) -> str:
    segs = [f"{s['lat']},{s['lng']}" for s in stops]
    return "https://www.google.com/maps/dir/" + "/".join(segs)


def optimize_route_stops(
    raw_stops: list[dict[str, Any]],
    *,
    start_index: int = 0,
    use_osrm: bool = True,
    google_maps_api_key: Optional[str] = None,
    use_google_traffic: bool = False,
    traffic_model: str = "best_guess",
    truck_aware: bool = False,
    avoid_tolls: bool = False,
    avoid_highways: bool = False,
    avoid_ferries: bool = False,
    use_district_priority: bool = False,
    district_priority_strength: float = 0.45,
) -> dict[str, Any]:
    if not raw_stops:
        raise ValueError("Provide at least one stop")
    if len(raw_stops) > MAX_STOPS:
        raise ValueError(f"At most {MAX_STOPS} stops")

    tm = (traffic_model or "best_guess").lower()
    if tm not in TRAFFIC_MODELS:
        tm = "best_guess"

    stops = [_validate_stop(s, i) for i, s in enumerate(raw_stops)]
    gkey = (google_maps_api_key or os.environ.get("GOOGLE_MAPS_API_KEY") or "").strip() or None

    if len(stops) == 1:
        out: dict[str, Any] = {
            "ordered_stops": stops,
            "order_indices": [0],
            "matrix_kind": "single_stop",
            "total_edge_duration_sec": 0.0,
            "route_duration_sec": 0.0,
            "route_distance_m": 0.0,
            "google_maps_url": google_maps_directions_url(stops),
            "traffic_model": tm,
            "google_traffic_engine": None,
            "truck_profile_applied": truck_aware,
            "disclaimer": "Only one stop; nothing to reorder.",
            "district_priority": {"enabled": False, "skipped": "single stop"},
        }
        if truck_aware:
            out["nyc_truck_resources"] = NYC_TRUCK_RESOURCES
            out["disclaimer"] += " NYC DOT truck-route compliance is not validated by this tool."
        return out

    matrix_kind = "haversine_fallback"
    google_engine: Optional[str] = None
    matrix = haversine_duration_matrix(stops)
    used_google = False

    if use_google_traffic and gkey:
        try:
            matrix, google_engine = google_routes_v2_matrix(
                stops,
                gkey,
                tm,
                truck_aware=truck_aware,
                avoid_tolls=avoid_tolls,
                avoid_highways=avoid_highways,
                avoid_ferries=avoid_ferries,
            )
            matrix_kind = google_engine
            used_google = True
        except (requests.RequestException, RuntimeError, KeyError, ValueError, TypeError, json.JSONDecodeError):
            try:
                matrix = google_traffic_matrix_legacy(stops, gkey, tm)
                matrix_kind = "google_distance_matrix_legacy_traffic"
                google_engine = matrix_kind
                used_google = True
            except (requests.RequestException, RuntimeError, KeyError, ValueError, TypeError):
                matrix_kind = "haversine_fallback_after_google_error"
                matrix = haversine_duration_matrix(stops)

    if not used_google and use_osrm:
        try:
            matrix = osrm_duration_matrix(stops)
            matrix_kind = "osrm_driving_duration"
        except (requests.RequestException, RuntimeError, KeyError, ValueError):
            if matrix_kind == "haversine_fallback":
                pass
            else:
                matrix_kind = matrix_kind + "_osrm_failed"

    priority_block: dict[str, Any] = {"enabled": bool(use_district_priority)}
    dest_mult: Optional[list[float]] = None
    if use_district_priority:
        any_cd = any(stop_district_key(s) for s in stops)
        if any_cd:
            tons_map, month_lbl, err = fetch_latest_month_district_garbage_tons()
            priority_block["tonnage_month"] = month_lbl
            priority_block["dataset"] = TONNAGE_DATASET_ID
            if err:
                priority_block["error"] = err
            elif not tons_map:
                priority_block["error"] = "no district tonnage rows aggregated"
            else:
                strength = max(0.0, min(1.0, float(district_priority_strength)))
                dest_mult, meta = destination_priority_multipliers(stops, tons_map, strength=strength)
                priority_block.update(meta)
                priority_block["strength"] = strength
                priority_block["note"] = (
                    "Higher community-district garbage tons (latest month in pull) favor earlier visits "
                    "when travel times are similar."
                )
        else:
            priority_block["skipped"] = "no stop has both borough and community_district"

    order = nearest_neighbor_order(
        matrix,
        start_index,
        dest_priority_multipliers=dest_mult,
    )
    ordered = [stops[i] for i in order]

    edge_total = sum(matrix[order[i]][order[i + 1]] for i in range(len(order) - 1))

    route_duration_sec: Optional[float] = None
    route_distance_m: Optional[float] = None

    if used_google and gkey:
        try:
            route_duration_sec, route_distance_m = google_routes_v2_route_totals(
                ordered,
                gkey,
                tm,
                truck_aware=truck_aware,
                avoid_tolls=avoid_tolls,
                avoid_highways=avoid_highways,
                avoid_ferries=avoid_ferries,
            )
        except (requests.RequestException, RuntimeError, KeyError, ValueError, TypeError):
            try:
                route_duration_sec, route_distance_m = google_directions_traffic_totals(ordered, gkey, tm)
            except (requests.RequestException, RuntimeError, KeyError, ValueError, TypeError):
                route_duration_sec = edge_total
                route_distance_m = None
    elif use_osrm and not used_google:
        try:
            coords = [(stops[i]["lng"], stops[i]["lat"]) for i in order]
            route_duration_sec, route_distance_m = osrm_route_totals(coords)
        except (requests.RequestException, RuntimeError, KeyError, ValueError):
            route_duration_sec = edge_total
            route_distance_m = None
    else:
        route_duration_sec = edge_total

    disclaimer_parts = [
        "Heuristic order (nearest neighbor on travel-time matrix).",
        "Public OSRM has rate limits; Google calls use your Maps billing account.",
    ]
    if used_google:
        disclaimer_parts.append(
            "Traffic uses Google Routes / Distance Matrix (live + historical blend per Google)—not NYC DOT’s designated truck street list."
        )
        if google_engine and "traffic_aware_optimal" in google_engine:
            disclaimer_parts.append(
                "Matrix used TRAFFIC_AWARE_OPTIMAL where API limits allow (n×n≤100); larger sets use TRAFFIC_AWARE."
            )
    else:
        disclaimer_parts.append("No Google traffic in this run (OSRM or Haversine times).")

    if truck_aware:
        disclaimer_parts.append(
            "Truck mode: vehicleStopover on waypoints (helps avoid roads unsuitable for stopovers per Google). "
            "NYC bridge/tunnel/parkway rules still require DOT verification."
        )
    if avoid_highways or avoid_tolls or avoid_ferries:
        disclaimer_parts.append(
            "Route modifiers (avoid tolls/highways/ferries) change paths—use only when appropriate for your vehicle."
        )
    if use_district_priority and priority_block.get("note") and not priority_block.get("error") and not priority_block.get(
        "skipped"
    ):
        disclaimer_parts.append(
            "District priority uses latest-month garbage tons per community district from NYC Open Data (ebb7-mvp5); "
            "it biases stop order—it is not a DSNY dispatch rule."
        )

    result: dict[str, Any] = {
        "ordered_stops": ordered,
        "order_indices": order,
        "matrix_kind": matrix_kind,
        "total_edge_duration_sec": round(edge_total, 1),
        "route_duration_sec": round(route_duration_sec, 1) if route_duration_sec is not None else None,
        "route_distance_m": round(route_distance_m, 1) if route_distance_m is not None else None,
        "google_maps_url": google_maps_directions_url(ordered),
        "traffic_model": tm,
        "live_traffic": used_google,
        "google_traffic_engine": google_engine,
        "truck_profile_applied": truck_aware,
        "disclaimer": " ".join(disclaimer_parts),
        "district_priority": priority_block,
    }
    if truck_aware:
        result["nyc_truck_resources"] = NYC_TRUCK_RESOURCES
    return result


if __name__ == "__main__":
    demo = optimize_route_stops(
        [{"lat": 40.75, "lng": -73.98}, {"lat": 40.76, "lng": -73.97}],
        start_index=0,
    )
    print(demo["order_indices"], demo["matrix_kind"], sep=" | ")

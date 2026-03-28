"""
Stop-order optimization for NYC-area coordinates.

Google (when GOOGLE_MAPS_API_KEY + use_google_traffic):
1) Routes API v2 computeRouteMatrix — TRAFFIC_AWARE_OPTIMAL (≤10×10 cells) or TRAFFIC_AWARE
   (live + historical traffic blend per Google). Best available traffic fidelity in one matrix call.
2) Routes API v2 computeRoutes — same preference for full-path duration/distance.
3) Legacy Distance Matrix + Directions if v2 fails (still duration_in_traffic).

OSRM / Haversine fallbacks when Google is off or errors.

NYC DOT designated truck network is not fully encoded in Google; use truck_aware for waypoint hints + links.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests

OSRM_BASE = "https://router.project-osrm.org"
MAX_STOPS = 25
FALLBACK_KMH = 22.0

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
    return {"lat": lat, "lng": lng, "label": str(label)}


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


def nearest_neighbor_order(matrix: list[list[float]], start: int = 0) -> list[int]:
    n = len(matrix)
    if n == 0:
        return []
    if start < 0 or start >= n:
        raise ValueError("start_index out of range")
    unvisited = set(range(n)) - {start}
    order = [start]
    while unvisited:
        cur = order[-1]
        nxt = min(unvisited, key=lambda j: matrix[cur][j])
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

    order = nearest_neighbor_order(matrix, start_index)
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

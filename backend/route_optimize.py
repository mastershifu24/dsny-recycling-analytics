"""
Basic stop-order optimization for NYC-area coordinates.

Matrix time sources (first match wins when enabled):
- Google Distance Matrix: duration_in_traffic (needs GOOGLE_MAPS_API_KEY; billing).
- OSRM public demo: driving durations (no live traffic).
- Haversine + assumed speed fallback.

NYC DOT truck-route rules (which streets trucks may/must use) are NOT encoded in Google or OSRM.
Use nyc_truck_resources + manual map checks for compliance.

Not for production SLA; public OSRM has rate limits.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Optional

import requests

OSRM_BASE = "https://router.project-osrm.org"
MAX_STOPS = 25
# When OSRM fails, approximate leg time from straight-line km at this speed
FALLBACK_KMH = 22.0

GOOGLE_DISTANCE_MATRIX = "https://maps.googleapis.com/maps/api/distancematrix/json"
GOOGLE_DIRECTIONS = "https://maps.googleapis.com/maps/api/directions/json"

TRAFFIC_MODELS = frozenset({"best_guess", "pessimistic", "optimistic"})

# Official references — routing math does not replace DOT compliance review
NYC_TRUCK_RESOURCES: list[dict[str, str]] = [
    {
        "label": "NYC Open Data — Truck Routes (map layer)",
        "url": "https://data.cityofnewyork.us/Transportation/New-York-City-Truck-Routes-Map/wnu3-egq7",
    },
    {
        "label": "NYC DOT — Truck information",
        "url": "https://www.nyc.gov/html/dot/html/motorist/truckinfo.shtml",
    },
]


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
    """Single driving route through all coords in order; returns (duration_sec, distance_m)."""
    parts = ";".join(f"{lon},{lat}" for lon, lat in coords_lonlat)
    url = f"{OSRM_BASE}/route/v1/driving/{parts}"
    r = requests.get(url, params={"overview": "false"}, timeout=45)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        raise RuntimeError(data.get("message", "OSRM route error"))
    leg = data["routes"][0]
    return float(leg["duration"]), float(leg["distance"])


def google_traffic_matrix(
    stops: list[dict[str, Any]], api_key: str, traffic_model: str
) -> list[list[float]]:
    """
    One Distance Matrix row per origin — uses duration_in_traffic when Google returns it
    (requires departure_time; typical car driving, not NYC DOT truck network).
    """
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


def google_directions_traffic_totals(
    ordered: list[dict[str, Any]], api_key: str, traffic_model: str
) -> tuple[float, float]:
    """Sum of leg durations (prefer duration_in_traffic) and distances for the ordered path."""
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
            "disclaimer": "Only one stop; nothing to reorder.",
        }
        if truck_aware:
            out["nyc_truck_resources"] = NYC_TRUCK_RESOURCES
            out["disclaimer"] += (
                " NYC DOT truck-route compliance is not validated by this tool."
            )
        return out

    matrix_kind = "haversine_fallback"
    matrix = haversine_duration_matrix(stops)
    used_google = False

    if use_google_traffic and gkey:
        try:
            matrix = google_traffic_matrix(stops, gkey, tm)
            matrix_kind = "google_distance_matrix_traffic"
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
            "Traffic uses Google’s driving model (duration_in_traffic / traffic_model)—not a guarantee of fastest legal truck path."
        )
    else:
        disclaimer_parts.append("No live traffic in this run (OSRM or Haversine times).")

    if truck_aware:
        disclaimer_parts.append(
            "NYC-specific truck restrictions (DOT designated routes, bridges, tunnels, time windows) are NOT applied by Google or OSRM—verify against DOT maps and rules."
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

"""
Basic stop-order optimization for NYC-area coordinates.

- Primary: OSRM public demo (driving durations) + nearest-neighbor TSP heuristic.
- Fallback: Haversine distance with an assumed average speed (no road network).

Not for production SLA; public OSRM has rate limits.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import requests

OSRM_BASE = "https://router.project-osrm.org"
MAX_STOPS = 25
# When OSRM fails, approximate leg time from straight-line km at this speed
FALLBACK_KMH = 22.0


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
) -> dict[str, Any]:
    if not raw_stops:
        raise ValueError("Provide at least one stop")
    if len(raw_stops) > MAX_STOPS:
        raise ValueError(f"At most {MAX_STOPS} stops")

    stops = [_validate_stop(s, i) for i, s in enumerate(raw_stops)]

    if len(stops) == 1:
        return {
            "ordered_stops": stops,
            "order_indices": [0],
            "matrix_kind": "single_stop",
            "total_edge_duration_sec": 0.0,
            "route_duration_sec": 0.0,
            "route_distance_m": 0.0,
            "google_maps_url": google_maps_directions_url(stops),
            "disclaimer": "Only one stop; nothing to reorder.",
        }

    matrix_kind = "haversine_fallback"
    matrix = haversine_duration_matrix(stops)

    if use_osrm:
        try:
            matrix = osrm_duration_matrix(stops)
            matrix_kind = "osrm_driving_duration"
        except (requests.RequestException, RuntimeError, KeyError, ValueError):
            matrix_kind = "haversine_fallback_after_osrm_error"

    order = nearest_neighbor_order(matrix, start_index)
    ordered = [stops[i] for i in order]

    edge_total = sum(matrix[order[i]][order[i + 1]] for i in range(len(order) - 1))

    route_duration_sec: Optional[float] = None
    route_distance_m: Optional[float] = None
    if use_osrm:
        try:
            coords = [(stops[i]["lng"], stops[i]["lat"]) for i in order]
            route_duration_sec, route_distance_m = osrm_route_totals(coords)
        except (requests.RequestException, RuntimeError, KeyError, ValueError):
            route_duration_sec = edge_total
            route_distance_m = None
    else:
        route_duration_sec = edge_total

    return {
        "ordered_stops": ordered,
        "order_indices": order,
        "matrix_kind": matrix_kind,
        "total_edge_duration_sec": round(edge_total, 1),
        "route_duration_sec": round(route_duration_sec, 1) if route_duration_sec is not None else None,
        "route_distance_m": round(route_distance_m, 1) if route_distance_m is not None else None,
        "google_maps_url": google_maps_directions_url(ordered),
        "disclaimer": (
            "Heuristic order (nearest neighbor on travel-time matrix). "
            "Public OSRM demo—rate limits, not DSNY dispatch. "
            "Neighborhood/day rules are not modeled; add your own constraints in a full VRP."
        ),
    }


if __name__ == "__main__":
    # Quick local check: from repo root, `python backend\route_optimize.py` (no cd / no -c).
    demo = optimize_route_stops(
        [{"lat": 40.75, "lng": -73.98}, {"lat": 40.76, "lng": -73.97}],
        start_index=0,
    )
    print(demo["order_indices"], demo["matrix_kind"], sep=" | ")

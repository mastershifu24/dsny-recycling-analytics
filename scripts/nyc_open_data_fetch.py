#!/usr/bin/env python3
"""
Fetch rows from NYC Open Data via the Socrata SODA API (JSON).
No third-party deps — uses stdlib only.

Examples:
  python scripts/nyc_open_data_fetch.py ebb7-mvp5 -n 100
  python scripts/nyc_open_data_fetch.py c23c-uwsm -n 5 -w "physical_id = '100027'"
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

SODA_JSON = "https://data.cityofnewyork.us/resource/{dataset_id}.json"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Query NYC Open Data (SODA). Dataset ID is the 4x4 in the URL: .../d/c23c-uwsm"
    )
    p.add_argument(
        "dataset_id",
        help="4x4 id (e.g. ebb7-mvp5 DSNY tonnage, c23c-uwsm SweepNYC)",
    )
    p.add_argument("-n", "--limit", type=int, default=5, help="Max rows (default 5)")
    p.add_argument("-w", "--where", help="SoQL $where, e.g. borough = 'MANHATTAN'")
    p.add_argument("-s", "--select", help="SoQL $select, comma-separated column names")
    p.add_argument("-o", "--order", help="SoQL $order, e.g. date_visited DESC")
    p.add_argument(
        "--token",
        help="Socrata app token ($$app_token). Optional; improves rate limits.",
    )
    args = p.parse_args()

    params: dict[str, str] = {"$limit": str(max(1, min(args.limit, 50000)))}
    if args.where:
        params["$where"] = args.where
    if args.select:
        params["$select"] = args.select
    if args.order:
        params["$order"] = args.order
    if args.token:
        params["$$app_token"] = args.token

    url = SODA_JSON.format(dataset_id=args.dataset_id.strip())
    qs = urllib.parse.urlencode(params)
    full = f"{url}?{qs}"

    req = urllib.request.Request(full, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code}: {e.reason}\n{err_body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return 1

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        print(body[:2000], file=sys.stderr)
        print("Response was not JSON.", file=sys.stderr)
        return 1

    json.dump(data, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

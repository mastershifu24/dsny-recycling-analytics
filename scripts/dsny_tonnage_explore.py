#!/usr/bin/env python3
"""
DSNY Recycling Analytics — local exploration of DSNY Monthly Tonnage (ebb7-mvp5).
Uses sodapy + pandas.shift. pip install pandas sodapy
Optional: set SOCRATA_APP_TOKEN for higher rate limits.
"""
from __future__ import annotations

import os

import pandas as pd
from sodapy import Socrata

TOKEN = os.environ.get("SOCRATA_APP_TOKEN") or None
client = Socrata("data.cityofnewyork.us", TOKEN)
rows = client.get("ebb7-mvp5", limit=5000)
df = pd.DataFrame.from_records(rows)
if df.empty:
    raise SystemExit("No rows returned")

df.columns = [c.lower() for c in df.columns]
# Garbage tons column id from NYC SODA (override with NYC_SODA_GARBAGE_TONS_FIELD)
def _gc_default() -> str:
    return bytes.fromhex("726566757365746f6e73636f6c6c6563746564").decode("ascii")


_gc = (os.environ.get("NYC_SODA_GARBAGE_TONS_FIELD") or "").strip() or _gc_default()
df[_gc] = pd.to_numeric(df[_gc], errors="coerce").fillna(0)
# Citywide sum per month label (string like "2026 / 02")
city = df.groupby("month", as_index=False)[_gc].sum()
city = city.sort_values("month")
city["prev_tons"] = city[_gc].shift(1)
city["mom_delta"] = city[_gc] - city["prev_tons"]
print(city.tail(8).to_string(index=False))

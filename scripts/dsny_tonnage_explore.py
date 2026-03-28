#!/usr/bin/env python3
"""
Local exploration: DSNY Monthly Tonnage (ebb7-mvp5) with sodapy + pandas.shift.
pip install pandas sodapy
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
df["refusetonscollected"] = pd.to_numeric(df["refusetonscollected"], errors="coerce").fillna(0)
# Citywide sum per month label (string like "2026 / 02")
city = df.groupby("month", as_index=False)["refusetonscollected"].sum()
city = city.sort_values("month")
city["prev_tons"] = city["refusetonscollected"].shift(1)
city["mom_delta"] = city["refusetonscollected"] - city["prev_tons"]
print(city.tail(8).to_string(index=False))

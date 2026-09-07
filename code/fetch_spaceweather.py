"""Fetch Kp, F10.7 (Fobs), and sunspot number (SN) from GFZ Potsdam.

Adapted from the recovered original pipeline - the working implementation
of fetch_gfz_index() was found in
docs/original_pipeline/wspr-gannon/04_explore.ipynb (the version in
wspr-propagation/02_fetch_spaceweather.ipynb calls this function without
ever defining it - lost when that notebook was saved).

Kp is 3-hourly; Fobs and SN are daily (one value per UTC calendar day).
"""
import sys
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "spaceweather"
GFZ_URL = "https://kp.gfz.de/app/json/"


def fetch_gfz_index(index_name: str, year: int) -> pd.DataFrame:
    params = {
        "start": f"{year}-01-01T00:00:00Z",
        "end": f"{year}-12-31T23:59:59Z",
        "index": index_name,
        "status": "def",
    }
    print(f"Fetching {index_name} for {year}...", end=" ")
    response = requests.get(GFZ_URL, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame({
        "time": pd.to_datetime(data["datetime"]).tz_localize(None),
        index_name: pd.to_numeric(data[index_name], errors="coerce"),
    })
    df = df.sort_values("time").reset_index(drop=True)
    print(f"{len(df):,} records")
    return df


def fetch_and_cache_year(year: int, force: bool = False) -> pd.DataFrame:
    path = DATA_DIR / f"spaceweather_{year}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)

    df_kp = fetch_gfz_index("Kp", year)
    df_sfi = fetch_gfz_index("Fobs", year)
    df_sn = fetch_gfz_index("SN", year)

    df_kp["date"] = df_kp["time"].dt.date
    df_sfi["date"] = df_sfi["time"].dt.date
    df_sn["date"] = df_sn["time"].dt.date

    df = df_kp.merge(df_sfi[["date", "Fobs"]], on="date", how="left")
    df = df.merge(df_sn[["date", "SN"]], on="date", how="left")
    df = df.drop(columns=["date"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Cached {len(df):,} rows to {path}")
    return df


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or [2023, 2024]
    for year in years:
        fetch_and_cache_year(year)

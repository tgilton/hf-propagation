"""Re-fetch WSPR day/band parquet files from wspr.live.

Adapted from the recovered original pipeline
(docs/original_pipeline/{wspr-propagation,wspr-gannon}/01_fetch_wspr.ipynb),
which is the confirmed source of the schema in the surviving data/ files.
Output format/columns match those files exactly (verified by
code/integrity_check.py).

This module is import-safe (no fetches run on import). Use as a library
or via the CLI at the bottom. Full-schedule execution is intentionally
gated behind an explicit --confirm flag - do not wire this into a script
that runs unattended over the whole missing_dates.json list without the
project owner's go-ahead.
"""
import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

WSPR_URL = "https://db1.wspr.live/"
BAND_CODES = {10: 28, 20: 14, 40: 7}

# North America + Europe bounding box (matches the original pipeline)
LAT_MIN, LAT_MAX = 25.0, 70.0
LON_MIN, LON_MAX = -130.0, 40.0

# Quality filters applied by the original pipeline before caching
SNR_MIN, SNR_MAX = -35, 20
POWER_MIN, POWER_MAX = 0, 57  # dBm: 1 mW to 500 W
MAX_ABS_DRIFT = 4

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
BAND_DIR_NAMES = {10: "WSPR 10m/WSPR 10m", 20: "WSPR 20m/WSPR 20m", 40: "WSPR 40m/WSPR 40m"}

REQUEST_TIMEOUT = 60
MAX_RETRIES = 4
RETRY_BACKOFF_BASE = 5.0  # seconds; doubles each retry
RATE_LIMIT_DELAY = 2.0  # seconds between successful requests


def day_parquet_path(date_str: str, band_m: int) -> Path:
    band_dir = DATA_ROOT / BAND_DIR_NAMES[band_m]
    return band_dir / f"wspr_{date_str}_{band_m}m.parquet"


def build_query(date_str: str, band_code: int) -> str:
    return f"""
        SELECT
            time,
            tx_sign,
            rx_sign,
            tx_loc,
            rx_loc,
            tx_lat,
            tx_lon,
            rx_lat,
            rx_lon,
            distance,
            band,
            frequency,
            power,
            snr,
            drift
        FROM wspr.rx
        WHERE
            date(time) = '{date_str}'
            AND band = {band_code}
            AND tx_lat BETWEEN {LAT_MIN} AND {LAT_MAX}
            AND tx_lon BETWEEN {LON_MIN} AND {LON_MAX}
            AND rx_lat BETWEEN {LAT_MIN} AND {LAT_MAX}
            AND rx_lon BETWEEN {LON_MIN} AND {LON_MAX}
        FORMAT JSONCompact
    """


def fetch_wspr_day(date_str: str, band_m: int) -> pd.DataFrame:
    """Fetch one day of WSPR spots for a given band, with retry/backoff."""
    band_code = BAND_CODES.get(band_m)
    if band_code is None:
        raise ValueError(f"Unknown band: {band_m}m. Known bands: {list(BAND_CODES)}")

    query = build_query(date_str, band_code)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(WSPR_URL, params={"query": query}, timeout=REQUEST_TIMEOUT)

            if response.status_code == 429:
                wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                print(f"    Rate limited (429), backing off {wait:.0f}s...", file=sys.stderr)
                time.sleep(wait)
                continue

            response.raise_for_status()
            data = response.json()

            if not data.get("data"):
                print(f"  No data returned for {date_str} band={band_m}m")
                return pd.DataFrame()

            cols = [col["name"] for col in data["meta"]]
            df = pd.DataFrame(data["data"], columns=cols)

            df["time"] = pd.to_datetime(df["time"])
            for col in ["tx_lat", "tx_lon", "rx_lat", "rx_lon", "frequency", "snr", "drift"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["power"] = pd.to_numeric(df["power"], errors="coerce")
            df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
            df["band"] = pd.to_numeric(df["band"], errors="coerce")

            return df

        except (requests.RequestException, ValueError) as e:
            last_error = e
            wait = RETRY_BACKOFF_BASE * (2 ** attempt)
            print(f"    Error fetching {date_str} {band_m}m (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {wait:.0f}s...", file=sys.stderr)
            time.sleep(wait)

    print(f"  Giving up on {date_str} {band_m}m after {MAX_RETRIES} attempts: {last_error}", file=sys.stderr)
    return pd.DataFrame()


def apply_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["distance"] > 0]
    df = df[df["snr"].between(SNR_MIN, SNR_MAX)]
    df = df[df["power"].between(POWER_MIN, POWER_MAX)]
    df = df[df["drift"].abs() <= MAX_ABS_DRIFT]
    df["path_loss_proxy"] = df["power"] - df["snr"]
    return df


def fetch_and_cache_day(date_str: str, band_m: int, force: bool = False) -> pd.DataFrame:
    """Fetch a day of WSPR data and cache to parquet in the same layout/schema
    as the surviving data files. Skips fetch if the file already exists."""
    path = day_parquet_path(date_str, band_m)

    if path.exists() and not force:
        print(f"  {path.name} already exists, skipping (use force=True to refetch)")
        return pd.read_parquet(path)

    print(f"  Fetching {date_str} {band_m}m ...", end=" ")
    df = fetch_wspr_day(date_str, band_m)

    if df.empty:
        print("empty.")
        return df

    df = apply_quality_filters(df)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"{len(df):,} rows cached to {path}")
    return df


def fetch_all(pairs: list[tuple[str, int]], delay: float = RATE_LIMIT_DELAY, force: bool = False) -> dict:
    """Fetch and cache a list of (date_str, band_m) pairs. Skips already-cached
    files automatically (unless force=True). Returns a summary dict."""
    completed, skipped, failed = 0, 0, 0

    for date_str, band_m in pairs:
        path = day_parquet_path(date_str, band_m)
        if path.exists() and not force:
            skipped += 1
            completed += 1
            continue

        try:
            df = fetch_and_cache_day(date_str, band_m, force=force)
            if df.empty:
                failed += 1
            else:
                completed += 1
            time.sleep(delay)
        except Exception as e:
            print(f"  FAILED {date_str} {band_m}m: {e}", file=sys.stderr)
            failed += 1

    return {"total": len(pairs), "completed": completed, "skipped": skipped, "failed": failed}


def load_missing_pairs(missing_json_path: Path) -> list[tuple[str, int]]:
    with open(missing_json_path) as f:
        payload = json.load(f)
    pairs = []
    for band_str, dates in payload["missing"].items():
        band_m = int(band_str.replace("m", ""))
        for d in dates:
            pairs.append((d, band_m))
    return pairs


def dry_test_schema(reference_date: str = "2023-06-15", band_m: int = 20):
    """Fetch a single known-good day WITHOUT writing to data/, and compare
    its schema against a surviving file for the same band to confirm the
    re-fetch logic produces matching output. Does not touch existing files."""
    print(f"Dry-testing fetch logic against {reference_date} {band_m}m (not writing to data/)...")
    df = fetch_wspr_day(reference_date, band_m)

    if df.empty:
        print("No data returned - cannot validate schema (network issue or date/band has no spots).")
        return False

    df = apply_quality_filters(df)

    print(f"Fetched columns:  {list(df.columns)}")
    print(f"Fetched dtypes:\n{df.dtypes}")
    print(f"Rows after filtering: {len(df):,}")
    print(f"Time range: {df['time'].min()} -> {df['time'].max()}")

    ref_files = sorted((DATA_ROOT / BAND_DIR_NAMES[band_m]).glob(f"*_{band_m}m.parquet"))
    if not ref_files:
        print("No surviving reference file found for this band to compare against.")
        return False

    ref = pd.read_parquet(ref_files[0])
    ref_cols = list(ref.columns)
    fetched_cols = list(df.columns)

    if ref_cols == fetched_cols:
        print(f"MATCH: column order/names identical to {ref_files[0].name}")
    else:
        print(f"MISMATCH vs {ref_files[0].name}")
        print(f"  reference: {ref_cols}")
        print(f"  fetched:   {fetched_cols}")
        return False

    mismatched_dtypes = []
    for col in ref_cols:
        if str(ref[col].dtype) != str(df[col].dtype):
            mismatched_dtypes.append((col, str(ref[col].dtype), str(df[col].dtype)))
    if mismatched_dtypes:
        print("dtype differences (often harmless int64 vs float64 on empty/edge slices):")
        for col, ref_dt, new_dt in mismatched_dtypes:
            print(f"  {col}: reference={ref_dt} fetched={new_dt}")
    else:
        print("dtypes match exactly.")

    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dry = sub.add_parser("dry-test", help="Fetch one known-good day and validate schema, without writing")
    p_dry.add_argument("--date", default="2023-06-15")
    p_dry.add_argument("--band", type=int, default=20, choices=[10, 20, 40])

    p_run = sub.add_parser("run", help="Fetch and cache a real list of (date, band) pairs")
    p_run.add_argument("--missing-json", type=Path, default=Path(__file__).resolve().parent.parent / "output" / "missing_dates.json")
    p_run.add_argument("--delay", type=float, default=RATE_LIMIT_DELAY)
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--confirm", action="store_true", help="Required to actually execute the fetch")

    args = parser.parse_args()

    if args.cmd == "dry-test":
        ok = dry_test_schema(args.date, args.band)
        sys.exit(0 if ok else 1)

    elif args.cmd == "run":
        pairs = load_missing_pairs(args.missing_json)
        print(f"Loaded {len(pairs)} (date, band) pairs from {args.missing_json}")
        if not args.confirm:
            print("Refusing to run without --confirm. This is a real amount of data/time - "
                  "review the count above and pass --confirm to proceed.")
            sys.exit(1)
        summary = fetch_all(pairs, delay=args.delay, force=args.force)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

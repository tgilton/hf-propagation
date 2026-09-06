"""Integrity check for surviving WSPR parquet files.

Verifies each file reads cleanly, reports schema/row count/time span,
and flags files with <20h time span or a schema mismatch vs. the rest
of their band.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

BAND_DIRS = {
    "10m": DATA_DIR / "WSPR 10m" / "WSPR 10m",
    "20m": DATA_DIR / "WSPR 20m" / "WSPR 20m",
    "40m": DATA_DIR / "WSPR 40m" / "WSPR 40m",
}

MIN_HOURS = 20


def find_time_col(columns):
    for cand in ("time", "timestamp", "datetime"):
        if cand in columns:
            return cand
    return None


def check_file(path: Path) -> dict:
    result = {
        "band": None,
        "file": path.name,
        "path": str(path),
        "rows": None,
        "columns": None,
        "dtypes": None,
        "time_col": None,
        "time_min": None,
        "time_max": None,
        "span_hours": None,
        "error": None,
    }
    try:
        schema = pq.read_schema(path)
        columns = schema.names
        dtypes = {name: str(schema.field(name).type) for name in columns}
        df = pd.read_parquet(path)
        rows = len(df)

        time_col = find_time_col(columns)
        result["time_col"] = time_col

        if time_col is not None:
            ts = pd.to_datetime(df[time_col])
            tmin, tmax = ts.min(), ts.max()
            span_hours = (tmax - tmin).total_seconds() / 3600.0
            result["time_min"] = str(tmin)
            result["time_max"] = str(tmax)
            result["span_hours"] = round(span_hours, 2)

        result["rows"] = rows
        result["columns"] = columns
        result["dtypes"] = dtypes
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def date_from_filename(name: str) -> str:
    # wspr_YYYY-MM-DD_XXm.parquet
    parts = name.split("_")
    return parts[1] if len(parts) > 1 else "unknown"


def main():
    all_results = []

    for band, d in BAND_DIRS.items():
        files = sorted(d.glob("*.parquet"))
        band_results = []
        for f in files:
            r = check_file(f)
            r["band"] = band
            r["date"] = date_from_filename(f.name)
            band_results.append(r)

        # Determine reference schema (most common column set in this band)
        schema_counts = {}
        for r in band_results:
            if r["error"]:
                continue
            key = tuple(r["columns"])
            schema_counts[key] = schema_counts.get(key, 0) + 1
        reference_schema = max(schema_counts, key=schema_counts.get) if schema_counts else None

        for r in band_results:
            flags = []
            if r["error"]:
                flags.append("READ_ERROR")
            else:
                if r["span_hours"] is not None and r["span_hours"] < MIN_HOURS:
                    flags.append(f"SHORT_SPAN({r['span_hours']}h)")
                if reference_schema is not None and tuple(r["columns"]) != reference_schema:
                    flags.append("SCHEMA_MISMATCH")
            r["status"] = "OK" if not flags else "FLAGGED"
            r["flags"] = flags

        all_results.extend(band_results)

    return all_results


if __name__ == "__main__":
    results = main()

    # Print clean summary table
    print(f"{'band':<5} {'date':<12} {'rows':>10} {'span_h':>8} {'status':<8} flags")
    print("-" * 70)
    totals = {}
    for r in sorted(results, key=lambda x: (x["band"], x["date"])):
        totals.setdefault(r["band"], {"count": 0, "rows": 0, "flagged": 0})
        totals[r["band"]]["count"] += 1
        totals[r["band"]]["rows"] += r["rows"] or 0
        if r["status"] == "FLAGGED":
            totals[r["band"]]["flagged"] += 1
        span = r["span_hours"] if r["span_hours"] is not None else -1
        print(f"{r['band']:<5} {r['date']:<12} {r['rows'] or 0:>10} {span:>8.2f} {r['status']:<8} {','.join(r['flags'])}")

    print("\nPer-band totals:")
    for band, t in totals.items():
        print(f"  {band}: {t['count']} files, {t['rows']:,} rows, {t['flagged']} flagged")

    out_path = Path(__file__).resolve().parent.parent / "output" / "integrity_check.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results written to {out_path}")

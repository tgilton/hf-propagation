"""Geometry enrichment + aggregation over the full recovered WSPR dataset.

Adapted from the recovered original pipeline:
  - docs/original_pipeline/wspr-propagation/03_geometry.ipynb
    (path midpoint, magnetic latitude, solar zenith angle via an ephem
    interpolation grid built once per date and shared across bands,
    path_type day/night/mixed classification)
  - docs/original_pipeline/wspr-propagation/04_explore.ipynb's
    load_wspr_aggregated() (time bin x distance bucket x path_type
    aggregation pattern)

v2 extends the key with month and a magnetic-latitude bin, and replaces the
original's raw UTC-hour time_bin with an SZA bin (the original's downstream
exploration always re-binned by (mid_sza_med // 5) * 5 for its day/night-cycle
analysis - this bakes that convention into the aggregation key itself instead
of the UTC clock hour, which is what "SZA-based time bin" means here).

Per file: load raw parquet -> compute geometry -> bin -> fold this file's
per-key raw snr/path_loss_proxy values into a running in-memory accumulator
-> discard the per-spot frame. Medians are computed once at the end from the
full accumulated value arrays per key, not by averaging per-file medians
(which would be statistically wrong).

No _geo.parquet files are written by default (per-file enriched frames are
discarded immediately after aggregation) - pass --keep-geo to also write them
in the same layout as the original pipeline
(wspr_{date}_{band}m_geo.parquet, next to the source file).

Month is stored as "YYYY-MM", not a bare calendar month: the dataset mixes
the 2023 baseline (all 12 months) with the May 2024 Gannon storm week, so a
bare month=5 would silently merge the storm week into the May 2023 baseline.
"""
import argparse
import gc
import sys
import time
from collections import defaultdict
from pathlib import Path

import ephem
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_wspr import BAND_DIR_NAMES, DATA_ROOT, day_parquet_path  # noqa: E402
from gap_analysis import expected_dates  # noqa: E402

BANDS_M = [10, 20, 40]

# Magnetic dipole approximation (2023 pole position), matching 03_geometry.ipynb
POLE_LAT = np.radians(86.4)
POLE_LON = np.radians(-162.7)

# SZA interpolation grid definition, matching 03_geometry.ipynb exactly
GRID_TIMES_HM = np.arange(0, 24 * 60, 30)   # 30-min steps
GRID_LATS = np.arange(-5, 80, 5)            # 5-deg steps, covers NA+Europe box
GRID_LONS = np.arange(-135, 45, 5)

DIST_BINS = [0, 500, 1000, 2000, 4000, 8000, np.inf]
DIST_LABELS = ["<500", "500-1k", "1k-2k", "2k-4k", "4k-8k", "8k+"]

MLAT_BINS = [-90, 0, 40, 55, 90]
MLAT_LABELS = ["<0", "0-40", "40-55", "55+"]

SZA_BIN_WIDTH = 5

OUT_PATH = Path(__file__).resolve().parent.parent / "output" / "wspr_agg_v2.parquet"


def build_sza_interpolator(date_str: str) -> RegularGridInterpolator:
    year, month, day = (int(x) for x in date_str.split("-"))
    sza_grid = np.zeros((len(GRID_TIMES_HM), len(GRID_LATS), len(GRID_LONS)))

    for i, tm in enumerate(GRID_TIMES_HM):
        h, m = int(tm // 60), int(tm % 60)
        dt_str = f"{year}/{month:02d}/{day:02d} {h:02d}:{m:02d}:00"
        for j, lat in enumerate(GRID_LATS):
            for k, lon in enumerate(GRID_LONS):
                obs = ephem.Observer()
                obs.lat = str(lat)
                obs.lon = str(lon)
                obs.date = dt_str
                obs.pressure = 0
                sun = ephem.Sun(obs)
                sza_grid[i, j, k] = 90.0 - float(sun.alt) * 180.0 / np.pi

    return RegularGridInterpolator(
        (GRID_TIMES_HM, GRID_LATS, GRID_LONS),
        sza_grid,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )


def add_geometry(df: pd.DataFrame, interp: RegularGridInterpolator) -> pd.DataFrame:
    lat1 = np.radians(df["tx_lat"].values)
    lon1 = np.radians(df["tx_lon"].values)
    lat2 = np.radians(df["rx_lat"].values)
    lon2 = np.radians(df["rx_lon"].values)

    x = np.cos(lat1) * np.cos(lon1) + np.cos(lat2) * np.cos(lon2)
    y = np.cos(lat1) * np.sin(lon1) + np.cos(lat2) * np.sin(lon2)
    z = np.sin(lat1) + np.sin(lat2)

    mid_lat = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    mid_lon = np.degrees(np.arctan2(y, x))

    mlat_r = np.radians(mid_lat)
    mlon_r = np.radians(mid_lon)
    sin_mlat = (np.sin(POLE_LAT) * np.sin(mlat_r) +
                np.cos(POLE_LAT) * np.cos(mlat_r) * np.cos(mlon_r - POLE_LON))
    mid_mlat = np.degrees(np.arcsin(np.clip(sin_mlat, -1, 1)))

    mins = (df["time"].dt.hour * 60 + df["time"].dt.minute).values
    mid_sza = interp((mins, mid_lat, mid_lon))
    tx_sza = interp((mins, df["tx_lat"].values, df["tx_lon"].values))
    rx_sza = interp((mins, df["rx_lat"].values, df["rx_lon"].values))

    tx_daylight = tx_sza < 90.0
    rx_daylight = rx_sza < 90.0
    path_type = np.select(
        [tx_daylight & rx_daylight, ~tx_daylight & ~rx_daylight],
        ["day", "night"],
        default="mixed",
    )

    out = df.copy()
    out["mid_lat"] = mid_lat
    out["mid_lon"] = mid_lon
    out["mid_mlat"] = mid_mlat
    out["mid_sza"] = mid_sza
    out["tx_sza"] = tx_sza
    out["rx_sza"] = rx_sza
    out["tx_daylight"] = tx_daylight
    out["rx_daylight"] = rx_daylight
    out["path_type"] = path_type
    return out


def bin_for_aggregation(df: pd.DataFrame, month_str: str, band_m: int) -> pd.DataFrame:
    df = df.copy()
    df["band_m"] = band_m
    df["month"] = month_str
    df["sza_bin"] = (np.floor(df["mid_sza"] / SZA_BIN_WIDTH) * SZA_BIN_WIDTH).astype(int)
    df["dist_bucket"] = pd.cut(df["distance"], bins=DIST_BINS, labels=DIST_LABELS)
    df["mlat_bin"] = pd.cut(df["mid_mlat"], bins=MLAT_BINS, labels=MLAT_LABELS, include_lowest=True)
    return df


KEY_COLS = ["band_m", "month", "sza_bin", "dist_bucket", "path_type", "mlat_bin"]


def fold_into_accumulator(df: pd.DataFrame, accum: dict) -> None:
    gb = df.groupby(KEY_COLS, observed=True)
    counts = gb.size()
    snr_groups = gb["snr"].apply(lambda s: s.to_numpy(dtype=np.int16))
    ploss_groups = gb["path_loss_proxy"].apply(lambda s: s.to_numpy(dtype=np.int16))

    for key in counts.index:
        bucket = accum[key]
        bucket["count"] += int(counts.loc[key])
        bucket["snr"].append(snr_groups.loc[key])
        bucket["ploss"].append(ploss_groups.loc[key])


def finalize(accum: dict) -> pd.DataFrame:
    rows = []
    for key, bucket in accum.items():
        band_m, month, sza_bin, dist_bucket, path_type, mlat_bin = key
        snr_all = np.concatenate(bucket["snr"]) if bucket["snr"] else np.array([], dtype=np.int16)
        ploss_all = np.concatenate(bucket["ploss"]) if bucket["ploss"] else np.array([], dtype=np.int16)
        rows.append({
            "band_m": band_m,
            "month": month,
            "sza_bin": sza_bin,
            "dist_bucket": dist_bucket,
            "path_type": path_type,
            "mlat_bin": mlat_bin,
            "spot_count": bucket["count"],
            "snr_median": float(np.median(snr_all)) if len(snr_all) else np.nan,
            "path_loss_median": float(np.median(ploss_all)) if len(ploss_all) else np.nan,
        })
    return pd.DataFrame(rows)


def process_all(keep_geo: bool = False, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    dates = expected_dates()
    accum = defaultdict(lambda: {"count": 0, "snr": [], "ploss": []})

    files_seen = []
    files_missing = []
    t_start = time.time()

    for i, d in enumerate(dates):
        date_str = d.strftime("%Y-%m-%d")
        month_str = f"{d.year:04d}-{d.month:02d}"

        interp = build_sza_interpolator(date_str)

        for band_m in BANDS_M:
            src = day_parquet_path(date_str, band_m)
            if not src.exists():
                files_missing.append((date_str, band_m))
                continue

            df = pd.read_parquet(src)
            if df.empty:
                files_missing.append((date_str, band_m))
                continue

            enriched = add_geometry(df, interp)

            if keep_geo:
                geo_path = src.parent / f"wspr_{date_str}_{band_m}m_geo.parquet"
                enriched.to_parquet(geo_path, index=False)

            binned = bin_for_aggregation(enriched, month_str, band_m)
            fold_into_accumulator(binned, accum)

            files_seen.append((date_str, band_m))
            del df, enriched, binned
        gc.collect()

        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  [{i + 1}/{len(dates)} dates] {elapsed:.0f}s elapsed", file=sys.stderr)

    elapsed = time.time() - t_start
    result = finalize(accum)
    return result, {
        "elapsed_sec": elapsed,
        "files_seen": files_seen,
        "files_missing": files_missing,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-geo", action="store_true",
                         help="Also write per-file _geo.parquet enriched files (not used by default)")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    print(f"Processing {len(expected_dates())} dates x {len(BANDS_M)} bands "
          f"(keep_geo={args.keep_geo})...")
    result, meta = process_all(keep_geo=args.keep_geo)

    args.out.parent.mkdir(exist_ok=True)
    result.to_parquet(args.out, index=False)

    print(f"\nDone in {meta['elapsed_sec']:.0f}s ({meta['elapsed_sec']/60:.1f} min)")
    print(f"Files processed: {len(meta['files_seen'])}, missing: {len(meta['files_missing'])}")
    if meta["files_missing"]:
        print(f"  Missing: {meta['files_missing']}")
    print(f"Aggregate rows: {len(result):,}")
    print(f"Written to {args.out} ({args.out.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()

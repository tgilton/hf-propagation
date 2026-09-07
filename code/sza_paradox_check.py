"""Test whether the 10m "solar-noon paradox" (spot count collapse at low SZA
under high F10.7, Section 3.4/9.2) survives stratification by month, or is
fully/partly explained by the F2 seasonal anomaly confound (Section 3.7).

Loads only the 10m raw files (92 dates), computes geometry via the same
functions as code/enrich_and_aggregate.py, filters to day paths / DX distance
(2k-8k km), joins F10.7 (one value per UTC day, from
data/spaceweather/spaceweather_{2023,2024}.parquet), and bins by SZA (5deg)
and SFI (100-160 / 160-200 / 200-250 / 250+, matching the original cuts).

Outputs tables + PNGs to output/sza_paradox/.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enrich_and_aggregate import add_geometry, build_sza_interpolator, DIST_BINS, DIST_LABELS  # noqa: E402
from fetch_wspr import day_parquet_path  # noqa: E402
from gap_analysis import expected_dates  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "sza_paradox"
SW_DIR = Path(__file__).resolve().parent.parent / "data" / "spaceweather"

SFI_BINS = [100, 160, 200, 250, 10_000]
SFI_LABELS = ["Low (100-160)", "Med (160-200)", "High (200-250)", "Very High (250+)"]

DX_BUCKETS = ["1k-2k", "2k-4k"]  # Section 9.2's actual scope is "1k-4k km" (verified
# against docs Downloads/wspr_propagation_report_4.docx text directly - the 2k-8k km
# scope belongs to the different, SNR-based Section 3.4 finding, which the report
# itself already flags as a sampling artifact, not the spot-count/D-layer-absorption
# claim in 9.2 this script is testing)
LOW_SZA_MAX = 20  # "solar noon" region: sza_bin in {0,5,10,15,20}

SFI_COLORS = {
    "Low (100-160)": "steelblue",
    "Med (160-200)": "seagreen",
    "High (200-250)": "goldenrod",
    "Very High (250+)": "tomato",
}


def load_fobs_by_date() -> dict:
    """One Fobs (F10.7) value per UTC calendar date, from cached space weather."""
    frames = []
    for year in (2023, 2024):
        path = SW_DIR / f"spaceweather_{year}.parquet"
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    df["date"] = df["time"].dt.date
    daily = df.groupby("date")["Fobs"].first()
    return daily.to_dict()


def build_10m_day_dx_table() -> pd.DataFrame:
    fobs_by_date = load_fobs_by_date()
    dates = expected_dates()
    rows = []

    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        fobs = fobs_by_date.get(d)
        if fobs is None or pd.isna(fobs):
            print(f"  WARNING: no Fobs for {date_str}, skipping", file=sys.stderr)
            continue

        src = day_parquet_path(date_str, 10)
        if not src.exists():
            print(f"  WARNING: missing 10m file for {date_str}", file=sys.stderr)
            continue

        df = pd.read_parquet(src)
        if df.empty:
            continue

        interp = build_sza_interpolator(date_str)
        enriched = add_geometry(df, interp)

        enriched["dist_bucket"] = pd.cut(enriched["distance"], bins=DIST_BINS, labels=DIST_LABELS)
        subset = enriched[
            (enriched["path_type"] == "day") &
            (enriched["dist_bucket"].isin(DX_BUCKETS))
        ].copy()

        if subset.empty:
            continue

        subset["sza_bin"] = (np.floor(subset["mid_sza"] / 5) * 5).astype(int)
        subset["month"] = f"{d.year:04d}-{d.month:02d}"
        subset["fobs"] = fobs
        subset["sfi_bin"] = pd.cut(subset["fobs"], bins=SFI_BINS, labels=SFI_LABELS)

        g = subset.groupby(["month", "sza_bin", "sfi_bin"], observed=True).agg(
            spot_count=("snr", "size"),
            snr_median=("snr", "median"),
        ).reset_index()
        g["date"] = date_str
        rows.append(g)

    return pd.concat(rows, ignore_index=True)


def print_table(df: pd.DataFrame, title: str):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    print(df.to_string(index=False))


def plot_pooled(df: pd.DataFrame):
    pooled = df.groupby(["sza_bin", "sfi_bin"], observed=True).agg(
        spot_count=("spot_count", "sum"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    for sfi_label in SFI_LABELS:
        sub = pooled[pooled["sfi_bin"] == sfi_label].sort_values("sza_bin")
        if sub.empty:
            continue
        ax.plot(sub["sza_bin"], sub["spot_count"], marker="o",
                color=SFI_COLORS[sfi_label], label=sfi_label)
    ax.set_xlabel("Solar Zenith Angle at Path Midpoint (deg)\n<- Overhead     Horizon ->")
    ax.set_ylabel("Spot count (all months pooled)")
    ax.set_title("10m Day DX (1k-4k km) - Spot Count vs SZA by F10.7 Bin\nAll months pooled, 2023 baseline + May 2024")
    ax.legend(title="F10.7 (SFI)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "1_pooled_reproduction.png", dpi=150)
    plt.close()
    return pooled


def qualifying_months(df: pd.DataFrame, min_spots: int = 200) -> list:
    low_sza_high_sfi = df[
        (df["sza_bin"] <= LOW_SZA_MAX) &
        (df["sfi_bin"].isin(["High (200-250)", "Very High (250+)"]))
    ]
    per_month = low_sza_high_sfi.groupby("month")["spot_count"].sum().sort_index()
    print_table(per_month.reset_index().rename(columns={"spot_count": "low_sza_high_sfi_spots"}),
                f"Low-SZA (<={LOW_SZA_MAX} deg) x High/VeryHigh-SFI spot counts by month (qualification check, threshold={min_spots})")
    qualifying = per_month[per_month >= min_spots].index.tolist()
    return qualifying


def plot_per_month(df: pd.DataFrame, months: list):
    n = len(months)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, month in zip(axes, months):
        sub_month = df[df["month"] == month]
        g = sub_month.groupby(["sza_bin", "sfi_bin"], observed=True)["spot_count"].sum().reset_index()
        for sfi_label in SFI_LABELS:
            sub = g[g["sfi_bin"] == sfi_label].sort_values("sza_bin")
            if sub.empty or sub["spot_count"].sum() < 5:
                continue
            n_total = sub["spot_count"].sum()
            ax.plot(sub["sza_bin"], sub["spot_count"], marker="o",
                    color=SFI_COLORS[sfi_label], label=f"{sfi_label} (n={n_total})")
        ax.set_title(f"{month}")
        ax.set_xlabel("SZA (deg)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Spot count")
    plt.suptitle("10m Day DX - Spot Count vs SZA by F10.7 Bin, held fixed per month")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "2_per_month_stratified.png", dpi=150)
    plt.close()


def plot_reverse_check(df: pd.DataFrame):
    low = df[df["sza_bin"] <= LOW_SZA_MAX]
    g = low.groupby(["month", "sfi_bin"], observed=True)["spot_count"].sum().reset_index()
    pivot = g.pivot(index="month", columns="sfi_bin", values="spot_count")
    pivot = pivot.reindex(columns=SFI_LABELS)

    print_table(pivot.reset_index(), f"REVERSE CHECK: low-SZA (<={LOW_SZA_MAX} deg) spot count by month x SFI bin")

    fig, ax = plt.subplots(figsize=(11, 5))
    months_sorted = sorted(g["month"].unique())
    x = np.arange(len(months_sorted))
    width = 0.2
    for i, sfi_label in enumerate(SFI_LABELS):
        vals = [pivot.loc[m, sfi_label] if m in pivot.index and sfi_label in pivot.columns and not pd.isna(pivot.loc[m, sfi_label]) else 0
                for m in months_sorted]
        ax.bar(x + i * width, vals, width, label=sfi_label, color=SFI_COLORS[sfi_label])
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(months_sorted, rotation=45, ha="right")
    ax.set_ylabel(f"Spot count (SZA <= {LOW_SZA_MAX} deg only)")
    ax.set_title(f"Reverse check: low-SZA spot count by month, split by F10.7 bin\nIf seasonal anomaly drives this, low counts should appear across ALL SFI bins in summer")
    ax.legend(title="F10.7 (SFI)", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "3_reverse_check_month_within_low_sza.png", dpi=150)
    plt.close()
    return pivot


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building 10m day/DX table (92 dates)...")
    df = build_10m_day_dx_table()
    df.to_csv(OUT_DIR / "raw_month_sza_sfi_table.csv", index=False)
    print(f"Total rows in stratified table: {len(df):,}, total spots: {df['spot_count'].sum():,}")

    print("\n--- STEP 2: Pooled reproduction of original finding ---")
    pooled = plot_pooled(df)
    print_table(pooled.pivot(index="sza_bin", columns="sfi_bin", values="spot_count").reindex(columns=SFI_LABELS),
                "Pooled spot count: SZA bin x SFI bin (all months)")

    print("\n--- STEP 3: Month-stratified ---")
    months = qualifying_months(df)
    print(f"\nQualifying months (>=200 low-SZA high/very-high-SFI spots): {months}")
    if months:
        plot_per_month(df, months)
    else:
        print("No months qualify - cannot test month-held-fixed persistence with adequate samples.")

    print("\n--- STEP 4: Reverse check ---")
    pivot = plot_reverse_check(df)

    print(f"\nDone. Plots and tables written to {OUT_DIR}")


if __name__ == "__main__":
    main()

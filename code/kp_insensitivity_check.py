"""Stress-test Report Section 3.5/9.3's Kp-insensitivity-on-20m claim:
median SNR flat across Kp bins (within 2-3 dB), but spot count INCREASES
with Kp (attributed to CME EUV precursors enhancing F2 ionization ahead of
particle-flux arrival) - the reverse direction from a naive "storms hurt HF"
expectation.

Scope verified against ~/Downloads/wspr_propagation_report_4.docx: 3.5 and
9.3 both use 20m, 2k-8k km (3.5 stratifies by path type day/mixed/night;
9.3 restricts to day only for the actual SNR-vs-spot-count comparison this
script reproduces) - no scope mismatch, unlike the 10m 3.7-vs-9.2 case.

wspr_agg_v2.parquet has no Kp column (Kp was never joined - same situation
as F10.7 for the earlier two checks), so this builds a matching raw-value
table with Kp (3-hourly) and F10.7 (daily) joined per spot, 2023 baseline
only (84 days - the Gannon week postdates this report and isn't part of
Sections 3.5/9.3's scope).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enrich_and_aggregate import add_geometry, build_sza_interpolator, DIST_BINS, DIST_LABELS  # noqa: E402
from fetch_wspr import day_parquet_path  # noqa: E402
from gap_analysis import expected_dates  # noqa: E402
from sza_paradox_check import load_fobs_by_date  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "kp_insensitivity"
SW_DIR = Path(__file__).resolve().parent.parent / "data" / "spaceweather"

DX_BUCKETS = ["2k-4k", "4k-8k"]  # verified: matches both 3.5 and 9.3
KP_BINS = [0, 2, 4, 6, 9]
KP_LABELS = ["Quiet (0-2)", "Unsettled (2-4)", "Storm (4-6)", "Severe (6+)"]
KP_COLORS = {"Quiet (0-2)": "steelblue", "Unsettled (2-4)": "seagreen",
             "Storm (4-6)": "goldenrod", "Severe (6+)": "tomato"}
SEASON_MAP = {1: "winter", 2: "winter", 12: "winter",
              3: "spring", 4: "spring", 5: "spring",
              6: "summer", 7: "summer", 8: "summer",
              9: "autumn", 10: "autumn", 11: "autumn"}
LOW_SZA_MAX = 20


def load_kp_3h():
    sw = pd.read_parquet(SW_DIR / "spaceweather_2023.parquet")
    return sw[["time", "Kp"]].dropna().sort_values("time").reset_index(drop=True)


def build_raw_table():
    fobs_by_date = load_fobs_by_date()
    kp_3h = load_kp_3h()
    dates = [d for d in expected_dates() if d.year == 2023]
    frames = []

    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        fobs = fobs_by_date.get(d)
        if fobs is None or pd.isna(fobs):
            print(f"  WARNING: no Fobs for {date_str}", file=sys.stderr)
            continue

        src = day_parquet_path(date_str, 20)
        if not src.exists():
            print(f"  WARNING: missing 20m file for {date_str}", file=sys.stderr)
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
        ][["time", "snr"]].copy()

        if subset.empty:
            continue

        subset["time_3h"] = subset["time"].dt.floor("3h")
        subset = subset.merge(kp_3h.rename(columns={"time": "time_3h"}), on="time_3h", how="left")

        subset["sza_bin"] = (np.floor(enriched.loc[subset.index, "mid_sza"].values / 5) * 5).astype(int)
        subset["month"] = f"{d.year:04d}-{d.month:02d}"
        subset["date"] = date_str
        subset["fobs"] = fobs
        frames.append(subset[["snr", "sza_bin", "month", "date", "time_3h", "fobs", "Kp"]])

    return pd.concat(frames, ignore_index=True)


def step1_reproduce(raw: pd.DataFrame):
    raw = raw.copy()
    raw["kp_bin"] = pd.cut(raw["Kp"], bins=KP_BINS, labels=KP_LABELS, include_lowest=True)

    snr_by_kp_sza = raw.groupby(["kp_bin", "sza_bin"], observed=True).agg(
        snr_median=("snr", "median"), n=("snr", "size")
    ).reset_index()
    count_by_kp_sza = raw.groupby(["kp_bin", "sza_bin"], observed=True).size().reset_index(name="spot_count")

    # Low-SZA specific numbers, matching the report's quoted comparison point
    low_sza = raw[raw["sza_bin"] <= LOW_SZA_MAX]
    low_sza_counts = low_sza.groupby("kp_bin", observed=True).size()
    low_sza_snr = low_sza.groupby("kp_bin", observed=True)["snr"].median()
    n_days_per_kp = raw.groupby("kp_bin", observed=True)["date"].nunique()

    print("SNR median by Kp bin x SZA bin:")
    print(snr_by_kp_sza.pivot(index="sza_bin", columns="kp_bin", values="snr_median").reindex(columns=KP_LABELS).to_string())
    print("\nSpot count by Kp bin x SZA bin:")
    print(count_by_kp_sza.pivot(index="sza_bin", columns="kp_bin", values="spot_count").reindex(columns=KP_LABELS).to_string())
    print(f"\nAt SZA <= {LOW_SZA_MAX} (report's quoted comparison point):")
    print(f"  spot counts: {low_sza_counts.reindex(KP_LABELS).to_dict()}")
    print(f"  median SNR:  {low_sza_snr.reindex(KP_LABELS).to_dict()}")
    print(f"\nDistinct sampled days contributing to each Kp bin: {n_days_per_kp.reindex(KP_LABELS).to_dict()}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for kp_label in KP_LABELS:
        g = snr_by_kp_sza[snr_by_kp_sza["kp_bin"] == kp_label].sort_values("sza_bin")
        if not g.empty:
            axes[0].plot(g["sza_bin"], g["snr_median"], marker="o", color=KP_COLORS[kp_label], label=kp_label)
        g2 = count_by_kp_sza[count_by_kp_sza["kp_bin"] == kp_label].sort_values("sza_bin")
        if not g2.empty:
            axes[1].plot(g2["sza_bin"], g2["spot_count"], marker="o", color=KP_COLORS[kp_label], label=kp_label)
    axes[0].set_title("Median SNR vs SZA by Kp bin")
    axes[0].set_xlabel("SZA (deg)")
    axes[0].set_ylabel("Median SNR (dB)")
    axes[1].set_title("Spot count vs SZA by Kp bin (pooled, all 84 days)")
    axes[1].set_xlabel("SZA (deg)")
    axes[1].set_ylabel("Spot count")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, title="Kp bin")
    plt.suptitle("20m Day DX (2k-8k km) - Reproduction of Report Figure 20")
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_DIR / "1_reproduction.png", dpi=150)
    plt.close()

    return snr_by_kp_sza, count_by_kp_sza


def step2_3_month_stratified(raw: pd.DataFrame):
    raw = raw.copy()
    raw["kp_bin"] = pd.cut(raw["Kp"], bins=KP_BINS, labels=KP_LABELS, include_lowest=True)

    low_sza = raw[raw["sza_bin"] <= LOW_SZA_MAX]

    snr_table = low_sza.groupby(["month", "kp_bin"], observed=True)["snr"].agg(["median", "size"]).reset_index()
    snr_pivot = snr_table.pivot(index="month", columns="kp_bin", values="median").reindex(columns=KP_LABELS)
    n_pivot = snr_table.pivot(index="month", columns="kp_bin", values="size").reindex(columns=KP_LABELS)

    print(f"\nMedian SNR by month x Kp bin, SZA<={LOW_SZA_MAX}:")
    print(snr_pivot.to_string())
    print(f"\nSample sizes (n spots) by month x Kp bin, SZA<={LOW_SZA_MAX}:")
    print(n_pivot.to_string())

    count_table = low_sza.groupby(["month", "kp_bin"], observed=True).size().reset_index(name="spot_count")
    count_pivot = count_table.pivot(index="month", columns="kp_bin", values="spot_count").reindex(columns=KP_LABELS)
    print(f"\nSpot count by month x Kp bin, SZA<={LOW_SZA_MAX}:")
    print(count_pivot.to_string())

    # months with usable n (>=30) in at least Quiet and one higher bin
    print("\nMonths with n>=30 in both Quiet and Severe (usable for within-month comparison):")
    usable = n_pivot[(n_pivot["Quiet (0-2)"] >= 30) & (n_pivot["Severe (6+)"] >= 30)]
    print(usable.to_string() if not usable.empty else "  NONE")

    return snr_pivot, count_pivot, n_pivot


def step4_regression(raw: pd.DataFrame):
    raw = raw.copy()
    raw["season"] = raw["month"].str[5:7].astype(int).map(SEASON_MAP)

    # --- SNR regression (per spot) ---
    seasons = ["winter", "spring", "autumn"]
    X_cols = ["intercept", "Kp", "fobs", "sza_bin"] + [f"season_{s}" for s in seasons]
    n = len(raw)
    X = np.column_stack([
        np.ones(n), raw["Kp"].values, raw["fobs"].values, raw["sza_bin"].values,
        *[(raw["season"] == s).astype(float).values for s in seasons],
    ])
    y_snr = raw["snr"].values.astype(float)
    snr_result = ols(X, y_snr, X_cols)

    # --- Spot-count regression: aggregate to (date, time_3h, sza_bin) cells ---
    cell = raw.groupby(["date", "month", "time_3h", "sza_bin", "Kp", "fobs"], observed=True).size().reset_index(name="count")
    cell["season"] = cell["month"].str[5:7].astype(int).map(SEASON_MAP)
    cell["log_count"] = np.log1p(cell["count"])

    n2 = len(cell)
    X2 = np.column_stack([
        np.ones(n2), cell["Kp"].values, cell["fobs"].values, cell["sza_bin"].values,
        *[(cell["season"] == s).astype(float).values for s in seasons],
    ])
    y_count = cell["log_count"].values.astype(float)
    count_result = ols(X2, y_count, X_cols)

    return snr_result, count_result, {"n_snr": n, "n_cells": n2}


def ols(X, y, col_names):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    n, k = X.shape
    dof = n - k
    sigma2 = np.sum(resid ** 2) / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    t_stats = beta / se
    p_vals = 2 * stats.t.sf(np.abs(t_stats), dof)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return pd.DataFrame({"term": col_names, "coef": beta, "se": se, "t": t_stats, "p_value": p_vals}), r2, dof


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building raw 20m day/DX table (84 baseline days, Kp + F10.7 joined)...")
    raw = build_raw_table()
    print(f"Total spots: {len(raw):,}\n")

    print("=" * 70)
    print("STEP 1: Reproduce original Figure 20")
    print("=" * 70)
    step1_reproduce(raw)

    print("\n" + "=" * 70)
    print("STEP 2/3: Month-stratified SNR and spot count")
    print("=" * 70)
    step2_3_month_stratified(raw)

    print("\n" + "=" * 70)
    print("STEP 4: Regressions")
    print("=" * 70)
    snr_result, count_result, meta = step4_regression(raw)
    print(f"\n--- SNR ~ Kp + Fobs + sza_bin + season (summer=ref), n={meta['n_snr']:,} ---")
    print(snr_result[0].to_string(index=False))
    print(f"R^2={snr_result[1]:.4f}, dof={snr_result[2]:,}")

    print(f"\n--- log1p(spot_count per date x time_3h x sza_bin cell) ~ Kp + Fobs + sza_bin + season, n={meta['n_cells']:,} ---")
    print(count_result[0].to_string(index=False))
    print(f"R^2={count_result[1]:.4f}, dof={count_result[2]:,}")

    snr_result[0].to_csv(OUT_DIR / "snr_regression.csv", index=False)
    count_result[0].to_csv(OUT_DIR / "count_regression.csv", index=False)

    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()

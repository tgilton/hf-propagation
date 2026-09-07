"""Test whether Report Section 3.7's F2 seasonal anomaly on 10m (winter/autumn
outperform summer by ~10 dB median SNR, attributed to seasonal F-layer
ionization independent of SZA) survives the same confound that killed the
solar-noon paradox: with one week sampled per month, F10.7 level and "which
week got sampled" are nearly perfectly entangled.

Scope verified directly against ~/Downloads/wspr_propagation_report_4.docx:
Figure 7 is median SNR vs SZA per calendar month, 10m, day paths, 2k-8k km.
2023 baseline only (excludes the 2024-05 Gannon week - that's a storm event,
not a normal seasonal comparison point).

Step 1 reproduces the original finding directly from wspr_agg_v2.parquet.
Steps 2-4 rebuild a raw-value-correct table (band=10, day, 2k-8k km) with
F10.7 joined per date, for the actual confound test.
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
from sza_paradox_check import load_fobs_by_date, SFI_BINS, SFI_LABELS  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "seasonal_anomaly"
V2_PATH = Path(__file__).resolve().parent.parent / "output" / "wspr_agg_v2.parquet"

DX_BUCKETS = ["2k-4k", "4k-8k"]  # verified: Section 3.7's actual "2k-8k km" scope
SEASON_MAP = {1: "winter", 2: "winter", 12: "winter",
              3: "spring", 4: "spring", 5: "spring",
              6: "summer", 7: "summer", 8: "summer",
              9: "autumn", 10: "autumn", 11: "autumn"}
SEASON_COLORS = {"winter": "steelblue", "spring": "seagreen", "summer": "tomato", "autumn": "goldenrod"}

BASELINE_MONTHS = [f"2023-{m:02d}" for m in range(1, 13)]  # excludes 2024-05 Gannon


def step1_reproduce_from_v2():
    df = pd.read_parquet(V2_PATH)
    sub = df[
        (df["band_m"] == 10) &
        (df["path_type"] == "day") &
        (df["dist_bucket"].isin(DX_BUCKETS)) &
        (df["month"].isin(BASELINE_MONTHS))
    ].copy()

    # Weighted-average snr_median across mlat_bin (v2 only stores per-bin
    # medians, so a spot-count-weighted mean across mlat_bin is the best
    # available combination without re-deriving from raw values)
    def wavg(g):
        w = g["spot_count"]
        return pd.Series({
            "snr_median_wavg": np.average(g["snr_median"], weights=w),
            "spot_count": w.sum(),
        })

    combined = sub.groupby(["month", "sza_bin"]).apply(wavg, include_groups=False).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    month_colors = plt.cm.hsv(np.linspace(0, 0.85, 12))
    for i, month in enumerate(BASELINE_MONTHS):
        g = combined[combined["month"] == month].sort_values("sza_bin")
        if g.empty:
            continue
        ax.plot(g["sza_bin"], g["snr_median_wavg"], color=month_colors[i],
                linewidth=1.5, marker="o", markersize=3, label=month)
    ax.set_xlabel("Solar Zenith Angle at Path Midpoint (deg)\n<- Overhead     Horizon ->")
    ax.set_ylabel("Median SNR, spot-count-weighted across mlat_bin (dB)")
    ax.set_title("10m Day DX (2k-8k km) - Reproduction of Report Figure 7\nMedian SNR vs SZA per calendar month, from wspr_agg_v2.parquet")
    ax.legend(fontsize=7, ncol=2, title="Month")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_DIR / "1_reproduction_from_v2.png", dpi=150)
    plt.close()

    # Single-number summary per month: spot-count-weighted mean of snr_median across all SZA
    per_month = combined.groupby("month").apply(
        lambda g: pd.Series({
            "snr_summary": np.average(g["snr_median_wavg"], weights=g["spot_count"]),
            "spot_count": g["spot_count"].sum(),
        }), include_groups=False
    ).reset_index()
    per_month["season"] = per_month["month"].str[5:7].astype(int).map(SEASON_MAP)
    return per_month


def build_raw_table():
    """10m day, 2k-8k km, per-spot snr with month/sza_bin/fobs, 2023 baseline only."""
    fobs_by_date = load_fobs_by_date()
    dates = [d for d in expected_dates() if d.year == 2023]
    frames = []

    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        fobs = fobs_by_date.get(d)
        if fobs is None or pd.isna(fobs):
            print(f"  WARNING: no Fobs for {date_str}", file=sys.stderr)
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
        ][["snr", "mid_sza"]].copy()

        if subset.empty:
            continue

        subset["sza_bin"] = (np.floor(subset["mid_sza"] / 5) * 5).astype(int)
        subset["month"] = f"{d.year:04d}-{d.month:02d}"
        subset["date"] = date_str
        subset["fobs"] = fobs
        frames.append(subset[["snr", "sza_bin", "month", "date", "fobs"]])

    return pd.concat(frames, ignore_index=True)


def step2_fobs_by_sampled_week(raw: pd.DataFrame):
    per_date = raw.groupby(["month", "date"])["fobs"].first().reset_index()
    per_month = per_date.groupby("month")["fobs"].agg(["mean", "median", "min", "max", "count"]).reset_index()
    per_month["season"] = per_month["month"].str[5:7].astype(int).map(SEASON_MAP)
    return per_month.sort_values("month")


def step3a_within_sfi_bin(raw: pd.DataFrame):
    raw = raw.copy()
    raw["sfi_bin"] = pd.cut(raw["fobs"], bins=SFI_BINS, labels=SFI_LABELS)
    raw["season"] = raw["month"].str[5:7].astype(int).map(SEASON_MAP)

    g = raw.groupby(["sfi_bin", "season"], observed=True).agg(
        snr_median=("snr", "median"),
        n=("snr", "size"),
    ).reset_index()
    return g


def step3b_regression(raw: pd.DataFrame):
    raw = raw.copy()
    raw["season"] = raw["month"].str[5:7].astype(int).map(SEASON_MAP)

    # Design matrix: intercept + fobs + sza_bin + season dummies (summer = reference)
    seasons = ["winter", "spring", "autumn"]  # summer is reference
    X_cols = ["intercept", "fobs", "sza_bin"] + [f"season_{s}" for s in seasons]

    n = len(raw)
    X = np.zeros((n, len(X_cols)))
    X[:, 0] = 1.0
    X[:, 1] = raw["fobs"].values
    X[:, 2] = raw["sza_bin"].values
    for i, s in enumerate(seasons):
        X[:, 3 + i] = (raw["season"] == s).astype(float).values
    y = raw["snr"].values.astype(float)

    beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    dof = n - X.shape[1]
    sigma2 = np.sum(resid ** 2) / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    t_stats = beta / se
    p_vals = 2 * stats.t.sf(np.abs(t_stats), dof)

    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Also fit WITHOUT season, to see how much R^2 season adds
    X_nosea = X[:, :3]
    beta2, _, _, _ = np.linalg.lstsq(X_nosea, y, rcond=None)
    resid2 = y - X_nosea @ beta2
    r2_nosea = 1 - np.sum(resid2 ** 2) / ss_tot

    result = pd.DataFrame({
        "term": X_cols,
        "coef": beta,
        "se": se,
        "t": t_stats,
        "p_value": p_vals,
    })
    return result, {"n": n, "r2_full": r2, "r2_without_season": r2_nosea, "dof": dof}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 1: Reproduce original Figure 7 from wspr_agg_v2.parquet")
    print("=" * 70)
    per_month_v2 = step1_reproduce_from_v2()
    print(per_month_v2.sort_values("snr_summary").to_string(index=False))
    winter_autumn = per_month_v2[per_month_v2["season"].isin(["winter", "autumn"])]["snr_summary"]
    summer = per_month_v2[per_month_v2["season"] == "summer"]["snr_summary"]
    print(f"\nWinter+autumn mean summary SNR: {winter_autumn.mean():.1f} dB")
    print(f"Summer mean summary SNR:        {summer.mean():.1f} dB")
    print(f"Gap: {winter_autumn.mean() - summer.mean():.1f} dB")

    print("\nBuilding raw per-spot table (2023 baseline, 10m day 2k-8k km)...")
    raw = build_raw_table()
    raw.to_csv(OUT_DIR / "raw_snr_month_sza_fobs.csv", index=False)
    print(f"Total spots: {len(raw):,}")

    print("\n" + "=" * 70)
    print("STEP 2: F10.7 during each actually-sampled week")
    print("=" * 70)
    fobs_table = step2_fobs_by_sampled_week(raw)
    print(fobs_table.to_string(index=False))
    fobs_table.to_csv(OUT_DIR / "fobs_by_sampled_week.csv", index=False)

    print("\n" + "=" * 70)
    print("STEP 3a: Median SNR by season WITHIN each F10.7 bin")
    print("=" * 70)
    within_sfi = step3a_within_sfi_bin(raw)
    print(within_sfi.to_string(index=False))
    within_sfi.to_csv(OUT_DIR / "within_sfi_bin_season_comparison.csv", index=False)

    print("\n" + "=" * 70)
    print("STEP 3b: Regression - snr ~ intercept + fobs + sza_bin + season (summer=ref)")
    print("=" * 70)
    reg_result, reg_meta = step3b_regression(raw)
    print(reg_result.to_string(index=False))
    print(f"\nn={reg_meta['n']:,}, dof={reg_meta['dof']:,}")
    print(f"R^2 with season:    {reg_meta['r2_full']:.4f}")
    print(f"R^2 without season: {reg_meta['r2_without_season']:.4f}")
    print(f"R^2 gain from season: {reg_meta['r2_full'] - reg_meta['r2_without_season']:.4f}")
    reg_result.to_csv(OUT_DIR / "regression_result.csv", index=False)

    print(f"\nDone. Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()

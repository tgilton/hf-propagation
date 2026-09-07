"""Resolve the 10 dB (report Figure 7) vs 2.8 dB (our pooled check) discrepancy
on the 10m seasonal anomaly.

Figure 7 was extracted directly from
~/Downloads/wspr_propagation_report_4.docx (word/media/058329ba...png,
matched to caption "Figure 7" by document order) and viewed directly: the
10m panel's curves separate most sharply at the high-SZA end (~SZA 80,
near the horizon per the numeric ticks - the axis caption's "Overhead/
Horizon" arrows appear to be mislabeled relative to the tick direction,
which doesn't affect this analysis), not as an all-SZA average.

Tests, all from wspr_agg_v2.parquet (10m, day, 2k-8k km, 2023 baseline):
  a. min-max: best month vs worst month, all-SZA summary (not group means)
  b. same comparison restricted to the SZA slice where Figure 7 visually
     separates most (SZA bin 80, checked against neighbors)
  c. best-vs-worst AND winter/autumn-vs-summer at that same slice
"""
from pathlib import Path

import numpy as np
import pandas as pd

V2_PATH = Path(__file__).resolve().parent.parent / "output" / "wspr_agg_v2.parquet"
DX_BUCKETS = ["2k-4k", "4k-8k"]
BASELINE_MONTHS = [f"2023-{m:02d}" for m in range(1, 13)]
SEASON_MAP = {1: "winter", 2: "winter", 12: "winter",
              3: "spring", 4: "spring", 5: "spring",
              6: "summer", 7: "summer", 8: "summer",
              9: "autumn", 10: "autumn", 11: "autumn"}


def load_scoped():
    df = pd.read_parquet(V2_PATH)
    return df[
        (df["band_m"] == 10) &
        (df["path_type"] == "day") &
        (df["dist_bucket"].isin(DX_BUCKETS)) &
        (df["month"].isin(BASELINE_MONTHS))
    ].copy()


def wavg_by(df, group_cols):
    def f(g):
        w = g["spot_count"]
        return pd.Series({"snr_wavg": np.average(g["snr_median"], weights=w), "n": w.sum()})
    return df.groupby(group_cols).apply(f, include_groups=False).reset_index()


def main():
    sub = load_scoped()
    sub["season"] = sub["month"].str[5:7].astype(int).map(SEASON_MAP)

    print("=" * 70)
    print("(a) MIN-MAX vs MEAN, all-SZA summary per month")
    print("=" * 70)
    per_month = wavg_by(sub, ["month"])
    per_month["season"] = per_month["month"].str[5:7].astype(int).map(SEASON_MAP)
    per_month = per_month.sort_values("snr_wavg")
    print(per_month.to_string(index=False))
    best = per_month.iloc[-1]
    worst = per_month.iloc[0]
    print(f"\nBest month:  {best['month']} ({best['season']}) = {best['snr_wavg']:.1f} dB, n={best['n']:,.0f}")
    print(f"Worst month: {worst['month']} ({worst['season']}) = {worst['snr_wavg']:.1f} dB, n={worst['n']:,.0f}")
    print(f"Min-max gap (all-SZA): {best['snr_wavg'] - worst['snr_wavg']:.1f} dB")

    print("\n" + "=" * 70)
    print("(b) Per-month summary AT EACH SZA BIN - find where separation is widest")
    print("=" * 70)
    per_month_sza = wavg_by(sub, ["month", "sza_bin"])
    per_month_sza["season"] = per_month_sza["month"].str[5:7].astype(int).map(SEASON_MAP)
    spread_by_sza = per_month_sza.groupby("sza_bin").agg(
        spread=("snr_wavg", lambda s: s.max() - s.min()),
        n_months=("snr_wavg", "size"),
        total_n=("n", "sum"),
    ).reset_index().sort_values("sza_bin")
    print(spread_by_sza.to_string(index=False))

    widest_bin = spread_by_sza.loc[spread_by_sza["spread"].idxmax(), "sza_bin"]
    print(f"\nWidest month-to-month spread is at sza_bin={widest_bin}")

    print("\n" + "=" * 70)
    print(f"(c) At sza_bin={widest_bin}: best-vs-worst AND winter/autumn-vs-summer")
    print("=" * 70)
    slice_df = per_month_sza[per_month_sza["sza_bin"] == widest_bin].sort_values("snr_wavg")
    print(slice_df.to_string(index=False))

    best_s = slice_df.iloc[-1]
    worst_s = slice_df.iloc[0]
    print(f"\nBest at this slice:  {best_s['month']} ({best_s['season']}) = {best_s['snr_wavg']:.1f} dB, n={best_s['n']:,.0f}")
    print(f"Worst at this slice: {worst_s['month']} ({worst_s['season']}) = {worst_s['snr_wavg']:.1f} dB, n={worst_s['n']:,.0f}")
    print(f"Min-max gap at sza_bin={widest_bin}: {best_s['snr_wavg'] - worst_s['snr_wavg']:.1f} dB")

    wa = slice_df[slice_df["season"].isin(["winter", "autumn"])]
    su = slice_df[slice_df["season"] == "summer"]
    wa_mean = np.average(wa["snr_wavg"], weights=wa["n"]) if len(wa) else np.nan
    su_mean = np.average(su["snr_wavg"], weights=su["n"]) if len(su) else np.nan
    print(f"\nWinter+autumn mean at sza_bin={widest_bin}: {wa_mean:.1f} dB (n={wa['n'].sum():,.0f})")
    print(f"Summer mean at sza_bin={widest_bin}:        {su_mean:.1f} dB (n={su['n'].sum():,.0f})")
    print(f"Group-mean gap at sza_bin={widest_bin}: {wa_mean - su_mean:.1f} dB")

    # Also explicitly check sza_bin=80 regardless (visual read of Figure 7), if different from widest_bin
    if 80 in per_month_sza["sza_bin"].values and widest_bin != 80:
        print("\n--- explicit check at sza_bin=80 (visual read of Figure 7) ---")
        s80 = per_month_sza[per_month_sza["sza_bin"] == 80].sort_values("snr_wavg")
        print(s80.to_string(index=False))
        print(f"Min-max gap at sza_bin=80: {s80['snr_wavg'].iloc[-1] - s80['snr_wavg'].iloc[0]:.1f} dB")


if __name__ == "__main__":
    main()

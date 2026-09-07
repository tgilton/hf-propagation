"""Methodological check on Report Section 7's gray-line analysis: did testing
solar zenith angle AT THE PATH MIDPOINT actually test the folklore claim
operators mean by "gray-line enhancement"?

The folklore claim is about paths running ALONG the terminator - both
endpoints in twilight simultaneously. Section 7 classifies paths by mid_sza
alone, which admits paths where the midpoint crosses SZA=90 while the
endpoints are anywhere (e.g. one endpoint in full daylight, the other in
full darkness, on a long enough path). This tests whether that's a
materially different population from "true" gray-line paths, using actual
endpoint SZA (tx_sza, rx_sza - already computed by add_geometry(), so no new
geometry code is needed; this is a simpler and more direct proxy for "in
twilight" than ITU-R P.533 control points ~1000 km inward, which would
require new great-circle interpolation code and can invert/overlap for the
shorter end of a 1k-8k km path anyway).

Scope verified against ~/Downloads/wspr_propagation_report_4.docx Section 7:
equinox weeks 2023 (March 20-26, September 18-24 - these are exactly the
sampled 2023-03 and 2023-09 baseline weeks), DX paths 1k-8k km, 20m and 40m
(10m too sparse per the report itself).
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

OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "gray_line"
V2_PATH = Path(__file__).resolve().parent.parent / "output" / "wspr_agg_v2.parquet"

DX_BUCKETS = ["1k-2k", "2k-4k", "4k-8k"]  # matches report's "1k-8k km"
EQUINOX_MONTHS = ["2023-03", "2023-09"]
BANDS = [20, 40]

EQUINOX_DATES = (
    [f"2023-03-{d:02d}" for d in range(20, 27)] +
    [f"2023-09-{d:02d}" for d in range(18, 25)]
)

GRAYLINE_LO, GRAYLINE_HI = 85, 95
DAY_MAX = 80
NIGHT_MIN = 100


def step1_reproduce_from_v2():
    df = pd.read_parquet(V2_PATH)
    sub = df[
        (df["band_m"].isin(BANDS)) &
        (df["month"].isin(EQUINOX_MONTHS)) &
        (df["dist_bucket"].isin(DX_BUCKETS))
    ].copy()

    def wavg(g):
        w = g["spot_count"]
        return pd.Series({"snr_wavg": np.average(g["snr_median"], weights=w), "n": w.sum()})

    combined = sub.groupby(["band_m", "sza_bin"]).apply(wavg, include_groups=False).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    for band, color in [(20, "steelblue"), (40, "seagreen")]:
        g = combined[combined["band_m"] == band].sort_values("sza_bin")
        ax.plot(g["sza_bin"], g["snr_wavg"], marker="o", color=color, label=f"{band}m")
    ax.axvline(90, color="black", linestyle=":", label="Terminator (SZA=90)")
    ax.set_xlabel("Solar Zenith Angle at Path Midpoint (deg, 5-deg bins)")
    ax.set_ylabel("Median SNR, spot-count-weighted (dB)")
    ax.set_title("Reproduction of Report Figure 11 from wspr_agg_v2.parquet\nEquinox weeks 2023, DX 1k-8k km")
    ax.legend()
    ax.grid(True, alpha=0.3)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "1_reproduction_from_v2.png", dpi=150)
    plt.close()

    print(combined.pivot(index="sza_bin", columns="band_m", values="snr_wavg").to_string())
    return combined


def build_raw_equinox_table():
    frames = []
    for date_str in EQUINOX_DATES:
        interp = build_sza_interpolator(date_str)
        for band_m in BANDS:
            src = day_parquet_path(date_str, band_m)
            if not src.exists():
                print(f"  WARNING: missing {band_m}m file for {date_str}", file=sys.stderr)
                continue
            df = pd.read_parquet(src)
            if df.empty:
                continue
            enriched = add_geometry(df, interp)
            enriched["dist_bucket"] = pd.cut(enriched["distance"], bins=DIST_BINS, labels=DIST_LABELS)
            sub = enriched[enriched["dist_bucket"].isin(DX_BUCKETS)][
                ["snr", "mid_sza", "tx_sza", "rx_sza", "dist_bucket"]
            ].copy()
            if sub.empty:
                continue
            sub["band_m"] = band_m
            sub["date"] = date_str
            frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def classify(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["midpoint_grayline"] = raw["mid_sza"].between(GRAYLINE_LO, GRAYLINE_HI)
    raw["true_grayline"] = (
        raw["tx_sza"].between(GRAYLINE_LO, GRAYLINE_HI) &
        raw["rx_sza"].between(GRAYLINE_LO, GRAYLINE_HI)
    )
    raw["both_day"] = (raw["tx_sza"] < DAY_MAX) & (raw["rx_sza"] < DAY_MAX)
    raw["both_night"] = (raw["tx_sza"] > NIGHT_MIN) & (raw["rx_sza"] > NIGHT_MIN)
    return raw


def step3_comparison(raw: pd.DataFrame):
    rows = []
    for band in BANDS:
        for bucket in DX_BUCKETS:
            sub = raw[(raw["band_m"] == band) & (raw["dist_bucket"] == bucket)]
            for label, mask_col in [
                ("midpoint gray-line (original def.)", "midpoint_grayline"),
                ("TRUE gray-line (both endpoints)", "true_grayline"),
                ("both daylight (baseline)", "both_day"),
                ("both night (baseline)", "both_night"),
            ]:
                g = sub[sub[mask_col]]
                rows.append({
                    "band_m": band, "dist_bucket": bucket, "population": label,
                    "n": len(g), "snr_median": g["snr"].median() if len(g) else np.nan,
                })
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    return result


def step4_overlap_fraction(raw: pd.DataFrame):
    rows = []
    for band in BANDS:
        sub = raw[raw["band_m"] == band]
        midpoint_pop = sub[sub["midpoint_grayline"]]
        n_midpoint = len(midpoint_pop)
        n_also_true = midpoint_pop["true_grayline"].sum()
        frac = n_also_true / n_midpoint if n_midpoint else np.nan
        rows.append({"band_m": band, "n_midpoint_grayline": n_midpoint,
                      "n_also_true_grayline": int(n_also_true), "overlap_fraction": frac})
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 1: Reproduce original midpoint-SZA finding from wspr_agg_v2.parquet")
    print("=" * 70)
    step1_reproduce_from_v2()

    print("\nBuilding raw equinox-week table (14 days x 2 bands, with endpoint geometry)...")
    raw = build_raw_equinox_table()
    print(f"Total spots (DX 1k-8k km, equinox weeks, 20m+40m): {len(raw):,}")
    raw = classify(raw)

    print("\n" + "=" * 70)
    print("STEP 3: Median SNR - true gray-line vs midpoint gray-line vs baselines")
    print("=" * 70)
    comparison = step3_comparison(raw)
    comparison.to_csv(OUT_DIR / "comparison_by_population.csv", index=False)

    print("\n" + "=" * 70)
    print("STEP 4: What fraction of midpoint-gray-line spots are TRUE gray-line?")
    print("=" * 70)
    overlap = step4_overlap_fraction(raw)
    overlap.to_csv(OUT_DIR / "overlap_fraction.csv", index=False)

    # Plot: median SNR by population, per band, faceted by distance bucket
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    pop_order = ["both daylight (baseline)", "TRUE gray-line (both endpoints)",
                 "midpoint gray-line (original def.)", "both night (baseline)"]
    pop_colors = {"both daylight (baseline)": "goldenrod",
                  "TRUE gray-line (both endpoints)": "tomato",
                  "midpoint gray-line (original def.)": "purple",
                  "both night (baseline)": "steelblue"}
    for ax, band in zip(axes, BANDS):
        g = comparison[comparison["band_m"] == band]
        x = np.arange(len(DX_BUCKETS))
        width = 0.2
        for i, pop in enumerate(pop_order):
            gp = g[g["population"] == pop].set_index("dist_bucket").reindex(DX_BUCKETS)
            ax.bar(x + i * width, gp["snr_median"], width, label=pop, color=pop_colors[pop])
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(DX_BUCKETS)
        ax.set_title(f"{band}m")
        ax.set_xlabel("Distance bucket")
        ax.grid(True, alpha=0.3, axis="y")
    axes[0].set_ylabel("Median SNR (dB)")
    axes[0].legend(fontsize=7)
    plt.suptitle("True Gray-Line vs Midpoint-Defined Gray-Line vs Day/Night Baselines\nEquinox weeks 2023, DX 1k-8k km")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "3_population_comparison.png", dpi=150)
    plt.close()

    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()

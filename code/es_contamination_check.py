"""Stress-test Report Section 8's sporadic-E (Es) contamination claim on 40m.

Verified scope directly from ~/Downloads/wspr_propagation_report_4.docx:
  8.1 Bimodal SNR distribution: summer (May-Aug) vs winter (Jan-Feb only,
      NOT Dec-Feb) 40m daytime (SZA<80) DX at 500-1k/1k-2k/2k-4k km.
      Es threshold = -5 dB (valley between the two peaks at 500-1k km).
  8.2 Daytime confirmation: SZA<70, 500-1k and 1k-2k km.
  8.3 Es fraction by UTC hour and month: 500-2k km, daytime, summer 2023,
      "4.9 million daytime spots", "1.1-1.3M spots per month".
      June Es fraction (10.3%) reported anomalously low vs
      May/July/Aug (12.7/11.7/13.5%) - the report ITSELF already flags this
      as possibly "the specific week sampled in June (19-25)" rather than a
      real monthly effect. That's exactly what this script tests.

2023 baseline only (84 days) - Section 8 predates the Gannon week.
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
from sza_paradox_check import load_fobs_by_date  # noqa: E402
from kp_insensitivity_check import load_kp_3h  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "es_contamination"

ES_THRESHOLD = -5
SUMMER_MONTHS = [5, 6, 7, 8]
WINTER_MONTHS = [1, 2]  # report's exact definition: Jan-Feb only, not Dec
BIMODAL_BUCKETS = ["500-1k", "1k-2k", "2k-4k"]
ES_FRACTION_BUCKETS = ["500-1k", "1k-2k"]  # "500-2k km" per Section 8.3


def build_raw_table(months_needed):
    fobs_by_date = load_fobs_by_date()
    kp_3h = load_kp_3h()
    dates = [d for d in expected_dates() if d.year == 2023 and d.month in months_needed]
    frames = []

    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        fobs = fobs_by_date.get(d)
        src = day_parquet_path(date_str, 40)
        if not src.exists():
            print(f"  WARNING: missing 40m file for {date_str}", file=sys.stderr)
            continue

        df = pd.read_parquet(src)
        if df.empty:
            continue

        interp = build_sza_interpolator(date_str)
        enriched = add_geometry(df, interp)
        enriched["dist_bucket"] = pd.cut(enriched["distance"], bins=DIST_BINS, labels=DIST_LABELS)

        sub = enriched[enriched["path_type"] == "day"][["time", "snr", "mid_sza", "dist_bucket"]].copy()
        if sub.empty:
            continue

        sub["time_3h"] = sub["time"].dt.floor("3h")
        sub = sub.merge(kp_3h.rename(columns={"time": "time_3h"}), on="time_3h", how="left")
        sub["hour"] = sub["time"].dt.hour
        sub["month"] = d.month
        sub["date"] = date_str
        sub["fobs"] = fobs
        frames.append(sub[["snr", "mid_sza", "dist_bucket", "hour", "month", "date", "fobs", "Kp"]])

    return pd.concat(frames, ignore_index=True)


def step1_bimodal(raw: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    summary_rows = []

    for ax, bucket in zip(axes, BIMODAL_BUCKETS):
        for label, months, color in [("summer (May-Aug)", SUMMER_MONTHS, "tomato"),
                                      ("winter (Jan-Feb)", WINTER_MONTHS, "steelblue")]:
            sub = raw[(raw["dist_bucket"] == bucket) & (raw["month"].isin(months)) & (raw["mid_sza"] < 80)]
            if sub.empty:
                continue
            n = len(sub)
            es_frac = (sub["snr"] >= ES_THRESHOLD).mean()
            ax.hist(sub["snr"], bins=60, range=(-35, 10), density=True, alpha=0.5, color=color,
                     label=f"{label} (n={n:,}, Es frac={es_frac:.1%})")
            ax.axvline(sub["snr"].median(), color=color, linestyle="--", linewidth=1.5)
            summary_rows.append({"dist_bucket": bucket, "season": label, "n": n, "es_fraction": es_frac,
                                  "median_snr": sub["snr"].median()})
        ax.axvline(ES_THRESHOLD, color="black", linestyle=":", linewidth=1, label=f"Es threshold ({ES_THRESHOLD} dB)")
        ax.set_title(f"{bucket} km, day (SZA<80)")
        ax.set_xlabel("SNR (dB)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Density")
    plt.suptitle("40m SNR Distribution: Summer vs Winter, Day Paths (SZA<80) - Reproduction of Report Figure 15")
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_DIR / "1_bimodal_reproduction.png", dpi=150)
    plt.close()

    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    return summary


def step2_monthly_es_fraction(raw: pd.DataFrame):
    sub = raw[(raw["dist_bucket"].isin(ES_FRACTION_BUCKETS)) & (raw["mid_sza"] < 80) & (raw["month"].isin(SUMMER_MONTHS))].copy()

    per_month = sub.groupby("month").agg(
        n=("snr", "size"),
        n_days=("date", "nunique"),
        es_fraction=("snr", lambda s: (s >= ES_THRESHOLD).mean()),
    ).reset_index()
    print("\nPer-month Es fraction (500-2k km, day SZA<80, 2023 summer):")
    print(per_month.to_string(index=False))

    # Day-by-day breakdown within June specifically
    june = sub[sub["month"] == 6]
    per_day = june.groupby("date").agg(
        n=("snr", "size"),
        es_fraction=("snr", lambda s: (s >= ES_THRESHOLD).mean()),
        fobs=("fobs", "first"),
        kp_mean=("Kp", "mean"),
    ).reset_index()
    print("\nJune 2023 day-by-day breakdown:")
    print(per_day.to_string(index=False))

    # F10.7/Kp by month, to check the report's own alternative hypothesis
    fobs_kp_by_month = sub.groupby("month").agg(
        fobs_mean=("fobs", "mean"),
        kp_mean=("Kp", "mean"),
    ).reset_index()
    print("\nMean F10.7 / Kp by month (summer weeks actually sampled):")
    print(fobs_kp_by_month.to_string(index=False))

    return per_month, per_day, fobs_kp_by_month


def step3_diurnal_by_month(raw: pd.DataFrame):
    sub = raw[(raw["dist_bucket"].isin(ES_FRACTION_BUCKETS)) & (raw["mid_sza"] < 80) & (raw["month"].isin(SUMMER_MONTHS))].copy()

    by_month_hour = sub.groupby(["month", "hour"]).agg(
        n=("snr", "size"),
        es_fraction=("snr", lambda s: (s >= ES_THRESHOLD).mean()),
    ).reset_index()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    month_names = {5: "May", 6: "June", 7: "July", 8: "August"}
    for ax, month in zip(axes, SUMMER_MONTHS):
        g = by_month_hour[by_month_hour["month"] == month].sort_values("hour")
        ax.plot(g["hour"], g["es_fraction"] * 100, marker="o", color="tomato")
        ax.set_title(f"{month_names[month]} (n={g['n'].sum():,})")
        ax.set_xlabel("UTC hour")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 25)
    axes[0].set_ylabel("Es fraction (%)")
    plt.suptitle("Es Fraction by UTC Hour, Held Fixed per Month (500-2k km, day SZA<80)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "3_diurnal_by_month.png", dpi=150)
    plt.close()

    print("\nEs fraction by month x hour (sample sizes):")
    pivot_n = by_month_hour.pivot(index="hour", columns="month", values="n")
    print(pivot_n.to_string())

    return by_month_hour


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    months_needed = sorted(set(SUMMER_MONTHS) | set(WINTER_MONTHS))
    print(f"Building raw 40m day-path table for months {months_needed} (2023 baseline)...")
    raw = build_raw_table(months_needed)
    print(f"Total day-path spots loaded: {len(raw):,}\n")

    print("=" * 70)
    print("STEP 1: Reproduce bimodal SNR distribution (Report Fig 15)")
    print("=" * 70)
    bimodal_summary = step1_bimodal(raw)
    bimodal_summary.to_csv(OUT_DIR / "bimodal_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("STEP 2: Monthly Es fraction - day-count/sampling scrutiny")
    print("=" * 70)
    per_month, per_day_june, fobs_kp = step2_monthly_es_fraction(raw)
    per_month.to_csv(OUT_DIR / "es_fraction_by_month.csv", index=False)
    per_day_june.to_csv(OUT_DIR / "june_day_by_day.csv", index=False)
    fobs_kp.to_csv(OUT_DIR / "fobs_kp_by_month.csv", index=False)

    print("\n" + "=" * 70)
    print("STEP 3: Diurnal pattern held fixed per month")
    print("=" * 70)
    diurnal = step3_diurnal_by_month(raw)
    diurnal.to_csv(OUT_DIR / "diurnal_by_month.csv", index=False)

    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()

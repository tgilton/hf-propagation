"""Quick diagnostic (not the full test): is Kp level entangled with
sampled-week/month the same way F10.7 was for the solar-noon paradox?

For the solar-noon check, High/Very-High SFI was concentrated in essentially
2 of 12 sampled 2023 months. This checks whether Storm/Severe Kp shows the
same concentration pattern before investing in a full month-stratified
treatment of Report Section 3.5/9.3's Kp-on-20m claim.
"""
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gap_analysis import expected_dates  # noqa: E402

SW_DIR = Path(__file__).resolve().parent.parent / "data" / "spaceweather"
OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "kp_entanglement"

KP_BINS = [0, 2, 4, 6, 9]
KP_LABELS = ["Quiet (0-2)", "Unsettled (2-4)", "Storm (4-6)", "Severe (6+)"]


def main():
    dates_2023 = [d for d in expected_dates() if d.year == 2023]
    sw = pd.read_parquet(SW_DIR / "spaceweather_2023.parquet")
    sw["date"] = sw["time"].dt.date
    sw["month"] = sw["time"].dt.strftime("%Y-%m")
    sw["kp_bin"] = pd.cut(sw["Kp"], bins=KP_BINS, labels=KP_LABELS, include_lowest=True)

    sampled = sw[sw["date"].isin(set(dates_2023))]

    tab = sampled.groupby(["month", "kp_bin"], observed=True).size().unstack("kp_bin").fillna(0).astype(int)
    print(f"Total 3-hourly Kp readings within the 12 sampled weeks: {len(sampled)}")
    print(tab.to_string())

    storm_severe = sampled[sampled["kp_bin"].isin(["Storm (4-6)", "Severe (6+)"])]
    per_month = storm_severe.groupby("month", observed=True).size()
    print(f"\nStorm+Severe 3h readings by month:\n{per_month.to_string()}")
    print(f"\nTotal Storm+Severe readings: {len(storm_severe)}")
    print(f"Distinct sampled months contributing any Storm+Severe reading: {storm_severe['month'].nunique()}/12")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT_DIR / "kp_bin_by_sampled_month.csv")

    verdict = (
        "LOW RISK: Storm+Severe Kp is spread across most sampled months, not "
        "cornered into 1-2 weeks the way High/Very-High SFI was."
        if storm_severe["month"].nunique() >= 8
        else "ELEVATED RISK: Storm+Severe Kp is concentrated in a small number "
             "of sampled months, similar to the SFI confound."
    )
    print(f"\nVerdict: {verdict}")


if __name__ == "__main__":
    main()

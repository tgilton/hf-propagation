"""Gap analysis: diff expected (band, date) coverage against what survived.

Expected scope:
  - 84 baseline days in 2023: the 3rd Mon-Sun week of every calendar month.
  - 8 Gannon storm days: 2024-05-07 through 2024-05-14.
  - All of the above x 3 bands (10m, 20m, 40m).
"""
import json
from datetime import date, timedelta
from pathlib import Path

from integrity_check import main as run_integrity_check

BANDS = ["10m", "20m", "40m"]
BASELINE_YEAR = 2023
GANNON_START = date(2024, 5, 7)
GANNON_DAYS = 8


def third_monday(year: int, month: int) -> date:
    d = date(year, month, 1)
    days_until_monday = (7 - d.weekday()) % 7
    first_monday = d + timedelta(days=days_until_monday)
    return first_monday + timedelta(weeks=2)


def expected_dates() -> list[date]:
    dates = []
    for month in range(1, 13):
        start = third_monday(BASELINE_YEAR, month)
        dates.extend(start + timedelta(days=offset) for offset in range(7))
    dates.extend(GANNON_START + timedelta(days=offset) for offset in range(GANNON_DAYS))
    return dates


def main():
    expected = expected_dates()
    assert len(expected) == 92, f"expected 92 dates, got {len(expected)}"
    expected_strs = {d.strftime("%Y-%m-%d") for d in expected}

    results = run_integrity_check()
    good_by_band = {b: set() for b in BANDS}
    extra_by_band = {b: set() for b in BANDS}

    for r in results:
        if r["status"] != "OK":
            continue
        band, dt = r["band"], r["date"]
        if dt in expected_strs:
            good_by_band[band].add(dt)
        else:
            extra_by_band[band].add(dt)

    missing = {}
    for band in BANDS:
        missing_dates = sorted(expected_strs - good_by_band[band])
        missing[band] = missing_dates

    print(f"Expected days: {len(expected_strs)} (84 baseline + {GANNON_DAYS} Gannon) x {len(BANDS)} bands = {len(expected_strs) * len(BANDS)} band-days\n")

    for band in BANDS:
        have = len(good_by_band[band])
        need = len(expected_strs)
        print(f"[{band}] {have}/{need} present, {len(missing[band])} missing")
        if extra_by_band[band]:
            print(f"       extra out-of-scope good files present: {sorted(extra_by_band[band])}")
        if missing[band]:
            print(f"       missing dates: {missing[band]}")
        print()

    out_path = Path(__file__).resolve().parent.parent / "output" / "missing_dates.json"
    out_path.parent.mkdir(exist_ok=True)
    payload = {
        "expected_total_days": len(expected_strs),
        "baseline_days": 84,
        "gannon_days": GANNON_DAYS,
        "bands": BANDS,
        "missing": missing,
        "extra_out_of_scope_present": {b: sorted(extra_by_band[b]) for b in BANDS},
        "counts": {b: {"present": len(good_by_band[b]), "missing": len(missing[b])} for b in BANDS},
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()

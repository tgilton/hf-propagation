from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

MIN_REPORTS = 10
MAX_DELTA_POWER = 10
RATIO_CAP = 20
N_BOOT = 10_000


def unzip_all():
    for z in DATA_DIR.glob("*.zip"):
        out = DATA_DIR / z.stem
        out.mkdir(exist_ok=True)
        with zipfile.ZipFile(z) as archive:
            archive.extractall(out)


def band_from_name(path: Path) -> str:
    name = path.name.lower()
    for band in ["10m", "20m", "40m"]:
        if band in name:
            return band
    return "unknown"


def day_from_file(path: Path) -> str:
    parts = path.stem.split("_")
    return parts[1] if len(parts) > 1 else "unknown"


def analyze_file(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(
        path,
        columns=["tx_sign", "rx_sign", "power", "snr"],
    ).dropna()

    df["power"] = df["power"].astype(int)
    df["snr"] = df["snr"].astype(float)

    txp = (
        df.groupby(["tx_sign", "power"])
        .agg(
            unique_rx=("rx_sign", "nunique"),
            median_snr=("snr", "median"),
            reports=("snr", "size"),
        )
        .reset_index()
    )

    txp = txp[txp["reports"] >= MIN_REPORTS]

    rows = []

    for tx, g in txp.groupby("tx_sign"):
        if g["power"].nunique() < 2:
            continue

        g = g.sort_values("power")
        recs = g.to_dict("records")

        for i in range(len(recs) - 1):
            for j in range(i + 1, len(recs)):
                lo, hi = recs[i], recs[j]
                dp = hi["power"] - lo["power"]

                if dp < 1 or dp > MAX_DELTA_POWER:
                    continue

                raw_ratio = hi["unique_rx"] / max(lo["unique_rx"], 1)
                clipped_ratio = min(raw_ratio, RATIO_CAP)
                log_rx_gain_per_db = np.log(raw_ratio) / dp

                rows.append({
                    "band": band_from_name(path),
                    "day": day_from_file(path),
                    "file": path.name,
                    "tx_sign": tx,
                    "p_lo": lo["power"],
                    "p_hi": hi["power"],
                    "delta_power": dp,
                    "rx_lo": lo["unique_rx"],
                    "rx_hi": hi["unique_rx"],
                    "rx_ratio": raw_ratio,
                    "rx_ratio_clipped": clipped_ratio,
                    "log_rx_gain_per_db": log_rx_gain_per_db,
                    "delta_rx_per_db": (hi["unique_rx"] - lo["unique_rx"]) / dp,
                    "start_snr": lo["median_snr"],
                    "reports_lo": lo["reports"],
                    "reports_hi": hi["reports"],
                })

    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    bins = [-50, -30, -28, -26, -24, -22, -20, -15, 50]
    labels = [
        "≤-30",
        "-30..-28",
        "-28..-26",
        "-26..-24",
        "-24..-22",
        "-22..-20",
        "-20..-15",
        ">-15",
    ]

    temp = results.copy()
    temp["snr_bin"] = pd.cut(temp["start_snr"], bins=bins, labels=labels)

    return (
        temp.groupby(["band", "snr_bin"], observed=True)
        .agg(
            n=("rx_ratio", "size"),
            median_rx_ratio=("rx_ratio_clipped", "median"),
            q25_rx_ratio=("rx_ratio_clipped", lambda s: np.percentile(s, 25)),
            q75_rx_ratio=("rx_ratio_clipped", lambda s: np.percentile(s, 75)),
            median_log_gain_per_db=("log_rx_gain_per_db", "median"),
            q25_log_gain_per_db=("log_rx_gain_per_db", lambda s: np.percentile(s, 25)),
            q75_log_gain_per_db=("log_rx_gain_per_db", lambda s: np.percentile(s, 75)),
            median_delta_rx_per_db=("delta_rx_per_db", "median"),
            median_start_snr=("start_snr", "median"),
            median_delta_power=("delta_power", "median"),
            median_rx_lo=("rx_lo", "median"),
            median_rx_hi=("rx_hi", "median"),
        )
        .reset_index()
    )


def summarize_three_regions(results: pd.DataFrame) -> pd.DataFrame:
    bins = [-50, -22, -15, 50]
    labels = ["Weak (< -22)", "Medium (-22..-15)", "Strong (> -15)"]

    temp = results.copy()
    temp["snr_region"] = pd.cut(temp["start_snr"], bins=bins, labels=labels)

    return (
        temp.groupby(["band", "snr_region"], observed=True)
        .agg(
            n=("log_rx_gain_per_db", "size"),
            median_log_gain_per_db=("log_rx_gain_per_db", "median"),
            q25_log_gain_per_db=("log_rx_gain_per_db", lambda s: np.percentile(s, 25)),
            q75_log_gain_per_db=("log_rx_gain_per_db", lambda s: np.percentile(s, 75)),
            median_rx_ratio=("rx_ratio_clipped", "median"),
            median_start_snr=("start_snr", "median"),
        )
        .reset_index()
    )


def bootstrap_three_regions(results: pd.DataFrame, n_boot: int = N_BOOT) -> pd.DataFrame:
    bins = [-50, -22, -15, 50]
    labels = ["Weak (< -22)", "Medium (-22..-15)", "Strong (> -15)"]

    temp = results.copy()
    temp["snr_region"] = pd.cut(temp["start_snr"], bins=bins, labels=labels)

    rng = np.random.default_rng(123)
    rows = []

    for (band, region), g in temp.groupby(["band", "snr_region"], observed=True):
        vals = g["log_rx_gain_per_db"].dropna().to_numpy()

        if len(vals) < 5:
            continue

        boots = np.median(
            rng.choice(vals, size=(n_boot, len(vals)), replace=True),
            axis=1,
        )

        rows.append({
            "band": band,
            "snr_region": str(region),
            "n": len(vals),
            "median_log_gain_per_db": np.median(vals),
            "ci_low": np.percentile(boots, 2.5),
            "ci_high": np.percentile(boots, 97.5),
        })

    return pd.DataFrame(rows)


def plot_receiver_ratio(summary: pd.DataFrame):
    plt.figure(figsize=(9, 5))

    for band, g in summary.groupby("band"):
        g = g.sort_values("snr_bin")
        x = np.arange(len(g))
        y = g["median_rx_ratio"].to_numpy()
        ylo = y - g["q25_rx_ratio"].to_numpy()
        yhi = g["q75_rx_ratio"].to_numpy() - y

        plt.errorbar(x, y, yerr=[ylo, yhi], marker="o", capsize=4, label=band)

    labels = summary["snr_bin"].drop_duplicates().astype(str).tolist()
    plt.xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("Starting SNR bin, dB")
    plt.ylabel("Receiver ratio, higher power / lower power")
    plt.title("High-Value dB Test: Receiver Gain vs Starting SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "receiver_ratio_vs_starting_snr.png", dpi=200)


def plot_log_gain_per_db(summary: pd.DataFrame):
    plt.figure(figsize=(9, 5))

    for band, g in summary.groupby("band"):
        g = g.sort_values("snr_bin")
        x = np.arange(len(g))
        y = g["median_log_gain_per_db"].to_numpy()
        ylo = y - g["q25_log_gain_per_db"].to_numpy()
        yhi = g["q75_log_gain_per_db"].to_numpy() - y

        plt.errorbar(x, y, yerr=[ylo, yhi], marker="o", capsize=4, label=band)

    labels = summary["snr_bin"].drop_duplicates().astype(str).tolist()
    plt.xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Starting SNR bin, dB")
    plt.ylabel("ln(receiver ratio) per dB")
    plt.title("High-Value dB Test: Normalized Receiver Gain")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "log_receiver_gain_per_db_vs_starting_snr.png", dpi=200)


def plot_three_regions(region_summary: pd.DataFrame):
    plt.figure(figsize=(8, 5))

    for band, g in region_summary.groupby("band"):
        x = np.arange(len(g))
        y = g["median_log_gain_per_db"].to_numpy()
        ylo = y - g["q25_log_gain_per_db"].to_numpy()
        yhi = g["q75_log_gain_per_db"].to_numpy() - y

        plt.errorbar(x, y, yerr=[ylo, yhi], marker="o", capsize=4, label=band)

    labels = region_summary["snr_region"].drop_duplicates().astype(str).tolist()
    plt.xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.ylabel("ln(receiver ratio) per dB")
    plt.xlabel("Starting SNR region")
    plt.title("High-Value dB: Weak vs Medium vs Strong Signals")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "three_region_gain_per_db.png", dpi=200)


def plot_bootstrap_three_regions(boot: pd.DataFrame):
    plt.figure(figsize=(8, 5))

    for band, g in boot.groupby("band"):
        x = np.arange(len(g))
        y = g["median_log_gain_per_db"].to_numpy()
        yerr = np.vstack([
            y - g["ci_low"].to_numpy(),
            g["ci_high"].to_numpy() - y,
        ])

        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4, label=band)

    labels = boot["snr_region"].drop_duplicates().astype(str).tolist()
    plt.xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.ylabel("Median ln(receiver ratio) per dB")
    plt.xlabel("Starting SNR region")
    plt.title("Bootstrap 95% CI: High-Value dB Effect")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "three_region_bootstrap_ci.png", dpi=200)


def main():
    # unzip_all()

    files = sorted(DATA_DIR.rglob("*.parquet"))
    print(f"Found {len(files)} parquet files")

    all_results = []

    for path in files:
        print(f"Analyzing {path}")
        result = analyze_file(path)

        if len(result):
            all_results.append(result)

    if not all_results:
        raise RuntimeError("No usable multi-power transmitter records found.")

    results = pd.concat(all_results, ignore_index=True)

    summary = summarize(results)
    region_summary = summarize_three_regions(results)
    boot_summary = bootstrap_three_regions(results)

    results.to_csv(OUT_DIR / "receiver_gain_pairs.csv", index=False)
    summary.to_csv(OUT_DIR / "receiver_gain_summary.csv", index=False)
    region_summary.to_csv(OUT_DIR / "three_region_gain_summary.csv", index=False)
    boot_summary.to_csv(OUT_DIR / "three_region_bootstrap_ci.csv", index=False)

    plot_receiver_ratio(summary)
    plot_log_gain_per_db(summary)
    plot_three_regions(region_summary)
    plot_bootstrap_three_regions(boot_summary)

    print("\nDetailed bin summary:")
    print(summary.to_string(index=False))

    print("\nThree-region summary:")
    print(region_summary.to_string(index=False))

    print("\nBootstrap CI summary:")
    print(boot_summary.to_string(index=False))


if __name__ == "__main__":
    main()
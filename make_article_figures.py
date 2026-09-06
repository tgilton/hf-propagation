from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("output")
FIG = OUT / "article_figures"
FIG.mkdir(exist_ok=True)

summary = pd.read_csv(OUT / "three_region_gain_summary.csv")
boot = pd.read_csv(OUT / "three_region_bootstrap_ci.csv")


def save_fig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=300)
    plt.close()


# Figure 1: Main result — gain per dB with bootstrap CI
plt.figure(figsize=(8, 5))

regions = ["Weak (< -22)", "Medium (-22..-15)", "Strong (> -15)"]
x = np.arange(len(regions))

for band, g in boot.groupby("band"):
    g = g.set_index("snr_region").loc[regions].reset_index()
    y = g["median_log_gain_per_db"].to_numpy()
    err = np.vstack([
        y - g["ci_low"].to_numpy(),
        g["ci_high"].to_numpy() - y,
    ])
    plt.errorbar(x, y, yerr=err, marker="o", capsize=4, label=band)

plt.axhline(0, linestyle="--", linewidth=1)
plt.xticks(x, ["Weak\n< -22 dB", "Medium\n-22 to -15 dB", "Strong\n> -15 dB"])
plt.ylabel("Median ln(receiver ratio) per dB")
plt.xlabel("Starting SNR region")
plt.title("High-Value dB Effect by Starting SNR")
plt.grid(True, alpha=0.3)
plt.legend(title="Band")
save_fig("fig1_high_value_db_bootstrap_ci.png")


# Figure 2: Receiver ratio by SNR region
plt.figure(figsize=(8, 5))

for band, g in summary.groupby("band"):
    g = g.set_index("snr_region").loc[regions].reset_index()
    plt.plot(
        x,
        g["median_rx_ratio"],
        marker="o",
        label=band,
    )

plt.axhline(1, linestyle="--", linewidth=1)
plt.xticks(x, ["Weak\n< -22 dB", "Medium\n-22 to -15 dB", "Strong\n> -15 dB"])
plt.ylabel("Median receiver ratio")
plt.xlabel("Starting SNR region")
plt.title("Additional Receivers Gained from Higher Power")
plt.grid(True, alpha=0.3)
plt.legend(title="Band")
save_fig("fig2_receiver_ratio_by_region.png")


# Figure 3: Conceptual decoder waterfall
snr = np.linspace(-32, -10, 500)

def logistic(s, s50=-24, width=1.5):
    return 1 / (1 + np.exp(-(s - s50) / width))

p = logistic(snr)

plt.figure(figsize=(8, 5))
plt.plot(snr, p, linewidth=2)
plt.axvspan(-30, -22, alpha=0.15, label="High-value region")
plt.axvline(-22, linestyle="--", linewidth=1)
plt.axvline(-15, linestyle="--", linewidth=1)
plt.xlabel("SNR, dB")
plt.ylabel("Probability of successful decode")
plt.title("Decoder Waterfall and the High-Value dB Region")
plt.grid(True, alpha=0.3)
plt.legend()
save_fig("fig3_conceptual_decoder_waterfall.png")


# Figure 4: Combined article table
article_table = boot.merge(
    summary[["band", "snr_region", "median_rx_ratio", "median_start_snr"]],
    on=["band", "snr_region"],
    how="left",
)

article_table = article_table[
    ["band", "snr_region", "n", "median_start_snr",
     "median_rx_ratio", "median_log_gain_per_db", "ci_low", "ci_high"]
]

article_table.to_csv(FIG / "article_results_table.csv", index=False)

print("Saved figures to:", FIG)
print(article_table.to_string(index=False))
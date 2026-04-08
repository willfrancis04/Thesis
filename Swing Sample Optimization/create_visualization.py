"""Build sample_size_analysis_figure.png from calculations/sample_size_results_v2_*.csv."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TARGET_ACCURACY = 0.90

players = ["Austin_R.", "Condon_C.", "Harber_P.", "Lebron_J.", "Madera_A"]
player_names_display = [
    "Austin, R.",
    "Condon, C.",
    "Harber, P.",
    "Lebron, J.",
    "Madera, A.",
]

BASE_DIR = Path(__file__).resolve().parent
CALC_DIR = BASE_DIR / "calculations"


def min_sample_size_for_target(df: pd.DataFrame, target: float = TARGET_ACCURACY):
    """Match sample_size_analysis_v2.py qualifying logic."""
    qualifying = df[
        (df["mean_accuracy"] >= target) & (df["p90_accuracy"] >= target)
    ]
    if len(qualifying) > 0:
        return int(qualifying["sample_size"].min())
    qualifying = df[df["mean_accuracy"] >= target]
    if len(qualifying) > 0:
        return int(qualifying["sample_size"].min())
    return None


all_data = {}
min_sizes = {}
for player in players:
    path = CALC_DIR / f"sample_size_results_v2_{player}.csv"
    df = pd.read_csv(path)
    all_data[player] = df
    min_sizes[player] = min_sample_size_for_target(df)

conservative_n = max(s for s in min_sizes.values() if s is not None)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle(
    "Sample Size Analysis: Accuracy vs. Number of Data Points",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

ax1 = axes[0]
colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]

for i, (player, display_name) in enumerate(zip(players, player_names_display)):
    df = all_data[player]
    min_size = min_sizes[player]

    ax1.plot(
        df["sample_size"],
        df["mean_accuracy"],
        label=f"{display_name} (min: {min_size})",
        linewidth=2.5,
        color=colors[i],
        alpha=0.8,
    )
    ax1.fill_between(
        df["sample_size"],
        df["mean_accuracy"] - df["std_accuracy"],
        df["mean_accuracy"] + df["std_accuracy"],
        alpha=0.15,
        color=colors[i],
    )
    ax1.axvline(x=min_size, color=colors[i], linestyle="--", linewidth=1.5, alpha=0.6)
    y_pos = df[df["sample_size"] == min_size]["mean_accuracy"].values[0]
    ax1.annotate(
        f"{min_size}",
        xy=(min_size, y_pos),
        xytext=(min_size, y_pos - 0.05),
        fontsize=9,
        ha="center",
        color=colors[i],
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=colors[i],
            alpha=0.7,
        ),
    )

ax1.axhline(
    y=TARGET_ACCURACY,
    color="red",
    linestyle="-",
    linewidth=2,
    alpha=0.7,
    label="90% Accuracy Threshold",
    zorder=0,
)
ax1.axvline(
    x=conservative_n,
    color="black",
    linestyle=":",
    linewidth=2.5,
    alpha=0.8,
    label=f"Conservative Recommendation ({conservative_n})",
    zorder=0,
)
ax1.text(
    conservative_n,
    0.95,
    f"Recommended ({conservative_n} samples)",
    fontsize=10,
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="yellow",
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    ),
    verticalalignment="top",
    horizontalalignment="center",
    zorder=10,
)

ax1.set_xlabel("Sample Size (Number of Data Points)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Mean Combined Accuracy", fontsize=12, fontweight="bold")
ax1.set_title("Mean Accuracy with Standard Deviation Bands", fontsize=13, pad=10)
ax1.grid(True, alpha=0.3, linestyle="--")
ax1.legend(loc="lower right", fontsize=10, framealpha=0.9)
ax1.set_xlim(5, 30)
ax1.set_ylim(0.70, 1.0)
ax1.set_xticks(range(5, 31, 2))

ax2 = axes[1]
for i, (player, display_name) in enumerate(zip(players, player_names_display)):
    df = all_data[player]
    min_size = min_sizes[player]
    ax2.plot(
        df["sample_size"],
        df["p90_accuracy"],
        label=f"{display_name} (min: {min_size})",
        linewidth=2.5,
        color=colors[i],
        alpha=0.8,
    )
    ax2.axvline(x=min_size, color=colors[i], linestyle="--", linewidth=1.5, alpha=0.6)

ax2.axhline(
    y=TARGET_ACCURACY,
    color="red",
    linestyle="-",
    linewidth=2,
    alpha=0.7,
    label="90% Accuracy Threshold",
    zorder=0,
)
ax2.axvline(
    x=conservative_n,
    color="black",
    linestyle=":",
    linewidth=2.5,
    alpha=0.8,
    label=f"Conservative Recommendation ({conservative_n})",
    zorder=0,
)

ax2.set_xlabel("Sample Size (Number of Data Points)", fontsize=12, fontweight="bold")
ax2.set_ylabel("90th Percentile Accuracy", fontsize=12, fontweight="bold")
ax2.set_title("90th Percentile Accuracy (Worst-Case Scenarios)", fontsize=13, pad=10)
ax2.grid(True, alpha=0.3, linestyle="--")
ax2.legend(loc="lower right", fontsize=10, framealpha=0.9)
ax2.set_xlim(5, 30)
ax2.set_ylim(0.70, 1.0)
ax2.set_xticks(range(5, 31, 2))

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = BASE_DIR / "sample_size_analysis_figure.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out.name}")
plt.close(fig)

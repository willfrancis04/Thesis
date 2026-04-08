#!/usr/bin/env python3
"""
Thesis Results Figures
======================
Generates five PNGs for the results section:

  fig1_three_model_mae.png
  fig3_scatter_comparison_part1.png, fig3_scatter_comparison_part2.png
  fig6_nn_stability.png
  fig7_logo_cv.png

Run from the project root:
    python3 thesis_figures/generate_all_figures.py

Fig. 6 uses ten new random ``base_seed`` values each run (OS entropy). Set
``THESIS_FIG6_MASTER_SEED`` to an integer to fix the batch of drawn seeds.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from ridge_regression import (
    METRIC_NAMES,
    build_arrays,
    load_players,
    run_loocv_ridge,
)
from neural_predictor.predict_outcomes import (
    run_loocv as run_loocv_nn,
)
from logo_for_fig7 import (
    STRATEGIES,
    attach_metadata,
    run_logo_cv,
    run_logo_cv_nn,
)

OUT = HERE

LOGO_XLABEL = {
    "Archetype": "Archetype",
    "Conference": "Conference",
    "Mean Swing Time": "Mean ST",
    "Swing-Time Variance": "Swing σ",
}


def run_loocv_mean_target_baseline(players):
    """Per fold: predict each outcome as the training-set mean (no swing features)."""
    results = []
    for i, held in enumerate(players):
        _, tgts = build_arrays(players, exclude=i)
        pred = tgts.mean(axis=0)
        actual = held["targets"]
        results.append(
            {
                "name": held["name"],
                "actual": actual,
                "pred": pred,
                "mean_st": float(held["swings"].mean()),
            }
        )
    return results


def _metrics_from_results(results):
    actual = np.array([r["actual"] for r in results])
    pred = np.array([r["pred"] for r in results])
    err = np.abs(actual - pred)
    out = {}
    for c, m in enumerate(METRIC_NAMES):
        mae = err[:, c].mean()
        rmse = np.sqrt(((actual[:, c] - pred[:, c]) ** 2).mean())
        ss_res = np.sum((actual[:, c] - pred[:, c]) ** 2)
        ss_tot = np.sum((actual[:, c] - actual[:, c].mean()) ** 2) + 1e-12
        r2 = 1 - ss_res / ss_tot
        out[m] = {"mae": mae, "rmse": rmse, "r2": r2}
    out["_mean_mae"] = float(err.mean())
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Three-model grouped MAE bar chart
# ═════════════════════════════════════════════════════════════════════════════
def fig1_three_model_mae(base_m, ridge_m, nn_m):
    x = np.arange(len(METRIC_NAMES), dtype=float)
    w = 0.25

    base_vals = [base_m[m]["mae"] for m in METRIC_NAMES]
    ridge_vals = [ridge_m[m]["mae"] for m in METRIC_NAMES]
    nn_vals = [nn_m[m]["mae"] for m in METRIC_NAMES]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w, base_vals, w, label="Baseline (train-mean)", color="#B0BEC5",
           edgecolor="#455A64", linewidth=0.8)
    ax.bar(x, ridge_vals, w, label="Ridge Regression", color="#1565C0",
           edgecolor="#0D47A1", linewidth=0.8)
    ax.bar(x + w, nn_vals, w, label="Neural Network (MLP)", color="#E65100",
           edgecolor="#BF360C", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_NAMES, fontsize=12)
    ax.set_ylabel("Mean Absolute Error (LOOCV)", fontsize=12)
    ax.set_title("LOOCV Performance: Baseline vs Ridge vs Neural Network",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    path = os.path.join(OUT, "fig1_three_model_mae.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1] {path}")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Predicted vs Actual: split for thesis pages (Ridge | NN per row)
# ═════════════════════════════════════════════════════════════════════════════
def _fig3_scatter_one_page(ridge_res, nn_res, metrics_subset, filename, suptitle):
    """metrics_subset: ordered list of names from METRIC_NAMES."""
    r_actual = np.array([r["actual"] for r in ridge_res])
    r_pred = np.array([r["pred"] for r in ridge_res])
    n_actual = np.array([r["actual"] for r in nn_res])
    n_pred = np.array([r["pred"] for r in nn_res])

    n_rows = len(metrics_subset)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 4.2 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=1.002)

    for local_row, m in enumerate(metrics_subset):
        c = METRIC_NAMES.index(m)
        for col, (actual, pred, color, name) in enumerate(
            [
                (r_actual, r_pred, "#1565C0", "Ridge"),
                (n_actual, n_pred, "#E65100", "Neural Network"),
            ]
        ):
            ax = axes[local_row, col]
            a, p = actual[:, c], pred[:, c]
            ax.scatter(
                a, p, s=85, alpha=0.88, edgecolors="k", linewidth=0.55,
                color=color, zorder=5,
            )
            lo = min(a.min(), p.min()) - 0.02
            hi = max(a.max(), p.max()) + 0.02
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.45, lw=1)

            mae = np.mean(np.abs(a - p))
            ss_res = np.sum((a - p) ** 2)
            ss_tot = np.sum((a - a.mean()) ** 2) + 1e-12
            r2 = 1 - ss_res / ss_tot
            if local_row == 0:
                ax.set_title(
                    f"{name}\n{m}  —  MAE = {mae:.3f}   R² = {r2:.3f}",
                    fontsize=12,
                    fontweight="bold",
                )
            else:
                ax.set_title(
                    f"{m}  —  MAE = {mae:.3f}   R² = {r2:.3f}",
                    fontsize=12,
                    fontweight="bold",
                )
            ax.set_xlabel(f"Actual {m}", fontsize=11)
            ax.set_ylabel(f"Predicted {m}", fontsize=11)
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=10)

    path = os.path.join(OUT, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3] {path}")


def fig3_scatter_comparison(ridge_res, nn_res):
    """Part 1: AVG, OBP, SLG — Part 2: OPS, JOPS (fits thesis pages)."""
    _fig3_scatter_one_page(
        ridge_res,
        nn_res,
        ["AVG", "OBP", "SLG"],
        "fig3_scatter_comparison_part1.png",
        "Predicted vs Actual (LOOCV) — Part 1: AVG, OBP, SLG",
    )
    _fig3_scatter_one_page(
        ridge_res,
        nn_res,
        ["OPS", "JOPS"],
        "fig3_scatter_comparison_part2.png",
        "Predicted vs Actual (LOOCV) — Part 2: OPS, JOPS",
    )


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 — LOOCV stability (stacked vertically)
# ═════════════════════════════════════════════════════════════════════════════
def fig6_nn_replicate_stability(players, ridge_mae, ridge_per, n_reps=10):
    print(f"  [6] Running {n_reps} NN LOOCV replicates (random seeds)...")
    master = os.environ.get("THESIS_FIG6_MASTER_SEED", "").strip()
    rng = (
        np.random.default_rng(int(master))
        if master.isdigit()
        else np.random.default_rng()
    )
    rep_maes = []
    per_metric_maes = {m: [] for m in METRIC_NAMES}

    for k in range(n_reps):
        base_seed = int(rng.integers(0, 2**31))
        res = run_loocv_nn(players, verbose=False, base_seed=base_seed)
        actual = np.array([r["actual"] for r in res])
        pred = np.array([r["pred"] for r in res])
        err = np.abs(actual - pred)
        rep_maes.append(float(err.mean()))
        for c, m in enumerate(METRIC_NAMES):
            per_metric_maes[m].append(float(err[:, c].mean()))
        print(
            f"       rep {k+1}/{n_reps}  seed={base_seed}  MAE={rep_maes[-1]:.4f}"
        )

    rep_arr = np.array(rep_maes)
    nn_loocv_mean = float(rep_arr.mean())

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 9.5),
        height_ratios=[1.0, 1.12],
    )
    fig.suptitle(
        "LOOCV — neural network stability across random seeds",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    # ── Row 1: overall LOOCV replicates ───────────────────────────────────
    ax1 = axes[0]
    bp = ax1.boxplot(rep_arr, vert=True, patch_artist=True, widths=0.45)
    bp["boxes"][0].set_facecolor("#FFCCBC")
    bp["boxes"][0].set_edgecolor("#BF360C")
    bp["medians"][0].set_color("#BF360C")
    ax1.axhline(ridge_mae, color="#1565C0", linewidth=2, linestyle="--",
                label=f"Ridge LOOCV = {ridge_mae:.4f}")
    ax1.scatter([1] * n_reps, rep_arr, color="#E65100", alpha=0.65, zorder=5, s=42)
    ax1.set_ylabel("Overall mean MAE", fontsize=11)
    ax1.set_title(
        "(a) Overall mean MAE across seeds (one held-out player per fold)",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    ax1.set_xticks([1])
    ax1.set_xticklabels(["Compact NN"])
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    std_nn = rep_arr.std(ddof=1)
    ax1.text(
        0.02, 0.02,
        f"NN: {nn_loocv_mean:.4f} \u00b1 {std_nn:.4f}\n"
        f"Beats Ridge LOOCV in {np.sum(rep_arr < ridge_mae)}/{n_reps} runs",
        transform=ax1.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ccc", alpha=0.95),
    )

    # ── Row 2: per-metric LOOCV replicate spread ───────────────────────────
    ax2 = axes[1]
    positions = np.arange(len(METRIC_NAMES))
    data = [per_metric_maes[m] for m in METRIC_NAMES]
    bp2 = ax2.boxplot(data, positions=positions, vert=True, patch_artist=True, widths=0.5)
    for box in bp2["boxes"]:
        box.set_facecolor("#FFCCBC")
        box.set_edgecolor("#BF360C")
    for med in bp2["medians"]:
        med.set_color("#BF360C")
    for i, m in enumerate(METRIC_NAMES):
        ax2.plot(
            [i - 0.3, i + 0.3],
            [ridge_per[m]["mae"]] * 2,
            color="#1565C0",
            linewidth=2,
            linestyle="--",
        )
    ax2.set_xticks(positions)
    ax2.set_xticklabels(METRIC_NAMES, fontsize=11)
    ax2.set_ylabel("MAE", fontsize=11)
    ax2.set_title(
        "(b) Per-metric MAE across seeds (blue dash = Ridge LOOCV)",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    ax2.grid(axis="y", alpha=0.3)

    path = os.path.join(OUT, "fig6_nn_stability.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [6] {path}")
    return nn_loocv_mean


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 7 — Leave-one-group-out CV (separate figure)
# ═════════════════════════════════════════════════════════════════════════════
def fig7_logo_cv(players, ridge_loocv_mae, nn_loocv_mean):
    print("  [7] Running LOGO-CV (Ridge + NN, 4 grouping strategies) ...")
    attach_metadata(players, ROOT)
    strategy_names = list(STRATEGIES.keys())
    logo_ridge_mae = []
    logo_nn_mae = []
    for sname in strategy_names:
        labels = STRATEGIES[sname](players)
        _, m_r = run_logo_cv(players, labels, sname, verbose=False)
        _, m_n = run_logo_cv_nn(
            players, labels, sname, base_seed=42, verbose=False
        )
        logo_ridge_mae.append(m_r["mean_mae_all"])
        logo_nn_mae.append(m_n["mean_mae_all"])
        print(f"       LOGO {sname}: Ridge {m_r['mean_mae_all']:.4f}  NN {m_n['mean_mae_all']:.4f}")

    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(strategy_names))
    w = 0.36
    ax.bar(
        x - w / 2,
        logo_ridge_mae,
        w,
        label="Ridge LOGO-CV",
        color="#1565C0",
        edgecolor="#0D47A1",
        linewidth=0.8,
    )
    ax.bar(
        x + w / 2,
        logo_nn_mae,
        w,
        label="NN LOGO-CV",
        color="#E65100",
        edgecolor="#BF360C",
        linewidth=0.8,
    )
    ax.axhline(ridge_loocv_mae, color="#1565C0", linestyle="--", linewidth=1.8, alpha=0.85, zorder=0)
    ax.axhline(nn_loocv_mean, color="#E65100", linestyle="--", linewidth=1.8, alpha=0.85, zorder=0)
    ax.text(
        0.99,
        ridge_loocv_mae,
        f" Ridge LOOCV {ridge_loocv_mae:.3f}",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#1565C0",
    )
    ax.text(
        0.99,
        nn_loocv_mean,
        f" NN LOOCV mean {nn_loocv_mean:.3f}",
        ha="right",
        va="top",
        fontsize=8,
        color="#BF360C",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([LOGO_XLABEL[s] for s in strategy_names], fontsize=10)
    ax.set_ylabel("Overall mean MAE", fontsize=11)
    ax.set_title(
        "Leave-one-group-out CV (entire group held out)\n"
        "Dashed lines: standard player-wise LOOCV (Ridge; NN mean over seeds)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    path = os.path.join(OUT, "fig7_logo_cv.png")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [7] {path}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 65)
    print("  THESIS FIGURES — generating results plots")
    print("=" * 65 + "\n")

    players = load_players(ROOT)
    if not players:
        print("  ERROR: no players found.")
        sys.exit(1)
    print(f"  Loaded {len(players)} players\n")

    print("  Running Baseline LOOCV ...")
    base_res = run_loocv_mean_target_baseline(players)
    base_m = _metrics_from_results(base_res)
    print(f"    Baseline overall MAE: {base_m['_mean_mae']:.4f}")

    print("  Running Ridge LOOCV ...")
    ridge_res = run_loocv_ridge(players, verbose=False)
    ridge_m = _metrics_from_results(ridge_res)
    print(f"    Ridge overall MAE: {ridge_m['_mean_mae']:.4f}")

    print("  Running Neural Network LOOCV ...")
    nn_res = run_loocv_nn(players, verbose=False, base_seed=42)
    nn_m = _metrics_from_results(nn_res)
    print(f"    NN overall MAE: {nn_m['_mean_mae']:.4f}")

    print(f"\n  Generating figures → {OUT}/\n")

    fig1_three_model_mae(base_m, ridge_m, nn_m)
    fig3_scatter_comparison(ridge_res, nn_res)
    nn_loocv_mean = fig6_nn_replicate_stability(
        players,
        ridge_m["_mean_mae"],
        ridge_m,
        n_reps=10,
    )
    fig7_logo_cv(players, ridge_m["_mean_mae"], nn_loocv_mean)

    print(f"\n  Done — 5 image files saved to {OUT}/\n")


if __name__ == "__main__":
    main()

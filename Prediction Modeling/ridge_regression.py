#!/usr/bin/env python3
"""
Ridge Regression → Hitting Outcome Predictor
=============================================
First modelling stage: Ridge (L2) linear regression from swing-time
summary statistics to hitting outcomes (AVG, OBP, SLG, OPS, JOPS).
Each player is represented by seven statistics of their swing times:
mean, std, min, max, median, skewness, and kurtosis. Player rows are
built from the thesis CSVs via the loader below (file patterns and
20-swing selection).

λ (L2 strength) is chosen automatically per LOOCV fold via sklearn's efficient
leave-one-out GCV (`RidgeCV`; sklearn names the parameter `alphas` / attribute
`alpha_`, same quantity as λ in textbooks).
"""

import csv
import glob
import os
from typing import List

import numpy as np
from scipy import stats as sp_stats
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

#  File layout (search for these headers to jump):
#    CONFIG  →  DATA: load CSVs into player dicts
#    FEATURES  →  swing vector → 7 summary numbers; stack train matrices
#    LOOCV  →  Ridge fit + held-out predict per fold; aggregate metrics
#    (no CLI — use thesis_figures/generate_all_figures.py for result figures)

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG — paths, targets, feature count
# ═════════════════════════════════════════════════════════════════════════════
SEQ_LEN = 20
METRIC_NAMES = ["AVG", "OBP", "SLG", "OPS", "JOPS"]
N_STAT_FEATURES = 7

# λ candidates for GCV (log-spaced). Texts call this λ; sklearn uses `alphas=`.
RIDGE_LAMBDA_GRID = np.logspace(-4, 4, 50)

HERE = os.path.dirname(os.path.abspath(__file__))


def thesis_data_search_roots(csv_dir: str) -> List[str]:
    """Siblings of *csv_dir*'s parent: only ``Manual Data Collection`` and ``Starter Data``.

    *csv_dir* is typically ``.../Prediction Modeling``; data live next to it under
    ``Thesis_Data/Manual Data Collection`` and ``Thesis_Data/Starter Data``.
    """
    parent = os.path.dirname(os.path.abspath(csv_dir))
    out: List[str] = []
    seen: set[str] = set()
    for name in ("Manual Data Collection", "Starter Data"):
        p = os.path.join(parent, name)
        if os.path.isdir(p) and p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  DATA — discover thesis CSVs, parse swings + AVG/OBP/SLG, pick 20 swings
# ═════════════════════════════════════════════════════════════════════════════
def load_players(csv_dir: str) -> List[dict]:
    """Parse every player CSV and return swing time arrays + outcome targets.
    Searches only :func:`thesis_data_search_roots` (``Manual Data Collection`` and
    ``Starter Data``) for ``Thesis Data - 2_16_26 - *.csv``. Ignores other files.

    Raises ValueError if any parsed swing time (end − start) is not finite or is ≤ 0;
    if fewer than SEQ_LEN valid swings are found; or if AVG, OBP, SLG cannot be read
    from the first row that yields a valid swing (player name and file path in message).
    """
    search_roots = thesis_data_search_roots(csv_dir)
    players = []
    seen = set()

    # --- Each matching file → one player row ---
    for root in search_roots:
        pattern = os.path.join(root, "**", "Thesis Data - 2_16_26 - *.csv")
        for fp in sorted(glob.glob(pattern, recursive=True)):
            if fp in seen:
                continue
            seen.add(fp)
            name = (
                os.path.basename(fp)
                .replace("Thesis Data - 2_16_26 - ", "")
                .replace(".csv", "")
            )
            swings: List[float] = []
            avg = obp = slg = None

            # --- Read rows: swing time = col2 − col1; stats from first good row ---
            with open(fp, newline="") as f:
                reader = csv.reader(f)
                next(reader)
                for i, row in enumerate(reader):
                    try:
                        st = float(row[2]) - float(row[1])
                    except (ValueError, IndexError):
                        continue
                    if not np.isfinite(st) or st <= 0:
                        line_no = i + 2  # header is line 1; first data row is line 2
                        raise ValueError(
                            f"Invalid swing time for player {name!r} in:\n  {fp}\n"
                            f"  Data row index {i} (line {line_no} in file): "
                            f"swing_time = end − start = {st!r} "
                            f"(column index 2 − 1 → {row[2]!r} − {row[1]!r}). "
                            f"Expected a finite value > 0. Fix the CSV."
                        )
                    swings.append(st)
                    if avg is None:
                        try:
                            avg, obp, slg = float(row[5]), float(row[6]), float(row[7])
                        except (ValueError, IndexError):
                            pass

            s_arr = np.array(swings, dtype=np.float32)

            # --- Hard requirements (no silent skip) ---
            if len(s_arr) < SEQ_LEN:
                raise ValueError(
                    f"Not enough swing times for player {name!r} in:\n  {fp}\n"
                    f"  Need at least {SEQ_LEN} rows with valid end−start swing times; "
                    f"found {len(s_arr)}."
                )
            if avg is None:
                raise ValueError(
                    f"Missing hitting stats (AVG, OBP, SLG) for player {name!r} in:\n  {fp}\n"
                    f"  Could not read numeric values from columns 6–8 on the first row "
                    f"that had a valid swing time. Fix the CSV."
                )

            # --- Subset to SEQ_LEN swings (rule differs for one named player) ---
            if name == "Austin, R.":
                idx = np.argsort(s_arr)[-SEQ_LEN:]
                idx = np.sort(idx)
                sel_swings = s_arr[idx]
            else:
                mean_st = s_arr.mean()
                diffs = np.abs(s_arr - mean_st)
                idx = np.argsort(diffs)[:SEQ_LEN]
                idx = np.sort(idx)
                sel_swings = s_arr[idx]

            # --- Derived targets OPS, JOPS ---
            ops = obp + slg
            jops = 3.27 * obp + slg
            players.append(
                {
                    "name": name,
                    "swings": sel_swings,
                    "targets": np.array([avg, obp, slg, ops, jops], dtype=np.float32),
                }
            )

    return players


# ═════════════════════════════════════════════════════════════════════════════
#  FEATURES — one swing vector → 7 numbers; many players → X and y matrices
# ═════════════════════════════════════════════════════════════════════════════

# --- Single player: 20 swings → feature vector for Ridge input ---
def stat_features(s: np.ndarray) -> np.ndarray:
    """7 descriptive statistics from a swing-time vector."""
    return np.array(
        [
            s.mean(),
            s.std(),
            s.min(),
            s.max(),
            np.median(s),
            sp_stats.skew(s),
            sp_stats.kurtosis(s),
        ],
        dtype=np.float32,
    )


# --- Training fold: stack features/targets for all players except `exclude` ---
def build_arrays(players, exclude=None):
    """Return (stat_features, targets) arrays from real player rows only."""
    feats, tgts = [], []
    for i, p in enumerate(players):
        if i == exclude:
            continue
        s, t = p["swings"], p["targets"]
        feats.append(stat_features(s))
        tgts.append(t)
    return np.stack(feats), np.stack(tgts)


# ═════════════════════════════════════════════════════════════════════════════
#  LOOCV — leave-one-player-out; RidgeCV inside each fold
# ═════════════════════════════════════════════════════════════════════════════
def run_loocv_ridge(players, verbose: bool = True):
    results = []

    for i, held in enumerate(players):
        if verbose:
            print(
                f"  Fold {i + 1:>2}/{len(players)}: {held['name']:<22}",
                end="",
                flush=True,
            )

        # --- Train matrices (everyone except held-out) ---
        feats, tgts = build_arrays(players, exclude=i)

        # --- Z-score X and y on the training fold only ---
        scaler_x = StandardScaler().fit(feats)
        scaler_y = StandardScaler().fit(tgts)

        X_train = scaler_x.transform(feats)
        y_train = scaler_y.transform(tgts)

        # --- Ridge: GCV picks λ per fold (sklearn: RidgeCV(..., alphas=...), alpha_) ---
        model = RidgeCV(alphas=RIDGE_LAMBDA_GRID)
        model.fit(X_train, y_train)

        # --- Held-out player: same scaling, then predict in original y scale ---
        x_test = scaler_x.transform(
            stat_features(held["swings"]).reshape(1, -1)
        )
        pred = scaler_y.inverse_transform(model.predict(x_test)).squeeze()

        actual = held["targets"]
        err = np.abs(actual - pred)
        results.append(
            {
                "name": held["name"],
                "actual": actual,
                "pred": pred,
                "mean_st": held["swings"].mean(),
            }
        )
        if verbose:
            parts = "  ".join(
                f"{m}:{err[c]:.3f}" for c, m in enumerate(METRIC_NAMES)
            )
            print(f"  MAE: {err.mean():.4f}  ({parts})")

    return results


# --- Collapse LOOCV predictions into MAE / RMSE / R² per metric ---
def aggregate_loocv_metrics(results):
    """Return dict of per-metric MAE, RMSE, R² and overall mean MAE."""
    actual_all = np.array([r["actual"] for r in results])
    pred_all = np.array([r["pred"] for r in results])
    err_all = np.abs(actual_all - pred_all)
    names = METRIC_NAMES
    out = {"per_metric": {}, "mean_mae_all": float(err_all.mean())}
    for c, m in enumerate(names):
        mae = err_all[:, c].mean()
        rmse = np.sqrt(((actual_all[:, c] - pred_all[:, c]) ** 2).mean())
        ss_res = np.sum((actual_all[:, c] - pred_all[:, c]) ** 2)
        ss_tot = np.sum((actual_all[:, c] - actual_all[:, c].mean()) ** 2) + 1e-12
        r2 = 1 - ss_res / ss_tot
        out["per_metric"][m] = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}
    return out

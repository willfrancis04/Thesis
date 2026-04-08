#!/usr/bin/env python3
"""
LOGO-CV helpers for thesis Fig. 7 only (Ridge + NN). Not a standalone CLI.
"""

from __future__ import annotations

import csv
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from ridge_regression import (
    METRIC_NAMES,
    RIDGE_LAMBDA_GRID,
    stat_features,
    thesis_data_search_roots,
)
from neural_predictor.predict_outcomes import (
    Normalizer,
    SwingDataset,
    train_one,
)

SHEET3_META: Dict[str, Dict[str, str]] = {
    "Madera, A": {"archetype": "On-Baser", "conference": "ACC"},
    "Condon, C.": {"archetype": "Run Producer", "conference": "SEC"},
    "Harber, P.": {"archetype": "Barrel Controller", "conference": "ACC"},
    "Austin, R.": {"archetype": "On-Baser", "conference": "SEC"},
    "Lebron, J.": {"archetype": "Run Producer", "conference": "SEC"},
}

_ARCHETYPE_CANON = {
    "barrel controler": "Barrel Controller",
    "barrel controller": "Barrel Controller",
    "run producer": "Run Producer",
    "on-baser": "On-Baser",
}


def _normalise_archetype(raw: str) -> str:
    return _ARCHETYPE_CANON.get(raw.strip().lower(), raw.strip())


def _looks_like_text(s: str) -> bool:
    cleaned = s.strip().replace(".", "").replace(",", "").replace(" ", "")
    return bool(cleaned) and not cleaned.replace("-", "").isdigit()


def _extract_csv_metadata(filepath: str) -> Tuple[Optional[str], Optional[str]]:
    archetype: Optional[str] = None
    conference: Optional[str] = None
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        rows: list = []
        for row in reader:
            rows.append(row)
            if len(rows) >= 3:
                break

    if len(rows) >= 2 and rows[1] and _looks_like_text(rows[1][0]):
        archetype = _normalise_archetype(rows[1][0])
    if len(rows) >= 3 and rows[2] and _looks_like_text(rows[2][0]):
        conference = rows[2][0].strip()
    return archetype, conference


def attach_metadata(players: List[dict], project_root: str) -> List[dict]:
    search_roots = thesis_data_search_roots(project_root)
    csv_by_name: Dict[str, str] = {}
    for root in search_roots:
        pattern = os.path.join(root, "**", "Thesis Data - 2_16_26 - *.csv")
        for fp in sorted(glob.glob(pattern, recursive=True)):
            name = (
                os.path.basename(fp)
                .replace("Thesis Data - 2_16_26 - ", "")
                .replace(".csv", "")
            )
            csv_by_name[name] = fp

    for p in players:
        name = p["name"]
        archetype = conference = None
        if name in csv_by_name:
            archetype, conference = _extract_csv_metadata(csv_by_name[name])
        if archetype is None and name in SHEET3_META:
            archetype = SHEET3_META[name]["archetype"]
        if conference is None and name in SHEET3_META:
            conference = SHEET3_META[name]["conference"]
        p["archetype"] = archetype or "Unknown"
        p["conference"] = conference or "Unknown"
    return players


def groups_by_archetype(players: List[dict]) -> List[str]:
    return [p["archetype"] for p in players]


def groups_by_conference(players: List[dict]) -> List[str]:
    return [p["conference"] for p in players]


def groups_by_mean_swing_time(players: List[dict]) -> List[str]:
    means = np.array([p["swings"].mean() for p in players])
    med = float(np.median(means))
    return ["Above Median" if m >= med else "Below Median" for m in means]


def groups_by_swing_variance(players: List[dict]) -> List[str]:
    stds = np.array([p["swings"].std() for p in players])
    med = float(np.median(stds))
    return ["High Variance" if s >= med else "Low Variance" for s in stds]


STRATEGIES = {
    "Archetype": groups_by_archetype,
    "Conference": groups_by_conference,
    "Mean Swing Time": groups_by_mean_swing_time,
    "Swing-Time Variance": groups_by_swing_variance,
}


def _aggregate_logo(results: List[dict]) -> dict:
    actual_all = np.array([r["actual"] for r in results])
    pred_all = np.array([r["pred"] for r in results])
    err_all = np.abs(actual_all - pred_all)
    out: dict = {"per_metric": {}, "mean_mae_all": float(err_all.mean())}
    for c, m in enumerate(METRIC_NAMES):
        mae = float(err_all[:, c].mean())
        rmse = float(np.sqrt(((actual_all[:, c] - pred_all[:, c]) ** 2).mean()))
        ss_res = float(np.sum((actual_all[:, c] - pred_all[:, c]) ** 2))
        ss_tot = (
            float(np.sum((actual_all[:, c] - actual_all[:, c].mean()) ** 2)) + 1e-12
        )
        r2 = 1 - ss_res / ss_tot
        out["per_metric"][m] = {"mae": mae, "rmse": rmse, "r2": r2}
    return out


def run_logo_cv(
    players: List[dict],
    group_labels: List[str],
    strategy_name: str,
    verbose: bool = True,
) -> Tuple[List[dict], dict]:
    unique_groups = sorted(set(group_labels))
    all_results: List[dict] = []

    if verbose:
        print(f"\n{'=' * 65}")
        print(f"  LOGO-CV — {strategy_name}")
        print(f"  Groups ({len(unique_groups)}): {', '.join(unique_groups)}")
        print("=" * 65 + "\n")

    for g in unique_groups:
        held_idx = [i for i, gl in enumerate(group_labels) if gl == g]
        train_idx = [i for i, gl in enumerate(group_labels) if gl != g]

        if verbose:
            print(f"  Fold: hold out {g!r} ({len(held_idx)} players)")
            for i in held_idx:
                print(f"    - {players[i]['name']}")

        feats_train = np.stack(
            [stat_features(players[i]["swings"]) for i in train_idx]
        )
        tgts_train = np.stack([players[i]["targets"] for i in train_idx])

        scaler_x = StandardScaler().fit(feats_train)
        scaler_y = StandardScaler().fit(tgts_train)
        X_train = scaler_x.transform(feats_train)
        y_train = scaler_y.transform(tgts_train)

        model = RidgeCV(alphas=RIDGE_LAMBDA_GRID)
        model.fit(X_train, y_train)

        for i in held_idx:
            p = players[i]
            x_test = scaler_x.transform(stat_features(p["swings"]).reshape(1, -1))
            pred = scaler_y.inverse_transform(model.predict(x_test)).squeeze()
            actual = p["targets"]
            err = np.abs(actual - pred)
            all_results.append(
                {
                    "name": p["name"],
                    "group": g,
                    "actual": actual,
                    "pred": pred,
                    "mean_st": p["swings"].mean(),
                }
            )
            if verbose:
                parts = "  ".join(
                    f"{m}:{err[c]:.3f}" for c, m in enumerate(METRIC_NAMES)
                )
                print(f"      {p['name']:<25} MAE: {err.mean():.4f}  ({parts})")
        if verbose:
            print()

    metrics = _aggregate_logo(all_results)
    if verbose:
        print(f"  Overall MAE: {metrics['mean_mae_all']:.4f}")
    return all_results, metrics


def _build_train_arrays(players: List[dict], train_idx: List[int]):
    feats = np.stack([stat_features(players[i]["swings"]) for i in train_idx])
    tgts = np.stack([players[i]["targets"] for i in train_idx])
    return feats, tgts


def run_logo_cv_nn(
    players: List[dict],
    group_labels: List[str],
    strategy_name: str,
    base_seed: int = 42,
    verbose: bool = True,
) -> Tuple[List[dict], dict]:
    unique_groups = sorted(set(group_labels))
    all_results: List[dict] = []
    fold_counter = 0

    if verbose:
        print(f"\n{'=' * 65}")
        print(f"  LOGO-CV (NN) — {strategy_name}")
        print("=" * 65 + "\n")

    for g in unique_groups:
        held_idx = [i for i, gl in enumerate(group_labels) if gl == g]
        train_idx = [i for i, gl in enumerate(group_labels) if gl != g]

        feats_train, tgts_train = _build_train_arrays(players, train_idx)
        norm = Normalizer(feats_train, tgts_train)
        ds = SwingDataset(norm.norm_feat(feats_train), norm.norm_tgt(tgts_train))

        for i in held_idx:
            held = players[i]
            fold_counter += 1
            model = train_one(
                ds, norm, val_player=held, seed=base_seed + fold_counter
            )
            model.eval()
            with torch.no_grad():
                f = torch.as_tensor(
                    norm.norm_feat(stat_features(held["swings"]))
                ).unsqueeze(0)
                pred = norm.denorm_tgt(model(f).squeeze().numpy())
            actual = held["targets"]
            err = np.abs(actual - pred)
            all_results.append(
                {
                    "name": held["name"],
                    "group": g,
                    "actual": actual,
                    "pred": pred,
                    "mean_st": held["swings"].mean(),
                }
            )
            if verbose:
                parts = "  ".join(
                    f"{m}:{err[c]:.3f}" for c, m in enumerate(METRIC_NAMES)
                )
                print(
                    f"    Fold {fold_counter:>2}/{len(players)}: {held['name']:<22} "
                    f"MAE: {err.mean():.4f}  ({parts})"
                )

    metrics = _aggregate_logo(all_results)
    return all_results, metrics

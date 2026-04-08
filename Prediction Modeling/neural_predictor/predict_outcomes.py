#!/usr/bin/env python3
"""
Compact MLP for swing-time → hitting outcomes (AVG, OBP, SLG, OPS, JOPS).

Used by ``thesis_figures/generate_all_figures.py`` for NN LOOCV and Fig. 7 LOGO-CV.
"""

import csv
import glob
import os
import sys
import warnings
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats as sp_stats
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)


def _project_root() -> str:
    pkg = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(pkg)
    if os.path.isfile(os.path.join(parent, "ridge_regression.py")):
        return parent
    return pkg


PROJECT_ROOT = _project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ridge_regression import thesis_data_search_roots

SEQ_LEN = 20
N_STAT_FEATURES = 7
COMPACT_MLP_HIDDEN = 10
DROPOUT = 0.3
EPOCHS = 300
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 40

METRIC_NAMES = ["AVG", "OBP", "SLG", "OPS", "JOPS"]
N_TARGETS = len(METRIC_NAMES)


def load_players(csv_dir: str) -> List[dict]:
    """Same rules as ``ridge_regression.load_players``."""
    search_roots = thesis_data_search_roots(csv_dir)
    players = []
    seen = set()

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

            with open(fp, newline="") as f:
                reader = csv.reader(f)
                next(reader)
                for i, row in enumerate(reader):
                    try:
                        st = float(row[2]) - float(row[1])
                    except (ValueError, IndexError):
                        continue
                    if not np.isfinite(st) or st <= 0:
                        line_no = i + 2
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


def stat_features(s: np.ndarray) -> np.ndarray:
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


class SwingDataset(Dataset):
    def __init__(self, feats, tgts):
        self.feats = torch.as_tensor(feats)
        self.tgts = torch.as_tensor(tgts)

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return self.feats[i], self.tgts[i]


def build_arrays(players, exclude=None):
    feats, tgts = [], []
    for i, p in enumerate(players):
        if i == exclude:
            continue
        s, t = p["swings"], p["targets"]
        feats.append(stat_features(s))
        tgts.append(t)
    return np.stack(feats), np.stack(tgts)


class Normalizer:
    def __init__(self, feats, tgts):
        self.feat_mu = feats.mean(axis=0)
        self.feat_sig = feats.std(axis=0) + 1e-8
        self.tgt_mu = tgts.mean(axis=0)
        self.tgt_sig = tgts.std(axis=0) + 1e-8

    def norm_feat(self, x):
        return (x - self.feat_mu) / self.feat_sig

    def norm_tgt(self, x):
        return (x - self.tgt_mu) / self.tgt_sig

    def denorm_tgt(self, x):
        return x * self.tgt_sig + self.tgt_mu


class SwingPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        h = COMPACT_MLP_HIDDEN
        self.net = nn.Sequential(
            nn.Linear(N_STAT_FEATURES, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(h, N_TARGETS),
        )

    def forward(self, feat):
        return self.net(feat)


def train_one(dataset, norm, val_player=None, seed=42, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SwingPredictor()
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    loss_fn = nn.SmoothL1Loss()
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=True, drop_last=False)

    best_state, best_loss, wait = None, float("inf"), 0

    for ep in range(EPOCHS):
        model.train()
        running = 0.0
        for fb, tb in loader:
            opt.zero_grad()
            loss = loss_fn(model(fb), tb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
        sched.step()

        if val_player is not None:
            model.eval()
            with torch.no_grad():
                vf = torch.as_tensor(
                    norm.norm_feat(stat_features(val_player["swings"]))
                ).unsqueeze(0)
                vt = torch.as_tensor(norm.norm_tgt(val_player["targets"])).unsqueeze(0)
                vl = loss_fn(model(vf), vt).item()

            if vl < best_loss:
                best_loss = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait >= PATIENCE:
                if verbose:
                    print(f"    Early stop @ epoch {ep + 1}")
                break

        if verbose and (ep + 1) % 100 == 0:
            print(f"    Epoch {ep + 1:>3} | loss {running / len(loader):.6f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


def run_loocv(players, verbose: bool = True, base_seed: int = 42):
    results = []
    for i, held in enumerate(players):
        if verbose:
            print(
                f"  Fold {i + 1:>2}/{len(players)}: {held['name']:<22}",
                end="",
                flush=True,
            )
        feats, tgts = build_arrays(players, exclude=i)
        norm = Normalizer(feats, tgts)
        ds = SwingDataset(norm.norm_feat(feats), norm.norm_tgt(tgts))
        model = train_one(ds, norm, val_player=held, seed=base_seed + i)

        model.eval()
        with torch.no_grad():
            f = torch.as_tensor(
                norm.norm_feat(stat_features(held["swings"]))
            ).unsqueeze(0)
            pred = norm.denorm_tgt(model(f).squeeze().numpy())

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

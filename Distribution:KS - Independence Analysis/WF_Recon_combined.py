"""
Create a combined (old + new data) 2‑panel figure:
- Panel 1: Swing Time vs Pitch Speed
- Panel 2: Swing Time by Pitch Type

Starter sheet:  ../Starter Data/Thesis Data Collection - Sheet3.csv
New data:  Manual Data Collection folders (Export.csv + Thesis Data*.csv), default ../Manual Data Collection
"""

import os
import glob

import matplotlib.pyplot as plt
import numpy as np

from plot_helpers import add_figure_n_key, annotate_box_column_sample_counts
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


FONTSIZE = 14
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_THESIS_ROOT = os.path.dirname(_SCRIPT_DIR)
_DEFAULT_MANUAL_ROOT = os.path.join(_THESIS_ROOT, 'Manual Data Collection')
NEW_DATA_ROOT = os.environ.get('THESIS_MANUAL_DATA_ROOT', _DEFAULT_MANUAL_ROOT)
STARTER_THESIS_CSV = os.path.join(
    _THESIS_ROOT, 'Starter Data', 'Thesis Data Collection - Sheet3.csv'
)


def load_old_data(path: str) -> pd.DataFrame:
    """Load and clean the original thesis dataset, returning rows with SwingTime."""
    df = pd.read_csv(path)
    df['FPS_numeric'] = pd.to_numeric(df['FPS'], errors='coerce')
    df['SwingTime'] = df['Clicks - Hands Move'] / df['FPS_numeric']
    df_clean = df.dropna(subset=['SwingTime']).copy()
    df_clean['PitchSpeed_numeric'] = pd.to_numeric(df_clean['Pitch Speed'], errors='coerce')
    df_clean['Source'] = 'Old'
    return df_clean


def load_thesis_row_data(thesis_path: str):
    """Read Thesis Data CSV; return list of swing_time_sec values."""
    df = pd.read_csv(thesis_path)
    ts = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    te = pd.to_numeric(df.iloc[:, 2], errors='coerce')
    swing_times = (te - ts).values
    rows = []
    for val in swing_times:
        if np.isfinite(val) and val >= 0:
            rows.append(val)
    return rows


def load_export_data(export_path: str):
    """Read Export CSV; return list of (pitch_type, velocity_mph)."""
    df = pd.read_csv(export_path)
    pt_col = 'Pitch Type' if 'Pitch Type' in df.columns else [
        c for c in df.columns if 'pitch' in c.lower() and 'type' in c.lower()
    ][0]
    vel_col = 'Velocity (MPH)' if 'Velocity (MPH)' in df.columns else None
    if vel_col is None:
        for c in df.columns:
            if 'velocity' in c.lower() or 'mph' in c.lower():
                vel_col = c
                break
    if vel_col is None:
        vel_col = df.columns[17] if len(df.columns) > 17 else df.columns[-1]
    pitch_types = df[pt_col].astype(str).values
    velocities = pd.to_numeric(df[vel_col], errors='coerce').values
    return list(zip(pitch_types, velocities))


def load_new_data(root_dir: str) -> pd.DataFrame:
    """Load and merge swing-time rows from all 23 new‑data folders."""
    folders = [
        d
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')
    ]
    all_rows = []
    for folder_name in sorted(folders):
        folder_path = os.path.join(root_dir, folder_name)
        hitter = folder_name.split(' - ')[0].strip() if ' - ' in folder_name else folder_name
        thesis_files = glob.glob(os.path.join(folder_path, 'Thesis Data*.csv'))
        export_path = os.path.join(folder_path, 'Export.csv')
        if not thesis_files or not os.path.isfile(export_path):
            continue
        thesis_path = thesis_files[0]
        swing_times = load_thesis_row_data(thesis_path)
        export_rows = load_export_data(export_path)
        n = min(len(swing_times), len(export_rows))
        if n == 0:
            continue
        for i in range(n):
            swing_sec = swing_times[i]
            pt, vel = export_rows[i]
            all_rows.append(
                {
                    'Hitter': hitter,
                    'Pitch Type': pt if pt and pt != 'nan' else None,
                    'PitchSpeed_numeric': vel if np.isfinite(vel) and vel != 0 else np.nan,
                    'SwingTime': swing_sec,
                    'Source': 'New',
                }
            )
    df_new = pd.DataFrame(all_rows)
    return df_new.dropna(subset=['SwingTime']).copy()


def main():
    # Load datasets
    df_old = load_old_data(STARTER_THESIS_CSV)
    df_new = load_new_data(NEW_DATA_ROOT)

    # Align column names and subset to common fields
    cols = ['Hitter', 'Pitch Type', 'PitchSpeed_numeric', 'SwingTime', 'Source']
    df_old_sub = df_old[cols].copy()
    df_new_sub = df_new[cols].copy()
    df_combined = pd.concat([df_old_sub, df_new_sub], ignore_index=True)

    # Figure setup: 2 panels stacked vertically (pitch speed on top)
    _FS = 7
    fig, (ax_speed, ax_pt) = plt.subplots(2, 1, figsize=(3.8, 5.5))

    # ---------- Panel 1: Swing Time vs Pitch Speed (combined old + new) ----------
    df_speed = df_combined.dropna(subset=['PitchSpeed_numeric'])
    ax_speed.scatter(
        df_speed['PitchSpeed_numeric'],
        df_speed['SwingTime'],
        alpha=0.5,
        s=10,
        edgecolors='black',
        linewidth=0.2,
        color='tab:blue',
    )

    # Combined linear fit for R² only (no best-fit line drawn)
    if not df_speed.empty and df_speed['PitchSpeed_numeric'].notna().sum() > 1:
        X = df_speed['PitchSpeed_numeric'].values.reshape(-1, 1)
        y = df_speed['SwingTime'].values
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2_speed = r2_score(y, y_pred)
    else:
        r2_speed = 0.0

    # Annotate pitch count per 5-mph bin (same style as box-plot n= labels)
    speeds = df_speed['PitchSpeed_numeric'].dropna()
    bin_lo = int(np.floor(speeds.min() / 5) * 5)
    bin_hi = int(np.ceil(speeds.max() / 5) * 5)
    bins = np.arange(bin_lo, bin_hi + 5, 5)
    y0, y1 = ax_speed.get_ylim()
    span = y1 - y0
    ax_speed.set_ylim(y0 - 0.065 * span, y1)
    y0_new = ax_speed.get_ylim()[0]
    y_text = y0_new + 0.025 * (y1 - y0_new)
    x_lo, x_hi = ax_speed.get_xlim()
    for left, right in zip(bins[:-1], bins[1:]):
        n_bin = int(((speeds >= left) & (speeds < right)).sum())
        if n_bin > 0:
            x_center = min((left + right) / 2, x_hi - 1.5)
            ax_speed.text(
                x_center, y_text, f'n={n_bin}',
                ha='center', va='bottom', fontsize=_FS - 2,
            )

    ax_speed.set_xlabel('Pitch Speed (mph)', fontsize=_FS)
    ax_speed.set_ylabel('Swing Time (s)', fontsize=_FS)
    ax_speed.set_title(
        f'Swing Time vs Pitch Speed (R²={r2_speed:.3f})',
        fontsize=_FS + 1,
        fontweight='bold',
    )
    ax_speed.tick_params(axis='both', labelsize=_FS - 1)
    ax_speed.grid(True, alpha=0.3)

    # ---------- Panel 2: Swing Time by Pitch Type (combined old + new) ----------
    df_pt = df_combined.dropna(subset=['Pitch Type'])

    PITCH_TYPE_ORDER = ['Fastball', 'Curveball', 'Slider', 'Changeup', 'Cutter']
    pt_means = []
    for pt in df_pt['Pitch Type'].unique():
        data = df_pt[df_pt['Pitch Type'] == pt]['SwingTime'].values
        if len(data) > 0:
            pt_means.append((pt, data.mean()))
    pt_means.sort(key=lambda x: PITCH_TYPE_ORDER.index(x[0]) if x[0] in PITCH_TYPE_ORDER else 999)
    pitch_types_sorted = [pt for pt, _ in pt_means]

    box_data_pt = []
    positions_pt = []
    for i, pt in enumerate(pitch_types_sorted):
        data = df_pt[df_pt['Pitch Type'] == pt]['SwingTime'].values
        if len(data) > 0:
            box_data_pt.append(data)
            positions_pt.append(i)

    _bw = 0.22
    for i, data in enumerate(box_data_pt):
        p5, p25, p50, p75, p95 = np.percentile(data, [5, 25, 50, 75, 95])
        ax_pt.plot([positions_pt[i], positions_pt[i]], [p5, p95], 'k-', linewidth=0.8)
        box = plt.Rectangle(
            (positions_pt[i] - _bw, p25),
            2 * _bw,
            p75 - p25,
            facecolor='lightcoral',
            edgecolor='black',
            linewidth=0.8,
        )
        ax_pt.add_patch(box)
        ax_pt.plot(
            [positions_pt[i] - _bw, positions_pt[i] + _bw],
            [p50, p50],
            'r-',
            linewidth=1.0,
        )
        ax_pt.plot(
            [positions_pt[i] - _bw * 0.5, positions_pt[i] + _bw * 0.5],
            [p5, p5],
            'k-',
            linewidth=0.8,
        )
        ax_pt.plot(
            [positions_pt[i] - _bw * 0.5, positions_pt[i] + _bw * 0.5],
            [p95, p95],
            'k-',
            linewidth=0.8,
        )

    if not df_pt.empty:
        dummies_pt = pd.get_dummies(df_pt['Pitch Type'], drop_first=True)
        X_pt = dummies_pt.values
        y_pt = df_pt['SwingTime'].values
        r2_pt = (
            r2_score(y_pt, LinearRegression().fit(X_pt, y_pt).predict(X_pt))
            if X_pt.size > 0 and len(np.unique(y_pt)) > 1
            else 0.0
        )
    else:
        r2_pt = 0.0

    ax_pt.set_xlabel('Pitch Type', fontsize=_FS)
    ax_pt.set_ylabel('Swing Time (s)', fontsize=_FS)
    ax_pt.set_title(
        f'Swing Time by Pitch Type (R²={r2_pt:.3f})',
        fontsize=_FS + 1,
        fontweight='bold',
    )
    ax_pt.set_xticks(positions_pt)
    ax_pt.set_xticklabels(
        pitch_types_sorted, fontsize=_FS - 1, rotation=45, ha='right'
    )
    ax_pt.tick_params(axis='y', labelsize=_FS - 1)
    ax_pt.grid(True, alpha=0.3, axis='y')
    if box_data_pt:
        annotate_box_column_sample_counts(
            ax_pt, positions_pt, box_data_pt, _FS - 2
        )

    plt.tight_layout(rect=(0, 0.03, 1, 1))
    pos_top = ax_speed.get_position()
    ax_speed.set_position([pos_top.x0, pos_top.y0 + 0.02, pos_top.width, pos_top.height])
    _n_key = add_figure_n_key(fig, fontsize=_FS - 2)
    out_path = 'PitchCharacteristics_combined.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight',
                bbox_extra_artists=[_n_key], pad_inches=0.1)
    print(f"\nFigure saved as '{out_path}'")


if __name__ == '__main__':
    main()



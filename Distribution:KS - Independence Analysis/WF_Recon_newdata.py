# %%
"""
New-data pipeline: KDE by hitter, count/score/runners panels, and hitter archetype figure.
Expects per-hitter folders (Thesis Data*.csv + Export.csv) under ../Manual Data Collection
by default, or set THESIS_MANUAL_DATA_ROOT. Also uses ../Starter Data/Thesis Data Collection - Sheet3.csv
for KDE merge of legacy five hitters.
"""
import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from plot_helpers import add_figure_n_key, annotate_box_column_sample_counts
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_THESIS_ROOT = os.path.dirname(_SCRIPT_DIR)
_DEFAULT_MANUAL_ROOT = os.path.join(_THESIS_ROOT, 'Manual Data Collection')
NEW_DATA_ROOT = os.environ.get('THESIS_MANUAL_DATA_ROOT', _DEFAULT_MANUAL_ROOT)
STARTER_THESIS_CSV = os.path.join(
    _THESIS_ROOT, 'Starter Data', 'Thesis Data Collection - Sheet3.csv'
)

# %%
def _parse_score(s):
    """Parse score string '3-5' or '4-4' -> (int, int) or None."""
    if s is None or not str(s).strip() or str(s) == 'nan':
        return None
    s = str(s).strip()
    m = re.match(r'^(\d+)\s*[-–]\s*(\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


def _parse_matchup(notes):
    """Parse 'Miami vs Alabama' or 'Alabama vs. Miami' -> (team1, team2). team1 gets first score."""
    if notes is None or not str(notes).strip() or str(notes) == 'nan':
        return None
    n = str(notes).strip()
    for sep in (' vs ', ' vs. ', ' VS ', ' VS. '):
        if sep in n:
            parts = n.split(sep, 1)
            if len(parts) == 2:
                return (parts[0].strip(), parts[1].strip())
    return None


def _hitter_team_from_matchups(matchups):
    """Given list of (team1, team2) per row, return the team that appears in every row (hitter's team)."""
    if not matchups:
        return None
    first_set = set(matchups[0])
    for m in matchups[1:]:
        first_set &= set(m)
    if len(first_set) == 1:
        return first_set.pop()
    if len(first_set) == 2:
        return None
    return None


def _run_diff_to_label(diff):
    """Convert run differential to centered buckets: 0, ±1, ±2, ±3, ±4⁺ (4+ runs grouped)."""
    if diff == 0:
        return '0'
    if diff >= 4:
        return '+4⁺'
    if diff <= -4:
        return '-4⁺'
    if diff > 0:
        return f'+{diff}'
    return str(diff)


def load_thesis_row_data(thesis_path):
    """Read Thesis Data CSV; return list of (swing_time_sec, archetype, conference). Row 1=player, row 2=archetype, row 3=conference."""
    df = pd.read_csv(thesis_path)
    ts = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    te = pd.to_numeric(df.iloc[:, 2], errors='coerce')
    swing_times = (te - ts).values
    archetype = None
    conference = None
    if len(df) >= 2:
        val = df.iloc[1, 0]
        if pd.notna(val) and str(val).strip():
            archetype = str(val).strip()
    if len(df) >= 3:
        val = df.iloc[2, 0]  # 3rd data row = conference (SEC, West Coast, etc.)
        if pd.notna(val) and str(val).strip():
            conference = str(val).strip()
    rows = []
    for i in range(len(swing_times)):
        if np.isfinite(swing_times[i]) and swing_times[i] >= 0:
            rows.append((swing_times[i], archetype, conference))
    return rows


def _normalize_runners_label(v):
    """Map runners-on-base strings to consistent labels (e.g., dashes -> 'None')."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return v
    s = str(v).strip()
    # Treat any standalone dash variants as 'no runners'
    if s in ('-', '–', '—'):
        return 'None'
    return s


def load_export_data(export_path):
    """Read Export CSV; return list of (pitch_type, velocity_mph, count, score_raw, runners, notes).
    score_raw is the raw score string (e.g. '3-5'); notes for matchup (e.g. 'Miami vs Alabama')."""
    df = pd.read_csv(export_path)
    pt_col = 'Pitch Type' if 'Pitch Type' in df.columns else [c for c in df.columns if 'pitch' in c.lower() and 'type' in c.lower()][0]
    vel_col = 'Velocity (MPH)' if 'Velocity (MPH)' in df.columns else None
    if vel_col is None:
        for c in df.columns:
            if 'velocity' in c.lower() or 'mph' in c.lower():
                vel_col = c
                break
    if vel_col is None:
        vel_col = df.columns[17] if len(df.columns) > 17 else df.columns[-1]
    count_col = 'Count' if 'Count' in df.columns else None
    if count_col is None:
        for c in df.columns:
            if 'count' in c.lower():
                count_col = c
                break
    score_col = None
    for c in df.columns:
        if c.strip().lower() == 'score':
            score_col = c
            break
    runners_col = None
    for c in df.columns:
        low = c.strip().lower()
        if 'runner' in low and 'base' in low:
            runners_col = c
            break
    if runners_col is None:
        for c in df.columns:
            if c.strip().lower() == 'runners':
                runners_col = c
                break
    notes_col = None
    for c in df.columns:
        low = c.strip().lower()
        if low in ('notes', 'note', 'matchup', 'match') or 'notes' in low or 'matchup' in low:
            notes_col = c
            break
    if notes_col is None:
        for c in df.columns:
            if ' vs ' in str(df[c].iloc[0] if len(df) > 0 else ''):
                notes_col = c
                break
    pitch_types = df[pt_col].astype(str).values
    velocities = pd.to_numeric(df[vel_col], errors='coerce').values
    counts = df[count_col].astype(str).values if count_col else [''] * len(df)
    scores = df[score_col].astype(str).values if score_col else [None] * len(df)
    runners = df[runners_col].astype(str).values if runners_col else [None] * len(df)
    notes = df[notes_col].astype(str).values if notes_col else [None] * len(df)
    return list(zip(pitch_types, velocities, counts, scores, runners, notes))


def load_all_new_data(root_dir):
    """Load and merge data from all 23 named folders. Returns one DataFrame."""
    folders = [d for d in os.listdir(root_dir)
               if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')]
    all_rows = []
    for folder_name in sorted(folders):
        folder_path = os.path.join(root_dir, folder_name)
        # Hitter name: strip " - 2 18 26" etc.
        hitter = folder_name.split(' - ')[0].strip() if ' - ' in folder_name else folder_name
        thesis_files = glob.glob(os.path.join(folder_path, 'Thesis Data*.csv'))
        export_path = os.path.join(folder_path, 'Export.csv')
        if not thesis_files or not os.path.isfile(export_path):
            continue
        thesis_path = thesis_files[0]
        thesis_rows = load_thesis_row_data(thesis_path)
        export_rows = load_export_data(export_path)
        n = min(len(thesis_rows), len(export_rows))
        if n == 0:
            continue
        archetype = thesis_rows[0][1] if thesis_rows else None
        conference = thesis_rows[0][2] if thesis_rows and len(thesis_rows[0]) > 2 else None
        # Score situation: parse "X-Y" and "Team1 vs Team2"; hitter's team = university in every row
        matchups = []
        scores_parsed = []
        for i in range(n):
            pt, vel, count, score_raw, runners, notes = export_rows[i]
            matchups.append(_parse_matchup(notes))
            scores_parsed.append(_parse_score(score_raw))
        hitter_team = _hitter_team_from_matchups([m for m in matchups if m is not None])
        for i in range(n):
            swing_sec, _, _ = thesis_rows[i]
            pt, vel, count, score_raw, runners, notes = export_rows[i]
            score_situation = None
            if hitter_team and matchups[i] is not None and scores_parsed[i] is not None:
                t1, t2 = matchups[i]
                r1, r2 = scores_parsed[i]
                if hitter_team == t1:
                    score_situation = _run_diff_to_label(r1 - r2)
                elif hitter_team == t2:
                    score_situation = _run_diff_to_label(r2 - r1)
            all_rows.append({
                'Hitter': hitter,
                'Archetype': archetype,
                'Conference': conference,
                'Pitch Type': pt if pt and pt != 'nan' else None,
                'Pitch Speed': vel if np.isfinite(vel) and vel != 0 else np.nan,
                'Count': count if count and str(count).strip() and str(count) != 'nan' else None,
                'Score': score_situation,
                'Runners': runners if runners is not None and str(runners).strip() and str(runners) != 'nan' else None,
                'FPS': 60,
                'Clicks - Hands Move': swing_sec * 60,
            })
    df = pd.DataFrame(all_rows)
    # Normalize runners-on-base labels (combine different dash variants, trim spaces)
    if 'Runners' in df.columns:
        df['Runners'] = df['Runners'].apply(_normalize_runners_label)
    return df


# %%
# Load the new data
df = load_all_new_data(NEW_DATA_ROOT)
print(f"Loaded {len(df)} rows from {NEW_DATA_ROOT}")

# Calculate SwingTime in seconds (same as original script)
df['FPS_numeric'] = pd.to_numeric(df['FPS'], errors='coerce')
df['SwingTime'] = df['Clicks - Hands Move'] / df['FPS_numeric']
df_clean = df.dropna(subset=['SwingTime']).copy()

print(f"Rows with valid SwingTime: {len(df_clean)}")
print(f"Unique hitters: {sorted(df_clean['Hitter'].unique())}")

# Get unique hitters and sort by mean swing time (same as original)
hitter_stats = []
for hitter in df_clean['Hitter'].unique():
    swing_times = df_clean[df_clean['Hitter'] == hitter]['SwingTime'].values
    mean_st = np.mean(swing_times)
    hitter_stats.append((hitter, mean_st))
hitter_stats.sort(key=lambda x: x[1])
hitters_sorted = [h[0] for h in hitter_stats]

# Old thesis CSV (5 hitters): included in KDE figures only; KS / pitch plots stay new-data only
df_old = pd.read_csv(STARTER_THESIS_CSV)
df_old['FPS_numeric'] = pd.to_numeric(df_old['FPS'], errors='coerce')
df_old['SwingTime'] = df_old['Clicks - Hands Move'] / df_old['FPS_numeric']
df_old_clean = df_old.dropna(subset=['SwingTime']).copy()

kde_hitter_stats = []
for hitter in df_clean['Hitter'].unique():
    st = df_clean[df_clean['Hitter'] == hitter]['SwingTime'].values
    if len(st) > 0:
        kde_hitter_stats.append((hitter, np.mean(st)))
for hitter in df_old_clean['Hitter'].unique():
    if hitter not in df_clean['Hitter'].values:
        st = df_old_clean[df_old_clean['Hitter'] == hitter]['SwingTime'].values
        if len(st) > 0:
            kde_hitter_stats.append((hitter, np.mean(st)))
kde_hitter_stats.sort(key=lambda x: x[1])
kde_hitters_sorted = [h[0] for h in kde_hitter_stats]

# %%
# Swing Time KDE plot (new data + 5 old-data hitters, sorted by mean swing time)
FONTSIZE = 14
colors = plt.cm.tab10(np.linspace(0, 1, 10))
linestyles = ['-', '--', '-.', ':']

fig, ax = plt.subplots(figsize=(12, 8))
for idx, hitter in enumerate(kde_hitters_sorted):
    if hitter in df_clean['Hitter'].values:
        swing_times = df_clean[df_clean['Hitter'] == hitter]['SwingTime'].values
    else:
        swing_times = df_old_clean[df_old_clean['Hitter'] == hitter]['SwingTime'].values
    n = len(swing_times)
    mean_st = np.mean(swing_times)
    std_st = np.std(swing_times, ddof=1)
    kde = gaussian_kde(swing_times)
    x_range = np.linspace(swing_times.min() - 0.01, swing_times.max() + 0.01, 200)
    kde_values = kde(x_range)
    color = colors[idx % len(colors)]
    linestyle = linestyles[idx // len(colors) % len(linestyles)]
    ax.plot(x_range, kde_values, color=color, linestyle=linestyle,
            linewidth=2, label=f"{hitter} (n={n}): {mean_st:.3f}±{std_st:.3f}")
ax.set_xlabel('Swing Time (seconds)', fontsize=FONTSIZE)
ax.set_ylabel('Density', fontsize=FONTSIZE)
ax.set_title('Swing Time Distribution by Hitter', fontsize=FONTSIZE+2, fontweight='bold')
ax.tick_params(axis='both', labelsize=FONTSIZE-2)
ax.grid(True, alpha=0.3)
LEGEND_FONTSIZE = FONTSIZE - 4  # slightly smaller than axis labels for many hitters
ax.legend(
    loc='upper right',
    fontsize=LEGEND_FONTSIZE,
    frameon=True,
    title='Hitter (n): µ±σ (s)',
    title_fontsize=FONTSIZE - 3,
    ncol=2,
    labelspacing=0.35,
    borderpad=0.4,
    columnspacing=0.8,
)
plt.tight_layout()
plt.savefig('SwingTimeKDEs_newdata.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'SwingTimeKDEs_newdata.png'")
plt.close(fig)

# %%
# Count order for box plot (balls-strikes)
COUNT_ORDER = ['0-0', '0-1', '0-2', '1-0', '1-1', '1-2', '2-0', '2-1', '2-2', '3-0', '3-1', '3-2']

# Standalone figure: Count, Score, Runners on base vs Swing Time (stacked vertically)
fig_csr, axes_csr = plt.subplots(3, 1, figsize=(10, 14))
ax_c, ax_s, ax_r = axes_csr[0], axes_csr[1], axes_csr[2]

def draw_box_panel(ax, df_sub, cat_col, cat_order, title, facecolor='lightblue'):
    """Draw box plot panel; cat_order can be list for order or None for sort by mean."""
    if df_sub.empty or df_sub[cat_col].isna().all():
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=FONTSIZE, transform=ax.transAxes)
        ax.set_title(title, fontsize=FONTSIZE+2, fontweight='bold')
        return
    means = [(c, df_sub[df_sub[cat_col] == c]['SwingTime'].mean()) for c in df_sub[cat_col].unique() if pd.notna(c)]
    if not means:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=FONTSIZE, transform=ax.transAxes)
        ax.set_title(title, fontsize=FONTSIZE+2, fontweight='bold')
        return
    if cat_order is not None:
        means.sort(key=lambda x: cat_order.index(x[0]) if x[0] in cat_order else 999)
    else:
        means.sort(key=lambda x: x[1])
    cats_sorted = [m[0] for m in means]
    box_data = []
    positions = []
    for i, c in enumerate(cats_sorted):
        d = df_sub[df_sub[cat_col] == c]['SwingTime'].values
        if len(d) > 0:
            box_data.append(d)
            positions.append(i)
    for i, data in enumerate(box_data):
        p5, p25, p50, p75, p95 = np.percentile(data, [5, 25, 50, 75, 95])
        ax.plot([positions[i], positions[i]], [p5, p95], 'k-', linewidth=1.5)
        box = plt.Rectangle((positions[i]-0.3, p25), 0.6, p75-p25, facecolor=facecolor, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.plot([positions[i]-0.3, positions[i]+0.3], [p50, p50], 'r-', linewidth=2)
        ax.plot([positions[i]-0.15, positions[i]+0.15], [p5, p5], 'k-', linewidth=1.5)
        ax.plot([positions[i]-0.15, positions[i]+0.15], [p95, p95], 'k-', linewidth=1.5)
    dummies = pd.get_dummies(df_sub[cat_col], drop_first=True)
    X = dummies.values
    y = df_sub['SwingTime'].values
    r2 = r2_score(y, LinearRegression().fit(X, y).predict(X)) if X.size > 0 and len(np.unique(y)) > 1 else 0.0
    ax.set_xlabel(cat_col, fontsize=FONTSIZE)
    ax.set_ylabel('Swing Time (s)', fontsize=FONTSIZE)
    # Slightly smaller title that includes R² so it doesn't crowd the panel
    ax.set_title(f'{title} (R²={r2:.3f})', fontsize=FONTSIZE, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels(cats_sorted, fontsize=FONTSIZE - 2, rotation=45, ha='right')
    ax.tick_params(axis='y', labelsize=FONTSIZE-2)
    ax.grid(True, alpha=0.3, axis='y')
    annotate_box_column_sample_counts(ax, positions, box_data, FONTSIZE - 3)

df_count_csr = df_clean.dropna(subset=['Count'])
draw_box_panel(ax_c, df_count_csr, 'Count', COUNT_ORDER, 'Swing Time by Count', 'lightblue')
# Score = run differential: 0 centered, ±1/±2/±3 around, ±4⁺ buckets for 4+ runs (hitter's team minus opponent)
SCORE_ORDER = ['-4⁺', '-3', '-2', '-1', '0', '+1', '+2', '+3', '+4⁺']
df_score = df_clean.dropna(subset=['Score'])
draw_box_panel(ax_s, df_score, 'Score', SCORE_ORDER, 'Swing Time by Run Differential', 'lavender')
df_runners = df_clean.dropna(subset=['Runners'])
RUNNERS_ORDER = ['None', '1B', '2B', '3B', '1B-2B', '1B-3B', '2B-3B', '1B-2B-3B']
draw_box_panel(ax_r, df_runners, 'Runners', RUNNERS_ORDER, 'Swing Time by Runners on Base', 'lightyellow')
plt.tight_layout(rect=(0, 0.02, 1, 1))
add_figure_n_key(fig_csr, fontsize=FONTSIZE - 3)
plt.savefig('SwingTimeByCountScoreRunners_newdata.png', dpi=300, bbox_inches='tight')
print("Figure saved as 'SwingTimeByCountScoreRunners_newdata.png' (Count, Score, Runners on base)")
plt.close(fig_csr)

# Swing Time by Hitter Archetype
df_archetype = df_clean.dropna(subset=['Archetype'])
archetype_means = []
for arch in df_archetype['Archetype'].unique():
    arch_data = df_archetype[df_archetype['Archetype'] == arch]['SwingTime'].values
    if len(arch_data) > 0:
        archetype_means.append((arch, np.mean(arch_data)))
archetype_means.sort(key=lambda x: x[1])
archetypes_sorted = [arch[0] for arch in archetype_means]

box_data_arch = []
positions_arch = []
for i, arch in enumerate(archetypes_sorted):
    arch_data = df_archetype[df_archetype['Archetype'] == arch]['SwingTime'].values
    if len(arch_data) > 0:
        box_data_arch.append(arch_data)
        positions_arch.append(i)

dummies_arch = pd.get_dummies(df_archetype['Archetype'], drop_first=True)
X_arch = dummies_arch.values
y_arch = df_archetype['SwingTime'].values
r2_arch = r2_score(y_arch, LinearRegression().fit(X_arch, y_arch).predict(X_arch)) if X_arch.size > 0 and len(np.unique(y_arch)) > 1 else 0.0

_FS_ARCH = 7
fig_arch, ax_arch = plt.subplots(1, 1, figsize=(3.8, 2.6))
_bw = 0.22
for i, data in enumerate(box_data_arch):
    p5, p25, p50, p75, p95 = np.percentile(data, [5, 25, 50, 75, 95])
    ax_arch.plot([positions_arch[i], positions_arch[i]], [p5, p95], 'k-', linewidth=0.8)
    box = plt.Rectangle((positions_arch[i]-_bw, p25), 2*_bw, p75-p25,
                         facecolor='lightgreen', edgecolor='black', linewidth=0.8)
    ax_arch.add_patch(box)
    ax_arch.plot([positions_arch[i]-_bw, positions_arch[i]+_bw], [p50, p50], 'r-', linewidth=1.0)
    ax_arch.plot([positions_arch[i]-_bw*0.5, positions_arch[i]+_bw*0.5], [p5, p5], 'k-', linewidth=0.8)
    ax_arch.plot([positions_arch[i]-_bw*0.5, positions_arch[i]+_bw*0.5], [p95, p95], 'k-', linewidth=0.8)
ax_arch.set_xlabel('Hitter Archetype', fontsize=_FS_ARCH)
ax_arch.set_ylabel('Swing Time (s)', fontsize=_FS_ARCH)
ax_arch.set_title(f'Swing Time by Hitter Archetype (R²={r2_arch:.3f})', fontsize=_FS_ARCH + 1, fontweight='bold')
ax_arch.set_xticks(positions_arch)
ax_arch.set_xticklabels(archetypes_sorted, fontsize=_FS_ARCH - 1, rotation=45, ha='right')
ax_arch.tick_params(axis='y', labelsize=_FS_ARCH - 1)
ax_arch.grid(True, alpha=0.3, axis='y')
annotate_box_column_sample_counts(ax_arch, positions_arch, box_data_arch, _FS_ARCH - 2)
fig_arch.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.22)
add_figure_n_key(fig_arch, fontsize=_FS_ARCH - 2, y_fig=0.03)
plt.savefig('HitterArchetype_newdata.png', dpi=300, bbox_inches='tight', pad_inches=0.03)
print("Figure saved as 'HitterArchetype_newdata.png'")
plt.close(fig_arch)

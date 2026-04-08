# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from plot_helpers import add_figure_n_key, annotate_box_column_sample_counts

# Output folder for starter-data figures
STARTER_DATA_FIGURES_DIR = 'Starter Data Figures'
os.makedirs(STARTER_DATA_FIGURES_DIR, exist_ok=True)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STARTER_THESIS_CSV = os.path.join(
    os.path.dirname(_SCRIPT_DIR), 'Starter Data', 'Thesis Data Collection - Sheet3.csv'
)

# %%
# Load the data
df = pd.read_csv(STARTER_THESIS_CSV)

# %%
# Calculate SwingTime in seconds
# Convert FPS to numeric, replacing '?' with NaN
df['FPS_numeric'] = pd.to_numeric(df['FPS'], errors='coerce')

# Calculate SwingTime = (Clicks - Hands Move) / FPS
df['SwingTime'] = df['Clicks - Hands Move'] / df['FPS_numeric']

# Remove rows where SwingTime couldn't be calculated
df_clean = df.dropna(subset=['SwingTime']).copy()

print(f"Total rows: {len(df)}")
print(f"Rows with valid SwingTime: {len(df_clean)}")
print(f"\nUnique hitters: {df_clean['Hitter'].unique()}")

# %%
# Create the KDE plot
# Font size parameter - easy to adjust
FONTSIZE = 14

# Set up the figure
fig, ax = plt.subplots(figsize=(12, 8))

# Get unique hitters and calculate their mean swing times for sorting
hitter_stats = []
for hitter in df_clean['Hitter'].unique():
    swing_times = df_clean[df_clean['Hitter'] == hitter]['SwingTime'].values
    mean_st = np.mean(swing_times)
    hitter_stats.append((hitter, mean_st))

# Sort hitters by mean swing time (ascending)
hitter_stats.sort(key=lambda x: x[1])
hitters_sorted = [h[0] for h in hitter_stats]

# Generate colors and linestyles for differentiation
colors = plt.cm.tab10(np.linspace(0, 1, 10))  # 10 distinct colors
linestyles = ['-', '--', '-.', ':']  # 4 distinct linestyles

# Plot KDE for each hitter in sorted order
for idx, hitter in enumerate(hitters_sorted):
    # Get swing times for this hitter
    swing_times = df_clean[df_clean['Hitter'] == hitter]['SwingTime'].values
    
    # Calculate mean, std, and count
    n = len(swing_times)
    mean_st = np.mean(swing_times)
    std_st = np.std(swing_times, ddof=1)
    
    # Generate KDE
    kde = gaussian_kde(swing_times)
    x_range = np.linspace(swing_times.min() - 0.01, swing_times.max() + 0.01, 200)
    kde_values = kde(x_range)
    
    # Select color and linestyle
    color = colors[idx % len(colors)]
    linestyle = linestyles[idx // len(colors) % len(linestyles)]
    
    # Plot the KDE
    ax.plot(x_range, kde_values, color=color, linestyle=linestyle, 
            linewidth=2, label=f"{hitter} (n={n}): {mean_st:.3f}±{std_st:.3f}")

# Customize the plot
ax.set_xlabel('Swing Time (seconds)', fontsize=FONTSIZE)
ax.set_ylabel('Density', fontsize=FONTSIZE)
ax.set_title('Swing Time Distribution by Hitter', fontsize=FONTSIZE+2, fontweight='bold')
ax.tick_params(axis='both', labelsize=FONTSIZE-2)
ax.grid(True, alpha=0.3)

# Add legend inside upper right
ax.legend(loc='upper right', fontsize=FONTSIZE-2, frameon=True, 
          title='Hitter (n): µ±σ (s)', title_fontsize=FONTSIZE-1)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(STARTER_DATA_FIGURES_DIR, 'SwingTimeKDEs_olddata.png'), dpi=300, bbox_inches='tight')
print("\nFigure saved as 'Starter Data Figures/SwingTimeKDEs_olddata.png'")

# %%
# Pairwise KS test heatmap
from scipy.stats import ks_2samp

# Font size parameter
FONTSIZE = 14

# Get swing times for each hitter (already sorted by mean)
hitter_data = {}
for hitter in hitters_sorted:
    hitter_data[hitter] = df_clean[df_clean['Hitter'] == hitter]['SwingTime'].values

n_hitters = len(hitters_sorted)

# Initialize matrices for KS statistic and p-value
ks_matrix = np.full((n_hitters, n_hitters), np.nan)
p_matrix = np.full((n_hitters, n_hitters), np.nan)

# Perform pairwise KS tests (lower triangle only)
for i in range(n_hitters):
    for j in range(i):  # Only lower triangle (j < i)
        stat, p_value = ks_2samp(hitter_data[hitters_sorted[i]], hitter_data[hitters_sorted[j]])
        ks_matrix[i, j] = stat
        p_matrix[i, j] = p_value

# Create the heatmap
fig, ax = plt.subplots(figsize=(12, 10))

# Create a masked array for the upper triangle and diagonal
mask = np.triu(np.ones_like(p_matrix, dtype=bool))
p_matrix_masked = np.ma.array(p_matrix, mask=mask)

# Plot the heatmap using p-values for color
# White (p ≥ 0.05) to Red (p < 0.05)
im = ax.imshow(p_matrix_masked, cmap='Reds_r', aspect='auto', vmin=0, vmax=0.2, 
               interpolation='nearest')

# Add text labels showing KS statistic
for i in range(n_hitters):
    for j in range(i):  # Only lower triangle
        ks_stat = ks_matrix[i, j]
        p_val = p_matrix[i, j]
        
        # Determine text color based on p-value (light vs dark background)
        text_color = 'white' if p_val < 0.05 else 'black'
        
        ax.text(j, i, f'{ks_stat:.2f}', ha='center', va='center', 
                color=text_color, fontsize=FONTSIZE-2, fontweight='bold')

# Set tick labels
ax.set_xticks(np.arange(n_hitters))
ax.set_yticks(np.arange(n_hitters))
ax.set_xticklabels(hitters_sorted, rotation=45, ha='right', fontsize=FONTSIZE-2)
ax.set_yticklabels(hitters_sorted, fontsize=FONTSIZE-2)

# Add title
ax.set_title('Pairwise Kolmogorov-Smirnov Test', fontsize=FONTSIZE+2, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('p-value', fontsize=FONTSIZE, rotation=270, labelpad=20)
cbar.ax.tick_params(labelsize=FONTSIZE-2)

# Add explanatory text in upper right empty space
explanation = ('Reading the Heatmap:\n\n'
               'Text = KS Statistic\n'
               '  • Larger values → distributions\n'
               '    are more different\n\n'
               'Color = p-value\n'
               '  • Tests if datasets are drawn\n'
               '    from the same distribution\n'
               '  • Red (p < 0.05): Statistically\n'
               '    different at 95% confidence\n'
               '  • White (p ≥ 0.05): No significant\n'
               '    difference detected')
ax.text(0.98, 0.98, explanation, transform=ax.transAxes, 
        fontsize=FONTSIZE-2, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
        family='monospace')

# Adjust layout and save
plt.tight_layout()
plt.savefig(os.path.join(STARTER_DATA_FIGURES_DIR, 'PairwiseKStest_olddata.png'), dpi=300, bbox_inches='tight')
print("\nFigure saved as 'Starter Data Figures/PairwiseKStest_olddata.png'")

# %%
# Pitch Speed and Location analysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Font size parameter
FONTSIZE = 14

# Assign colors, markers, and linestyles to hitters (consistent across all plots)
colors = plt.cm.tab10(np.linspace(0, 1, 10))
markers = ['o', 's', '^', 'D', 'v']  # Marker types - cycles after using all colors
linestyles = ['-', '--', '-.', ':']  # Same as KDE plot

# Color cycles through all 10 colors
hitter_color_map = {hitter: colors[i % len(colors)] for i, hitter in enumerate(hitters_sorted)}
# Marker stays the same for first 10 hitters, then changes for next batch
hitter_marker_map = {hitter: markers[i // len(colors)] for i, hitter in enumerate(hitters_sorted)}
# Linestyle changes every 10 hitters (aligned with color cycling)
hitter_linestyle_map = {hitter: linestyles[i // len(colors) % len(linestyles)] for i, hitter in enumerate(hitters_sorted)}

# Add color, marker, and linestyle columns to dataframe
df_clean['HitterColor'] = df_clean['Hitter'].map(hitter_color_map)
df_clean['HitterMarker'] = df_clean['Hitter'].map(hitter_marker_map)
df_clean['HitterLinestyle'] = df_clean['Hitter'].map(hitter_linestyle_map)

# Convert Pitch Speed to numeric
df_clean['PitchSpeed_numeric'] = pd.to_numeric(df_clean['Pitch Speed'], errors='coerce')

# Categorize Pitch Location
def categorize_location(location_str):
    """Categorize pitch location into Middle, Low, High, In, Away"""
    # Extract the location part (before " - ")
    if ' - ' in location_str:
        loc = location_str.split(' - ')[0]
        words = loc.split()
        
        if len(words) >= 2:
            # Check if both words are "Middle"
            if words[0] == 'Middle' and words[1] == 'Middle':
                return 'Middle'
            # Priority to first word if it's Low or High
            elif words[0] in ['Low', 'High']:
                return words[0]
            # Otherwise, first word is Middle, use second word
            elif words[1] == 'In':
                return 'In'
            elif words[1] == 'Away':
                return 'Away'
    return None

df_clean['LocationCategory'] = df_clean['Pitch Location'].apply(categorize_location)
# Count from Balls-Strikes for "Swing Time by Count" panel
df_clean['Count'] = df_clean['Balls'].astype(str) + '-' + df_clean['Strikes'].astype(str)


def _draw_percentile_boxplots(ax, positions, box_data_list, facecolor='lightblue'):
    """5th–95th whiskers, 25–75 box, red median."""
    for i, data in enumerate(box_data_list):
        p5, p25, p50, p75, p95 = np.percentile(data, [5, 25, 50, 75, 95])
        ax.plot([positions[i], positions[i]], [p5, p95], 'k-', linewidth=1.5)
        box = plt.Rectangle((positions[i] - 0.3, p25), 0.6, p75 - p25,
                            facecolor=facecolor, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.plot([positions[i] - 0.3, positions[i] + 0.3], [p50, p50], 'r-', linewidth=2)
        ax.plot([positions[i] - 0.15, positions[i] + 0.15], [p5, p5], 'k-', linewidth=1.5)
        ax.plot([positions[i] - 0.15, positions[i] + 0.15], [p95, p95], 'k-', linewidth=1.5)


# Swing time by count + situation: stacked figure only (full counts on top, buckets below)
COUNT_ORDER = ['0-0', '0-1', '0-2', '1-0', '1-1', '1-2', '2-0', '2-1', '2-2', '3-0', '3-1', '3-2']
df_count = df_clean.dropna(subset=['Count'])
count_means = []
for c in df_count['Count'].unique():
    c_data = df_count[df_count['Count'] == c]['SwingTime'].values
    if len(c_data) > 0:
        count_means.append((c, np.mean(c_data)))
count_means.sort(key=lambda x: COUNT_ORDER.index(x[0]) if x[0] in COUNT_ORDER else 999)
counts_sorted = [c[0] for c in count_means]
box_data_count = []
positions_count = []
for i, c in enumerate(counts_sorted):
    c_data = df_count[df_count['Count'] == c]['SwingTime'].values
    if len(c_data) > 0:
        box_data_count.append(c_data)
        positions_count.append(i)
dummies_count = pd.get_dummies(df_count['Count'], drop_first=True)
X_count = dummies_count.values
y_count = df_count['SwingTime'].values
r2_count = r2_score(y_count, LinearRegression().fit(X_count, y_count).predict(X_count)) if X_count.size > 0 and len(np.unique(y_count)) > 1 else 0.0

# Count buckets (1-0 and 2-0 with hitter-friendly counts)
def _count_to_situation(c):
    if pd.isna(c):
        return None
    s = str(c).strip()
    if s in ('3-1', '3-0', '2-1', '2-0', '1-0'):
        return 'Hitter friendly'
    if s in ('0-0', '0-1', '1-1', '3-2'):
        return 'Neutral'
    if s in ('0-2', '1-2', '2-2'):
        return 'Pitcher friendly'
    return None


SITUATION_ORDER = ['Pitcher friendly', 'Neutral', 'Hitter friendly']
df_sit = df_count.copy()
df_sit['Count situation'] = df_sit['Count'].map(_count_to_situation)
df_sit = df_sit.dropna(subset=['Count situation'])
sit_means = []
for sit in df_sit['Count situation'].unique():
    d = df_sit[df_sit['Count situation'] == sit]['SwingTime'].values
    if len(d) > 0:
        sit_means.append((sit, np.mean(d)))
sit_means.sort(key=lambda x: SITUATION_ORDER.index(x[0]) if x[0] in SITUATION_ORDER else 999)
sits_sorted = [m[0] for m in sit_means]
box_data_sit = []
positions_sit = []
for i, sit in enumerate(sits_sorted):
    d = df_sit[df_sit['Count situation'] == sit]['SwingTime'].values
    if len(d) > 0:
        box_data_sit.append(d)
        positions_sit.append(i)
dummies_sit = pd.get_dummies(df_sit['Count situation'], drop_first=True)
X_sit = dummies_sit.values
y_sit = df_sit['SwingTime'].values
r2_sit = r2_score(y_sit, LinearRegression().fit(X_sit, y_sit).predict(X_sit)) if X_sit.size > 0 and len(np.unique(y_sit)) > 1 else 0.0

# Stacked figure: full count breakdown (top) + hitter / neutral / pitcher-friendly (bottom)
fig_stack, (ax_all, ax_brk) = plt.subplots(
    2, 1, figsize=(10, 9.5), gridspec_kw={'height_ratios': [1.45, 1.15]}
)
_draw_percentile_boxplots(ax_all, positions_count, box_data_count)
_draw_percentile_boxplots(ax_brk, positions_sit, box_data_sit)
ax_all.set_xlabel('Count', fontsize=FONTSIZE)
ax_all.set_ylabel('Swing Time (s)', fontsize=FONTSIZE)
ax_all.set_title(f'Swing Time by Count (R²={r2_count:.3f})', fontsize=FONTSIZE + 2, fontweight='bold')
ax_all.set_xticks(positions_count)
ax_all.set_xticklabels(counts_sorted, fontsize=FONTSIZE - 2, rotation=45, ha='right')
ax_all.tick_params(axis='y', labelsize=FONTSIZE - 2)
ax_all.grid(True, alpha=0.3, axis='y')
ax_brk.set_xlabel('Count situation', fontsize=FONTSIZE)
ax_brk.set_ylabel('Swing Time (s)', fontsize=FONTSIZE)
ax_brk.set_title(f'Swing Time by Count Situation (R²={r2_sit:.3f})', fontsize=FONTSIZE + 2, fontweight='bold')
ax_brk.set_xticks(positions_sit)
ax_brk.set_xticklabels(sits_sorted, fontsize=FONTSIZE - 2, ha='center')
ax_brk.tick_params(axis='y', labelsize=FONTSIZE - 2)
ax_brk.grid(True, alpha=0.3, axis='y')
ylim_edges = []
for d in box_data_count + box_data_sit:
    ylim_edges.extend(np.percentile(d, [5, 95]).tolist())
y_lo, y_hi = min(ylim_edges), max(ylim_edges)
y_pad = (y_hi - y_lo) * 0.06 if y_hi > y_lo else 0.01
span = (y_hi - y_lo) if y_hi > y_lo else 0.01
# Extra bottom range so n= can sit low without sitting under whiskers / clipping
n_bottom_band = 0.07 * span
ax_all.set_ylim(y_lo - y_pad - n_bottom_band, y_hi + y_pad)
ax_brk.set_ylim(y_lo - y_pad - n_bottom_band, y_hi + y_pad)
annotate_box_column_sample_counts(
    ax_all,
    positions_count,
    box_data_count,
    FONTSIZE - 3,
    pad_data_bottom_frac=0,
    y_span_frac_from_bottom=0.028,
)
annotate_box_column_sample_counts(
    ax_brk,
    positions_sit,
    box_data_sit,
    FONTSIZE - 3,
    pad_data_bottom_frac=0,
    y_span_frac_from_bottom=0.028,
)
# Bottom margin: room for rotated count labels, bottom xlabel, and figure-level n key (no clipping)
fig_stack.subplots_adjust(left=0.09, right=0.96, top=0.95, bottom=0.10, hspace=0.35)
_n_key_stack = add_figure_n_key(fig_stack, fontsize=FONTSIZE - 3, y_fig=0.035)
plt.savefig(
    os.path.join(STARTER_DATA_FIGURES_DIR, 'SwingTimeByCount_stacked_olddata.png'),
    dpi=300,
    bbox_inches='tight',
    bbox_extra_artists=[_n_key_stack],
    pad_inches=0.12,
)
print("\nFigure saved as 'Starter Data Figures/SwingTimeByCount_stacked_olddata.png' (counts + situation)")
plt.close(fig_stack)

# --- 3-panel figure: Pitch Speed, Pitch Type, Pitch Location (stacked vertically) ---
fig, axes = plt.subplots(3, 1, figsize=(10, 14))
ax1, ax2, ax3 = axes[0], axes[1], axes[2]  # Speed, Type, Location

# PANEL 1: SwingTime vs Pitch Speed
# Remove rows with missing pitch speed for this analysis
df_speed = df_clean.dropna(subset=['PitchSpeed_numeric'])

# Plot horizontal lines first (behind markers), then scatter points
# First pass: plot all horizontal lines
for hitter in hitters_sorted:
    hitter_data = df_speed[df_speed['Hitter'] == hitter]
    if len(hitter_data) > 0:
        mean_swing_time = hitter_data['SwingTime'].mean()
        min_speed = hitter_data['PitchSpeed_numeric'].min()
        max_speed = hitter_data['PitchSpeed_numeric'].max()
        ax1.plot([min_speed, max_speed], [mean_swing_time, mean_swing_time],
                 color=hitter_color_map[hitter], linestyle=hitter_linestyle_map[hitter],
                 linewidth=1.2, alpha=0.4, zorder=1)

# Second pass: plot scatter points on top
for hitter in hitters_sorted:
    hitter_data = df_speed[df_speed['Hitter'] == hitter]
    ax1.scatter(hitter_data['PitchSpeed_numeric'], hitter_data['SwingTime'],
                color=hitter_color_map[hitter], marker=hitter_marker_map[hitter],
                label=hitter, alpha=0.8, s=80, edgecolors='black', linewidth=0.5, zorder=2)

# Calculate R2
X = df_speed['PitchSpeed_numeric'].values.reshape(-1, 1)
y = df_speed['SwingTime'].values
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# Draw best fit line if R2 > 0.2
if r2 > 0.2:
    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))
    ax1.plot(x_range, y_range, 'k--', linewidth=2, label=f'Best Fit (R²={r2:.3f})')

# Customize left subplot
ax1.set_xlabel('Pitch Speed (mph)', fontsize=FONTSIZE)
ax1.set_ylabel('Swing Time (s)', fontsize=FONTSIZE)
ax1.set_title(f'Swing Time vs Pitch Speed (R²={r2:.3f})', fontsize=FONTSIZE+2, fontweight='bold')
ax1.tick_params(axis='both', labelsize=FONTSIZE-2)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=FONTSIZE-4, loc='best', ncol=2)

# PANEL 2: SwingTime by Pitch Type
df_pitch_type = df_clean.dropna(subset=['Pitch Type'])

# Calculate mean swing time for each pitch type and sort
pitch_type_means = []
for pt in df_pitch_type['Pitch Type'].unique():
    pt_data = df_pitch_type[df_pitch_type['Pitch Type'] == pt]['SwingTime'].values
    if len(pt_data) > 0:
        pitch_type_means.append((pt, np.mean(pt_data)))

# Sort pitch types by fixed order: Fastball, Curveball, Slider, Changeup, Cutter
PITCH_TYPE_ORDER = ['Fastball', 'Curveball', 'Slider', 'Changeup', 'Cutter']
pitch_type_means.sort(key=lambda x: PITCH_TYPE_ORDER.index(x[0]) if x[0] in PITCH_TYPE_ORDER else 999)
pitch_types_sorted = [pt[0] for pt in pitch_type_means]

# Prepare data for box plot
box_data_pt = []
positions_pt = []
for i, pt in enumerate(pitch_types_sorted):
    pt_data = df_pitch_type[df_pitch_type['Pitch Type'] == pt]['SwingTime'].values
    if len(pt_data) > 0:
        box_data_pt.append(pt_data)
        positions_pt.append(i)

# Create custom box and whisker plot using percentiles
for i, data in enumerate(box_data_pt):
    p5 = np.percentile(data, 5)
    p25 = np.percentile(data, 25)
    p50 = np.percentile(data, 50)
    p75 = np.percentile(data, 75)
    p95 = np.percentile(data, 95)
    
    # Draw whiskers (5th to 95th)
    ax2.plot([positions_pt[i], positions_pt[i]], [p5, p95], 'k-', linewidth=1.5)
    
    # Draw box (25th to 75th)
    box = plt.Rectangle((positions_pt[i]-0.3, p25), 0.6, p75-p25, 
                         facecolor='lightcoral', edgecolor='black', linewidth=1.5)
    ax2.add_patch(box)
    
    # Draw median line
    ax2.plot([positions_pt[i]-0.3, positions_pt[i]+0.3], [p50, p50], 'r-', linewidth=2)
    
    # Draw whisker caps
    ax2.plot([positions_pt[i]-0.15, positions_pt[i]+0.15], [p5, p5], 'k-', linewidth=1.5)
    ax2.plot([positions_pt[i]-0.15, positions_pt[i]+0.15], [p95, p95], 'k-', linewidth=1.5)

# R² for Pitch Type
df_pt = df_pitch_type.copy()
dummies_pt = pd.get_dummies(df_pt['Pitch Type'], drop_first=True)
X_pt = dummies_pt.values
y_pt = df_pt['SwingTime'].values
r2_pt = r2_score(y_pt, LinearRegression().fit(X_pt, y_pt).predict(X_pt)) if X_pt.size > 0 and len(np.unique(y_pt)) > 1 else 0.0

# Customize subplot (Pitch Type)
ax2.set_xlabel('Pitch Type', fontsize=FONTSIZE)
ax2.set_ylabel('Swing Time (s)', fontsize=FONTSIZE)
ax2.set_title(f'Swing Time by Pitch Type (R²={r2_pt:.3f})', fontsize=FONTSIZE+2, fontweight='bold')
ax2.set_xticks(positions_pt)
ax2.set_xticklabels(pitch_types_sorted, fontsize=FONTSIZE - 2, rotation=45, ha='right')
ax2.tick_params(axis='y', labelsize=FONTSIZE-2)
ax2.grid(True, alpha=0.3, axis='y')
annotate_box_column_sample_counts(ax2, positions_pt, box_data_pt, FONTSIZE - 3)

# PANEL 3: SwingTime by Location Category
df_loc = df_clean.dropna(subset=['LocationCategory'])
category_means = []
for cat in df_loc['LocationCategory'].unique():
    cat_data = df_loc[df_loc['LocationCategory'] == cat]['SwingTime'].values
    if len(cat_data) > 0:
        category_means.append((cat, np.mean(cat_data)))
LOCATION_ORDER = ['High', 'Low', 'Middle', 'In', 'Away']
category_means.sort(key=lambda x: LOCATION_ORDER.index(x[0]) if x[0] in LOCATION_ORDER else 999)
categories_sorted = [cat[0] for cat in category_means]
box_data = []
positions = []
for i, cat in enumerate(categories_sorted):
    cat_data = df_loc[df_loc['LocationCategory'] == cat]['SwingTime'].values
    if len(cat_data) > 0:
        box_data.append(cat_data)
        positions.append(i)
for i, data in enumerate(box_data):
    p5 = np.percentile(data, 5)
    p25 = np.percentile(data, 25)
    p50 = np.percentile(data, 50)
    p75 = np.percentile(data, 75)
    p95 = np.percentile(data, 95)
    ax3.plot([positions[i], positions[i]], [p5, p95], 'k-', linewidth=1.5)
    box = plt.Rectangle((positions[i]-0.3, p25), 0.6, p75-p25, facecolor='lightblue', edgecolor='black', linewidth=1.5)
    ax3.add_patch(box)
    ax3.plot([positions[i]-0.3, positions[i]+0.3], [p50, p50], 'r-', linewidth=2)
    ax3.plot([positions[i]-0.15, positions[i]+0.15], [p5, p5], 'k-', linewidth=1.5)
    ax3.plot([positions[i]-0.15, positions[i]+0.15], [p95, p95], 'k-', linewidth=1.5)
dummies_loc = pd.get_dummies(df_loc['LocationCategory'], drop_first=True)
X_loc = dummies_loc.values
y_loc = df_loc['SwingTime'].values
r2_loc = r2_score(y_loc, LinearRegression().fit(X_loc, y_loc).predict(X_loc)) if X_loc.size > 0 and len(np.unique(y_loc)) > 1 else 0.0
ax3.set_xlabel('Pitch Location Category', fontsize=FONTSIZE)
ax3.set_ylabel('Swing Time (s)', fontsize=FONTSIZE)
ax3.set_title(f'Swing Time by Pitch Location (R²={r2_loc:.3f})', fontsize=FONTSIZE+2, fontweight='bold')
ax3.set_xticks(positions)
ax3.set_xticklabels(categories_sorted, fontsize=FONTSIZE - 2)
ax3.tick_params(axis='y', labelsize=FONTSIZE-2)
ax3.grid(True, alpha=0.3, axis='y')
annotate_box_column_sample_counts(ax3, positions, box_data, FONTSIZE - 3)

plt.tight_layout(rect=(0, 0.02, 1, 1))
# Nudge only the top panel down to close the gap with the middle panel
pos0 = axes[0].get_position()
axes[0].set_position([pos0.x0, pos0.y0 - 0.03, pos0.width, pos0.height])
_n_key_pc = add_figure_n_key(fig, fontsize=FONTSIZE - 3)
plt.savefig(os.path.join(STARTER_DATA_FIGURES_DIR, 'PitchCharacteristics_olddata.png'), dpi=300, bbox_inches='tight', bbox_extra_artists=[_n_key_pc], pad_inches=0.1)
print("\nFigure saved as 'Starter Data Figures/PitchCharacteristics_olddata.png'")
plt.close(fig)

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "MLB Data for PCA"

# Load the data
df = pd.read_csv(DATA_DIR / "bat-tracking.csv")

# Extract the features (bat speed and swing length)
X = df[['avg_bat_speed', 'swing_length']].values

# Standardize the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Get principal components
components = pca.components_
explained_variance = pca.explained_variance_ratio_

print("PCA Results:")
print(f"Explained variance ratio: {explained_variance}")
print(f"Principal components:\n{components}")
print(f"\nFirst PC explains {explained_variance[0]*100:.2f}% of variance")
print(f"Second PC explains {explained_variance[1]*100:.2f}% of variance")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Original data scatter plot
axes[0, 0].scatter(X[:, 0], X[:, 1], alpha=0.6, s=50)
axes[0, 0].set_xlabel('Average Bat Speed', fontsize=12)
axes[0, 0].set_ylabel('Average Swing Length', fontsize=12)
axes[0, 0].set_title('Original Data', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 2. Standardized data with principal components
mean = np.mean(X_scaled, axis=0)
axes[0, 1].scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.6, s=50, label='Data points')
# Draw principal components
for i, (comp, var) in enumerate(zip(components, explained_variance)):
    axes[0, 1].arrow(mean[0], mean[1], comp[0]*3*np.sqrt(var), comp[1]*3*np.sqrt(var),
                     head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2,
                     label=f'PC{i+1} ({var*100:.1f}% variance)')
axes[0, 1].set_xlabel('Standardized Average Bat Speed', fontsize=12)
axes[0, 1].set_ylabel('Standardized Average Swing Length', fontsize=12)
axes[0, 1].set_title('Standardized Data with Principal Components', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_aspect('equal')

# 3. PCA transformed data
axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=50)
axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)', fontsize=12)
axes[1, 0].set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)', fontsize=12)
axes[1, 0].set_title('PCA Transformed Data', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 4. Ratio of standardized bat speed to swing length in each PC
axes[1, 1].axis('off')
# Component loadings: PC1 and PC2 coefficients for bat speed and swing length
c1_bs, c1_sl = components[0, 0], components[0, 1]
c2_bs, c2_sl = components[1, 0], components[1, 1]
ratio_pc1 = c1_bs / c1_sl if c1_sl != 0 else np.inf
# Format ratios like 1.00/-1.00 as 1/-1 for readability
def _fmt_ratio(v: float) -> str:
    if not np.isfinite(v):
        return "—"
    r = int(np.round(v))
    if abs(v - r) < 0.005:
        return str(r)
    return f"{v:.2f}"

# PC2 ratio text: opposite signs mean trade-off
if abs(c2_sl) < 1e-10:
    ratio_pc2_str = "— (swing length ≈ 0)"
else:
    ratio_pc2_str = f"1 : {_fmt_ratio(c2_sl/c2_bs)}" if abs(c2_bs) >= 1e-10 else "—"
text = (
    "Ratio of standardized average bat speed to swing length\n"
    "in each principal component (loadings):\n\n"
    f"PC1 ({explained_variance[0]*100:.1f}% variance):\n"
    f"  Bat speed loading: {c1_bs:.4f}\n"
    f"  Swing length loading: {c1_sl:.4f}\n"
    f"  Ratio (bat speed : swing length) ≈ 1 : {_fmt_ratio(c1_sl/c1_bs)}\n"
    "  → PC1 weights both equally; higher PC1 = higher\n"
    "    combined bat speed and swing length.\n\n"
    f"PC2 ({explained_variance[1]*100:.1f}% variance):\n"
    f"  Bat speed loading: {c2_bs:.4f}\n"
    f"  Swing length loading: {c2_sl:.4f}\n"
    f"  Ratio (bat speed : swing length) ≈ {ratio_pc2_str}\n"
    "  → PC2 captures the trade-off: positive = higher speed,\n"
    "    shorter swing; negative = lower speed, longer swing."
)
axes[1, 1].text(0.5, 0.5, text, transform=axes[1, 1].transAxes,
                fontsize=10, verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                family='monospace')
axes[1, 1].set_title('Standardized Bat Speed vs Swing Length in PCs', fontsize=14, fontweight='bold')

plt.tight_layout()
out_png = SCRIPT_DIR / "pca_results.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"\nVisualization saved as '{out_png}'")

# Print detailed component information
print("\n" + "="*60)
print("Principal Component Details:")
print("="*60)
print(f"\nPC1 (explains {explained_variance[0]*100:.2f}% of variance):")
print(f"  Bat Speed coefficient: {components[0, 0]:.4f}")
print(f"  Swing Length coefficient: {components[0, 1]:.4f}")
print(f"\nPC2 (explains {explained_variance[1]*100:.2f}% of variance):")
print(f"  Bat Speed coefficient: {components[1, 0]:.4f}")
print(f"  Swing Length coefficient: {components[1, 1]:.4f}")

# Calculate conversion factors
bat_speed_mean = scaler.mean_[0]
bat_speed_std = scaler.scale_[0]
swing_length_mean = scaler.mean_[1]
swing_length_std = scaler.scale_[1]

print("\n" + "="*60)
print("Standardization Conversion Factors:")
print("="*60)
print(f"\nBat Speed:")
print(f"  Mean: {bat_speed_mean:.4f} mph")
print(f"  Standard Deviation: {bat_speed_std:.4f} mph")
print(f"  → 1 standardized unit = {bat_speed_std:.4f} mph")
print(f"  → 1 standardized unit = {bat_speed_std:.2f} mph (rounded)")

print(f"\nSwing Length:")
print(f"  Mean: {swing_length_mean:.4f} feet")
print(f"  Standard Deviation: {swing_length_std:.4f} feet")
print(f"  → 1 standardized unit = {swing_length_std:.4f} feet")
print(f"  → 1 standardized unit = {swing_length_std:.2f} feet (rounded)")

print("\n" + "="*60)
print("Example Conversions:")
print("="*60)
print(f"If a player has standardized bat speed = +1.0:")
print(f"  Original bat speed = {bat_speed_mean:.2f} + (1.0 × {bat_speed_std:.2f}) = {bat_speed_mean + bat_speed_std:.2f} mph")
print(f"  (This player is 1 standard deviation above average)")

print(f"\nIf a player has standardized swing length = +1.0:")
print(f"  Original swing length = {swing_length_mean:.2f} + (1.0 × {swing_length_std:.2f}) = {swing_length_mean + swing_length_std:.2f} feet")
print(f"  (This player is 1 standard deviation above average)")

plt.show()


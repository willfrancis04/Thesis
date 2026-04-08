from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "MLB Data for PCA"

# Load the data
df = pd.read_csv(DATA_DIR / "Thesis Data Collection - MLB Raw Data.csv")

# Extract the features
X = df[['Bat Speed (MPH)', 'Swing Length (ft)', 'Clicks - Hands Move']].values

# Remove any rows with missing values
mask = ~np.isnan(X).any(axis=1)
X_clean = X[mask]
swing_time = X_clean[:, 2]  # Swing time (Clicks - Hands Move)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Get principal components
components = pca.components_
explained_variance = pca.explained_variance_ratio_

# Extract PC1 and PC2 values
PC1_values = X_pca[:, 0]
PC2_values = X_pca[:, 1]

# Calculate correlations
corr_PC1_swing_time = np.corrcoef(PC1_values, swing_time)[0, 1]
corr_PC2_swing_time = np.corrcoef(PC2_values, swing_time)[0, 1]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Swing Time vs PC1
ax1 = axes[0, 0]
scatter1 = ax1.scatter(swing_time, PC1_values, alpha=0.6, s=60, c=swing_time, cmap='viridis')
# Add trend line
z1 = np.polyfit(swing_time, PC1_values, 1)
p1 = np.poly1d(z1)
ax1.plot(swing_time, p1(swing_time), "r--", alpha=0.8, linewidth=2, label=f'Trend (r={corr_PC1_swing_time:.3f})')
ax1.set_xlabel('Swing Time (Clicks - Hands Move)', fontsize=12, fontweight='bold')
ax1.set_ylabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)', fontsize=12, fontweight='bold')
ax1.set_title('Swing Time Effect on PC1', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.2)
ax1.axvline(x=np.mean(swing_time), color='k', linestyle='-', alpha=0.2)
plt.colorbar(scatter1, ax=ax1, label='Swing Time')

# 2. Swing Time vs PC2
ax2 = axes[0, 1]
scatter2 = ax2.scatter(swing_time, PC2_values, alpha=0.6, s=60, c=swing_time, cmap='plasma')
# Add trend line
z2 = np.polyfit(swing_time, PC2_values, 1)
p2 = np.poly1d(z2)
ax2.plot(swing_time, p2(swing_time), "r--", alpha=0.8, linewidth=2, label=f'Trend (r={corr_PC2_swing_time:.3f})')
ax2.set_xlabel('Swing Time (Clicks - Hands Move)', fontsize=12, fontweight='bold')
ax2.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)', fontsize=12, fontweight='bold')
ax2.set_title('Swing Time Effect on PC2', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
ax2.axvline(x=np.mean(swing_time), color='k', linestyle='-', alpha=0.2)
plt.colorbar(scatter2, ax=ax2, label='Swing Time')

# 3. PC1 vs PC2 colored by Swing Time
ax3 = axes[1, 0]
scatter3 = ax3.scatter(PC1_values, PC2_values, c=swing_time, cmap='coolwarm', alpha=0.6, s=60)
ax3.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)', fontsize=12, fontweight='bold')
ax3.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)', fontsize=12, fontweight='bold')
ax3.set_title('PC1 vs PC2 (Colored by Swing Time)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax3.axvline(x=0, color='k', linestyle='--', alpha=0.3)
cbar3 = plt.colorbar(scatter3, ax=ax3)
cbar3.set_label('Swing Time (Clicks - Hands Move)', fontsize=10)

# 4. Summary statistics and coefficients
ax4 = axes[1, 1]
ax4.axis('off')

# Create text summary
summary_text = f"""
SWING TIME EFFECT ON PRINCIPAL COMPONENTS
{'='*50}

PC1 COEFFICIENT: {components[0, 2]:.4f}
  • Small negative contribution
  • Correlation with swing time: {corr_PC1_swing_time:.4f}
  • Swing time explains minimal variance in PC1

PC2 COEFFICIENT: {components[1, 2]:.4f}
  • Strong positive contribution (largest in PC2)
  • Correlation with swing time: {corr_PC2_swing_time:.4f}
  • Swing time is the PRIMARY driver of PC2

INTERPRETATION:
  • PC1 = Overall power (speed + length)
    → Swing time has minimal effect
    
  • PC2 = Swing time dimension
    → Higher PC2 = Longer swing time
    → Captures speed vs. time trade-off

VARIANCE EXPLAINED:
  • PC1: {explained_variance[0]*100:.2f}%
  • PC2: {explained_variance[1]*100:.2f}%
  • Combined: {(explained_variance[0] + explained_variance[1])*100:.2f}%
"""

ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
out_png = SCRIPT_DIR / "swing_time_effect.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
print(f"Visualization saved as '{out_png}'")

# Print correlation details
print("\n" + "="*60)
print("Swing Time Correlations:")
print("="*60)
print(f"PC1 vs Swing Time correlation: {corr_PC1_swing_time:.4f}")
print(f"PC2 vs Swing Time correlation: {corr_PC2_swing_time:.4f}")
print(f"\nPC1 coefficient for swing time: {components[0, 2]:.4f}")
print(f"PC2 coefficient for swing time: {components[1, 2]:.4f}")

plt.show()


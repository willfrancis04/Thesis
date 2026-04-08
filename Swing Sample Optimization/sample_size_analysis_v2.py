import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp

BASE_DIR = Path(__file__).resolve().parent
CALC_DIR = BASE_DIR / "calculations"
THESIS_ROOT = BASE_DIR.parent
DATA_CSV = THESIS_ROOT / "Starter Data" / "Thesis Data Collection - Sheet3.csv"

# Load the data
df = pd.read_csv(DATA_CSV)

# Extract swing time data (Clicks - Hands Move column)
print("Data loaded successfully!")
print(f"Total rows: {len(df)}")
print(f"\nPlayers and their data counts:")
print(df['Hitter'].value_counts())

# Get swing times for each player
players = df['Hitter'].unique()
player_data = {}

for player in players:
    player_swing_times = df[df['Hitter'] == player]['Clicks - Hands Move'].values
    player_data[player] = player_swing_times
    print(f"\n{player}: {len(player_swing_times)} data points")
    print(f"  Mean: {np.mean(player_swing_times):.2f}")
    print(f"  Std: {np.std(player_swing_times):.2f}")
    print(f"  Range: [{np.min(player_swing_times):.2f}, {np.max(player_swing_times):.2f}]")

def calculate_key_metrics(data):
    """Calculate key distribution metrics for swing time analysis"""
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'median': np.median(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25)
    }

def compare_distributions_ks(original_data, sample_data):
    """Use Kolmogorov-Smirnov test to compare distributions"""
    ks_stat, p_value = ks_2samp(original_data, sample_data)
    # Lower KS statistic means more similar distributions
    # We want KS statistic < 0.1 for 90% similarity (roughly)
    similarity = 1 - min(ks_stat, 1.0)
    return similarity, ks_stat, p_value

def compare_metrics_accuracy(original_metrics, sample_metrics):
    """Compare key metrics and return accuracy percentage"""
    # Focus on the most important metrics: mean, std, median, IQR
    key_metrics = ['mean', 'std', 'median', 'iqr']
    
    errors = {}
    for key in key_metrics:
        if original_metrics[key] != 0:
            relative_error = abs(sample_metrics[key] - original_metrics[key]) / abs(original_metrics[key])
            errors[key] = relative_error
        else:
            errors[key] = abs(sample_metrics[key] - original_metrics[key])
    
    # Calculate accuracy (1 - average relative error)
    avg_error = np.mean(list(errors.values()))
    accuracy = max(0, 1 - avg_error)
    
    return accuracy, errors

def bootstrap_sample_size_analysis(player_name, swing_times, min_sample_size=5, max_sample_size=None, 
                                   n_iterations=1000, target_accuracy=0.90):
    """
    Determine minimum sample size needed to retain target_accuracy of distribution
    Uses both KS test and metric comparison
    """
    if max_sample_size is None:
        max_sample_size = len(swing_times) - 1
    
    # Calculate original distribution metrics
    original_metrics = calculate_key_metrics(swing_times)
    
    # Test different sample sizes
    sample_sizes = range(min_sample_size, max_sample_size + 1)
    results = []
    
    for sample_size in sample_sizes:
        metric_accuracies = []
        ks_similarities = []
        
        # Bootstrap: sample multiple times and check accuracy
        for _ in range(n_iterations):
            # Random sample without replacement
            sample = np.random.choice(swing_times, size=sample_size, replace=False)
            sample_metrics = calculate_key_metrics(sample)
            
            # Metric-based accuracy
            metric_accuracy, _ = compare_metrics_accuracy(original_metrics, sample_metrics)
            metric_accuracies.append(metric_accuracy)
            
            # KS test similarity
            ks_sim, _, _ = compare_distributions_ks(swing_times, sample)
            ks_similarities.append(ks_sim)
        
        # Combined accuracy: average of metric accuracy and KS similarity
        combined_accuracies = [(m + k) / 2 for m, k in zip(metric_accuracies, ks_similarities)]
        
        mean_accuracy = np.mean(combined_accuracies)
        std_accuracy = np.std(combined_accuracies)
        min_accuracy = np.min(combined_accuracies)
        p90_accuracy = np.percentile(combined_accuracies, 90)
        p95_accuracy = np.percentile(combined_accuracies, 95)
        
        mean_metric_acc = np.mean(metric_accuracies)
        mean_ks_sim = np.mean(ks_similarities)
        
        results.append({
            'sample_size': sample_size,
            'mean_accuracy': mean_accuracy,
            'mean_metric_accuracy': mean_metric_acc,
            'mean_ks_similarity': mean_ks_sim,
            'std_accuracy': std_accuracy,
            'min_accuracy': min_accuracy,
            'p90_accuracy': p90_accuracy,
            'p95_accuracy': p95_accuracy
        })
        
        # Early stopping if we consistently meet target
        if mean_accuracy >= target_accuracy and p90_accuracy >= target_accuracy:
            print(f"  Sample size {sample_size}: Combined accuracy = {mean_accuracy:.3f} "
                  f"(Metric: {mean_metric_acc:.3f}, KS: {mean_ks_sim:.3f}), P90 = {p90_accuracy:.3f}")
    
    results_df = pd.DataFrame(results)
    
    # Find minimum sample size that meets target
    # Require both mean and P90 to meet target
    qualifying_sizes = results_df[
        (results_df['mean_accuracy'] >= target_accuracy) & 
        (results_df['p90_accuracy'] >= target_accuracy)
    ]
    
    if len(qualifying_sizes) > 0:
        min_required_size = qualifying_sizes['sample_size'].min()
    else:
        # Try with just mean accuracy
        qualifying_sizes = results_df[results_df['mean_accuracy'] >= target_accuracy]
        if len(qualifying_sizes) > 0:
            min_required_size = qualifying_sizes['sample_size'].min()
        else:
            min_required_size = None
    
    return results_df, min_required_size, original_metrics

# Analyze each player
print("\n" + "="*80)
print("SAMPLE SIZE ANALYSIS (Improved Method)")
print("="*80)
print("Using combined metric: (Metric Accuracy + KS Similarity) / 2")
print("="*80)

all_results = {}

for player in players:
    print(f"\n{'='*80}")
    print(f"Analyzing: {player}")
    print(f"{'='*80}")
    
    swing_times = player_data[player]
    results_df, min_size, original_metrics = bootstrap_sample_size_analysis(
        player, swing_times, min_sample_size=5, n_iterations=1000, target_accuracy=0.90
    )
    
    all_results[player] = {
        'results': results_df,
        'min_size': min_size,
        'original_metrics': original_metrics,
        'total_data_points': len(swing_times)
    }
    
    if min_size:
        print(f"\n✓ Minimum sample size for 90% accuracy: {min_size} data points")
        print(f"  (Reduction from {len(swing_times)} to {min_size} = {len(swing_times) - min_size} fewer data points)")
        print(f"  (Percentage reduction: {(1 - min_size/len(swing_times))*100:.1f}%)")
        
        # Show the accuracy at that sample size
        result_at_min = results_df[results_df['sample_size'] == min_size].iloc[0]
        print(f"  Accuracy at {min_size} samples: Mean = {result_at_min['mean_accuracy']:.3f}, "
              f"P90 = {result_at_min['p90_accuracy']:.3f}")
    else:
        print(f"\n✗ Could not achieve 90% accuracy with available sample sizes")
        best_result = results_df.loc[results_df['mean_accuracy'].idxmax()]
        print(f"  Best accuracy achieved: {best_result['mean_accuracy']:.3f} at sample size {best_result['sample_size']}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n{'Player':<30} {'Total Data':<12} {'Min Sample':<12} {'Reduction':<12} {'% Reduction':<12}")
print("-"*80)

for player in players:
    result = all_results[player]
    total = result['total_data_points']
    min_size = result['min_size']
    
    if min_size:
        reduction = total - min_size
        pct_reduction = (1 - min_size/total) * 100
        print(f"{player:<30} {total:<12} {min_size:<12} {reduction:<12} {pct_reduction:<12.1f}%")
    else:
        print(f"{player:<30} {total:<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

# Overall recommendation
print("\n" + "="*80)
print("OVERALL RECOMMENDATION")
print("="*80)

min_sizes = [all_results[p]['min_size'] for p in players if all_results[p]['min_size'] is not None]
if min_sizes:
    overall_min = max(min_sizes)  # Use the maximum to be conservative
    avg_min = np.mean(min_sizes)
    print(f"\nConservative estimate (max of all players): {overall_min} data points per player")
    print(f"Average across players: {avg_min:.1f} data points per player")
    print(f"\nConservative reduction: from 30 to {overall_min} = {30 - overall_min} fewer data points")
    print(f"Percentage reduction: {(1 - overall_min/30)*100:.1f}%")
    
    if overall_min < 30:
        print(f"\n✓ You can reduce data collection by {30 - overall_min} data points per player")
        print(f"  while maintaining 90% accuracy of the swing time distribution.")
else:
    print("\n⚠ Could not determine minimum sample size for all players.")
    print("  Consider collecting more data or adjusting the accuracy threshold.")

# Save detailed results
print("\n" + "="*80)
print("Saving detailed results to CSV files...")
print("="*80)

CALC_DIR.mkdir(parents=True, exist_ok=True)
for player in players:
    result = all_results[player]
    filename = f"sample_size_results_v2_{player.replace(', ', '_').replace(' ', '_')}.csv"
    result["results"].to_csv(CALC_DIR / filename, index=False)
    print(f"Saved: calculations/{filename}")

print("\nAnalysis complete!")


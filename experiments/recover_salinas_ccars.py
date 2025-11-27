"""
Recovery Script for Salinas CCARS Experiment

This script reconstructs the CCARS results from the partial files saved during
the previous run, avoiding the need to re-run the expensive 50 Monte Carlo iterations.
It then proceeds to the evaluation and visualization steps that failed previously.
"""

import sys
import os
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyperspectral_loader import load_salinas, SALINAS_CLASSES
from nicola_ccars_hyperspectral import HyperspectralCCARS
from CARS_analysis.multiclass_extension import (
    plot_multiclass_confusion_matrix,
    print_classification_report
)

# Configuration
N_COMPONENTS = 4
N_RUNS = 50
N_ITERATIONS = 100
CALIBRATION = True

print("="*80)
print("RECOVERING SALINAS CCARS EXPERIMENT RESULTS")
print("="*80)

# ============================================================================
# STEP 1: Load Dataset (Needed for evaluation)
# ============================================================================
print("\nLoading Salinas Dataset...")
df_salinas, X_salinas, y_salinas = load_salinas(verbose=False)

# SNV Normalization
print("Applying SNV normalization...")
X_normalized = np.zeros_like(X_salinas, dtype=float)
for i in range(X_salinas.shape[0]):
    mean = np.mean(X_salinas[i, :])
    std = np.std(X_salinas[i, :])
    if std > 0:
        X_normalized[i, :] = (X_salinas[i, :] - mean) / std
    else:
        X_normalized[i, :] = X_salinas[i, :] - mean

# ============================================================================
# STEP 2: Initialize CCARS and Reconstruct State
# ============================================================================
output_dir = "experiments/salinas_ccars_results"
print(f"\nReconstructing state from {output_dir}...")

ccars = HyperspectralCCARS(
    output_path=output_dir,
    n_components=N_COMPONENTS,
    cv_folds=5,
    test_percentage=0.3,
    calibration=CALIBRATION
)

# Manually trigger the split to set up X_train, X_test, etc.
# We need to replicate the exact split logic from fit()
from sklearn.model_selection import train_test_split

ccars.wavelengths = np.arange(X_salinas.shape[1])
ccars.class_names = list(SALINAS_CLASSES.values())
ccars.n_features = X_salinas.shape[1]

# Replicate split logic
if CALIBRATION:
    X_calib, X_valid, y_calib, y_valid = train_test_split(
        X_normalized, y_salinas, test_size=0.5, random_state=42, stratify=y_salinas
    )
else:
    X_calib, y_calib = X_normalized, y_salinas

X_train, X_test, y_train, y_test = train_test_split(
    X_calib, y_calib, test_size=0.3, 
    random_state=42, stratify=y_calib
)

ccars.X_train, ccars.y_train = X_train, y_train
ccars.X_test, ccars.y_test = X_test, y_test

# Reconstruct variable frequencies from partial coefficient files
print("Reading partial coefficient files...")
coef_files = glob.glob(f"{output_dir}/coefficients/coefficients_*.csv")
print(f"Found {len(coef_files)} coefficient files")

variable_frequencies = np.zeros(ccars.n_features)
all_stats = []
all_coefs = []

for file_path in tqdm(coef_files):
    # Read coefficients
    df = pd.read_csv(file_path)
    all_coefs.append(df)
    
    # Get variables selected in the final iteration (100)
    final_vars = df[df['Iteration'] == N_ITERATIONS]['Wavelength_Index'].values
    
    # Update frequencies
    for var_idx in final_vars:
        variable_frequencies[int(var_idx)] += 1

# Reconstruct statistics dataframe
stat_files = glob.glob(f"{output_dir}/statistics/statistics_*.csv")
for file_path in stat_files:
    all_stats.append(pd.read_csv(file_path))

ccars.variable_frequencies = variable_frequencies
ccars.statistics_df = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
ccars.coefficients_df = pd.concat(all_coefs, ignore_index=True) if all_coefs else pd.DataFrame()

print(f"\nReconstruction Complete!")
print(f"Most frequent variable selected: {variable_frequencies.max()}/{len(coef_files)} times")

# ============================================================================
# STEP 3: Resume Evaluation
# ============================================================================
print("\n" + "-" *80)
print("STEP 3: Evaluating Selected Wavelengths (Resumed)")
print("-"*80)

thresholds = [10, 15, 20, 25, 30]
results = []

for threshold in thresholds:
    selected_indices, selected_wavelengths = ccars.get_selected_wavelengths(threshold)
    
    if len(selected_indices) < 2:
        continue
    
    print(f"\n{'='*60}")
    print(f"Threshold: {threshold} (minimum frequency)")
    print(f"{'='*60}")
    
    # This calls the FIXED evaluate_selected_wavelengths method
    metrics, indices, clf = ccars.evaluate_selected_wavelengths(threshold, verbose=True)
    
    results.append({
        'Threshold': threshold,
        'N_Wavelengths': len(indices),
        'Reduction_%': 100 * (1 - len(indices) / X_salinas.shape[1]),
        'Accuracy': metrics['overall_accuracy'],
        'Kappa': metrics['kappa'],
        'Macro_F1': metrics['macro_f1'],
        'Weighted_F1': metrics['weighted_f1']
    })

# Save threshold comparison
results_df = pd.DataFrame(results)
results_df.to_csv(f"{output_dir}/threshold_comparison.csv", index=False)

print("\n" + "="*80)
print("THRESHOLD COMPARISON SUMMARY")
print("="*80)
print(results_df.to_string(index=False))

# ============================================================================
# STEP 4: Visualizations & Saving
# ============================================================================
print("\nGenerating visualizations...")

# Wavelength frequency plot
ccars.plot_wavelength_frequencies(
    threshold=20,
    save_path=f"{output_dir}/wavelength_frequencies.png"
)

# Convergence plot
ccars.plot_convergence(
    run_index=0,
    save_path=f"{output_dir}/convergence_analysis.png"
)

# Confusion matrix with optimal threshold
if not results_df.empty:
    optimal_threshold = results_df.loc[results_df['Macro_F1'].idxmax(), 'Threshold']
    metrics, indices, clf = ccars.evaluate_selected_wavelengths(int(optimal_threshold), verbose=False)

    class_names = [SALINAS_CLASSES.get(c, f'Class_{c}') for c in clf.classes_]
    plot_multiclass_confusion_matrix(
        metrics['confusion_matrix'],
        class_names=class_names,
        save_path=f"{output_dir}/confusion_matrix.png",
        figsize=(14, 12),
        normalize=True
    )
    
    # Save optimal wavelengths
    optimal_wavelengths_df = pd.DataFrame({
        'Band_Index': indices,
        'Band_Number': indices + 1,
        'Frequency': ccars.variable_frequencies[indices]
    })
    optimal_wavelengths_df.to_csv(f"{output_dir}/optimal_wavelengths.csv", index=False)

# Save full results
ccars.save_results()

print(f"\n✓ Recovery and completion successful! Results saved to: {output_dir}/")

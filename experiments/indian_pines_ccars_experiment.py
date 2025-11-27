"""
Nicola Dilillo's CCARS Method - Indian Pines Dataset Experiment

Applies CCARS to the Indian Pines hyperspectral dataset for crop/land cover classification.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyperspectral_loader import load_indian_pines, INDIAN_PINES_CLASSES
from nicola_ccars_hyperspectral import HyperspectralCCARS  
from CARS_analysis.multiclass_extension import (
    plot_multiclass_confusion_matrix,
    print_classification_report
)

# Nicola's configuration
N_COMPONENTS = 4
N_RUNS = 500      # Monte Carlo runs (Strictly matched to Nicola's R=500)
N_ITERATIONS = 100
SAMPLING_RATIO = 0.8
USE_ARS = True
CALIBRATION = True
APPLY_LOG = True  # Log10 transform

print("="*80)
print("NICOLA DILILLO'S CCARS METHOD - INDIAN PINES DATASET")
print("="*80)
print("\nMethod: Calibrated Competitive Adaptive Reweighted Sampling")
print(f"Configuration (Strictly matched): {N_COMPONENTS} components, {N_RUNS} runs, Log10+SNV\n")

# Load dataset
print("-"*80)
print("STEP 1: Loading Indian Pines Dataset")
print("-"*80)

df_indian, X_indian, y_indian = load_indian_pines(verbose=True)

# Preprocessing
print("\n" + "-"*80)
print("STEP 2: Preprocessing (Log10 -> SNV)")
print("-"*80)

print("Applying Log10 transform...")
X_log = np.log10(np.maximum(X_indian, 1e-6))

print("Applying SNV normalization...")
X_normalized = np.zeros_like(X_log, dtype=float)
for i in range(X_log.shape[0]):
    mean = np.mean(X_log[i, :])
    std = np.std(X_log[i, :])
    if std > 0:
        X_normalized[i, :] = (X_log[i, :] - mean) / std
    else:
        X_normalized[i, :] = X_log[i, :] - mean

print("[OK] Preprocessing complete")

# Run CCARS
print("\n" + "-"*80)
print("STEP 3: Running Nicola's CCARS")
print("-"*80)

output_dir = "experiments/indian_pines_ccars_results_full"
os.makedirs(output_dir, exist_ok=True)

ccars = HyperspectralCCARS(
    output_path=output_dir,
    n_components=N_COMPONENTS,
    test_percentage=0.3,
    calibration=CALIBRATION
)

wavelengths = np.arange(X_indian.shape[1])

ccars.fit(
    X=X_normalized,
    y=y_indian,
    wavelengths=wavelengths,
    n_runs=N_RUNS,
    n_iterations=N_ITERATIONS,
    sampling_ratio=SAMPLING_RATIO,
    use_ars=USE_ARS,
    apply_log=False, # Already applied manually above
    class_names=list(INDIAN_PINES_CLASSES.values()),
    verbose=True
)

# Evaluate with thresholds
print("\n" + "-"*80)
print("STEP 4: Wavelength Selection Evaluation")
print("-"*80)

thresholds = [10, 15, 20, 25, 30]
results = []

for threshold in thresholds:
    selected_indices, _ = ccars.get_selected_wavelengths(threshold)
    if len(selected_indices) < 2:
        continue
    
    print(f"\n{'='*60}")
    print(f"Threshold: {threshold}")
    print(f"{'='*60}")
    
    metrics, indices, clf = ccars.evaluate_selected_wavelengths(threshold, verbose=True)
    
    results.append({
        'Threshold': threshold,
        'N_Wavelengths': len(indices),
        'Reduction_%': 100 * (1 - len(indices) / X_indian.shape[1]),
        'Accuracy': metrics['overall_accuracy'],
        'Kappa': metrics['kappa'],
        'Macro_F1': metrics['macro_f1'],
        'Weighted_F1': metrics['weighted_f1']
    })

# Save results
import pandas as pd
results_df = pd.DataFrame(results)
results_df.to_csv(f"{output_dir}/threshold_comparison.csv", index=False)

print("\n" + "="*80)
print("THRESHOLD COMPARISON")
print("="*80)
print(results_df.to_string(index=False))

# ============================================================================
# STEP 5: Permutation Test (Validation)
# ============================================================================
print("\n" + "-"*80)
print("STEP 5: Permutation Test (Validation)")
print("-"*80)

if not results_df.empty:
    optimal_threshold = results_df.loc[results_df['Macro_F1'].idxmax(), 'Threshold']
    selected_indices, selected_wavelengths = ccars.get_selected_wavelengths(int(optimal_threshold))
    
    print(f"Running permutation test on {len(selected_wavelengths)} selected wavelengths...")
    
    perm_df, p_values = ccars.permutation_test(
        wavelengths=selected_wavelengths, 
        n_permutations=100, 
        verbose=True
    )

# Visualizations
print("\n" + "-"*80)
print("STEP 6: Generating Visualizations")
print("-"*80)

ccars.plot_wavelength_frequencies(threshold=20, save_path=f"{output_dir}/wavelength_frequencies.png")
ccars.plot_convergence(run_index=0, save_path=f"{output_dir}/convergence_analysis.png")

# Confusion matrix
optimal_threshold = results_df.loc[results_df['Macro_F1'].idxmax(), 'Threshold']
metrics, indices, clf = ccars.evaluate_selected_wavelengths(int(optimal_threshold), verbose=False)

class_names = [INDIAN_PINES_CLASSES.get(c, f'Class_{c}') for c in clf.classes_]
plot_multiclass_confusion_matrix(
    metrics['confusion_matrix'],
    class_names=class_names,
    save_path=f"{output_dir}/confusion_matrix.png",
    figsize=(14, 12),
    normalize=True
)

print_classification_report(ccars.y_test, clf.predict(ccars.X_test[:, indices]), class_names)

# Save results
print("\n" + "-"*80)
print("STEP 6: Saving Results")
print("-"*80)

ccars.save_results()

optimal_wavelengths_df = pd.DataFrame({
    'Band_Index': indices,
    'Band_Number': indices + 1,
    'Frequency': ccars.variable_frequencies[indices]
})
optimal_wavelengths_df.to_csv(f"{output_dir}/optimal_wavelengths.csv", index=False)

# Final summary
print("\n" + "="*80)
print("NICOLA'S CCARS - INDIAN PINES COMPLETE")
print("="*80)

best_result = results_df.loc[results_df['Macro_F1'].idxmax()]

print(f"\nOPTIMAL CONFIGURATION:")
print(f"  Selected: {int(best_result['N_Wavelengths'])} / {X_indian.shape[1]} wavelengths")
print(f"  Reduction: {best_result['Reduction_%']:.1f}%")
print(f"\nPERFORMANCE:")
print(f"  Accuracy: {best_result['Accuracy']*100:.2f}%")
print(f"  Kappa: {best_result['Kappa']:.4f}")
print(f"  Macro F1: {best_result['Macro_F1']:.4f}")

print(f"\n✓ Results saved to: {output_dir}/")
print("\n" + "="*80)

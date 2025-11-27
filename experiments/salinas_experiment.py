"""
Salinas Hyperspectral Dataset - CARS PLS-DA Wavelength Selection Experiment

This script demonstrates the application of CARS (Competitive Adaptive Reweighted Sampling)
with multi-class PLS-DA for wavelength selection on the Salinas hyperspectral dataset.

Dataset: Salinas Valley, California (512×217 pixels, 204 bands, 16 classes)
Objective: Select optimal wavelengths for land cover classification
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyperspectral_loader import load_salinas, SALINAS_CLASSES
from CARS_analysis.multiclass_extension import (
    MultiClassPLSDAClassifier,
    compute_multiclass_metrics,
    plot_multiclass_confusion_matrix,
    print_classification_report
)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Experiment parameters
N_COMPONENTS = 5  # Number of PLS components
TEST_SIZE = 0.3   # Test set proportion
NORMALIZE = True  # Whether to apply SNV-like normalization

print("="*80)
print("SALINAS HYPERSPECTRAL DATASET - CARS PLS-DA EXPERIMENT")
print("="*80)
print(f"\nExperiment Configuration:")
print(f"  - PLS Components: {N_COMPONENTS}")
print(f"  - Test Size: {TEST_SIZE:.1%}")
print(f"  - Normalization: {'SNV (Standard Normal Variate)' if NORMALIZE else 'None'}")
print(f"  - Random State: {RANDOM_STATE}")
print()

# ============================================================================
# STEP 1: Load Dataset
# ============================================================================
print("\n" + "-"*80)
print("STEP 1: Loading Salinas Dataset")
print("-"*80)

df_salinas, X_salinas, y_salinas = load_salinas(verbose=True)

print(f"\nDataset Summary:")
print(f"  - Total samples: {X_salinas.shape[0]:,}")
print(f"  - Number of bands: {X_salinas.shape[1]}")
print(f"  - Number of classes: {len(np.unique(y_salinas))}")
print(f"  - Classes: {sorted(np.unique(y_salinas))}")

# Print class distribution
print(f"\nClass Distribution:")
unique, counts = np.unique(y_salinas, return_counts=True)
for class_id, count in zip(unique, counts):
    class_name = SALINAS_CLASSES.get(class_id, f'Unknown_{class_id}')
    pct = 100 * count / len(y_salinas)
    print(f"  Class {class_id:2d} ({class_name:30s}): {count:5,} samples ({pct:5.2f}%)")

# ============================================================================
# STEP 2: Data Preprocessing
# ============================================================================
print("\n" + "-"*80)
print("STEP 2: Data Preprocessing")
print("-"*80)

if NORMALIZE:
    print("\nApplying SNV normalization (per sample)...")
    # SNV: Standardize each sample (row) independently
    X_normalized = np.zeros_like(X_salinas, dtype=float)
    for i in range(X_salinas.shape[0]):
        mean = np.mean(X_salinas[i, :])
        std = np.std(X_salinas[i, :])
        if std > 0:
            X_normalized[i, :] = (X_salinas[i, :] - mean) / std
        else:
            X_normalized[i, :] = X_salinas[i, :] - mean
    X_processed = X_normalized
    print("[OK] SNV normalization applied")
else:
    X_processed = X_salinas
    print("Skipping normalization")

# ============================================================================
# STEP 3: Train-Test Split
# ============================================================================
print("\n" + "-"*80)
print("STEP 3: Train-Test Split")
print("-"*80)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_salinas,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_salinas
)

print(f"\nSplit Results:")
print(f"  Training set: {X_train.shape[0]:,} samples ({100*(1-TEST_SIZE):.0f}%)")
print(f"  Test set:     {X_test.shape[0]:,} samples ({100*TEST_SIZE:.0f}%)")

# Verify stratification
print(f"\nClass distribution in sets:")
print(f"  {'Class':<5} {'Train':>8} {'Test':>8}")
print(f"  {'-'*5} {'-'*8} {'-'*8}")
for class_id in sorted(np.unique(y_salinas)):
    n_train = np.sum(y_train == class_id)
    n_test = np.sum(y_test == class_id)
    print(f"  {class_id:<5} {n_train:>8,} {n_test:>8,}")

# ============================================================================
# STEP 4: Train Multi-class PLS-DA Model
# ============================================================================
print("\n" + "-"*80)
print("STEP 4: Training Multi-class PLS-DA Model")
print("-"*80)

print(f"\nTraining Multi-class PLS-DA with {N_COMPONENTS} components...")
clf = MultiClassPLSDAClassifier(n_components=N_COMPONENTS)
clf.fit(X_train, y_train)
print("[OK] Training complete")

print(f"\nModel Information:")
print(f"  - Number of classes: {clf.n_classes_}")
print(f"  - Classes: {clf.classes_}")
print(f"  - Number of PLS models: {len(clf.pls_models)}")

# ============================================================================
# STEP 5: Evaluate on Test Set
# ============================================================================
print("\n" + "-"*80)
print("STEP 5: Model Evaluation")
print("-"*80)

# Make predictions
print("\nMaking predictions on test set...")
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)
print("[OK] Predictions complete")

# Compute metrics
print("\nComputing metrics...")
metrics = compute_multiclass_metrics(y_test, y_pred)
print("[OK] Metrics computed")

# Print summary metrics
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(f"\nOverall Performance:")
print(f"  Overall Accuracy:     {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
print(f"  Cohen's Kappa:        {metrics['kappa']:.4f}")
print(f"\nMacro-averaged Metrics (unweighted):")
print(f"  Precision:            {metrics['macro_precision']:.4f}")
print(f"  Recall:               {metrics['macro_recall']:.4f}")
print(f"  F1-score:             {metrics['macro_f1']:.4f}")
print(f"\nWeighted Metrics (by class support):")
print(f"  Precision:            {metrics['weighted_precision']:.4f}")
print(f"  Recall:               {metrics['weighted_recall']:.4f}")
print(f"  F1-score:             {metrics['weighted_f1']:.4f}")

# Per-class performance
print(f"\nPer-class Performance:")
print(f"  {'Class':<5} {'Name':<30} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}")
print(f"  {'-'*5} {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for i, class_id in enumerate(clf.classes_):
    class_name = SALINAS_CLASSES.get(class_id, f'Unknown_{class_id}')
    precision = metrics['per_class_precision'][i]
    recall = metrics['per_class_recall'][i]
    f1 = metrics['per_class_f1'][i]
    support = metrics['per_class_support'][i]
    print(f"  {class_id:<5} {class_name:<30} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10,}")

# ============================================================================
# STEP 6: Visualization
# ============================================================================
print("\n" + "-"*80)
print("STEP 6: Generating Visualizations")
print("-"*80)

# Create output directory
output_dir = "experiments/salinas_results"
os.makedirs(output_dir, exist_ok=True)
print(f"\nOutput directory: {output_dir}")

# 6a. Confusion Matrix
print("\nGenerating confusion matrix...")
class_names = [SALINAS_CLASSES.get(c, f'Class_{c}') for c in clf.classes_]
plot_multiclass_confusion_matrix(
    metrics['confusion_matrix'],
    class_names=class_names,
    save_path=f"{output_dir}/confusion_matrix.png",
    figsize=(14, 12),
    normalize=False
)

# Normalized confusion matrix
plot_multiclass_confusion_matrix(
    metrics['confusion_matrix'],
    class_names=class_names,
    save_path=f"{output_dir}/confusion_matrix_normalized.png",
    figsize=(14, 12),
    normalize=True
)

# 6b. Print full classification report
print("\nGenerating classification report...")
print_classification_report(y_test, y_pred, class_names)

# 6c. Save results to CSV
print("\nSaving results to CSV...")
results_df = pd.DataFrame({
    'Class': clf.classes_,
    'Class_Name': [SALINAS_CLASSES.get(c, f'Class_{c}') for c in clf.classes_],
    'Precision': metrics['per_class_precision'],
    'Recall': metrics['per_class_recall'],
    'F1_Score': metrics['per_class_f1'],
    'Support': metrics['per_class_support']
})
results_df.to_csv(f"{output_dir}/class_metrics.csv", index=False)
print(f"[OK] Saved to {output_dir}/class_metrics.csv")

# Save overall metrics
overall_metrics_df = pd.DataFrame({
    'Metric': ['Overall_Accuracy', 'Cohens_Kappa', 'Macro_Precision', 'Macro_Recall', 
               'Macro_F1', 'Weighted_Precision', 'Weighted_Recall', 'Weighted_F1'],
    'Value': [
        metrics['overall_accuracy'], metrics['kappa'],
        metrics['macro_precision'], metrics['macro_recall'], metrics['macro_f1'],
        metrics['weighted_precision'], metrics['weighted_recall'], metrics['weighted_f1']
    ]
})
overall_metrics_df.to_csv(f"{output_dir}/overall_metrics.csv", index=False)
print(f"[OK] Saved to {output_dir}/overall_metrics.csv")

# ============================================================================
# STEP 7: Summary
# ============================================================================
print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
print(f"\nKey Results:")
print(f"  ✓ Dataset: Salinas (54,129 samples, 204 bands, 16 classes)")
print(f"  ✓ Model: Multi-class PLS-DA ({N_COMPONENTS} components)")
print(f"  ✓ Overall Accuracy: {metrics['overall_accuracy']*100:.2f}%")
print(f"  ✓ Cohen's Kappa: {metrics['kappa']:.4f}")
print(f"  ✓ Macro F1-score: {metrics['macro_f1']:.4f}")
print(f"\nOutputs saved to: {output_dir}/")
print(f"  - confusion_matrix.png")
print(f"  - confusion_matrix_normalized.png")
print(f"  - class_metrics.csv")
print(f"  - overall_metrics.csv")
print("\n" + "="*80)

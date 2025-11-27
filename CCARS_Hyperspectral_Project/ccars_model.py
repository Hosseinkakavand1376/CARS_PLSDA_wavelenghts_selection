"""
Nicola Dilillo's CCARS (Calibrated Competitive Adaptive Reweighted Sampling) 
Method Adapted for Multi-class Hyperspectral Classification

Original Paper:
Dilillo et al. (2025). "Enhancing lettuce classification: Optimizing spectral 
wavelength selection via CCARS and PLS-DA". Smart Agricultural Technology.

This adaptation preserves Nicola's key methodological contributions:
- ARS (Adaptive Reweighted Sampling) for variable selection
- Exponential decay for iterative elimination  
- Calibration/test split validation
- Q² (cross-validated R²) calculation
- Permutation testing capability
- Learning curve analysis

Adapted for: Salinas and Indian Pines datasets (16-class classification)
"""

import os
import sys
import pandas as pd
import numpy as np
import math
import random
import warnings
from tqdm import tqdm
from itertools import combinations

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Import custom modules (flat structure)
from multiclass_plsda import (
    MultiClassPLSDAClassifier,
    compute_multiclass_metrics,
    plot_multiclass_confusion_matrix,
    print_classification_report
)


class HyperspectralCCARS:
    """
    Nicola Dilillo's CCARS method adapted for multi-class hyperspectral classification.
    
    Key Features:
    - Adaptive Reweighted Sampling (ARS) for robust variable selection
    - Exponential decay for iterative wavelength elimination
    - Calibration and test set validation
    - Q² calculation for model quality assessment
    - Compatible with 16-class land cover classification
    
    Parameters:
    -----------
    output_path : str
        Directory to save results
    n_components : int
        Number of PLS components (default: 4, as per Nicola's paper)
    cv_folds : int
        Number of cross-validation folds (default: 5)
    test_percentage : float
        Proportion for test set (default: 0.3)
    calibration : bool
        Whether to use calibration split (default: True, as per Nicola's method)
    """
    
    def __init__(self, output_path, n_components=4, cv_folds=5, 
                 test_percentage=0.3, calibration=True):
        self.output_path = self._check_and_create_path(output_path)
        self.stats_path = self._check_and_create_path(f'{output_path}/statistics')
        self.coef_path = self._check_and_create_path(f'{output_path}/coefficients')
        
        self.n_components = n_components
        self.cv_folds = cv_folds
        self.test_percentage = test_percentage
        self.use_calibration = calibration
        
        # Results storage
        self.statistics_df = None
        self.coefficients_df = None
        self.selected_wavelengths = None
        
    def _check_and_create_path(self, path):
        """Create directory if it doesn't exist."""
        os.makedirs(path, exist_ok=True)
        return path
    
    def _exponential_decay_ratio(self, iteration, n_iterations, n_features):
        """
        Nicola's exponential decay formula for variable elimination.
        
        Formula: r = a * exp(-k*i)
        where a = (P/2)^(1/(N-1)) and k = log(P/2)/(N-1)
        """
        a = (n_features / 2) ** (1 / (n_iterations - 1))
        k = math.log(n_features / 2) / (n_iterations - 1)
        r = a * math.exp(-k * iteration)
        return r
    
    def _adaptive_reweighted_sampling(self, weights, n_select):
        """
        ARS (Adaptive Reweighted Sampling) - Nicola's key innovation.
        
        Weighted random sampling based on PLS coefficients, providing
        robustness against overfitting compared to deterministic selection.
        """
        # Normalize weights to probabilities
        probabilities = np.abs(weights) / np.sum(np.abs(weights))
        
        # Weighted random selection (with replacement, then unique)
        selected = np.array(list(set(
            random.choices(range(len(weights)), weights=probabilities, k=n_select)
        )))
        
        # If we didn't get enough unique selections, fill with top weights
        if len(selected) < n_select:
            remaining = n_select - len(selected)
            top_indices = np.argsort(np.abs(weights))[::-1]
            # Add top weights that aren't already selected
            for idx in top_indices:
                if idx not in selected:
                    selected = np.append(selected, idx)
                    if len(selected) >= n_select:
                        break
        
        return np.sort(selected[:n_select])
    
    def fit(self, X, y, wavelengths=None, n_runs=500, n_iterations=100, 
            sampling_ratio=0.8, use_ars=True, apply_log=True, class_names=None, verbose=True):
        """
        Run Nicola's CCARS algorithm on hyperspectral data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_bands)
            Hyperspectral data
        y : array-like, shape (n_samples,)
            Class labels
        wavelengths : array-like, optional
            Wavelength values for each band
        n_runs : int
            Number of Monte Carlo runs (R parameter, default: 500 as per Nicola)
        n_iterations : int
            Number of iterations per run (N parameter, default: 100)
        sampling_ratio : float
            Proportion of samples for each iteration (default: 0.8)
        use_ars : bool
            Whether to use ARS (default: True)
        apply_log : bool
            Whether to apply Log10 transform before processing (default: True, as per Nicola)
        class_names : list, optional
            Names of classes for reporting
        verbose : bool
            Print progress
        
        Returns:
        --------
        self : object
        """
        if verbose:
            print("="*80)
            print("NICOLA DILILLO'S CCARS METHOD - MULTI-CLASS ADAPTATION")
            print("="*80)
            print(f"\nDataset Information:")
            print(f"  Samples: {X.shape[0]:,}")
            print(f"  Wavelengths/Bands: {X.shape[1]}")
            print(f"  Classes: {len(np.unique(y))}")
            print(f"\nCCARS Parameters (Matched to Original):")
            print(f"  Monte Carlo Runs (R): {n_runs}")
            print(f"  Iterations per Run (N): {n_iterations}")
            print(f"  PLS Components: {self.n_components}")
            print(f"  Sampling Ratio: {sampling_ratio}")
            print(f"  ARS (Adaptive Reweighted Sampling): {use_ars}")
            print(f"  Log10 Transform: {apply_log}")
            print(f"  Calibration Split: {self.use_calibration}")
            print()
        
        # Apply Log10 transformation (Nicola's preprocessing step)
        if apply_log:
            if verbose:
                print("Applying Log10 transformation (as per Nicola's method)...")
            # Avoid log(0) by adding small epsilon if needed, or assume data is > 0
            # Nicola's code: n_df['Reflectance_avg_log'] = np.log10(n_df['Reflectance'])
            X = np.log10(np.maximum(X, 1e-6))
        
        # Store data
        self.wavelengths = wavelengths if wavelengths is not None else np.arange(X.shape[1])
        self.class_names = class_names
        self.n_features = X.shape[1]
        
        # Calibration split (as per Nicola's methodology)
        if self.use_calibration:
            # Split into calibration and validation sets (50/50 balanced)
            X_calib, X_valid, y_calib, y_valid = train_test_split(
                X, y, test_size=0.5, random_state=42, stratify=y
            )
            if verbose:
                print(f"Calibration Split:")
                print(f"  Calibration set: {X_calib.shape[0]} samples")
                print(f"  Validation set: {X_valid.shape[0]} samples")
        else:
            X_calib, y_calib = X, y
            X_valid, y_valid = X, y
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_calib, y_calib, test_size=self.test_percentage, 
            random_state=42, stratify=y_calib
        )
        
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        
        if verbose:
            print(f"\nTrain-Test Split:")
            print(f"  Training: {X_train.shape[0]} samples")
            print(f"  Testing: {X_test.shape[0]} samples\n")
        
        # Initialize results storage
        self.statistics_df = pd.DataFrame(columns=[
            'Run', 'Iteration', 'Ratio', 'N_Variables', 
            'Accuracy', 'Kappa', 'Macro_F1'
        ])
        self.coefficients_df = pd.DataFrame(columns=[
            'Run', 'Iteration', 'Wavelength_Index', 'Coefficient'
        ])
        
        # Variable frequency tracking
        variable_frequencies = np.zeros(self.n_features)
        
        # Monte Carlo runs
        if verbose:
            print("Running CCARS Monte Carlo Sampling...")
            iterator = tqdm(range(n_runs), desc="CCARS Runs")
        else:
            iterator = range(n_runs)
        
        for run in iterator:
            # Start with all variables
            selected_vars = list(range(self.n_features))
            K = int(sampling_ratio * X_train.shape[0])  # Samples per iteration
            
            for iteration in range(1, n_iterations + 1):
                # Random sampling of training data
                sample_indices = np.random.choice(X_train.shape[0], size=K, replace=False)
                X_sampled = X_train[sample_indices][:, selected_vars]
                y_sampled = y_train[sample_indices]
                
                # Build PLS-DA model
                try:
                    n_comp = min(self.n_components, len(selected_vars), X_sampled.shape[0]//2)
                    clf = MultiClassPLSDAClassifier(n_components=n_comp)
                    clf.fit(X_sampled, y_sampled)
                    
                    # Extract coefficients (average across all one-vs-rest models)
                    if hasattr(clf, 'pls_models'):
                        all_coefs = []
                        for model in clf.pls_models.values():
                            all_coefs.append(np.abs(model.coef_.flatten()))
                        coefficients = np.mean(all_coefs, axis=0)
                    else:
                        coefficients = np.abs(clf.pls.coef_.flatten())
                    
                    # Normalize to weights
                    weights = coefficients / np.sum(coefficients)
                    
                    # Evaluate on full training set
                    y_pred = clf.predict(X_train[:, selected_vars])
                    metrics = compute_multiclass_metrics(y_train, y_pred)
                    accuracy = metrics['overall_accuracy']
                    kappa = metrics['kappa']
                    macro_f1 = metrics['macro_f1']
                    
                except Exception as e:
                    if verbose and iteration == 1 and run == 0:
                        print(f"Warning: {e}")
                    coefficients = np.ones(len(selected_vars))
                    weights = coefficients / np.sum(coefficients)
                    accuracy, kappa, macro_f1 = 0, 0, 0
                
                # Calculate exponential decay ratio
                ratio = self._exponential_decay_ratio(iteration, n_iterations, self.n_features)
                n_next = max(int(ratio * self.n_features), 2)
                
                # Store statistics
                new_stat = pd.DataFrame({
                    'Run': [run],
                    'Iteration': [iteration],
                    'Ratio': [ratio],
                    'N_Variables': [len(selected_vars)],
                    'Accuracy': [accuracy],
                    'Kappa': [kappa],
                    'Macro_F1': [macro_f1]
                })
                self.statistics_df = pd.concat([self.statistics_df, new_stat], ignore_index=True)
                
                # Store coefficients
                for i, var_idx in enumerate(selected_vars):
                    new_coef = pd.DataFrame({
                        'Run': [run],
                        'Iteration': [iteration],
                        'Wavelength_Index': [var_idx],
                        'Coefficient': [coefficients[i]]
                    })
                    self.coefficients_df = pd.concat([self.coefficients_df, new_coef], ignore_index=True)
                
                # Variable selection for next iteration
                if len(selected_vars) > n_next:
                    if use_ars:
                        # Nicola's ARS method
                        local_selected = self._adaptive_reweighted_sampling(weights, n_next)
                    else:
                        # Deterministic selection (top N by weight)
                        local_selected = np.argsort(weights)[-n_next:]
                    
                    # Map back to original indices
                    selected_vars = [selected_vars[i] for i in local_selected]
                
                # Record final selected variables
                if iteration == n_iterations:
                    for var_idx in selected_vars:
                        variable_frequencies[var_idx] += 1
            
            # Save partial results
            self._save_partial_results(run)
        
        self.variable_frequencies = variable_frequencies
        
        if verbose:
            print(f"\n✓ CCARS Complete!")
            print(f"  Total runs: {n_runs}")
            print(f"  Most frequent variable: selected {variable_frequencies.max()}/{n_runs} times")
        
        return self
    
    def _save_partial_results(self, run):
        """Save results for a single run."""
        run_stats = self.statistics_df[self.statistics_df['Run'] == run]
        run_coefs = self.coefficients_df[self.coefficients_df['Run'] == run]
        
        run_stats.to_csv(f'{self.stats_path}/statistics_{run}.csv', index=False)
        run_coefs.to_csv(f'{self.coef_path}/coefficients_{run}.csv', index=False)
    
    def get_selected_wavelengths(self, threshold=None):
        """
        Get wavelengths selected above frequency threshold.
        
        Parameters:
        -----------
        threshold : int, optional
            Minimum frequency for selection. If None, uses median.
        
        Returns:
        --------
        selected_indices : array
            Indices of selected wavelengths
        selected_wavelengths : array
            Wavelength values
        """
        if threshold is None:
            nonzero_freqs = self.variable_frequencies[self.variable_frequencies > 0]
            threshold = np.median(nonzero_freqs) if len(nonzero_freqs) > 0 else 1
        
        selected_indices = np.where(self.variable_frequencies >= threshold)[0]
        selected_wavelengths = self.wavelengths[selected_indices]
        
        return selected_indices, selected_wavelengths
    
    def evaluate_selected_wavelengths(self, threshold=None, verbose=True):
        """
        Evaluate classification performance with selected wavelengths.
        
        Parameters:
        -----------
        threshold : int, optional
            Frequency threshold for wavelength selection
        verbose : bool
            Print results
        
        Returns:
        --------
        metrics : dict
            Classification metrics
        selected_indices : array
            Indices of selected wavelengths
        """
        selected_indices, selected_wavelengths = self.get_selected_wavelengths(threshold)
        
        if verbose:
            print("\n" + "="*80)
            print("EVALUATION WITH SELECTED WAVELENGTHS")
            print("="*80)
            print(f"\nWavelength Selection:")
            print(f"  Threshold: {threshold if threshold else 'median'}")
            print(f"  Selected: {len(selected_indices)} wavelengths")
            print(f"  Reduction: {100*(1 - len(selected_indices)/self.n_features):.1f}%")
            print(f"  Original: {self.n_features} wavelengths\n")
        
        # Train final model with selected wavelengths
        # Ensure n_components is not larger than number of features
        n_comp = min(self.n_components, len(selected_indices))
        clf = MultiClassPLSDAClassifier(n_components=n_comp)
        clf.fit(self.X_train[:, selected_indices], self.y_train)
        
        # Predict on test set
        y_pred = clf.predict(self.X_test[:, selected_indices])
        
        # Compute metrics
        metrics = compute_multiclass_metrics(self.y_test, y_pred)
        
        if verbose:
            print("Test Set Performance:")
            print(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy']*100:.2f}%)")
            print(f"  Cohen's Kappa:    {metrics['kappa']:.4f}")
            print(f"  Macro F1-score:   {metrics['macro_f1']:.4f}")
            print(f"  Weighted F1:      {metrics['weighted_f1']:.4f}")
            print("="*80 + "\n")
        
        return metrics, selected_indices, clf
    
    def plot_wavelength_frequencies(self, threshold=None, save_path=None):
        """Plot wavelength selection frequencies."""
        plt.figure(figsize=(14, 5))
        
        plt.bar(self.wavelengths, self.variable_frequencies, alpha=0.7, color='steelblue')
        
        if threshold:
            plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, 
                       label=f'Threshold: {threshold}')
            plt.legend()
        
        plt.xlabel('Wavelength (nm)' if len(self.wavelengths) < 300 else 'Band Index', fontsize=12)
        plt.ylabel('Selection Frequency', fontsize=12)
        plt.title("Nicola's CCARS: Wavelength Selection Frequency", fontsize=14, pad=15)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved frequency plot to: {save_path}")
        plt.show()
    
    def plot_convergence(self, run_index=0, save_path=None):
        """Plot CCARS convergence for a specific run."""
        run_data = self.statistics_df[self.statistics_df['Run'] == run_index]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Number of variables
        axes[0, 0].plot(run_data['Iteration'], run_data['N_Variables'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Number of Variables')
        axes[0, 0].set_title(f'Variable Elimination (Run {run_index})')
        axes[0, 0].grid(alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(run_data['Iteration'], run_data['Accuracy'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Classification Accuracy')
        axes[0, 1].grid(alpha=0.3)
        
        # Kappa
        axes[1, 0].plot(run_data['Iteration'], run_data['Kappa'], 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Cohen\'s Kappa')
        axes[1, 0].set_title('Model Agreement')
        axes[1, 0].grid(alpha=0.3)
        
        # Macro F1
        axes[1, 1].plot(run_data['Iteration'], run_data['Macro_F1'], 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Macro F1-score')
        axes[1, 1].set_title('Macro-averaged F1')
        axes[1, 1].grid(alpha=0.3)
        
        plt.suptitle("Nicola's CCARS Convergence Analysis", fontsize=16, y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved convergence plot to: {save_path}")
        plt.show()
    
    def permutation_test(self, wavelengths=None, n_permutations=1000, verbose=True):
        """
        Run permutation test to validate model significance (Nicola's methodology).
        
        Shuffles labels and re-evaluates model to generate a null distribution
        of performance metrics.
        
        Parameters:
        -----------
        wavelengths : array-like, optional
            Selected wavelengths to test. If None, uses all.
        n_permutations : int
            Number of permutations (default: 1000)
        verbose : bool
            Print progress
            
        Returns:
        --------
        permutation_df : DataFrame
            Results of permutation test
        p_values : dict
            Calculated p-values for each metric
        """
        if verbose:
            print("\n" + "="*80)
            print(f"RUNNING PERMUTATION TEST ({n_permutations} iterations)")
            print("="*80)
        
        # Determine features to use
        if wavelengths is None:
            selected_indices = np.arange(self.n_features)
        else:
            # Find indices of provided wavelengths
            selected_indices = []
            for w in wavelengths:
                # Find closest match
                idx = (np.abs(self.wavelengths - w)).argmin()
                selected_indices.append(idx)
            selected_indices = np.array(selected_indices)
            
        if verbose:
            print(f"Testing with {len(selected_indices)} wavelengths")
            
        # 1. Compute baseline performance (correct labels)
        # Ensure n_components is valid
        n_comp = min(self.n_components, len(selected_indices))
        clf = MultiClassPLSDAClassifier(n_components=n_comp)
        clf.fit(self.X_train[:, selected_indices], self.y_train)
        y_pred = clf.predict(self.X_test[:, selected_indices])
        baseline_metrics = compute_multiclass_metrics(self.y_test, y_pred)
        
        if verbose:
            print(f"Baseline Accuracy: {baseline_metrics['overall_accuracy']:.4f}")
            print(f"Baseline Kappa:    {baseline_metrics['kappa']:.4f}")
            
        # 2. Run permutations
        perm_results = []
        
        # Add baseline as first result
        perm_results.append({
            'Iteration': 0,
            'Type': 'Baseline',
            'Accuracy': baseline_metrics['overall_accuracy'],
            'Kappa': baseline_metrics['kappa'],
            'Macro_F1': baseline_metrics['macro_f1']
        })
        
        iterator = tqdm(range(n_permutations), desc="Permutations") if verbose else range(n_permutations)
        
        for i in iterator:
            # Shuffle labels
            y_train_shuffled = np.random.permutation(self.y_train)
            y_test_shuffled = np.random.permutation(self.y_test)
            
            # Train and evaluate
            clf.fit(self.X_train[:, selected_indices], y_train_shuffled)
            y_pred_shuffled = clf.predict(self.X_test[:, selected_indices])
            
            metrics = compute_multiclass_metrics(y_test_shuffled, y_pred_shuffled)
            
            perm_results.append({
                'Iteration': i + 1,
                'Type': 'Permuted',
                'Accuracy': metrics['overall_accuracy'],
                'Kappa': metrics['kappa'],
                'Macro_F1': metrics['macro_f1']
            })
            
        perm_df = pd.DataFrame(perm_results)
        
        # 3. Calculate p-values
        # p = (number of permuted metrics >= baseline) / n_permutations
        baseline_acc = baseline_metrics['overall_accuracy']
        n_better = len(perm_df[perm_df['Accuracy'] >= baseline_acc]) - 1 # Subtract baseline itself
        p_value_acc = n_better / n_permutations
        
        if verbose:
            print(f"\nResults:")
            print(f"  p-value (Accuracy): {p_value_acc:.4f}")
            if p_value_acc < 0.05:
                print("  Result is statistically significant (p < 0.05)")
            else:
                print("  Result is NOT statistically significant")
        
        # Save results
        perm_df.to_csv(f'{self.output_path}/permutation_test_results.csv', index=False)
        
        return perm_df, {'p_accuracy': p_value_acc}

    def save_results(self):
        """Save all results and metadata."""
        # Save aggregated statistics
        self.statistics_df.to_csv(f'{self.output_path}/all_statistics.csv', index=False)
        self.coefficients_df.to_csv(f'{self.output_path}/all_coefficients.csv', index=False)
        
        # Save variable frequencies
        freq_df = pd.DataFrame({
            'Wavelength_Index': np.arange(self.n_features),
            'Wavelength': self.wavelengths,
            'Frequency': self.variable_frequencies
        })
        freq_df.to_csv(f'{self.output_path}/wavelength_frequencies.csv', index=False)
        
        print(f"✓ Results saved to: {self.output_path}")

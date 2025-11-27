"""
CARS (Competitive Adaptive Reweighted Sampling) Wavelength Selection
for Hyperspectral Classification

This script integrates the full CARS algorithm to select optimal wavelengths
for multi-class hyperspectral classification on Salinas and Indian Pines datasets.

The CARS algorithm iteratively:
1. Randomly samples training data
2. Builds PLS models  
3. Weights variables by PLS regression coefficients
4. Competitively selects variables with exponential decay
5. Evaluates model performance with reduced wavelengths

Result: Optimal subset of spectral bands that maintains or improves classification accuracy
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyperspectral_loader import load_salinas, load_indian_pines
from CARS_analysis.multiclass_extension import (
    MultiClassPLSDAClassifier,
    compute_multiclass_metrics,
    print_classification_report
)

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class CARSWavelengthSelector:
    """
    CARS algorithm for wavelength selection in hyperspectral data.
    
    Parameters:
    -----------
    n_iterations : int
        Number of iterations (N) for variable elimination
    n_runs : int
        Number of Monte Carlo runs (R)
    sampling_ratio : float
        Proportion of samples to use in each iteration
    n_components : int
        Number of PLS components
    """
    
    def __init__(self, n_iterations=100, n_runs=50, sampling_ratio=0.8, n_components=5):
        self.n_iterations = n_iterations
        self.n_runs = n_runs
        self.sampling_ratio = sampling_ratio
        self.n_components = n_components
        self.variable_frequencies = None
        self.iteration_history = []
        
    def _exponential_decay_ratio(self, iteration, n_iterations):
        """Calculate exponential decay ratio for iteration i."""
        r = np.exp(-iteration * np.log(2) / (n_iterations - 1))
        return r
    
    def _select_variables_by_weights(self, weights, n_select, use_weighted_sampling=False):
        """Select variables based on weights."""
        if use_weighted_sampling:
            # Weighted random sampling (ARS - Adaptive Reweighted Sampling)
            probabilities = np.abs(weights) / np.sum(np.abs(weights))
            selected = np.random.choice(len(weights), size=n_select, replace=False, p=probabilities)
        else:
            # Deterministic selection (top N by weight)
            selected = np.argsort(np.abs(weights))[-n_select:]
        
        return sorted(selected)
    
    def fit(self, X, y, verbose=True):
        """
        Run CARS algorithm to select optimal wavelengths.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
           Target values
        verbose : bool
            Whether to print progress
        
        Returns:
        --------
        self : object
        """
        n_samples, n_features = X.shape
        n_sampled = int(self.sampling_ratio * n_samples)
        
        # Track variable frequencies across runs
        variable_frequencies = np.zeros(n_features)
        
        # Store history for analysis
        all_run_history = []
        
        if verbose:
            print(f"\nRunning CARS Wavelength Selection:")
            print(f"  Total variables: {n_features}")
            print(f"  Iterations per run: {self.n_iterations}")
            print(f"  Number of runs: {self.n_runs}")
            print(f"  Sampling ratio: {self.sampling_ratio}")
            print(f"  PLS components: {self.n_components}")
            print()
        
        # Monte Carlo runs
        for run in tqdm(range(self.n_runs), desc="CARS Runs", disable=not verbose):
            # Initialize: all variables selected
            selected_vars = list(range(n_features))
            run_history = []
            
            for iteration in range(1, self.n_iterations + 1):
                # Random sampling
                sample_indices = np.random.choice(n_samples, size=n_sampled, replace=False)
                X_sampled = X[sample_indices][:, selected_vars]
                y_sampled = y[sample_indices]
                
                # Build PLS model
                try:
                    clf = MultiClassPLSDAClassifier(n_components=min(self.n_components, len(selected_vars), X_sampled.shape[0]//2))
                    clf.fit(X_sampled, y_sampled)
                    
                    # Extract regression coefficients (weights)
                    if hasattr(clf, 'pls_models'):
                        # Multi-class: average coefficients across all models
                        all_coefs = []
                        for model in clf.pls_models.values():
                            all_coefs.append(np.abs(model.coef_.flatten()))
                        weights = np.mean(all_coefs, axis=0)
                    else:
                        # Binary
                        weights = np.abs(clf.pls.coef_.flatten())
                    
                    # Evaluate on full training set
                    y_pred = clf.predict(X[:, selected_vars])
                    accuracy = accuracy_score(y, y_pred)
                    
                except Exception as e:
                    if verbose and iteration == 1:
                        print(f"Warning in run {run}, iteration {iteration}: {e}")
                    weights = np.ones(len(selected_vars))
                    accuracy = 0
                
                # Calculate number of variables for next iteration
                ratio = self._exponential_decay_ratio(iteration, self.n_iterations)
                n_next = max(int(ratio * n_features), 2)  # Keep at least 2 variables
                
                # Record history
                run_history.append({
                    'iteration': iteration,
                    'n_variables': len(selected_vars),
                    'accuracy': accuracy,
                    'ratio': ratio
                })
                
                # Select variables for next iteration
                if len(selected_vars) > n_next:
                    local_selected = self._select_variables_by_weights(weights, n_next, use_weighted_sampling=True)
                    # Map back to original indices
                    selected_vars = [selected_vars[i] for i in local_selected]
                
                if iteration == self.n_iterations:
                    # Record final selected variables for this run
                    for var_idx in selected_vars:
                        variable_frequencies[var_idx] += 1
            
            all_run_history.append(run_history)
        
        self.variable_frequencies = variable_frequencies
        self.iteration_history = all_run_history
        
        if verbose:
            print(f"\n✓ CARS completed: {self.n_runs} runs finished")
            print(f"  Most frequent variable selected: {variable_frequencies.max()}/{self.n_runs} times")
        
        return self
    
    def get_selected_variables(self, threshold=None):
        """
        Get selected variables based on frequency threshold.
        
        Parameters:
        -----------
        threshold : int, optional
            Minimum frequency for variable selection. If None, uses median frequency.
        
        Returns:
        --------
        selected_indices : array
            Indices of selected variables
        """
        if threshold is None:
            threshold = np.median(self.variable_frequencies[self.variable_frequencies > 0])
        
        selected_indices = np.where(self.variable_frequencies >= threshold)[0]
        return selected_indices
    
    def plot_frequency(self, wavelengths=None, threshold=None, save_path=None):
        """Plot variable selection frequencies."""
        plt.figure(figsize=(14, 5))
        
        if wavelengths is not None:
            x = wavelengths
            xlabel = 'Wavelength (nm)'
        else:
            x = np.arange(len(self.variable_frequencies))
            xlabel = 'Variable Index'
        
        plt.bar(x, self.variable_frequencies, alpha=0.7)
        
        if threshold is not None:
            plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')
            plt.legend()
        
        plt.xlabel(xlabel)
        plt.ylabel('Selection Frequency')
        plt.title(f'CARS Variable Selection Frequency (N={self.n_runs} runs)')
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_convergence(self, run_index=0, save_path=None):
        """Plot convergence of a specific run."""
        if run_index >= len(self.iteration_history):
            run_index = 0
        
        history = pd.DataFrame(self.iteration_history[run_index])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Number of variables vs iteration
        axes[0].plot(history['iteration'], history['n_variables'], 'b-', linewidth=2)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Number of Variables')
        axes[0].set_title(f'CARS Variable Elimination (Run {run_index})')
        axes[0].grid(alpha=0.3)
        
        # Accuracy vs iteration
        axes[1].plot(history['iteration'], history['accuracy'], 'g-', linewidth=2)
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'Model Accuracy During Selection (Run {run_index})')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def run_cars_experiment(dataset_name='salinas', output_dir=None):
    """
    Run complete CARS experiment on a dataset.
    
    Parameters:
    -----------
    dataset_name : str
        'salinas' or 'indian_pines'
    output_dir : str, optional
        Directory to save results
    """
    print("="*80)
    print(f"CARS WAVELENGTH SELECTION EXPERIMENT: {dataset_name.upper()}")
    print("="*80)
    
    # Load dataset
    if dataset_name == 'salinas':
        df, X, y = load_salinas(verbose=True)
    else:
        df, X, y = load_indian_pines(verbose=True)
    
    # Apply SNV normalization
    print("\nApplying SNV normalization...")
    X_normalized = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]):
        mean = np.mean(X[i, :])
        std = np.std(X[i, :])
        if std > 0:
            X_normalized[i, :] = (X[i, :] - mean) / std
        else:
            X_normalized[i, :] = X[i, :] - mean
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    # Create output directory
    if output_dir is None:
        output_dir = f"experiments/{dataset_name}_cars_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run CARS
    print("\n" + "-"*80)
    print("Running CARS Algorithm")
    print("-"*80)
    
    cars = CARSWavelengthSelector(
        n_iterations=100,
        n_runs=50,
        sampling_ratio=0.8,
        n_components=5
    )
    
    cars.fit(X_train, y_train, verbose=True)
    
    # Analyze results with different thresholds
    thresholds = [10, 15, 20, 25]
    results = []
    
    print("\n" + "-"*80)
    print("Evaluating Selected Wavelengths")
    print("-"*80)
    
    for threshold in thresholds:
        selected_vars = cars.get_selected_variables(threshold=threshold)
        
        if len(selected_vars) < 2:
            continue
        
        # Train model with selected wavelengths
        clf = MultiClassPLSDAClassifier(n_components=min(5, len(selected_vars)))
        clf.fit(X_train[:, selected_vars], y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test[:, selected_vars])
        metrics = compute_multiclass_metrics(y_test, y_pred)
        
        results.append({
            'threshold': threshold,
            'n_wavelengths': len(selected_vars),
            'accuracy': metrics['overall_accuracy'],
            'kappa': metrics['kappa'],
            'macro_f1': metrics['macro_f1']
        })
        
        print(f"\nThreshold {threshold}: {len(selected_vars)} wavelengths selected")
        print(f"  Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"  Kappa: {metrics['kappa']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/cars_threshold_comparison.csv", index=False)
    
    # Plot frequency
    band_indices = np.arange(X.shape[1])
    cars.plot_frequency(wavelengths=band_indices, threshold=20, 
                       save_path=f"{output_dir}/wavelength_frequency.png")
    
    #Plot convergence
    cars.plot_convergence(run_index=0, save_path=f"{output_dir}/convergence_plot.png")
    
    print(f"\n✓ Results saved to: {output_dir}/")
    print("="*80)
    
    return cars, results_df


if __name__ == '__main__':
    print("\n" + "="*80)
    print("CARS WAVELENGTH SELECTION FOR HYPERSPECTRAL CLASSIFICATION")
    print("="*80)
    
    # Run for both datasets
    print("\n### Running Salinas Dataset ###\n")
    cars_salinas, results_salinas = run_cars_experiment('salinas')
    
    print("\n\n### Running Indian Pines Dataset ###\n")
    cars_indian, results_indian = run_cars_experiment('indian_pines')
    
    print("\n" + "="*80)
    print("All CARS experiments completed successfully!")
    print("="*80)

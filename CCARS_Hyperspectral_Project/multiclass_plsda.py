"""
Multi-class Classification Extension for CARS PLS-DA

This module extends the original CARS_model.py to support multi-class classification
for hyperspectral datasets like Salinas (16 classes) and Indian Pines (16 classes).

The original PLSDAClassifier was designed for binary classification. This module provides:
- MultiClassPLSDAClassifier: Supports multi-class using one-vs-all strategy
- Multi-class metrics: Overall accuracy, per-class metrics, Cohen's Kappa
- Multi-class confusion matrix visualization
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                              cohen_kappa_score, precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


class MultiClassPLSDAClassifier(BaseEstimator, ClassifierMixin):
    """
    PLS-Discriminant Analysis classifier for multi-class classification.
    
    Uses one-vs-rest (OvR) strategy where a separate PLS model is trained
    for each class against all other classes.
    
    Parameters:
    -----------
    n_components : int, default=2
        Number of PLS components to use
    """
    
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls_models = {}
        
    def fit(self, X, y):
        """
        Fit multi-class PLS-DA model using one-vs-rest strategy.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (class labels)
        
        Returns:
        --------
        self : object
        """
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # For binary classification, fall back to simple binary PLS-DA
        if self.n_classes_ == 2:
            self.pls = PLSRegression(n_components=self.n_components)
            y_binary = (y == self.classes_[1]).astype(int)
            self.pls.fit(X, y_binary.reshape(-1, 1))
            self.binary_mode = True
        else:
            # Multi-class: train one PLS model per class (one-vs-rest)
            self.binary_mode = False
            for class_label in self.classes_:
                y_binary = (y == class_label).astype(int)
                pls = PLSRegression(n_components=self.n_components)
                pls.fit(X, y_binary.reshape(-1, 1))
                self.pls_models[class_label] = pls
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
        
        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted class labels
        """
        if self.binary_mode:
            # Binary classification with threshold
            y_pred_continuous = self.pls.predict(X)
            y_pred_binary = (y_pred_continuous >= 0.5).astype(int).ravel()
            return self.classes_[y_pred_binary]
        else:
            # Multi-class: predict using all models and select class with highest score
            scores = np.zeros((X.shape[0], self.n_classes_))
            for i, class_label in enumerate(self.classes_):
                scores[:, i] = self.pls_models[class_label].predict(X).ravel()
            
            # Select class with maximum score
            y_pred_indices = np.argmax(scores, axis=1)
            return self.classes_[y_pred_indices]
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
        
        Returns:
        --------
        y_pred_proba : array-like, shape (n_samples, n_classes)
            Predicted class probabilities
        """
        if self.binary_mode:
            # Binary mode
            y_pred_continuous = self.pls.predict(X)
            y_pred_proba = np.hstack([1 - y_pred_continuous, y_pred_continuous])
            return y_pred_proba
        else:
            # Multi-class: get scores from all models
            scores = np.zeros((X.shape[0], self.n_classes_))
            for i, class_label in enumerate(self.classes_):
                scores[:, i] = self.pls_models[class_label].predict(X).ravel()
            
            # Convert scores to probabilities using softmax
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            y_pred_proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            return y_pred_proba


def compute_multiclass_metrics(y_true, y_pred, class_names=None):
    """
    Compute comprehensive metrics for multi-class classification.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    class_names : list, optional
        Names of classes for better readability
    
    Returns:
    --------
    metrics : dict
        Dictionary containing various metrics:
        - overall_accuracy: Overall classification accuracy
        - kappa: Cohen's Kappa coefficient
        - per_class_precision: Precision for each class
        - per_class_recall: Recall for each class
        - per_class_f1: F1-score for each class
        - confusion_matrix: Confusion matrix
    """
    # Overall accuracy
    overall_accuracy = accuracy_score(y_true, y_pred)
    
    # Cohen's Kappa (agreement metric, accounts for chance)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Macro-averaged metrics (unweighted mean of per-class metrics)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Weighted metrics (weighted by support)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    metrics = {
        'overall_accuracy': overall_accuracy,
        'kappa': kappa,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'per_class_support': support,
        'confusion_matrix': cm
    }
    
    return metrics


def plot_multiclass_confusion_matrix(cm, class_names=None, save_path=None, 
                                     figsize=(12, 10), normalize=False):
    """
    Plot confusion matrix for multi-class classification.
    
    Parameters:
    -----------
    cm : array-like
        Confusion matrix
    class_names : list, optional
        Names of classes for axis labels
    save_path : str, optional
        Path to save the figure
    figsize : tuple
        Figure size (width, height)
    normalize : bool
        Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=figsize)
    
    # Use seaborn for better looking heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto',
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    
    plt.title(title, fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()
    plt.close()


def print_classification_report(y_true, y_pred, class_names=None):
    """
    Print detailed classification report.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    class_names : list, optional
        Names of classes for better readability
    """
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    
    if class_names is not None:
        # Map class labels to names
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        target_names = [class_names[int(c)] if int(c) < len(class_names) else f"Class {int(c)}" 
                       for c in unique_classes]
    else:
        target_names = None
    
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    
    # Additional metrics
    metrics = compute_multiclass_metrics(y_true, y_pred)
    
    print("\nADDITIONAL METRICS")
    print("-" * 70)
    print(f"Overall Accuracy:     {metrics['overall_accuracy']:.4f}")
    print(f"Cohen's Kappa:        {metrics['kappa']:.4f}")
    print(f"Macro-avg Precision:  {metrics['macro_precision']:.4f}")
    print(f"Macro-avg Recall:     {metrics['macro_recall']:.4f}")
    print(f"Macro-avg F1-score:   {metrics['macro_f1']:.4f}")
    print(f"Weighted Precision:   {metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall:      {metrics['weighted_recall']:.4f}")
    print(f"Weighted F1-score:    {metrics['weighted_f1']:.4f}")
    print("="*70 + "\n")

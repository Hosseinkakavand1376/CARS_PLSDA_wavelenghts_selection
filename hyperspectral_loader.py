"""
Hyperspectral Dataset Loader

This module provides functions to load and preprocess the Salinas and Indian Pines
hyperspectral datasets for use with the CARS PLS-DA wavelength selection pipeline.

The loader handles:
- Loading .mat files
- Reshaping 3D hyperspectral cubes to 2D (samples × bands)
- Filtering background pixels
- Converting to pandas DataFrame format compatible with existing CARS code
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from typing import Tuple, Optional
import warnings


# Dataset paths
DATASETS_DIR = 'datasets'
SALINAS_DIR = os.path.join(DATASETS_DIR, 'salinas')
INDIAN_PINES_DIR = os.path.join(DATASETS_DIR, 'indian_pines')

# Class names for reference
SALINAS_CLASSES = {
    0: 'Background',
    1: 'Brocoli_green_weeds_1',
    2: 'Brocoli_green_weeds_2',
    3: 'Fallow',
    4: 'Fallow_rough_plow',
    5: 'Fallow_smooth',
    6: 'Stubble',
    7: 'Celery',
    8: 'Grapes_untrained',
    9: 'Soil_vinyard_develop',
    10: 'Corn_senesced_green_weeds',
    11: 'Lettuce_romaine_4wk',
    12: 'Lettuce_romaine_5wk',
    13: 'Lettuce_romaine_6wk',
    14: 'Lettuce_romaine_7wk',
    15: 'Vinyard_untrained',
    16: 'Vinyard_vertical_trellis'
}

INDIAN_PINES_CLASSES = {
    0: 'Background',
    1: 'Alfalfa',
    2: 'Corn-notill',
    3: 'Corn-mintill',
    4: 'Corn',
    5: 'Grass-pasture',
    6: 'Grass-trees',
    7: 'Grass-pasture-mowed',
    8: 'Hay-windrowed',
    9: 'Oats',
    10: 'Soybean-notill',
    11: 'Soybean-mintill',
    12: 'Soybean-clean',
    13: 'Wheat',
    14: 'Woods',
    15: 'Buildings-Grass-Trees-Drives',
    16: 'Stone-Steel-Towers'
}


def load_mat_file(filepath: str, verbose: bool = True) -> dict:
    """
    Load a MATLAB .mat file.
    
    Args:
        filepath: Path to .mat file
        verbose: Whether to print loading information
    
    Returns:
        Dictionary containing the loaded data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if verbose:
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"Loading {os.path.basename(filepath)} ({file_size_mb:.2f} MB)...")
    
    data = loadmat(filepath)
    
    if verbose:
        print(f"  Keys in file: {[k for k in data.keys() if not k.startswith('__')]}")
    
    return data


def reshape_hyperspectral_data(data_cube: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape 3D hyperspectral data cube to 2D (samples × bands) format.
    
    Args:
        data_cube: Hyperspectral data cube of shape (height, width, bands)
        gt: Ground truth array of shape (height, width)
    
    Returns:
        Tuple of (X, y) where:
            X: Reshaped data of shape (height*width, bands)
            y: Reshaped labels of shape (height*width,)
    """
    height, width, n_bands = data_cube.shape
    
    # Reshape from (H, W, Bands) to (H*W, Bands)
    X = data_cube.reshape(-1, n_bands)
    
    # Reshape ground truth from (H, W) to (H*W,)
    y = gt.reshape(-1)
    
    return X, y


def filter_background_pixels(X: np.ndarray, y: np.ndarray, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove background pixels (class 0) from the dataset.
    
    Args:
        X: Feature matrix of shape (n_samples, n_bands)
        y: Label vector of shape (n_samples,)
        verbose: Whether to print filtering information
    
    Returns:
        Tuple of (X_filtered, y_filtered) with background pixels removed
    """
    # Find non-background pixels
    mask = y > 0
    
    if verbose:
        n_total = len(y)
        n_labeled = np.sum(mask)
        n_background = n_total - n_labeled
        print(f"  Total pixels: {n_total:,}")
        print(f"  Labeled pixels: {n_labeled:,} ({100*n_labeled/n_total:.1f}%)")
        print(f"  Background pixels removed: {n_background:,}")
    
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    return X_filtered, y_filtered


def prepare_dataframe(X: np.ndarray, y: np.ndarray, dataset_name: str, 
                     class_names: Optional[dict] = None) -> pd.DataFrame:
    """
    Convert hyperspectral data to pandas DataFrame format compatible with CARS.
    
    This creates a long-format DataFrame where each row represents one pixel-band combination.
    This format is similar to the original lettuce spectral data format.
    
    Args:
        X: Feature matrix of shape (n_samples, n_bands)
        y: Label vector of shape (n_samples,)
        dataset_name: Name of the dataset (e.g., 'salinas', 'indian_pines')
        class_names: Optional dictionary mapping class IDs to names
    
    Returns:
        DataFrame with columns: Class, Class_Name, Sample_ID, Band, Reflectance
    """
    n_samples, n_bands = X.shape
    
    # Create a list to store rows
    rows = []
    
    print(f"Converting to DataFrame format...")
    print(f"  Creating {n_samples:,} samples × {n_bands} bands = {n_samples * n_bands:,} rows")
    
    # For each sample (pixel)
    for sample_idx in range(n_samples):
        class_label = int(y[sample_idx])
        class_name = class_names.get(class_label, f'Class_{class_label}') if class_names else f'Class_{class_label}'
        
        # For each band
        for band_idx in range(n_bands):
            rows.append({
                'Class': class_label,
                'Class_Name': class_name,
                'Sample_ID': sample_idx,
                'Band': band_idx,
                'Reflectance': X[sample_idx, band_idx]
            })
    
    df = pd.DataFrame(rows)
    
    print(f"  DataFrame created: {len(df):,} rows")
    print(f"  Classes: {sorted(df['Class'].unique())}")
    print(f"  Class distribution:")
    class_counts = df.groupby('Class')['Sample_ID'].nunique().sort_index()
    for class_id, count in class_counts.items():
        class_name = class_names.get(class_id, f'Class_{class_id}') if class_names else f'Class_{class_id}'
        print(f"    Class {class_id} ({class_name}): {count:,} samples")
    
    return df


def load_salinas(datasets_dir: str = DATASETS_DIR, verbose: bool = True) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load the Salinas hyperspectral dataset.
    
    Args:
        datasets_dir: Base directory containing datasets
        verbose: Whether to print loading information
    
    Returns:
        Tuple of (df, X, y) where:
            df: DataFrame in CARS-compatible format
            X: Feature matrix (n_samples, 204)
            y: Label vector (n_samples,)
    """
    if verbose:
        print("="*60)
        print("Loading Salinas Dataset")
        print("="*60)
    
    # Construct file paths
    data_path = os.path.join(datasets_dir, 'salinas', 'Salinas_corrected.mat')
    gt_path = os.path.join(datasets_dir, 'salinas', 'Salinas_gt.mat')
    
    # Load .mat files
    data_dict = load_mat_file(data_path, verbose=verbose)
    gt_dict = load_mat_file(gt_path, verbose=verbose)
    
    # Extract data and ground truth
    # The key name varies, so we need to find the right one
    data_key = [k for k in data_dict.keys() if not k.startswith('__')][0]
    gt_key = [k for k in gt_dict.keys() if not k.startswith('__')][0]
    
    data_cube = data_dict[data_key]
    gt = gt_dict[gt_key]
    
    if verbose:
        print(f"  Data cube shape: {data_cube.shape}")
        print(f"  Ground truth shape: {gt.shape}")
        print(f"  Data type: {data_cube.dtype}")
        print(f"  Value range: [{data_cube.min():.2f}, {data_cube.max():.2f}]")
    
    # Reshape to 2D format
    X, y = reshape_hyperspectral_data(data_cube, gt)
    
    # Filter background pixels
    X, y = filter_background_pixels(X, y, verbose=verbose)
    
    # Create DataFrame
    df = prepare_dataframe(X, y, 'salinas', SALINAS_CLASSES)
    
    if verbose:
        print("="*60)
        print("Salinas Dataset Loaded Successfully!")
        print("="*60)
    
    return df, X, y


def load_indian_pines(datasets_dir: str = DATASETS_DIR, verbose: bool = True) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load the Indian Pines hyperspectral dataset.
    
    Args:
        datasets_dir: Base directory containing datasets
        verbose: Whether to print loading information
    
    Returns:
        Tuple of (df, X, y) where:
            df: DataFrame in CARS-compatible format
            X: Feature matrix (n_samples, 200)
            y: Label vector (n_samples,)
    """
    if verbose:
        print("="*60)
        print("Loading Indian Pines Dataset")
        print("="*60)
    
    # Construct file paths
    data_path = os.path.join(datasets_dir, 'indian_pines', 'Indian_pines_corrected.mat')
    gt_path = os.path.join(datasets_dir, 'indian_pines', 'Indian_pines_gt.mat')
    
    # Load .mat files
    data_dict = load_mat_file(data_path, verbose=verbose)
    gt_dict = load_mat_file(gt_path, verbose=verbose)
    
    # Extract data and ground truth
    data_key = [k for k in data_dict.keys() if not k.startswith('__')][0]
    gt_key = [k for k in gt_dict.keys() if not k.startswith('__')][0]
    
    data_cube = data_dict[data_key]
    gt = gt_dict[gt_key]
    
    if verbose:
        print(f"  Data cube shape: {data_cube.shape}")
        print(f"  Ground truth shape: {gt.shape}")
        print(f"  Data type: {data_cube.dtype}")
        print(f"  Value range: [{data_cube.min():.2f}, {data_cube.max():.2f}]")
    
    # Reshape to 2D format
    X, y = reshape_hyperspectral_data(data_cube, gt)
    
    # Filter background pixels
    X, y = filter_background_pixels(X, y, verbose=verbose)
    
    # Create DataFrame
    df = prepare_dataframe(X, y, 'indian_pines', INDIAN_PINES_CLASSES)
    
    if verbose:
        print("="*60)
        print("Indian Pines Dataset Loaded Successfully!")
        print("="*60)
    
    return df, X, y


def get_dataset_info(dataset_name: str) -> dict:
    """
    Get information about a dataset.
    
    Args:
        dataset_name: Name of dataset ('salinas' or 'indian_pines')
    
    Returns:
        Dictionary with dataset information
    """
    if dataset_name.lower() == 'salinas':
        return {
            'name': 'Salinas',
            'shape': (512, 217, 204),
            'n_bands': 204,
            'n_classes': 16,
            'wavelength_range': '360-2500 nm',
            'classes': SALINAS_CLASSES
        }
    elif dataset_name.lower() == 'indian_pines':
        return {
            'name': 'Indian Pines',
            'shape': (145, 145, 200),
            'n_bands': 200,
            'n_classes': 16,
            'wavelength_range': '400-2500 nm',
            'classes': INDIAN_PINES_CLASSES
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == '__main__':
    """Test the loader functions."""
    print("\n" + "="*60)
    print("Testing Hyperspectral Dataset Loader")
    print("="*60 + "\n")
    
    # Test Salinas
    try:
        df_salinas, X_salinas, y_salinas = load_salinas()
        print(f"\n[OK] Salinas loaded: {X_salinas.shape[0]:,} samples, {X_salinas.shape[1]} bands")
    except Exception as e:
        print(f"\n[ERROR] Error loading Salinas: {e}")
    
    print("\n")
    
    # Test Indian Pines
    try:
        df_indian, X_indian, y_indian = load_indian_pines()
        print(f"\n[OK] Indian Pines loaded: {X_indian.shape[0]:,} samples, {X_indian.shape[1]} bands")
    except Exception as e:
        print(f"\n[ERROR] Error loading Indian Pines: {e}")
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)

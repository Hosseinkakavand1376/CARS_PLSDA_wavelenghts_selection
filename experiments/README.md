# Hyperspectral Dataset Experiments

This directory contains experiment scripts for applying CARS PLS-DA wavelength selection to hyperspectral remote sensing datasets.

## Experiment Scripts

### 1. Salinas Dataset (`salinas_experiment.py`)
- **Dataset**: Salinas Valley, California
- **Size**: 54,129 samples, 204 bands, 16 classes
- **Classes**: Vegetables, bare soils, vineyard fields
- **Normalization**: SNV (Standard Normal Variate)
- **Model**: Multi-class PLS-DA with 5 components
- **Split**: 70% train, 30% test

**Usage:**
```bash
cd experiments
python salinas_experiment.py
```

**Outputs** (saved to `experiments/salinas_results/`):
- `confusion_matrix.png` - Confusion matrix visualization
- `confusion_matrix_normalized.png` - Normalized confusion matrix
- `class_metrics.csv` - Per-class performance metrics
- `overall_metrics.csv` - Overall performance summary

### 2. Indian Pines Dataset (`indian_pines_experiment.py`)
- **Dataset**: Northwestern Indiana agricultural area
- **Size**: 10,249 samples, 200 bands, 16 classes
- **Classes**: Various crops, grass, trees, buildings
- **Normalization**: SNV (Standard Normal Variate)
- **Model**: Multi-class PLS-DA with 5 components
- **Split**: 70% train, 30% test
- **Note**: Significant class imbalance (ratio up to 122:1)

**Usage:**
```bash
cd experiments
python indian_pines_experiment.py
```

**Outputs** (saved to `experiments/indian_pines_results/`):
- `confusion_matrix.png` - Confusion matrix visualization
- `confusion_matrix_normalized.png` - Normalized confusion matrix
- `class_metrics.csv` - Per-class performance metrics
- `overall_metrics.csv` - Overall performance summary

## Metrics Explained

### Overall Metrics
- **Overall Accuracy**: Percentage of correctly classified samples
- **Cohen's Kappa**: Agreement metric accounting for chance (0=random, 1=perfect)

### Macro-averaged Metrics
Simple average of per-class metrics (treats all classes equally, regardless of size)

### Weighted Metrics
Weighted average by class support (larger classes contribute more to the average)

### Per-class Metrics
- **Precision**: Of predicted class X, how many were actually class X?
- **Recall**: Of actual class X, how many were correctly identified?
- **F1-score**: Harmonic mean of precision and recall

## Customization

You can modify the following parameters in each script:
- `N_COMPONENTS`: Number of PLS components (default: 5)
- `TEST_SIZE`: Proportion of data for testing (default: 0.3)
- `NORMALIZE`: Whether to apply SNV normalization (default: True)
- `RANDOM_STATE`: Seed for reproducibility (default: 42)

## Expected Results

Based on literature, you can expect:
- **Salinas**: 85-95% accuracy (well-balanced classes)
- **Indian Pines**: 75-90% accuracy (class imbalance affects performance)

## Troubleshooting

If you encounter import errors, make sure you're running from the root directory:
```bash
cd c:\Users\hosse\CARS_PLSDA_wavelenghts_selection
python experiments/salinas_experiment.py
```

Or from the experiments directory:
```bash
cd experiments
python salinas_experiment.py
```

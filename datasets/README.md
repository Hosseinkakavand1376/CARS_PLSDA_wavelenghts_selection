# Hyperspectral Datasets

This directory contains the downloaded hyperspectral remote sensing datasets for CARS PLS-DA wavelength selection experiments.

## Datasets

### Salinas (512×217 pixels, 204 bands)
- **Location**: `salinas/`
- **Files**:
  - `Salinas_corrected.mat` (22.2 MB) - Hyperspectral data cube
  - `Salinas_gt.mat` (4.3 KB) - Ground truth labels (16 classes)
- **Sensor**: AVIRIS
- **Wavelength range**: 360-2500nm
- **Classes**: Vegetables, bare soils, vineyard fields

### Indian Pines (145×145 pixels, 200 bands)
- **Location**: `indian_pines/`
- **Files**:
  - `Indian_pines_corrected.mat` (5.95 MB) - Hyperspectral data cube
  - `Indian_pines_gt.mat` (1.1 KB) - Ground truth labels (16 classes)
- **Sensor**: AVIRIS
- **Wavelength range**: 400-2500nm
- **Classes**: Various crops (corn, soybean, wheat), grass, trees, buildings

## Usage

To load these datasets, use the `hyperspectral_loader.py` module:

```python
from hyperspectral_loader import load_salinas, load_indian_pines

# Load Salinas dataset
X_salinas, y_salinas = load_salinas()

# Load Indian Pines dataset
X_indian, y_indian = load_indian_pines()
```

## Data Sources

Datasets downloaded from:
- **EHU Hyperspectral Remote Sensing Scenes**: https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes

## Re-download

To re-download the datasets (e.g., if files are corrupted), delete the dataset files and run:

```bash
python download_datasets.py
```

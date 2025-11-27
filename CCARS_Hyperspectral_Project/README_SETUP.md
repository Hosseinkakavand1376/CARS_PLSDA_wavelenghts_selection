# Nicola's CCARS Method for Hyperspectral Classification

This project implements Nicola Dilillo's **CCARS (Calibrated Competitive Adaptive Reweighted Sampling)** method, adapted for multi-class hyperspectral datasets (Salinas and Indian Pines).

## 🚀 Setup on a New Computer

Follow these steps to run the project on a fresh machine.

### 1. Prerequisites
Ensure you have **Python 3.8+** installed.

### 2. Install Dependencies
Open a terminal/command prompt in this folder and run:
```bash
pip install -r requirements.txt
```

### 3. Download Datasets
Run the download script to automatically fetch the Salinas and Indian Pines datasets:
```bash
python download_data.py
```
*This will create a `datasets/` folder and download the necessary `.mat` files.*

### 4. Run Experiments
To run the full CCARS analysis (wavelength selection + classification):

**For Salinas Dataset:**
```bash
python run_salinas.py
```

**For Indian Pines Dataset:**
```bash
python run_indian_pines.py
```

### 📂 Output
Results will be saved in the `results/` directory:
- `results/salinas/`: Plots, statistics, and optimal wavelengths for Salinas.
- `results/indian_pines/`: Results for Indian Pines.

### ⚙️ Configuration
You can modify parameters (e.g., number of runs, components) directly in the `run_salinas.py` or `run_indian_pines.py` scripts.
- Default: `N_RUNS = 500` (High robustness, takes time)
- For quick testing: Change to `N_RUNS = 50`

---
*Adapted from Dilillo et al. (2025), Smart Agricultural Technology.*

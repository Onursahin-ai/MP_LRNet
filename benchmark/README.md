# MP-LRNet Benchmark: UCI Household Electric Power Consumption

This directory contains the reproducibility materials for the **generalizability evaluation** of MP-LRNet on a public benchmark dataset, as described in Section 3.X of the manuscript.

## Dataset

**UCI Individual Household Electric Power Consumption**
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- **Description:** Measurements of electric power consumption in one household (Sceaux, France) with a one-minute sampling rate over approximately 4 years (Dec 2006 - Nov 2010)
- **Preprocessing:** Aggregated to **daily mean** values, resulting in **1,433 daily observations** with 7 features
- **Target variable:** `Global_active_power` (daily average in kW)
- **Features:** `Global_active_power`, `Global_reactive_power`, `Voltage`, `Global_intensity`, `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`
- **Weekly pattern:** Weekend consumption (~1.23 kW) is notably higher than weekdays (~1.04 kW), making the [7, 14] multi-patch configuration meaningful

## Experimental Setup

The same experimental protocol as the original furniture sales study was applied:

| Parameter | Value |
|-----------|-------|
| Train/Test Split | 80% / 20% (chronological) |
| Normalization | Min-Max Scaling [0, 1] |
| LSTM Neurons | 200 |
| Dense Layer | 200 neurons, ReLU |
| Dropout / Recurrent Dropout | 0.2 / 0.2 |
| Optimizer | Adam (lr=0.001) |
| Loss Function | Mean Squared Error |
| Epochs | 20 |
| Batch Size | 64 |
| RF Estimators | 100 (random_state=42) |
| Iterations | 10 independent runs per configuration |
| Patch Sizes | [7, 14] (weekly / biweekly) |

## Results

### UCI Power Consumption Dataset (10 independent runs, mean ± std)

| Model | Multi-Patch | Ensemble | R² | MSE | MAE | RMSE | MAPE (%) |
|-------|------------|----------|-----|-----|-----|------|----------|
| **MP-LRNet** | **(7, 14)** | **RF** | **0.9201±0.0024** | **0.0076** | **0.0628** | **0.0872** | **6.95** |
| MP-LRNet (7,28) | (7, 28) | RF | 0.9116±0.0022 | 0.0085 | 0.0678 | 0.0922 | 7.69 |
| LSTM + RF | None | RF | 0.9066±0.0025 | 0.0088 | 0.0684 | 0.0940 | 7.54 |
| LSTM + MP | (7, 14) | None | 0.4581±0.0096 | 0.0516 | 0.1696 | 0.2271 | 20.03 |
| LSTM Only | None | None | 0.3796±0.0391 | 0.0587 | 0.1756 | 0.2423 | 20.15 |

### Cross-Domain Performance Comparison

| Model | Furniture Sales R² | UCI Power R² | Rank | Consistent? |
|-------|-------------------|-------------|------|-------------|
| MP-LRNet (LSTM + MP + RF) | 0.9918 | 0.9201 | 1st | ✓ |
| LSTM + RF (No MP) | 0.9803 | 0.9066 | 2nd | ✓ |
| LSTM + MP (No RF) | 0.9010 | 0.4581 | 3rd | ✓ |
| LSTM Only (Baseline) | 0.8168 | 0.3796 | 4th | ✓ |

**Key finding:** The performance ranking (MP-LRNet > LSTM+RF > LSTM+MP > LSTM-Only) is **identical across both datasets**, confirming the generalizability of the proposed hybrid architecture.

## Files

| File | Description |
|------|-------------|
| `run_power_benchmark.py` | Main benchmark script (downloads data automatically) |
| `results/power_results.csv` | Summary metrics (mean ± std for all configurations) |
| `results/power_detail.json` | Per-iteration detailed results (all 10 runs) |

## How to Run

### Option 1: Google Colab (Recommended)
```python
!python run_power_benchmark.py
```

### Option 2: Local
```bash
pip install tensorflow scikit-learn pandas numpy
python run_power_benchmark.py
```

The script automatically downloads the UCI dataset. No manual data download required.

## Requirements

- Python 3.9+
- TensorFlow 2.10+
- scikit-learn 1.1+
- pandas, numpy

## Citation

If you use this code, please cite:

```
[Authors], "A Hybrid Time Series Forecasting Model Combining Recurrent Neural Networks
and Ensemble Learning for Furniture Sales Prediction," Ain Shams Engineering Journal, 2026.
```

## License

This benchmark code is provided for academic reproducibility purposes.

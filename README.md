# MP-LRNet

**A Hybrid Time Series Forecasting Model Combining Recurrent Neural Networks and Ensemble Learning**

MP-LRNet (Multi-Patch LSTM–RF Network) integrates Long Short-Term Memory networks with Random Forest ensemble learning within a modular multi-patch architecture for accurate time series forecasting.

## Key Results

### Furniture Sales Dataset (Proprietary)
- **MP-LRNet R² = 0.9918** (best among 200 configurations)
- 10 independent runs, mean metrics: MAE=0.7321, RMSE=1.1896, MAPE=10.16%

### UCI Power Consumption Dataset (Public Benchmark)
- **MP-LRNet R² = 0.9201** (best among 5 configurations)
- Performance ranking identical to furniture dataset: MP-LRNet > LSTM+RF > LSTM+MP > LSTM-Only

See [`benchmark/`](benchmark/) for full reproducibility materials.

## Repository Structure

```
MP_LRNet/
├── README.md
├── 29.ipynb                    # Main model code (furniture sales)
├── 29.xlsx                     # Model results
├── hepsi2.xlsx                 # Full dataset
├── model_ve_sonuclar_*.h5      # Trained model weights
├── *.png                       # Visualization outputs
└── benchmark/                  # Public dataset evaluation
    ├── README.md               # Benchmark documentation
    ├── run_power_benchmark.py  # Reproducible benchmark script
    └── results/
        ├── power_results.csv   # Summary metrics
        └── power_detail.json   # Per-iteration details
```

## Citation

If you use this code, please cite:

```
[Authors], "A Hybrid Time Series Forecasting Model Combining Recurrent Neural Networks
and Ensemble Learning for Furniture Sales Prediction," Ain Shams Engineering Journal, 2026.
```

## License

Academic use only.

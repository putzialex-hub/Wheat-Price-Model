# Wheat Price Model (Single-Series BL2c1)

## Run the pipeline

Use the provided BL2c1 CSV (semicolon-separated with `Exchange Date`, `Close`, `Settlement Price`):

```bash
python cli.py --csv wheat_prices.csv
```

The CLI will load the single-series data, build features, run the walk-forward backtest, and print MAE/RMSE/MAPE plus the number of quarters/rows used.

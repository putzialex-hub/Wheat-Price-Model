from __future__ import annotations

import argparse
from pathlib import Path

from config import AppConfig, DataPaths
from pipeline import run_training_pipeline, forecast_next_quarter_end


def main() -> None:
    """
    Single-series CLI:
      - loads BL2c1 CSV
      - trains model
      - prints backtest metrics
      - prints latest weekly forecast stub
    """
    parser = argparse.ArgumentParser(description="Wheat BL2c1 single-series pipeline")
    parser.add_argument(
        "--csv",
        default="data/wheat_prices.csv",
        help="Path to BL2c1 CSV (semicolon-separated). Default: data/wheat_prices.csv",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    cfg = AppConfig(data=DataPaths(contracts_path=csv_path))
    model, artifacts, bt = run_training_pipeline(cfg, str(csv_path))

    print("Backtest metrics:")
    for k, v in bt.metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"Rows used: {len(artifacts.dataset_meta)}")
    print(f"Quarters covered: {artifacts.dataset_meta['quarter'].nunique()}")

    preds = bt.predictions.copy()
    earliest = preds.sort_values("asof_date").groupby("quarter", as_index=False).first()
    earliest["abs_err_model"] = (earliest["y_true"] - earliest["y_pred_model"]).abs()
    worst = earliest.sort_values("abs_err_model", ascending=False).head(10)

    print("\nWorst 10 quarters (model abs error):")
    print(
        worst[
            ["quarter", "asof_date", "qend_date", "y_true", "y_pred_model", "abs_err_model"]
        ].to_string(index=False)
    )

    try:
        fc = forecast_next_quarter_end(model, artifacts, cfg)
        print("\nLatest weekly forecast (stub):")
        print(fc.to_string(index=False))
    except Exception as e:
        print(f"\nLive forecast not produced: {e}")


if __name__ == "__main__":
    main()

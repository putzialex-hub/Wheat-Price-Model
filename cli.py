from __future__ import annotations

import argparse
from pathlib import Path

from config import AppConfig, DataPaths
from data_loader import load_bl2c1_csv
from features import add_market_features
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

    prices = load_bl2c1_csv(csv_path)
    feats = add_market_features(prices)
    ret_20d = feats["ret_20d"]
    print(
        "ret_20d stats:",
        f"min={ret_20d.min():.6f}",
        f"median={ret_20d.median():.6f}",
        f"max={ret_20d.max():.6f}",
        f"missing={int(ret_20d.isna().sum())}",
    )
    if ret_20d.nunique(dropna=True) <= 2:
        raise ValueError("ret_20d appears constant; check parsing/feature computation")

    model, artifacts, bt = run_training_pipeline(cfg, str(csv_path))

    print("Backtest metrics:")
    for k, v in bt.metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"Rows used: {len(artifacts.dataset_meta)}")
    print(f"Quarters covered: {artifacts.dataset_meta['quarter'].nunique()}")

    preds = bt.predictions.copy()
    if "asof_close" not in preds.columns and "y_pred_naive" in preds.columns:
        preds["asof_close"] = preds["y_pred_naive"]
    if "ret_20d" not in preds.columns:
        preds["ret_20d"] = 0.0
    delta_pred = preds["y_pred_model"] - preds["asof_close"]
    print(f"delta_pred stats: mean={delta_pred.mean():.6f}, std={delta_pred.std():.6f}")
    earliest = preds.sort_values("asof_date").groupby("quarter", as_index=False).first()
    earliest["abs_err_model"] = (earliest["y_true"] - earliest["y_pred_model"]).abs()
    earliest["abs_err_naive"] = (earliest["y_true"] - earliest["y_pred_naive"]).abs()
    earliest["abs_err_mom"] = (earliest["y_true"] - earliest["y_pred_mom"]).abs()
    worst = earliest.sort_values("abs_err_model", ascending=False).head(10)

    print("\nWorst 10 quarters (model abs error):")
    print(
        worst[
            [
                "quarter",
                "asof_date",
                "qend_date",
                "asof_close",
                "y_true",
                "y_pred_model",
                "y_pred_naive",
                "y_pred_mom",
                "abs_err_model",
                "abs_err_naive",
                "abs_err_mom",
            ]
        ].to_string(index=False)
    )

    print("\nSanity sample (10 rows):")
    sample = preds.sort_values("asof_date").head(10)
    print(
        sample[
            [
                "quarter",
                "asof_date",
                "qend_date",
                "weeks_to_qend",
                "asof_close",
                "ret_20d",
                "y_true",
                "y_pred_model",
                "y_pred_naive",
                "y_pred_mom",
            ]
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

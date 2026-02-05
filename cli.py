from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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
    parser.add_argument(
        "--primary-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only the primary as-of row per quarter. Default: True.",
    )
    parser.add_argument(
        "--macro-csv",
        default=None,
        help="Optional macro CSV (available_at, eurusd, brent).",
    )
    parser.add_argument(
        "--hybrid-threshold",
        type=float,
        default=10.0,
        help="Hybrid threshold in EUR/t for switching to model prediction.",
    )
    parser.add_argument(
        "--hybrid-model",
        choices=["ridge", "tree", "both"],
        default="ridge",
        help="Which model drives hybrid selection. Default: ridge.",
    )
    parser.add_argument(
        "--grid-thresholds",
        default=None,
        help="Comma-separated list of thresholds for grid search (e.g. '5,10,15').",
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Disable hybrid reporting (use pure model outputs).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    macro_path = Path(args.macro_csv) if args.macro_csv else None
    if macro_path is None:
        default_macro = Path("data/macro_features.csv")
        macro_path = default_macro if default_macro.exists() else None
    elif not macro_path.exists():
        macro_path = None

    cfg = AppConfig(data=DataPaths(contracts_path=csv_path, macro_path=macro_path))

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

    if args.grid_thresholds:
        thresholds = [float(x.strip()) for x in args.grid_thresholds.split(",") if x.strip()]
        rows = []
        for threshold in thresholds:
            _, _, bt = run_training_pipeline(
                cfg,
                str(csv_path),
                primary_only=args.primary_only,
                hybrid_threshold=threshold,
                enable_hybrid=True,
                hybrid_model=args.hybrid_model,
            )
            preds = bt.predictions
            y_true = preds["y_true"].to_numpy()
            y_pred_naive = preds["y_pred_naive"].to_numpy()
            rows.append(
                {
                    "threshold": threshold,
                    "model": "naive",
                    "MAE": (abs(y_true - y_pred_naive)).mean(),
                    "MAPE": (abs(y_true - y_pred_naive) / preds["y_true"].abs().clip(lower=1e-9)).mean()
                    * 100.0,
                    "usage_rate": np.nan,
                }
            )
            if "y_pred_hybrid_ridge" in preds.columns:
                y_pred = preds["y_pred_hybrid_ridge"].to_numpy()
                rows.append(
                    {
                        "threshold": threshold,
                        "model": "hybrid_ridge",
                        "MAE": (abs(y_true - y_pred)).mean(),
                        "MAPE": (abs(y_true - y_pred) / preds["y_true"].abs().clip(lower=1e-9)).mean()
                        * 100.0,
                        "usage_rate": preds["use_model_ridge"].mean(),
                    }
                )
            if "y_pred_hybrid_tree" in preds.columns:
                y_pred = preds["y_pred_hybrid_tree"].to_numpy()
                rows.append(
                    {
                        "threshold": threshold,
                        "model": "hybrid_tree",
                        "MAE": (abs(y_true - y_pred)).mean(),
                        "MAPE": (abs(y_true - y_pred) / preds["y_true"].abs().clip(lower=1e-9)).mean()
                        * 100.0,
                        "usage_rate": preds["use_model_tree"].mean(),
                    }
                )
        grid = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
        print("\nHybrid threshold grid (sorted by MAE):")
        print(grid.to_string(index=False, float_format="{:.4f}".format))
        print("\nBest threshold per model:")
        best = grid.sort_values("MAE").groupby("model", as_index=False).first()
        print(best.to_string(index=False, float_format="{:.4f}".format))
        return

    model, artifacts, bt = run_training_pipeline(
        cfg,
        str(csv_path),
        primary_only=args.primary_only,
        hybrid_threshold=args.hybrid_threshold,
        enable_hybrid=not args.no_hybrid,
        hybrid_model=args.hybrid_model,
    )

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
    delta_pred = preds["y_pred_tree"] - preds["asof_close"]
    print(f"delta_pred stats: mean={delta_pred.mean():.6f}, std={delta_pred.std():.6f}")
    usage_tree = preds["use_model_tree"].mean() if "use_model_tree" in preds.columns else None
    usage_ridge = preds["use_model_ridge"].mean() if "use_model_ridge" in preds.columns else None
    usage_parts = []
    if usage_tree is not None:
        usage_parts.append(f"tree={usage_tree:.3f}")
    if usage_ridge is not None:
        usage_parts.append(f"ridge={usage_ridge:.3f}")
    if usage_parts:
        print("Hybrid usage rate:", " ".join(usage_parts))
    earliest = preds.sort_values("asof_date").groupby("quarter", as_index=False).first()
    earliest["abs_err_tree"] = (earliest["y_true"] - earliest["y_pred_tree"]).abs()
    earliest["abs_err_ridge"] = (earliest["y_true"] - earliest["y_pred_ridge"]).abs()
    if "y_pred_hybrid_tree" in earliest.columns:
        earliest["abs_err_hybrid_tree"] = (earliest["y_true"] - earliest["y_pred_hybrid_tree"]).abs()
    else:
        earliest["y_pred_hybrid_tree"] = np.nan
        earliest["abs_err_hybrid_tree"] = np.nan
    if "y_pred_hybrid_ridge" not in earliest.columns:
        earliest["y_pred_hybrid_ridge"] = np.nan
    earliest["abs_err_naive"] = (earliest["y_true"] - earliest["y_pred_naive"]).abs()
    earliest["abs_err_mom"] = (earliest["y_true"] - earliest["y_pred_mom"]).abs()
    worst = earliest.sort_values("abs_err_tree", ascending=False).head(10)

    print("\nWorst 10 quarters (tree abs error):")
    print(
        worst[
            [
                "quarter",
                "asof_date",
                "qend_date",
                "asof_close",
                "y_true",
                "y_pred_tree",
                "y_pred_ridge",
                "y_pred_hybrid_tree",
                "y_pred_naive",
                "y_pred_mom",
                "abs_err_tree",
                "abs_err_ridge",
                "abs_err_hybrid_tree",
                "abs_err_naive",
                "abs_err_mom",
            ]
        ].to_string(index=False)
    )

    worst5_quarters = set(earliest.sort_values("abs_err_tree", ascending=False).head(5)["quarter"])
    full_mae = {
        "tree": earliest["abs_err_tree"].mean(),
        "ridge": earliest["abs_err_ridge"].mean(),
        "hybrid_tree": earliest["abs_err_hybrid_tree"].mean(),
        "naive": earliest["abs_err_naive"].mean(),
        "mom": earliest["abs_err_mom"].mean(),
    }
    trimmed = earliest[~earliest["quarter"].isin(worst5_quarters)]
    trimmed_mae = {
        "tree": trimmed["abs_err_tree"].mean(),
        "ridge": trimmed["abs_err_ridge"].mean(),
        "hybrid_tree": trimmed["abs_err_hybrid_tree"].mean(),
        "naive": trimmed["abs_err_naive"].mean(),
        "mom": trimmed["abs_err_mom"].mean(),
    }

    print("\nMAE (full vs excl worst 5 quarters by tree):")
    for key in ["tree", "ridge", "hybrid_tree", "naive", "mom"]:
        print(f"  {key}: {full_mae[key]:.4f} | {trimmed_mae[key]:.4f}")

    subset_2022 = preds[preds["qend_date"].dt.year == 2022]
    if not subset_2022.empty:
        print("\n2022 subset (MAE/MAPE):")
        candidates = [
            ("naive", "y_pred_naive"),
            ("ridge", "y_pred_ridge"),
            ("tree", "y_pred_tree"),
            ("hybrid_ridge", "y_pred_hybrid_ridge"),
            ("hybrid_tree", "y_pred_hybrid_tree"),
        ]
        for key, col in candidates:
            if col not in subset_2022.columns:
                continue
            mae = (subset_2022["y_true"] - subset_2022[col]).abs().mean()
            denom = subset_2022["y_true"].abs().clip(lower=1e-9)
            mape = ((subset_2022["y_true"] - subset_2022[col]).abs() / denom).mean() * 100.0
            print(f"  {key}: MAE={mae:.4f} MAPE={mape:.4f}")

    print("\nSanity sample (10 rows):")
    sample = preds.sort_values("asof_date").head(10)
    if "y_pred_hybrid_ridge" not in sample.columns:
        sample["y_pred_hybrid_ridge"] = np.nan
    if "y_pred_hybrid_tree" not in sample.columns:
        sample["y_pred_hybrid_tree"] = np.nan
    with pd.option_context("display.float_format", "{:.6f}".format):
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
                    "y_pred_tree",
                    "y_pred_ridge",
                    "y_pred_hybrid_tree",
                    "y_pred_hybrid_ridge",
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
    parser.add_argument(
        "--primary-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use only the primary as-of row per quarter. Default: True.",
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

    model, artifacts, bt = run_training_pipeline(cfg, str(csv_path), primary_only=args.primary_only)

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
    delta_pred = preds["y_pred_tree"] - preds["asof_close"]
    print(f"delta_pred stats: mean={delta_pred.mean():.6f}, std={delta_pred.std():.6f}")
    earliest = preds.sort_values("asof_date").groupby("quarter", as_index=False).first()
    earliest["abs_err_tree"] = (earliest["y_true"] - earliest["y_pred_tree"]).abs()
    earliest["abs_err_ridge"] = (earliest["y_true"] - earliest["y_pred_ridge"]).abs()
    earliest["abs_err_naive"] = (earliest["y_true"] - earliest["y_pred_naive"]).abs()
    earliest["abs_err_mom"] = (earliest["y_true"] - earliest["y_pred_mom"]).abs()
    worst = earliest.sort_values("abs_err_tree", ascending=False).head(10)

    print("\nWorst 10 quarters (tree abs error):")
    print(
        worst[
            [
                "quarter",
                "asof_date",
                "qend_date",
                "asof_close",
                "y_true",
                "y_pred_tree",
                "y_pred_ridge",
                "y_pred_naive",
                "y_pred_mom",
                "abs_err_tree",
                "abs_err_ridge",
                "abs_err_naive",
                "abs_err_mom",
            ]
        ].to_string(index=False)
    )

    worst5_quarters = set(earliest.sort_values("abs_err_tree", ascending=False).head(5)["quarter"])
    full_mae = {
        "tree": earliest["abs_err_tree"].mean(),
        "ridge": earliest["abs_err_ridge"].mean(),
        "naive": earliest["abs_err_naive"].mean(),
        "mom": earliest["abs_err_mom"].mean(),
    }
    trimmed = earliest[~earliest["quarter"].isin(worst5_quarters)]
    trimmed_mae = {
        "tree": trimmed["abs_err_tree"].mean(),
        "ridge": trimmed["abs_err_ridge"].mean(),
        "naive": trimmed["abs_err_naive"].mean(),
        "mom": trimmed["abs_err_mom"].mean(),
    }

    print("\nMAE (full vs excl worst 5 quarters by tree):")
    for key in ["tree", "ridge", "naive", "mom"]:
        print(f"  {key}: {full_mae[key]:.4f} | {trimmed_mae[key]:.4f}")

    print("\nSanity sample (10 rows):")
    sample = preds.sort_values("asof_date").head(10)
    with pd.option_context("display.float_format", "{:.6f}".format):
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
                    "y_pred_tree",
                    "y_pred_ridge",
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
 main


if __name__ == "__main__":
    main()

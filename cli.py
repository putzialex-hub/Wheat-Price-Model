from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from config import AppConfig, DataPaths
from data_loader import load_bl2c1_csv, load_bl2c2_csv
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
        "--target-mode",
        choices=["level", "delta", "log_return"],
        default="level",
        help="Target mode for training (level, delta, log_return). Default: level.",
    )
    parser.add_argument(
        "--csv-c2",
        default=None,
        help="Optional BL2c2 CSV (same schema as BL2c1).",
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
        "--ridge-alpha",
        type=float,
        default=10.0,
        help="Ridge alpha (L2) regularization strength. Default: 10.0.",
    )
    parser.add_argument(
        "--ridge-alpha-grid",
        default=None,
        help="Comma-separated grid for ridge alpha (e.g. '0.1,1,10,100').",
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Disable hybrid reporting (use pure model outputs).",
    )
    parser.add_argument(
        "--intervals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Report prediction intervals and coverage. Default: True.",
    )
    parser.add_argument(
        "--interval-alpha",
        type=float,
        default=0.2,
        help="Interval alpha (0.2 -> 80% coverage). Default: 0.2.",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=["per_fold", "pooled", "rolling"],
        default="pooled",
        help="Conformal calibration mode. Default: pooled.",
    )
    parser.add_argument(
        "--calibration-buckets",
        type=int,
        default=3,
        help="Volatility bucket count (fixed to 3). Default: 3.",
    )
    parser.add_argument(
        "--rolling-calibration-size",
        type=int,
        default=200,
        help="Rolling calibration pool size (scores). Default: 200.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    csv_c2_path = Path(args.csv_c2) if args.csv_c2 else None
    if csv_c2_path is not None and not csv_c2_path.exists():
        csv_c2_path = None
    macro_path = Path(args.macro_csv) if args.macro_csv else None
    if macro_path is None:
        default_macro = Path("data/macro_features.csv")
        macro_path = default_macro if default_macro.exists() else None
    elif not macro_path.exists():
        macro_path = None

    cfg = AppConfig(data=DataPaths(contracts_path=csv_path, macro_path=macro_path))
    ridge_grid = None
    if args.ridge_alpha_grid:
        ridge_grid = [float(x.strip()) for x in args.ridge_alpha_grid.split(",") if x.strip()]
    cfg = replace(
        cfg,
        model=replace(
            cfg.model,
            ridge_alpha=args.ridge_alpha,
            ridge_alpha_grid=ridge_grid,
            interval_alpha=args.interval_alpha,
            calibration_mode=args.calibration_mode,
            rolling_calibration_size=args.rolling_calibration_size,
        ),
    )

    print(f"Target mode: {args.target_mode}")

    prices = load_bl2c1_csv(csv_path)
    feats = add_market_features(prices)
    if csv_c2_path is not None:
        prices_c2 = load_bl2c2_csv(csv_c2_path)
        overlap = prices.index.intersection(prices_c2.index)
        if not overlap.empty:
            print(
                "C2 overlap range:",
                f"{overlap.min().date()} to {overlap.max().date()}",
            )
        merged = prices.join(prices_c2[["close_c2"]], how="left")
        missing_frac = merged["close_c2"].isna().mean()
        spread = merged["close"] - merged["close_c2"]
        spread_stats = spread.dropna()
        if not spread_stats.empty:
            print(
                "C2 missing fraction:",
                f"{missing_frac:.3f}",
                "spread min/median/max:",
                f"{spread_stats.min():.4f}",
                f"{spread_stats.median():.4f}",
                f"{spread_stats.max():.4f}",
            )
        else:
            print("C2 missing fraction:", f"{missing_frac:.3f}", "spread: no overlap data")
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
                price_csv_c2_path=str(csv_c2_path) if csv_c2_path else None,
                primary_only=args.primary_only,
                hybrid_threshold=threshold,
                enable_hybrid=True,
                hybrid_model=args.hybrid_model,
                target_mode=args.target_mode,
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
        price_csv_c2_path=str(csv_c2_path) if csv_c2_path else None,
        primary_only=args.primary_only,
        hybrid_threshold=args.hybrid_threshold,
        enable_hybrid=not args.no_hybrid,
        hybrid_model=args.hybrid_model,
        target_mode=args.target_mode,
    )

    print("Backtest metrics:")
    for k, v in bt.metrics.items():
        print(f"  {k}: {v:.4f}")
    if args.intervals:
        print("\nInterval metrics (p10/p50/p90):")
        target = 1 - cfg.model.interval_alpha
        print(f"  coverage_target: {target:.4f}")
        metric_map = [
            ("MAE_p50", "MAE_p50"),
            ("MAPE_p50", "MAPE_p50"),
            ("MAE_target_p50", "MAE_target_p50"),
            ("coverage_80_raw", "coverage_raw"),
            ("avg_width_80_raw", "width_raw"),
            ("coverage_80_cal", "coverage_cal"),
            ("avg_width_80_cal", "width_cal"),
        ]
        for key, label in metric_map:
            if key in bt.metrics:
                print(f"  {label}: {bt.metrics[key]:.4f}")
        print("\nResidual-conformal intervals:")
        resid_map = [
            ("coverage_resid_naive", "coverage_resid_naive"),
            ("width_resid_naive", "width_resid_naive"),
            ("coverage_resid_p50", "coverage_resid_p50"),
            ("width_resid_p50", "width_resid_p50"),
        ]
        for key, label in resid_map:
            if key in bt.metrics:
                print(f"  {label}: {bt.metrics[key]:.4f}")
        print("\nRolling calibration (recent scores):")
        roll_map = [
            ("coverage_cal_roll", "coverage_roll"),
            ("width_cal_roll", "width_roll"),
            ("coverage_cal_roll_recent40", "coverage_roll_recent40"),
            ("width_cal_roll_recent40", "width_roll_recent40"),
            ("coverage_cal_recent40", "coverage_cal_recent40"),
            ("width_cal_recent40", "width_cal_recent40"),
        ]
        for key, label in roll_map:
            if key in bt.metrics:
                print(f"  {label}: {bt.metrics[key]:.4f}")

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
    if args.intervals and "q_hat" in preds.columns:
        q_hat_vals = preds["q_hat"]
        print(
            "q_hat stats:",
            f"mean={q_hat_vals.mean():.4f}",
            f"median={q_hat_vals.median():.4f}",
            f"p90={np.nanpercentile(q_hat_vals, 90):.4f}",
            f"max={q_hat_vals.max():.4f}",
        )
    if args.intervals and "q_hat_naive" in preds.columns:
        q_hat_vals = preds["q_hat_naive"]
        print(
            "q_hat_naive stats:",
            f"mean={q_hat_vals.mean():.4f}",
            f"median={q_hat_vals.median():.4f}",
            f"p90={np.nanpercentile(q_hat_vals, 90):.4f}",
            f"max={q_hat_vals.max():.4f}",
        )
    if args.intervals and "q_hat_roll" in preds.columns:
        q_hat_vals = preds["q_hat_roll"]
        print(
            "q_hat_roll stats:",
            f"mean={q_hat_vals.mean():.4f}",
            f"median={q_hat_vals.median():.4f}",
            f"p90={np.nanpercentile(q_hat_vals, 90):.4f}",
            f"max={q_hat_vals.max():.4f}",
        )
    earliest = preds.sort_values("asof_date").groupby("quarter", as_index=False).first()
    earliest["abs_err_tree"] = (earliest["y_true"] - earliest["y_pred_tree"]).abs()
    earliest["abs_err_ridge"] = (earliest["y_true"] - earliest["y_pred_ridge"]).abs()
    if "y_pred_hybrid_tree" in earliest.columns:
        earliest["abs_err_hybrid_tree"] = (earliest["y_true"] - earliest["y_pred_hybrid_tree"]).abs()
    else:
        earliest["y_pred_hybrid_tree"] = np.nan
        earliest["abs_err_hybrid_tree"] = np.nan
    if "y_pred_hybrid_ridge" in earliest.columns:
        earliest["abs_err_hybrid_ridge"] = (earliest["y_true"] - earliest["y_pred_hybrid_ridge"]).abs()
    if "y_pred_hybrid_ridge" not in earliest.columns:
        earliest["y_pred_hybrid_ridge"] = np.nan
    earliest["abs_err_naive"] = (earliest["y_true"] - earliest["y_pred_naive"]).abs()
    earliest["abs_err_mom"] = (earliest["y_true"] - earliest["y_pred_mom"]).abs()
    worst = earliest.sort_values("abs_err_tree", ascending=False).head(10)

    print("\nWorst 10 quarters (tree abs error):")
    worst_cols = [
        "quarter",
        "asof_date",
        "qend_date",
        "asof_close",
        "y_true",
        "y_pred_tree",
        "y_pred_ridge",
    ]
    if "y_pred_hybrid_tree" in worst.columns and not worst["y_pred_hybrid_tree"].isna().all():
        worst_cols.append("y_pred_hybrid_tree")
        worst_cols.append("abs_err_hybrid_tree")
    if "y_pred_hybrid_ridge" in worst.columns and not worst["y_pred_hybrid_ridge"].isna().all():
        worst_cols.append("y_pred_hybrid_ridge")
    worst_cols += [
        "y_pred_naive",
        "y_pred_mom",
        "abs_err_tree",
        "abs_err_ridge",
        "abs_err_naive",
        "abs_err_mom",
    ]
    print(worst[worst_cols].to_string(index=False))

    worst5_quarters = set(earliest.sort_values("abs_err_tree", ascending=False).head(5)["quarter"])
    full_mae = {
        "tree": earliest["abs_err_tree"].mean(),
        "ridge": earliest["abs_err_ridge"].mean(),
        "naive": earliest["abs_err_naive"].mean(),
        "mom": earliest["abs_err_mom"].mean(),
    }
    if "abs_err_hybrid_tree" in earliest.columns and not earliest["abs_err_hybrid_tree"].isna().all():
        full_mae["hybrid_tree"] = earliest["abs_err_hybrid_tree"].mean()
    if "abs_err_hybrid_ridge" in earliest.columns and not earliest["abs_err_hybrid_ridge"].isna().all():
        full_mae["hybrid_ridge"] = earliest["abs_err_hybrid_ridge"].mean()
    trimmed = earliest[~earliest["quarter"].isin(worst5_quarters)]
    trimmed_mae = {
        "tree": trimmed["abs_err_tree"].mean(),
        "ridge": trimmed["abs_err_ridge"].mean(),
        "naive": trimmed["abs_err_naive"].mean(),
        "mom": trimmed["abs_err_mom"].mean(),
    }
    if "abs_err_hybrid_tree" in trimmed.columns and not trimmed["abs_err_hybrid_tree"].isna().all():
        trimmed_mae["hybrid_tree"] = trimmed["abs_err_hybrid_tree"].mean()
    if "abs_err_hybrid_ridge" in trimmed.columns and not trimmed["abs_err_hybrid_ridge"].isna().all():
        trimmed_mae["hybrid_ridge"] = trimmed["abs_err_hybrid_ridge"].mean()

    print("\nMAE (full vs excl worst 5 quarters by tree):")
    for key in full_mae:
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
    sample_cols = [
        "quarter",
        "asof_date",
        "qend_date",
        "weeks_to_qend",
        "asof_close",
        "ret_20d",
        "y_true",
        "y_pred_tree",
        "y_pred_ridge",
    ]
    if "y_pred_hybrid_tree" in sample.columns and not sample["y_pred_hybrid_tree"].isna().all():
        sample_cols.append("y_pred_hybrid_tree")
    if "y_pred_hybrid_ridge" in sample.columns and not sample["y_pred_hybrid_ridge"].isna().all():
        sample_cols.append("y_pred_hybrid_ridge")
    sample_cols += ["y_pred_naive", "y_pred_mom"]
    with pd.option_context("display.float_format", "{:.6f}".format):
        print(sample[sample_cols].to_string(index=False))

    try:
        fc = forecast_next_quarter_end(model, artifacts, cfg)
        print("\nLatest weekly forecast (stub):")
        if {"p10_raw", "p90_raw", "p10_cal", "p90_cal"}.issubset(fc.columns):
            cols = [
                "asof_date",
                "forecast_p50",
                "forecast_target_p50",
                "p10_raw",
                "p90_raw",
                "p10_cal",
                "p90_cal",
                "risk_score_raw",
                "risk_score_cal",
                "q_hat",
                "p10_cal_roll",
                "p90_cal_roll",
                "risk_score_roll",
                "q_hat_roll",
                "forecast_point_naive",
                "low_naive",
                "high_naive",
                "risk_score_naive",
                "q_hat_naive",
                "low_p50",
                "high_p50",
                "risk_score_p50",
                "q_hat_p50",
            ]
            cols = [c for c in cols if c in fc.columns]
            print(fc[cols].to_string(index=False))
        else:
            print(fc.to_string(index=False))
    except Exception as e:
        print(f"\nLive forecast not produced: {e}")


if __name__ == "__main__":
    main()

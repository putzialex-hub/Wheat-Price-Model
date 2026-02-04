from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import AppConfig, DataPaths
from .pipeline import run_training_pipeline, forecast_next_quarter_end


def main() -> None:
    """
    Minimal CLI:
      - trains model
      - prints backtest metrics
      - prints latest weekly forecast stub
    """
    # Example config; adapt paths to your environment.
    cfg = AppConfig(
        data=DataPaths(
            contracts_path=Path("data/matif_wheat_contracts.csv"),
            macro_path=Path("data/macro_features.csv"),
            fundamentals_path=Path("data/fundamentals_features.csv"),
        )
    )

    # You must provide a contract expiry calendar.
    # Recommended: maintain as CSV and load it here.
    # For now, we load from JSON file if exists.
    expiry_path = Path("data/expiry_calendar.json")
    if not expiry_path.exists():
        raise SystemExit(
            "Missing data/expiry_calendar.json. Provide mapping {contract: 'YYYY-MM-DD'} "
            "for all contracts in contracts table."
        )

    expiry_calendar = json.loads(expiry_path.read_text(encoding="utf-8"))
    expiry_calendar = {k: pd.Timestamp(v) for k, v in expiry_calendar.items()}

    model, artifacts, metrics = run_training_pipeline(cfg, expiry_calendar)

    print("Backtest metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    try:
        fc = forecast_next_quarter_end(model, artifacts, cfg)
        print("\nLatest weekly forecast (stub):")
        print(fc.to_string(index=False))
    except Exception as e:
        print(f"\nLive forecast not produced: {e}")


if __name__ == "__main__":
    main()

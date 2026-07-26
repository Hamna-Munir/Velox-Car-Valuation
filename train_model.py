#!/usr/bin/env python3
"""
CLI entry point for training VELOX's models.

All real logic lives in the `velox` package (velox.data, velox.model) so it
can be imported and tested directly, and reused by both this CLI and the
FastAPI service. This script is just the "run it from the command line" glue.

Usage:
    python train_model.py
"""

from velox.logging_config import setup_logging
from velox.model import train_and_save


def main() -> None:
    setup_logging()
    metrics = train_and_save()
    print()
    print("=== Training complete ===")
    print(f"Test R²:        {metrics['r2']:.4f}")
    print(f"CV R² (search):  {metrics['cv_r2']:.4f}")
    print(f"Test MAE:        Rs {metrics['mae']:,.0f}")
    print(f"Test RMSE:       Rs {metrics['rmse']:,.0f}")
    print(f"P10-P90 coverage: {metrics['coverage']['p10_p90_coverage']:.1%} "
          f"(target ~80%)")
    print(f"Best hyperparameters: {metrics['best_params']}")


if __name__ == "__main__":
    main()

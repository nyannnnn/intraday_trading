from pathlib import Path

import pandas as pd

from config import (
    UNIVERSE,
    DATA_DIR,
    MODEL_DIR,
    LABEL_COLUMN,
    FEATURE_COLUMNS,
    TRAIN_VAL_SPLIT_DATE,
)
from quant.quant_model import build_panel_features_and_labels
from ML.ml_model import MLModelConfig, MLSignalModel


def load_panel_ohlcv(data_dir: Path, universe: list[str]) -> pd.DataFrame:
    """Load per-symbol OHLCV CSVs into a (symbol, datetime) panel DataFrame."""
    frames = []
    for sym in universe:
        csv_path = data_dir / f"{sym}.csv"
        if not csv_path.exists():
            print(f"[WARN] Missing CSV for {sym}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df.columns = [c.lower() for c in df.columns]

        if "datetime" not in df.columns:
            raise ValueError(f"{csv_path} has no 'datetime' column")

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

        needed = ["open", "high", "low", "close", "volume"]
        for col in needed:
            if col not in df.columns:
                raise ValueError(f"{csv_path} missing required column '{col}'")

        df = df[["datetime"] + needed].copy()
        df["symbol"] = sym

        frames.append(df)

    if not frames:
        raise ValueError("No data frames loaded. Check DATA_DIR and UNIVERSE.")

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.set_index(["symbol", "datetime"]).sort_index()
    return all_df


def main():
    """Train the intraday ML model with time-based split and persist it to disk."""
    print("=== Step 1: Load raw OHLCV data ===")
    panel_ohlcv = load_panel_ohlcv(DATA_DIR, UNIVERSE)
    print("Panel OHLCV shape:", panel_ohlcv.shape)
    print("Panel OHLCV index example:", panel_ohlcv.index[:5])

    print("\n=== Step 2: Build features & labels ===")
    panel_with_features = build_panel_features_and_labels(panel_ohlcv)
    print("Panel with features+labels shape:", panel_with_features.shape)
    print(
        "Panel columns (first 20):",
        list(panel_with_features.columns[:20]),
    )

    print("\n=== Step 3: Train ML model (time-based split) ===")
    cfg = MLModelConfig(
        label_col=LABEL_COLUMN,
        feature_cols=FEATURE_COLUMNS,
        train_val_split_date=TRAIN_VAL_SPLIT_DATE,
    )

    model = MLSignalModel(cfg)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "intraday_gbm.joblib"

    metrics = model.fit(panel_with_features, save_path=model_path)

    print("\n=== Validation metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

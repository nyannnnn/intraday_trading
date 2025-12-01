# ML/ml_model.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score


@dataclass
class MLModelConfig:
    """
    Configuration for the ML signal model.

    train_val_split_date:
        Calendar date (e.g. "2024-07-01") used to split train vs validation.
    label_col:
        Name of the binary label column (0/1).
    feature_cols:
        List of feature column names.
    """
    label_col: str
    feature_cols: List[str]
    train_val_split_date: str

    # Model hyper-params (tune later if you want)
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 3
    random_state: int = 42


class MLSignalModel:
    """
    Wraps a scikit-learn classifier + time-based train/validation logic.
    """

    def __init__(self, config: MLModelConfig):
        self.cfg = config
        self.clf: GradientBoostingClassifier | None = None

    # ---------- Internal helpers ----------

    def _time_based_train_val_split(
        self, panel_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Given a panel DataFrame indexed by (symbol, datetime),
        split into train/val by calendar date.
        """
        # Make sure required columns are present and non-NA
        needed = self.cfg.feature_cols + [self.cfg.label_col]
        df = panel_df.dropna(subset=needed).copy()

        # Extract the datetime level; handle tz-aware index safely
        dt_index = df.index.get_level_values("datetime")
        if getattr(dt_index, "tz", None) is not None:
            # convert to UTC then drop tz-info so we can compare to naive Timestamp
            dt_index = dt_index.tz_convert("UTC").tz_localize(None)

        split_ts = pd.to_datetime(self.cfg.train_val_split_date)

        train_mask = dt_index <= split_ts
        val_mask = dt_index > split_ts

        if train_mask.sum() == 0 or val_mask.sum() == 0:
            raise ValueError(
                f"Time-based split produced empty train/val sets.\n"
                f"  First timestamp: {dt_index.min()}, last: {dt_index.max()}, "
                f"split: {split_ts}"
            )

        df_train = df.loc[train_mask]
        df_val = df.loc[val_mask]

        X_train = df_train[self.cfg.feature_cols]
        y_train = df_train[self.cfg.label_col].astype(int)

        X_val = df_val[self.cfg.feature_cols]
        y_val = df_val[self.cfg.label_col].astype(int)

        return X_train, X_val, y_train, y_val

    # ---------- Public API ----------

    def fit(self, panel_df: pd.DataFrame, save_path: Path | None = None) -> Dict[str, float]:
        """
        Fit the classifier on the training portion of `panel_df` and
        evaluate on the validation portion.
        """
        X_train, X_val, y_train, y_val = self._time_based_train_val_split(panel_df)

        clf = GradientBoostingClassifier(
            n_estimators=self.cfg.n_estimators,
            learning_rate=self.cfg.learning_rate,
            max_depth=self.cfg.max_depth,
            random_state=self.cfg.random_state,
        )
        clf.fit(X_train, y_train)
        self.clf = clf

        # Validation metrics
        val_proba = clf.predict_proba(X_val)[:, 1]
        val_pred = (val_proba >= 0.5).astype(int)

        try:
            val_auc = roc_auc_score(y_val, val_proba)
        except ValueError:
            # If only one class is present in y_val
            val_auc = float("nan")

        val_acc = accuracy_score(y_val, val_pred)

        metrics = {
            "val_auc": float(val_auc),
            "val_accuracy": float(val_acc),
            "n_train": float(len(y_train)),
            "n_val": float(len(y_val)),
            "pos_rate_train": float(y_train.mean()),
            "pos_rate_val": float(y_val.mean()),
        }

        # Optionally persist to disk
        if save_path is not None:
            joblib.dump(clf, save_path)
        return metrics

    def predict_proba(self, panel_df: pd.DataFrame) -> pd.Series:
        """
        Given a panel feature DataFrame (same feature_cols, same index type),
        return a Series of predicted p_up (probability of label=1).
        """
        if self.clf is None:
            raise RuntimeError("Model is not fitted. Call fit(...) first.")

        df = panel_df.dropna(subset=self.cfg.feature_cols).copy()
        X = df[self.cfg.feature_cols]

        proba = self.clf.predict_proba(X)[:, 1]
        return pd.Series(proba, index=df.index, name="p_up")

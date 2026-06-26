from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

from snr_ml_features import default_feature_columns, make_training_frame


@dataclass
class ModelBundle:
    feature_columns: list[str]
    event_model: HistGradientBoostingClassifier
    direction_model: HistGradientBoostingClassifier
    metadata: dict


def _build_event_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=250,
        min_samples_leaf=40,
        random_state=42,
    )


def _build_direction_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=5,
        max_iter=200,
        min_samples_leaf=30,
        random_state=42,
    )


def train_model_bundle(
    feature_df: pd.DataFrame,
    feature_columns: Optional[Iterable[str]] = None,
    target_column: str = "large_move",
    direction_target_column: str = "large_up_move",
    split_ratio: float = 0.8,
) -> tuple[ModelBundle, dict]:
    columns = list(feature_columns or default_feature_columns())
    train_df = make_training_frame(feature_df, feature_columns=columns, target_column=target_column)
    split_idx = max(1, int(len(train_df) * split_ratio))

    X_train = train_df[columns].iloc[:split_idx]
    y_train = train_df[target_column].iloc[:split_idx]
    X_test = train_df[columns].iloc[split_idx:]
    y_test = train_df[target_column].iloc[split_idx:]

    event_model = _build_event_model()
    event_model.fit(X_train, y_train)

    event_prob = event_model.predict_proba(X_test)[:, 1] if len(X_test) > 0 else np.array([])
    event_pred = (event_prob >= 0.5).astype(int) if len(event_prob) > 0 else np.array([])

    direction_df = feature_df.copy()
    direction_df = direction_df.loc[
        (direction_df[direction_target_column] == 1) | (direction_df["large_down_move"] == 1)
    ].copy()
    direction_df = direction_df.dropna(subset=columns + [direction_target_column])
    dir_split_idx = max(1, int(len(direction_df) * split_ratio))

    X_dir_train = direction_df[columns].iloc[:dir_split_idx]
    y_dir_train = direction_df[direction_target_column].astype(int).iloc[:dir_split_idx]
    X_dir_test = direction_df[columns].iloc[dir_split_idx:]
    y_dir_test = direction_df[direction_target_column].astype(int).iloc[dir_split_idx:]

    direction_model = _build_direction_model()
    direction_model.fit(X_dir_train, y_dir_train)

    dir_prob = direction_model.predict_proba(X_dir_test)[:, 1] if len(X_dir_test) > 0 else np.array([])
    dir_pred = (dir_prob >= 0.5).astype(int) if len(dir_prob) > 0 else np.array([])

    metrics = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "event_roc_auc": float(roc_auc_score(y_test, event_prob)) if len(event_prob) > 0 and y_test.nunique() > 1 else None,
        "event_pr_auc": float(average_precision_score(y_test, event_prob)) if len(event_prob) > 0 else None,
        "event_report": classification_report(y_test, event_pred, digits=4, zero_division=0) if len(event_prob) > 0 else "",
        "direction_rows": int(len(direction_df)),
        "direction_roc_auc": float(roc_auc_score(y_dir_test, dir_prob)) if len(dir_prob) > 0 and y_dir_test.nunique() > 1 else None,
        "direction_report": classification_report(y_dir_test, dir_pred, digits=4, zero_division=0) if len(dir_prob) > 0 else "",
    }

    bundle = ModelBundle(
        feature_columns=columns,
        event_model=event_model,
        direction_model=direction_model,
        metadata=metrics,
    )
    return bundle, metrics


def save_model_bundle(bundle: ModelBundle, output_dir: str | Path) -> dict:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    event_path = path / "event_model.joblib"
    direction_path = path / "direction_model.joblib"
    meta_path = path / "model_metadata.joblib"

    joblib.dump(bundle.event_model, event_path)
    joblib.dump(bundle.direction_model, direction_path)
    joblib.dump({"feature_columns": bundle.feature_columns, "metadata": bundle.metadata}, meta_path)

    return {
        "event_model": str(event_path),
        "direction_model": str(direction_path),
        "metadata": str(meta_path),
    }


def load_model_bundle(model_dir: str | Path) -> ModelBundle:
    path = Path(model_dir)
    event_model = joblib.load(path / "event_model.joblib")
    direction_model = joblib.load(path / "direction_model.joblib")
    meta = joblib.load(path / "model_metadata.joblib")
    return ModelBundle(
        feature_columns=list(meta["feature_columns"]),
        event_model=event_model,
        direction_model=direction_model,
        metadata=dict(meta.get("metadata", {})),
    )


def predict_latest(bundle: ModelBundle, feature_df: pd.DataFrame) -> dict:
    latest = feature_df[bundle.feature_columns].dropna().iloc[[-1]]
    event_prob = float(bundle.event_model.predict_proba(latest)[0, 1])
    up_prob = float(bundle.direction_model.predict_proba(latest)[0, 1])
    return {
        "large_move_probability": event_prob,
        "up_move_probability": up_prob,
        "down_move_probability": 1.0 - up_prob,
        "timestamp": str(latest.index[-1]),
    }

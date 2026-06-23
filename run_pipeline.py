from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from integration_bridge import append_signal_csv, build_signal_payload, save_indicator_snapshot, save_payload_json, save_tradingview_bridge
from model_runtime import load_model_bundle, predict_latest, save_model_bundle, train_model_bundle
from screenshot_reader import analyze_screenshot, choose_latest_screenshot
from snr_ml_features import FeatureConfig, add_snr_features, build_impulse_targets, load_market_data
from telegram_bot_stub import send_signal_payload


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def build_feature_frame(config: dict[str, Any], base_dir: Path) -> pd.DataFrame:
    cfg = FeatureConfig(
        symbol=config["symbol"],
        interval=config["timeframe"],
        period=config["period"],
        timezone=config["timezone"],
        data_source=config.get("data_source", "auto"),
        target_horizon=int(config["target_horizon"]),
        target_threshold=float(config["large_move_threshold"]),
    )
    csv_path = config.get("csv_path") or None
    if csv_path:
        csv_path = str(resolve_path(base_dir, csv_path))

    df = load_market_data(
        csv_path=csv_path,
        symbol=cfg.symbol,
        interval=cfg.interval,
        period=cfg.period,
        timezone=cfg.timezone,
        data_source=cfg.data_source,
    )
    df = add_snr_features(df, config=cfg)
    df = build_impulse_targets(df, horizon=cfg.target_horizon, move_threshold=cfg.target_threshold)
    return df


def ensure_models(feature_df: pd.DataFrame, config: dict[str, Any], base_dir: Path):
    models_dir = resolve_path(base_dir, config["models_dir"])
    meta_file = models_dir / "model_metadata.joblib"
    if meta_file.exists():
        return load_model_bundle(models_dir)

    if not config.get("train_if_missing_models", True):
        raise FileNotFoundError("Model files are missing and auto-training is disabled.")

    bundle, metrics = train_model_bundle(feature_df)
    save_model_bundle(bundle, models_dir)
    metrics_path = models_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return bundle


def export_screenshot_snapshot(config: dict[str, Any], base_dir: Path) -> dict | None:
    if not config.get("screenshot_fallback_enabled", True):
        return None
    inbox = resolve_path(base_dir, config["screenshot_inbox"])
    latest = choose_latest_screenshot(inbox)
    if latest is None:
        return None

    snapshot = analyze_screenshot(latest)
    snapshot_dir = resolve_path(base_dir, config["snapshot_output"])
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / "latest_screenshot_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot.__dict__, indent=2), encoding="utf-8")
    return snapshot.__dict__


def main(config_path: str = "product_config.example.json") -> dict[str, Any]:
    base_dir = Path(__file__).resolve().parent
    config = load_config(resolve_path(base_dir, config_path))

    feature_df = build_feature_frame(config, base_dir)
    bundle = ensure_models(feature_df, config, base_dir)
    prediction = predict_latest(bundle, feature_df)

    latest_row = feature_df.loc[pd.to_datetime(prediction["timestamp"])]
    payload = build_signal_payload(
        latest_row,
        large_move_probability=prediction["large_move_probability"],
        up_move_probability=prediction["up_move_probability"],
        down_move_probability=prediction["down_move_probability"],
        symbol=config["symbol"],
        timeframe=config["timeframe"],
        notes="Pipeline export aligned with SNR_Line.pine",
    )

    outputs_dir = resolve_path(base_dir, config["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    save_payload_json(payload, outputs_dir / "latest_signal.json")
    save_indicator_snapshot(payload, outputs_dir / "indicator_snapshot.json")
    bridge_path = save_tradingview_bridge(payload, outputs_dir / "tradingview_ai_bridge.txt")
    append_signal_csv(payload, outputs_dir / "signal_history.csv")

    screenshot_snapshot = export_screenshot_snapshot(config, base_dir)

    telegram_sent = False
    if config.get("telegram_enabled", False) and payload.large_move_probability >= float(config.get("telegram_min_probability", 0.7)):
        send_signal_payload(
            payload,
            bot_token=config.get("telegram_bot_token") or None,
            chat_id=config.get("telegram_chat_id") or None,
        )
        telegram_sent = True

    result = {
        "prediction": prediction,
        "telegram_sent": telegram_sent,
        "screenshot_snapshot": screenshot_snapshot,
        "outputs_dir": str(outputs_dir),
        "tradingview_bridge_path": str(bridge_path),
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from integration_bridge import append_signal_csv, build_signal_payload, save_indicator_snapshot, save_payload_json, save_tradingview_bridge
from model_runtime import load_model_bundle, predict_latest
from run_pipeline import build_feature_frame, export_screenshot_snapshot, load_config, resolve_path
from telegram_bot_stub import send_signal_payload


def main(config_path: str = "product_config.example.json") -> dict:
    base_dir = Path(__file__).resolve().parent
    config = load_config(resolve_path(base_dir, config_path))

    feature_df = build_feature_frame(config, base_dir)
    models_dir = resolve_path(base_dir, config["models_dir"])
    bundle = load_model_bundle(models_dir)
    prediction = predict_latest(bundle, feature_df)

    latest_row = feature_df.loc[pd.to_datetime(prediction["timestamp"])]
    payload = build_signal_payload(
        latest_row,
        large_move_probability=prediction["large_move_probability"],
        up_move_probability=prediction["up_move_probability"],
        down_move_probability=prediction["down_move_probability"],
        symbol=config["symbol"],
        timeframe=config["timeframe"],
        notes="Inference export aligned with SNR_Line.pine",
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
        "status": "inference_complete",
        "prediction": prediction,
        "telegram_sent": telegram_sent,
        "screenshot_snapshot": screenshot_snapshot,
        "outputs_dir": str(outputs_dir),
        "tradingview_bridge_path": str(bridge_path),
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    arg_path = sys.argv[1] if len(sys.argv) > 1 else "product_config.example.json"
    main(arg_path)

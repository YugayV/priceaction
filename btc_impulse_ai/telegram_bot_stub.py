from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import requests

try:
    from integration_bridge import SignalPayload
except ImportError:  # Allows package-style execution later.
    from .integration_bridge import SignalPayload


def format_signal_message(payload: SignalPayload) -> str:
    return (
        f"BTC Impulse AI\n"
        f"Symbol: {payload.symbol}\n"
        f"TF: {payload.timeframe}\n"
        f"Time: {payload.timestamp}\n"
        f"Close: {payload.close:.2f}\n"
        f"Large Move Prob: {payload.large_move_probability:.2%}\n"
        f"Up Prob: {payload.up_move_probability:.2%}\n"
        f"Down Prob: {payload.down_move_probability:.2%}\n"
        f"SNR: {payload.snr_bias}\n"
        f"Session: {payload.session_state}\n"
        f"KZ: {payload.kill_zone_state}\n"
        f"SMC: {payload.smc_state}\n"
        f"Entry: {payload.entry_state}\n"
        f"Notes: {payload.notes or '-'}"
    )


def send_telegram_message(
    text: str,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    timeout: int = 20,
) -> dict:
    token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    target_chat = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not target_chat:
        raise ValueError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID or pass them directly.")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    response = requests.post(
        url,
        json={
            "chat_id": target_chat,
            "text": text,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def send_signal_payload(payload: SignalPayload, bot_token: Optional[str] = None, chat_id: Optional[str] = None) -> dict:
    return send_telegram_message(format_signal_message(payload), bot_token=bot_token, chat_id=chat_id)


def load_latest_snapshot(snapshot_path: str | Path) -> str:
    path = Path(snapshot_path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")

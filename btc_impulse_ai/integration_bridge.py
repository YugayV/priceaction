from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class SignalPayload:
    timestamp: str
    symbol: str
    timeframe: str
    close: float
    large_move_probability: float
    up_move_probability: float
    down_move_probability: float
    snr_bias: str
    session_state: str
    kill_zone_state: str
    smc_state: str
    entry_state: str
    pa_signal: str = "WAIT"
    pa_bias: str = "Neutral"
    pa_reason: str = ""
    pa_score: float = 0.0
    notes: str = ""


SESSION_TO_PINE = {
    "asia": "Asia",
    "london": "London",
    "new_york": "New York",
    "off_hours": "No Session",
}

KILL_ZONE_TO_PINE = {
    "silver": "Silver",
    "london_kz": "London KZ",
    "ny_kz": "NY KZ",
    "no_kz": "No Session",
}

ENTRY_TO_PINE = {
    "wait": "Wait",
    "snr_retest_long": "SNR Long",
    "snr_reclaim_long": "SNR Long",
    "snr_retest_short": "SNR Short",
    "snr_reject_short": "SNR Short",
    "long": "Long",
    "short": "Short",
    "fvg_long": "FVG Long",
    "fvg_short": "FVG Short",
}


def classify_snr_bias(row: pd.Series) -> str:
    if row.get("close", 0.0) > row.get("snr_upper", float("inf")):
        return "above_upper"
    if row.get("close", 0.0) < row.get("snr_lower", float("-inf")):
        return "below_lower"
    if row.get("close", 0.0) >= row.get("reg_basis", 0.0):
        return "above_basis"
    return "below_basis"


def classify_session_state(row: pd.Series) -> str:
    if int(row.get("is_ny", 0)) == 1:
        return "new_york"
    if int(row.get("is_london", 0)) == 1:
        return "london"
    if int(row.get("is_asia", 0)) == 1:
        return "asia"
    return "off_hours"


def classify_kill_zone_state(row: pd.Series) -> str:
    if int(row.get("is_silver", 0)) == 1:
        return "silver"
    if int(row.get("is_london_kz", 0)) == 1:
        return "london_kz"
    if int(row.get("is_ny_kz", 0)) == 1:
        return "ny_kz"
    return "no_kz"


def classify_smc_state(row: pd.Series) -> str:
    if int(row.get("bull_choch", 0)) == 1:
        return "bull_choch"
    if int(row.get("bear_choch", 0)) == 1:
        return "bear_choch"
    if int(row.get("in_bull_fvg", 0)) == 1:
        return "bull_fvg_active"
    if int(row.get("in_bear_fvg", 0)) == 1:
        return "bear_fvg_active"
    if int(row.get("bull_break", 0)) == 1:
        return "bull_bos"
    if int(row.get("bear_break", 0)) == 1:
        return "bear_bos"
    return "neutral"


def classify_entry_state(row: pd.Series) -> str:
    if int(row.get("snr_bull_retest", 0)) == 1:
        return "snr_retest_long"
    if int(row.get("snr_bear_retest", 0)) == 1:
        return "snr_retest_short"
    if int(row.get("snr_long_reclaim", 0)) == 1:
        return "snr_reclaim_long"
    if int(row.get("snr_short_reject", 0)) == 1:
        return "snr_reject_short"
    return "wait"


def classify_pa_signal(row: pd.Series) -> tuple[str, str, str, float]:
    pa_long_sweep_reversal = int(row.get("bull_sweep", 0)) == 1 and (
        int(row.get("bull_choch", 0)) == 1
        or int(row.get("snr_long_reclaim", 0)) == 1
        or int(row.get("in_bull_fvg", 0)) == 1
        or int(row.get("bull_pa_rejection", 0)) == 1
    )
    pa_short_sweep_reversal = int(row.get("bear_sweep", 0)) == 1 and (
        int(row.get("bear_choch", 0)) == 1
        or int(row.get("snr_short_reject", 0)) == 1
        or int(row.get("in_bear_fvg", 0)) == 1
        or int(row.get("bear_pa_rejection", 0)) == 1
    )
    pa_long_breakout = int(row.get("snr_bull_breakout", 0)) == 1 or (
        int(row.get("bull_break", 0)) == 1 and int(row.get("trend_bull", 0)) == 1
    )
    pa_short_breakout = int(row.get("snr_bear_breakout", 0)) == 1 or (
        int(row.get("bear_break", 0)) == 1 and int(row.get("trend_bear", 0)) == 1
    )
    pa_long_reclaim = int(row.get("snr_long_reclaim", 0)) == 1 or int(row.get("snr_bull_retest", 0)) == 1
    pa_short_reject = int(row.get("snr_short_reject", 0)) == 1 or int(row.get("snr_bear_retest", 0)) == 1
    pa_score = float(
        int(row.get("bull_sweep", 0))
        + int(row.get("bear_sweep", 0))
        + int(row.get("bull_choch", 0))
        + int(row.get("bear_choch", 0))
        + int(row.get("in_bull_fvg", 0))
        + int(row.get("in_bear_fvg", 0))
        + int(row.get("snr_long_reclaim", 0))
        + int(row.get("snr_short_reject", 0))
        + int(row.get("snr_bull_breakout", 0))
        + int(row.get("snr_bear_breakout", 0))
        + int(row.get("bull_pa_continuation", 0))
        + int(row.get("bear_pa_continuation", 0))
    )
    if pa_long_sweep_reversal:
        return ("Liquidity Sweep Long", "Long", "Lows swept and price reclaimed with bullish confirmation", pa_score)
    if pa_short_sweep_reversal:
        return ("Liquidity Sweep Short", "Short", "Highs swept and price rejected with bearish confirmation", pa_score)
    if pa_long_breakout:
        return ("Bull Breakout", "Long", "Price broke structure or SNR resistance", pa_score)
    if pa_short_breakout:
        return ("Bear Breakout", "Short", "Price broke structure or SNR support", pa_score)
    if pa_long_reclaim:
        return ("SNR Reclaim Long", "Long", "Support held and price reclaimed above basis", pa_score)
    if pa_short_reject:
        return ("SNR Reject Short", "Short", "Resistance held and price rejected below basis", pa_score)
    return ("WAIT", "Neutral", "No strong PA trigger", pa_score)


def build_signal_payload(
    row: pd.Series,
    large_move_probability: float,
    up_move_probability: float,
    down_move_probability: float,
    symbol: str = "BTC-USD",
    timeframe: str = "15m",
    notes: str = "",
) -> SignalPayload:
    pa_signal, pa_bias, pa_reason, pa_score = classify_pa_signal(row)
    return SignalPayload(
        timestamp=str(row.name),
        symbol=symbol,
        timeframe=timeframe,
        close=float(row["close"]),
        large_move_probability=float(large_move_probability),
        up_move_probability=float(up_move_probability),
        down_move_probability=float(down_move_probability),
        snr_bias=classify_snr_bias(row),
        session_state=classify_session_state(row),
        kill_zone_state=classify_kill_zone_state(row),
        smc_state=classify_smc_state(row),
        entry_state=classify_entry_state(row),
        pa_signal=pa_signal,
        pa_bias=pa_bias,
        pa_reason=pa_reason,
        pa_score=pa_score,
        notes=notes,
    )


def sanitize_bridge_text(value: str) -> str:
    return value.replace("|", "/").replace("\r", " ").replace("\n", " ").strip()


def payload_to_pine_inputs(payload: SignalPayload) -> Dict[str, Any]:
    direction = "Bullish" if payload.up_move_probability >= payload.down_move_probability else "Bearish"
    confidence = round(max(payload.up_move_probability, payload.down_move_probability) * 100, 1)
    session_value = KILL_ZONE_TO_PINE.get(payload.kill_zone_state, "No Session")
    if session_value == "No Session":
        session_value = SESSION_TO_PINE.get(payload.session_state, "No Session")
    entry_value = ENTRY_TO_PINE.get(payload.entry_state, "Wait")
    notes_value = sanitize_bridge_text(payload.notes)
    pa_signal_value = sanitize_bridge_text(payload.pa_signal)
    pa_bias_value = sanitize_bridge_text(payload.pa_bias)
    pa_reason_value = sanitize_bridge_text(payload.pa_reason)
    return {
        "ai_direction": direction,
        "ai_probability_large_move": round(payload.large_move_probability * 100, 1),
        "ai_confidence": confidence,
        "ai_session": session_value,
        "ai_entry_state": entry_value,
        "ai_notes": notes_value,
        "pa_signal": pa_signal_value,
        "pa_bias": pa_bias_value,
        "pa_reason": pa_reason_value,
    }


def build_pine_bridge_payload(payload: SignalPayload) -> str:
    pine_inputs = payload_to_pine_inputs(payload)
    return (
        "AIBRIDGE"
        f"|{pine_inputs['ai_direction']}"
        f"|{pine_inputs['ai_probability_large_move']:.1f}"
        f"|{pine_inputs['ai_confidence']:.1f}"
        f"|{pine_inputs['ai_session']}"
        f"|{pine_inputs['ai_entry_state']}"
        f"|{pine_inputs['ai_notes']}"
        f"|{pine_inputs['pa_signal']}"
        f"|{pine_inputs['pa_bias']}"
        f"|{pine_inputs['pa_reason']}"
    )


def payload_to_indicator_fields(payload: SignalPayload) -> Dict[str, Any]:
    pine_inputs = payload_to_pine_inputs(payload)
    direction = "bullish" if payload.up_move_probability >= payload.down_move_probability else "bearish"
    confidence = round(max(payload.up_move_probability, payload.down_move_probability) * 100, 1)
    return {
        "ai_symbol": payload.symbol,
        "ai_timeframe": payload.timeframe,
        "ai_probability_large_move": round(payload.large_move_probability * 100, 1),
        "ai_direction": direction,
        "ai_confidence": confidence,
        "ai_session": payload.session_state,
        "ai_kill_zone": payload.kill_zone_state,
        "ai_smc_state": payload.smc_state,
        "ai_entry_state": payload.entry_state,
        "pa_signal": payload.pa_signal,
        "pa_bias": payload.pa_bias,
        "pa_reason": payload.pa_reason,
        "pa_score": payload.pa_score,
        "ai_notes": payload.notes,
        "pine_ai_direction": pine_inputs["ai_direction"],
        "pine_ai_probability_large_move": pine_inputs["ai_probability_large_move"],
        "pine_ai_confidence": pine_inputs["ai_confidence"],
        "pine_ai_session": pine_inputs["ai_session"],
        "pine_ai_entry_state": pine_inputs["ai_entry_state"],
        "pine_ai_notes": pine_inputs["ai_notes"],
        "pine_pa_signal": pine_inputs["pa_signal"],
        "pine_pa_bias": pine_inputs["pa_bias"],
        "pine_pa_reason": pine_inputs["pa_reason"],
        "pine_bridge_payload": build_pine_bridge_payload(payload),
    }


def save_payload_json(payload: SignalPayload, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(payload), indent=2), encoding="utf-8")
    return path


def save_indicator_snapshot(payload: SignalPayload, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = payload_to_indicator_fields(payload)
    path.write_text(json.dumps(fields, indent=2), encoding="utf-8")
    return path


def save_tradingview_bridge(payload: SignalPayload, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_pine_bridge_payload(payload), encoding="utf-8")
    return path


def append_signal_csv(payload: SignalPayload, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([asdict(payload)])
    if path.exists():
        row.to_csv(path, mode="a", header=False, index=False)
    else:
        row.to_csv(path, index=False)
    return path

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone as dt_timezone
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
from typing import Iterable, Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass
class FeatureConfig:
    symbol: str = "BTC-USD"
    interval: str = "15m"
    period: str = "120d"
    timezone: str = "Asia/Seoul"
    data_source: str = "auto"
    snr_length: int = 200
    outer_band_width: int = 20
    atr_period: int = 14
    rsi_length: int = 14
    structure_lookback: int = 12
    session_range_length: int = 50
    snr_touch_atr: float = 0.35
    flow_len: int = 24
    flow_smoothing: int = 10
    flow_threshold: float = 15.0
    obos_band_factor: float = 1.8
    fvg_min_atr: float = 0.15
    target_horizon: int = 12
    target_threshold: float = 700.0


def _ensure_datetime_index(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    data = df.copy()
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
        data = data.set_index("timestamp")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex or a 'timestamp' column.")
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    data.index = data.index.tz_convert(timezone)
    data = data.sort_index()
    data.columns = [str(col).lower() for col in data.columns]
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return data


def _interval_to_minutes(interval: str) -> int:
    value = interval.strip().lower()
    if value.endswith("m"):
        return int(value[:-1])
    if value.endswith("h"):
        return int(value[:-1]) * 60
    if value.endswith("d"):
        return int(value[:-1]) * 1440
    raise ValueError(f"Unsupported interval: {interval}")


def _period_to_bars(period: str, interval: str, max_bars: int = 5000) -> int:
    value = period.strip().lower()
    if not value:
        return min(1000, max_bars)
    unit = value[-1]
    amount = int(value[:-1])
    minutes_per_bar = _interval_to_minutes(interval)
    if unit == "d":
        total_minutes = amount * 1440
    elif unit == "w":
        total_minutes = amount * 10080
    elif unit == "y":
        total_minutes = amount * 525600
    elif unit == "h":
        total_minutes = amount * 60
    elif unit == "m":
        total_minutes = amount
    else:
        total_minutes = 1440 * 30
    return max(50, min(max_bars, int(math.ceil(total_minutes / max(minutes_per_bar, 1)))))


def _period_to_timedelta(period: str) -> timedelta:
    value = period.strip().lower()
    if not value:
        return timedelta(days=30)
    unit = value[-1]
    amount = int(value[:-1])
    if unit == "d":
        return timedelta(days=amount)
    if unit == "w":
        return timedelta(weeks=amount)
    if unit == "y":
        return timedelta(days=365 * amount)
    if unit == "h":
        return timedelta(hours=amount)
    if unit == "m":
        return timedelta(minutes=amount)
    return timedelta(days=30)


def _normalize_binance_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper().replace("-", "").replace("/", "")
    if cleaned.endswith("USD") and not cleaned.endswith("USDT"):
        return cleaned[:-3] + "USDT"
    return cleaned


def _fetch_json(url: str) -> list:
    with urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_binance_data(symbol: str, interval: str, period: str, timezone: str) -> pd.DataFrame:
    interval_value = interval.strip().lower()
    supported_intervals = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"}
    if interval_value not in supported_intervals:
        raise ValueError(f"Binance does not support interval '{interval}'.")

    bars_to_fetch = _period_to_bars(period, interval_value)
    symbol_value = _normalize_binance_symbol(symbol)
    base_url = "https://api.binance.com/api/v3/klines"
    all_rows: list[list] = []
    end_time_ms: Optional[int] = None

    while len(all_rows) < bars_to_fetch:
        limit = min(1000, bars_to_fetch - len(all_rows))
        params = {"symbol": symbol_value, "interval": interval_value, "limit": limit}
        if end_time_ms is not None:
            params["endTime"] = end_time_ms
        raw_rows = _fetch_json(f"{base_url}?{urlencode(params)}")
        if not raw_rows:
            break
        all_rows = raw_rows + all_rows
        if len(raw_rows) < limit:
            break
        end_time_ms = int(raw_rows[0][0]) - 1

    if not all_rows:
        raise ValueError("No market data returned from Binance.")

    data = pd.DataFrame(
        all_rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms", utc=True)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return _ensure_datetime_index(data[["timestamp"] + numeric_cols], timezone)


def _load_yfinance_data(
    symbol: str,
    interval: str,
    period: str,
    timezone: str,
    end_datetime: Optional[datetime | pd.Timestamp] = None,
    use_datetime_now: bool = False,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Install yfinance or switch to Binance/CSV.") from exc

    download_kwargs: dict[str, object] = {
        "interval": interval,
        "auto_adjust": False,
        "progress": False,
    }
    if use_datetime_now or end_datetime is not None:
        resolved_end = end_datetime
        if resolved_end is None:
            resolved_end = datetime.now(dt_timezone.utc)
        resolved_end_ts = pd.Timestamp(resolved_end)
        if resolved_end_ts.tzinfo is None:
            resolved_end_ts = resolved_end_ts.tz_localize("UTC")
        else:
            resolved_end_ts = resolved_end_ts.tz_convert("UTC")
        resolved_start_ts = resolved_end_ts - _period_to_timedelta(period)
        download_kwargs["start"] = resolved_start_ts.to_pydatetime()
        download_kwargs["end"] = resolved_end_ts.to_pydatetime()
    else:
        download_kwargs["period"] = period

    raw = yf.download(symbol, **download_kwargs)
    if raw.empty:
        raise ValueError("No market data returned from yfinance.")
    raw = raw.rename(columns=str.lower)
    raw["volume"] = raw["volume"].fillna(0.0)
    raw.index = pd.to_datetime(raw.index, utc=True).tz_convert(timezone)
    return raw


def load_market_data(
    csv_path: Optional[str] = None,
    symbol: str = "BTC-USD",
    interval: str = "15m",
    period: str = "120d",
    timezone: str = "Asia/Seoul",
    data_source: str = "auto",
    end_datetime: Optional[datetime | pd.Timestamp] = None,
    use_yfinance_now: bool = False,
) -> pd.DataFrame:
    if csv_path:
        raw = pd.read_csv(csv_path)
        return _ensure_datetime_index(raw, timezone)

    source = data_source.strip().lower()
    errors: list[str] = []
    if source not in {"auto", "binance", "yfinance", "yf", "yfinance_now", "yf_now"}:
        raise ValueError(f"Unsupported data_source: {data_source}")

    if source in {"auto", "binance"}:
        try:
            return _load_binance_data(symbol=symbol, interval=interval, period=period, timezone=timezone)
        except Exception as exc:
            errors.append(f"binance: {exc}")
            if source == "binance":
                raise

    if source in {"auto", "yfinance", "yf", "yfinance_now", "yf_now"}:
        try:
            return _load_yfinance_data(
                symbol=symbol,
                interval=interval,
                period=period,
                timezone=timezone,
                end_datetime=end_datetime,
                use_datetime_now=use_yfinance_now or source in {"yfinance_now", "yf_now"},
            )
        except Exception as exc:
            errors.append(f"yfinance: {exc}")

    raise ValueError("No market data returned. Sources tried -> " + " | ".join(errors))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_down = down.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _rolling_linreg_endpoint(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = np.sum((x - x_mean) ** 2)

    def _endpoint(values: np.ndarray) -> float:
        y = np.asarray(values, dtype=float)
        y_mean = y.mean()
        slope = np.sum((x - x_mean) * (y - y_mean)) / denom
        intercept = y_mean - slope * x_mean
        return intercept + slope * (window - 1)

    return series.rolling(window, min_periods=window).apply(_endpoint, raw=True)


def _session_mask(index: pd.DatetimeIndex, start_hhmm: str, end_hhmm: str) -> pd.Series:
    start_hour = int(start_hhmm[:2])
    start_minute = int(start_hhmm[2:])
    end_hour = int(end_hhmm[:2])
    end_minute = int(end_hhmm[2:])
    minutes = index.hour * 60 + index.minute
    start_total = start_hour * 60 + start_minute
    end_total = end_hour * 60 + end_minute
    if start_total <= end_total:
        mask = (minutes >= start_total) & (minutes < end_total)
    else:
        mask = (minutes >= start_total) | (minutes < end_total)
    return pd.Series(mask.astype(int), index=index)


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    idx = data.index
    data["is_asia"] = _session_mask(idx, "0000", "0900")
    data["is_london"] = _session_mask(idx, "1600", "0100")
    data["is_ny"] = _session_mask(idx, "2100", "0600")
    data["is_london_kz"] = _session_mask(idx, "1600", "1900")
    data["is_ny_kz"] = _session_mask(idx, "2100", "0000")
    data["is_silver"] = _session_mask(idx, "2300", "0000")
    data["active_kz"] = ((data["is_london_kz"] + data["is_ny_kz"] + data["is_silver"]) > 0).astype(int)
    data["hour"] = idx.hour
    data["day_of_week"] = idx.dayofweek
    data["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    return data


def add_snr_features(df: pd.DataFrame, config: Optional[FeatureConfig] = None) -> pd.DataFrame:
    cfg = config or FeatureConfig()
    data = _ensure_datetime_index(df, cfg.timezone)
    data = add_session_features(data)

    typical_price = (data["high"] + data["low"] + data["close"]) / 3.0
    data["atr14"] = _atr(data, cfg.atr_period)
    data["rsi"] = _rsi(data["close"], cfg.rsi_length)
    data["reg_basis"] = _rolling_linreg_endpoint((data["high"] + data["low"]) / 2.0, cfg.snr_length)
    data["reg_slope"] = data["reg_basis"] - data["reg_basis"].shift(5)
    data["stdev"] = ((data["high"] + data["low"]) / 2.0).rolling(cfg.snr_length, min_periods=cfg.snr_length).std()
    data["dev"] = (data["stdev"] * 3.0) / max(cfg.outer_band_width, 1)
    data["snr_upper"] = data["reg_basis"] + (cfg.outer_band_width * data["dev"])
    data["snr_lower"] = data["reg_basis"] - (cfg.outer_band_width * data["dev"])
    vol_roll = data["volume"].rolling(min(cfg.snr_length, 100), min_periods=5)
    data["poc_proxy"] = (typical_price * data["volume"]).rolling(min(cfg.snr_length, 100), min_periods=5).sum() / vol_roll.sum().replace(0, np.nan)
    data["range_high"] = data["high"].rolling(min(cfg.snr_length, cfg.session_range_length), min_periods=5).max()
    data["range_low"] = data["low"].rolling(min(cfg.snr_length, cfg.session_range_length), min_periods=5).min()
    data["pd_eq"] = (data["range_high"] + data["range_low"]) / 2.0
    data["in_discount"] = (data["close"] <= data["pd_eq"]).astype(int)
    data["in_premium"] = (data["close"] >= data["pd_eq"]).astype(int)

    data["vol_pressure"] = ((data["close"] - data["open"]) / (data["high"] - data["low"]).replace(0, np.nan)) * data["volume"].replace(0, np.nan)
    data["fpi_raw"] = data["vol_pressure"].rolling(cfg.flow_len, min_periods=cfg.flow_len).mean()
    fpi_high = data["fpi_raw"].rolling(cfg.flow_len * 2, min_periods=cfg.flow_len).max()
    fpi_low = data["fpi_raw"].rolling(cfg.flow_len * 2, min_periods=cfg.flow_len).min()
    fpi_range = (fpi_high - fpi_low).replace(0, np.nan)
    data["fpi_norm"] = ((data["fpi_raw"] - fpi_low) / fpi_range * 200.0) - 100.0
    data["flow_main"] = _ema(data["fpi_norm"].clip(-100, 100), cfg.flow_smoothing)
    data["flow_signal"] = _ema(data["flow_main"], 5)
    data["flow_is_bull"] = (data["flow_main"] > 0).astype(int)
    data["flow_is_trending"] = (data["flow_main"].abs() > cfg.flow_threshold).astype(int)

    prior_high = data["high"].shift(1).rolling(cfg.structure_lookback, min_periods=cfg.structure_lookback).max()
    prior_low = data["low"].shift(1).rolling(cfg.structure_lookback, min_periods=cfg.structure_lookback).min()
    data["bull_break"] = (data["close"] > prior_high).astype(int)
    data["bear_break"] = (data["close"] < prior_low).astype(int)
    data["bull_sweep"] = ((data["low"] < prior_low) & (data["close"] > prior_low)).astype(int)
    data["bear_sweep"] = ((data["high"] > prior_high) & (data["close"] < prior_high)).astype(int)

    structure_dir = np.where(data["bull_break"] == 1, 1, np.where(data["bear_break"] == 1, -1, np.nan))
    data["structure_dir"] = pd.Series(structure_dir, index=data.index).ffill().fillna(0)
    data["prev_structure_dir"] = data["structure_dir"].shift(1).fillna(0)
    data["bull_choch"] = ((data["bull_break"] == 1) & (data["prev_structure_dir"] < 0)).astype(int)
    data["bear_choch"] = ((data["bear_break"] == 1) & (data["prev_structure_dir"] > 0)).astype(int)

    gap_up = data["low"] - data["high"].shift(2)
    gap_down = data["low"].shift(2) - data["high"]
    data["bull_fvg"] = ((data["low"] > data["high"].shift(2)) & (gap_up >= data["atr14"] * cfg.fvg_min_atr)).astype(int)
    data["bear_fvg"] = ((data["high"] < data["low"].shift(2)) & (gap_down >= data["atr14"] * cfg.fvg_min_atr)).astype(int)
    data["last_bull_fvg_top"] = np.where(data["bull_fvg"] == 1, data["low"], np.nan)
    data["last_bull_fvg_bottom"] = np.where(data["bull_fvg"] == 1, data["high"].shift(2), np.nan)
    data["last_bear_fvg_top"] = np.where(data["bear_fvg"] == 1, data["low"].shift(2), np.nan)
    data["last_bear_fvg_bottom"] = np.where(data["bear_fvg"] == 1, data["high"], np.nan)
    data["last_bull_fvg_top"] = data["last_bull_fvg_top"].ffill()
    data["last_bull_fvg_bottom"] = data["last_bull_fvg_bottom"].ffill()
    data["last_bear_fvg_top"] = data["last_bear_fvg_top"].ffill()
    data["last_bear_fvg_bottom"] = data["last_bear_fvg_bottom"].ffill()
    data["recent_bull_fvg"] = data["bull_fvg"].rolling(cfg.structure_lookback, min_periods=1).max().fillna(0).astype(int)
    data["recent_bear_fvg"] = data["bear_fvg"].rolling(cfg.structure_lookback, min_periods=1).max().fillna(0).astype(int)
    data["in_bull_fvg"] = (
        (data["recent_bull_fvg"] == 1)
        & (data["close"] <= data["last_bull_fvg_top"])
        & (data["close"] >= data["last_bull_fvg_bottom"])
    ).astype(int)
    data["in_bear_fvg"] = (
        (data["recent_bear_fvg"] == 1)
        & (data["close"] <= data["last_bear_fvg_top"])
        & (data["close"] >= data["last_bear_fvg_bottom"])
    ).astype(int)

    data["bar_range"] = (data["high"] - data["low"]).replace(0, np.nan)
    data["candle_body"] = (data["close"] - data["open"]).abs()
    data["upper_wick"] = data["high"] - data[["close", "open"]].max(axis=1)
    data["lower_wick"] = data[["close", "open"]].min(axis=1) - data["low"]
    data["trend_bull"] = ((data["structure_dir"] == 1) | (data["reg_slope"] > 0)).astype(int)
    data["trend_bear"] = ((data["structure_dir"] == -1) | (data["reg_slope"] < 0)).astype(int)

    touch_buffer = data["atr14"] * cfg.snr_touch_atr
    data["snr_long_touch"] = (
        (data["in_discount"] == 1)
        & ((data["low"] <= data["reg_basis"] + touch_buffer) | (data["low"] <= data["snr_lower"] + touch_buffer))
    ).astype(int)
    data["snr_short_touch"] = (
        (data["in_premium"] == 1)
        & ((data["high"] >= data["reg_basis"] - touch_buffer) | (data["high"] >= data["snr_upper"] - touch_buffer))
    ).astype(int)
    data["snr_long_reclaim"] = (
        (data["trend_bull"] == 1)
        & (data["snr_long_touch"] == 1)
        & (data["close"] > data["open"])
        & (data["close"] >= data["reg_basis"] - touch_buffer * 0.35)
    ).astype(int)
    data["snr_short_reject"] = (
        (data["trend_bear"] == 1)
        & (data["snr_short_touch"] == 1)
        & (data["close"] < data["open"])
        & (data["close"] <= data["reg_basis"] + touch_buffer * 0.35)
    ).astype(int)
    data["snr_bull_breakout"] = ((data["trend_bull"] == 1) & (data["close"] > data["snr_upper"]) & (data["close"].shift(1) <= data["snr_upper"].shift(1))).astype(int)
    data["snr_bear_breakout"] = ((data["trend_bear"] == 1) & (data["close"] < data["snr_lower"]) & (data["close"].shift(1) >= data["snr_lower"].shift(1))).astype(int)
    data["recent_snr_bull_breakout"] = data["snr_bull_breakout"].rolling(cfg.structure_lookback, min_periods=1).max().fillna(0).astype(int)
    data["recent_snr_bear_breakout"] = data["snr_bear_breakout"].rolling(cfg.structure_lookback, min_periods=1).max().fillna(0).astype(int)
    data["snr_bull_retest"] = (
        (data["trend_bull"] == 1)
        & (data["recent_snr_bull_breakout"] == 1)
        & (data["low"] <= data["snr_upper"] + touch_buffer)
        & (data["close"] >= data["snr_upper"])
    ).astype(int)
    data["snr_bear_retest"] = (
        (data["trend_bear"] == 1)
        & (data["recent_snr_bear_breakout"] == 1)
        & (data["high"] >= data["snr_lower"] - touch_buffer)
        & (data["close"] <= data["snr_lower"])
    ).astype(int)

    data["bull_pa_rejection"] = (
        (data["active_kz"] == 1)
        & (data["in_discount"] == 1)
        & (data["close"] > data["open"])
        & (data["lower_wick"] > data["candle_body"] * 1.2)
        & (data["close"] >= data["reg_basis"])
    ).astype(int)
    data["bear_pa_rejection"] = (
        (data["active_kz"] == 1)
        & (data["in_premium"] == 1)
        & (data["close"] < data["open"])
        & (data["upper_wick"] > data["candle_body"] * 1.2)
        & (data["close"] <= data["reg_basis"])
    ).astype(int)
    data["bull_pa_continuation"] = (
        (data["active_kz"] == 1)
        & (data["trend_bull"] == 1)
        & (data["close"] > data["open"])
        & (data["close"] > data["high"].shift(1))
        & ((data["candle_body"] / data["bar_range"]) > 0.55)
        & (data["close"] > data["poc_proxy"])
    ).astype(int)
    data["bear_pa_continuation"] = (
        (data["active_kz"] == 1)
        & (data["trend_bear"] == 1)
        & (data["close"] < data["open"])
        & (data["close"] < data["low"].shift(1))
        & ((data["candle_body"] / data["bar_range"]) > 0.55)
        & (data["close"] < data["poc_proxy"])
    ).astype(int)

    data["kz_priority_bonus"] = np.where(data["is_silver"] == 1, 2, 0)
    data["kz_structure_long"] = (
        (data["bull_choch"] == 1)
        | (data["bull_sweep"] == 1)
        | (data["bull_break"].rolling(cfg.structure_lookback, min_periods=1).max() == 1)
        | (data["in_bull_fvg"] == 1)
        | (data["snr_long_reclaim"] == 1)
        | (data["snr_bull_retest"] == 1)
    ).astype(int)
    data["kz_structure_short"] = (
        (data["bear_choch"] == 1)
        | (data["bear_sweep"] == 1)
        | (data["bear_break"].rolling(cfg.structure_lookback, min_periods=1).max() == 1)
        | (data["in_bear_fvg"] == 1)
        | (data["snr_short_reject"] == 1)
        | (data["snr_bear_retest"] == 1)
    ).astype(int)

    data["kz_long_score"] = (
        data["active_kz"]
        + data["trend_bull"]
        + data["in_discount"]
        + data["kz_structure_long"]
        + (data["close"] <= data["poc_proxy"]).fillna(False).astype(int)
        + data["flow_is_bull"]
        + data["in_bull_fvg"]
        + data["snr_long_reclaim"]
        + data["snr_bull_retest"]
        + data["kz_priority_bonus"]
    )
    data["kz_short_score"] = (
        data["active_kz"]
        + data["trend_bear"]
        + data["in_premium"]
        + data["kz_structure_short"]
        + (data["close"] >= data["poc_proxy"]).fillna(False).astype(int)
        + (1 - data["flow_is_bull"])
        + data["in_bear_fvg"]
        + data["snr_short_reject"]
        + data["snr_bear_retest"]
        + data["kz_priority_bonus"]
    )

    data = data.replace([np.inf, -np.inf], np.nan)
    return data


def build_impulse_targets(
    df: pd.DataFrame,
    horizon: int = 12,
    move_threshold: float = 700.0,
    percent_threshold: Optional[float] = None,
) -> pd.DataFrame:
    data = df.copy()
    future_high = pd.concat([data["high"].shift(-i) for i in range(1, horizon + 1)], axis=1).max(axis=1)
    future_low = pd.concat([data["low"].shift(-i) for i in range(1, horizon + 1)], axis=1).min(axis=1)
    data["future_max_high"] = future_high
    data["future_min_low"] = future_low
    data["future_up_move"] = data["future_max_high"] - data["close"]
    data["future_down_move"] = data["close"] - data["future_min_low"]
    data["future_up_return_pct"] = data["future_up_move"] / data["close"] * 100.0
    data["future_down_return_pct"] = data["future_down_move"] / data["close"] * 100.0

    if percent_threshold is not None:
        up_mask = data["future_up_return_pct"] >= percent_threshold
        down_mask = data["future_down_return_pct"] >= percent_threshold
    else:
        up_mask = data["future_up_move"] >= move_threshold
        down_mask = data["future_down_move"] >= move_threshold

    data["large_up_move"] = up_mask.astype(int)
    data["large_down_move"] = down_mask.astype(int)
    data["large_move"] = (up_mask | down_mask).astype(int)
    data["move_direction"] = np.select([up_mask, down_mask], [1, -1], default=0)
    return data


def default_feature_columns() -> list[str]:
    return [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "hour",
        "day_of_week",
        "is_asia",
        "is_london",
        "is_ny",
        "is_london_kz",
        "is_ny_kz",
        "is_silver",
        "active_kz",
        "atr14",
        "rsi",
        "reg_basis",
        "reg_slope",
        "snr_upper",
        "snr_lower",
        "poc_proxy",
        "pd_eq",
        "in_discount",
        "in_premium",
        "flow_main",
        "flow_signal",
        "flow_is_bull",
        "flow_is_trending",
        "bull_break",
        "bear_break",
        "bull_choch",
        "bear_choch",
        "bull_sweep",
        "bear_sweep",
        "bull_fvg",
        "bear_fvg",
        "in_bull_fvg",
        "in_bear_fvg",
        "snr_long_reclaim",
        "snr_short_reject",
        "snr_bull_breakout",
        "snr_bear_breakout",
        "snr_bull_retest",
        "snr_bear_retest",
        "bull_pa_rejection",
        "bear_pa_rejection",
        "bull_pa_continuation",
        "bear_pa_continuation",
        "kz_long_score",
        "kz_short_score",
    ]


def make_training_frame(
    df: pd.DataFrame,
    feature_columns: Optional[Iterable[str]] = None,
    target_column: str = "large_move",
) -> pd.DataFrame:
    columns = list(feature_columns or default_feature_columns())
    selected = df[columns + [target_column]].dropna().copy()
    return selected


def latest_signal_snapshot(df: pd.DataFrame) -> dict:
    last = df.dropna().iloc[-1]
    return {
        "timestamp": str(last.name),
        "close": float(last["close"]),
        "reg_basis": float(last["reg_basis"]),
        "snr_upper": float(last["snr_upper"]),
        "snr_lower": float(last["snr_lower"]),
        "poc_proxy": float(last["poc_proxy"]),
        "active_kz": int(last["active_kz"]),
        "is_silver": int(last["is_silver"]),
        "bull_choch": int(last["bull_choch"]),
        "bear_choch": int(last["bear_choch"]),
        "in_bull_fvg": int(last["in_bull_fvg"]),
        "in_bear_fvg": int(last["in_bear_fvg"]),
        "snr_long_reclaim": int(last["snr_long_reclaim"]),
        "snr_short_reject": int(last["snr_short_reject"]),
        "snr_bull_retest": int(last["snr_bull_retest"]),
        "snr_bear_retest": int(last["snr_bear_retest"]),
        "kz_long_score": float(last["kz_long_score"]),
        "kz_short_score": float(last["kz_short_score"]),
    }

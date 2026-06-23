from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Optional

import pandas as pd


def normalize_stream_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper().replace("-", "").replace("/", "")
    if cleaned.endswith("USD") and not cleaned.endswith("USDT"):
        return cleaned[:-3] + "USDT"
    return cleaned


@dataclass
class BinanceKlineEvent:
    symbol: str
    interval: str
    event_time: pd.Timestamp
    open_time: pd.Timestamp
    close_time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool
    trade_count: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["event_time"] = self.event_time.isoformat()
        payload["open_time"] = self.open_time.isoformat()
        payload["close_time"] = self.close_time.isoformat()
        return payload


class BinanceKlineStream:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        max_events: int = 1000,
    ) -> None:
        self.symbol = normalize_stream_symbol(symbol)
        self.interval = interval.strip().lower()
        self.max_events = max_events
        self._events: deque[BinanceKlineEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._ws = None
        self._started = False
        self._error: Optional[str] = None
        self._last_message_time: Optional[pd.Timestamp] = None

    @property
    def stream_url(self) -> str:
        return f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_{self.interval}"

    def _on_message(self, _ws, message: str) -> None:
        payload = json.loads(message)
        kline = payload.get("k", {})
        event = BinanceKlineEvent(
            symbol=str(kline.get("s", self.symbol)),
            interval=str(kline.get("i", self.interval)),
            event_time=pd.to_datetime(payload.get("E"), unit="ms", utc=True),
            open_time=pd.to_datetime(kline.get("t"), unit="ms", utc=True),
            close_time=pd.to_datetime(kline.get("T"), unit="ms", utc=True),
            open=float(kline.get("o", 0.0)),
            high=float(kline.get("h", 0.0)),
            low=float(kline.get("l", 0.0)),
            close=float(kline.get("c", 0.0)),
            volume=float(kline.get("v", 0.0)),
            is_closed=bool(kline.get("x", False)),
            trade_count=int(kline.get("n", 0)),
        )
        with self._lock:
            self._events.append(event)
            self._last_message_time = event.event_time

    def _on_error(self, _ws, error: Any) -> None:
        self._error = str(error)

    def _on_close(self, _ws, _close_status_code, _close_msg) -> None:
        self._started = False

    def _on_open(self, _ws) -> None:
        self._started = True

    def start(self) -> "BinanceKlineStream":
        if self._thread and self._thread.is_alive():
            return self

        try:
            import websocket
        except ImportError as exc:
            raise ImportError("Install websocket-client to use BinanceKlineStream.") from exc

        self._error = None
        self._started = False
        self._ws = websocket.WebSocketApp(
            self.stream_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._ws is not None:
            self._ws.close()
        self._started = False

    def wait_for_messages(self, min_messages: int = 1, timeout: float = 10.0) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._lock:
                if len(self._events) >= min_messages:
                    return True
            if self._error:
                return False
            time.sleep(0.2)
        return False

    def latest_event(self) -> Optional[BinanceKlineEvent]:
        with self._lock:
            return self._events[-1] if self._events else None

    def latest_snapshot(self) -> dict[str, Any]:
        event = self.latest_event()
        if event is None:
            return {
                "symbol": self.symbol,
                "interval": self.interval,
                "status": "waiting",
                "events_received": 0,
                "last_error": self._error or "",
            }
        snapshot = event.to_dict()
        snapshot["status"] = "running" if self._started else "stopped"
        snapshot["events_received"] = len(self._events)
        snapshot["last_error"] = self._error or ""
        return snapshot

    def recent_frame(self, limit: int = 200) -> pd.DataFrame:
        with self._lock:
            events = list(self._events)[-limit:]
        if not events:
            return pd.DataFrame(
                columns=[
                    "event_time",
                    "open_time",
                    "close_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "is_closed",
                    "trade_count",
                ]
            )
        frame = pd.DataFrame([event.to_dict() for event in events])
        frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
        frame["close_time"] = pd.to_datetime(frame["close_time"], utc=True)
        return frame

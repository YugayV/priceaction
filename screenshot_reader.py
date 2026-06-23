from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import easyocr
except ImportError:
    easyocr = None


@dataclass
class ScreenshotSnapshot:
    image_path: str
    width: int
    height: int
    mean_brightness: float
    ocr_text: str
    symbol_hint: str = ""
    timeframe_hint: str = ""
    price_hint: float | None = None
    session_hint: str = ""
    kill_zone_hint: str = ""
    smc_hint: str = ""
    indicator_hint: str = ""


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def extract_text_from_image(image_path: str | Path, language_list: Optional[list[str]] = None) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(path)
    if easyocr is None:
        return ""

    reader = easyocr.Reader(language_list or ["en"], gpu=False)
    results = reader.readtext(str(path), detail=0, paragraph=True)
    return _normalize_text(" ".join(results))


def parse_tradingview_text(raw_text: str) -> dict:
    text = _normalize_text(raw_text)
    text_upper = text.upper()
    symbol_match = re.search(r"\b(BTC(?:USD|USDT)?|XBTUSD|BTC-USD)\b", text_upper)
    timeframe_match = re.search(r"\b(1M|3M|5M|15M|30M|45M|1H|2H|4H|1D)\b", text_upper)
    price_matches = re.findall(r"\b\d{2,6}(?:[.,]\d{1,2})?\b", text)

    session_hint = ""
    if "NEW YORK" in text_upper:
        session_hint = "new_york"
    elif "LONDON" in text_upper:
        session_hint = "london"
    elif "ASIA" in text_upper:
        session_hint = "asia"

    kill_zone_hint = ""
    if "SILVER" in text_upper:
        kill_zone_hint = "silver"
    elif "NY KZ" in text_upper or "NEW YORK KZ" in text_upper:
        kill_zone_hint = "ny_kz"
    elif "LONDON KZ" in text_upper:
        kill_zone_hint = "london_kz"

    smc_parts = []
    for token in ["CHOCH", "BOS", "FVG", "SWEEP"]:
        if token in text_upper:
            smc_parts.append(token.lower())

    indicator_hint = "snr_smc" if "SNR" in text_upper or "SMART MONEY" in text_upper else ""
    price_hint = None
    if price_matches:
        try:
            price_hint = float(price_matches[-1].replace(",", ""))
        except ValueError:
            price_hint = None

    return {
        "symbol_hint": symbol_match.group(1) if symbol_match else "",
        "timeframe_hint": timeframe_match.group(1).lower() if timeframe_match else "",
        "price_hint": price_hint,
        "session_hint": session_hint,
        "kill_zone_hint": kill_zone_hint,
        "smc_hint": "|".join(smc_parts),
        "indicator_hint": indicator_hint,
    }


def analyze_screenshot(image_path: str | Path) -> ScreenshotSnapshot:
    path = Path(image_path)
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32)
    ocr_text = extract_text_from_image(path)
    parsed = parse_tradingview_text(ocr_text)

    return ScreenshotSnapshot(
        image_path=str(path),
        width=image.width,
        height=image.height,
        mean_brightness=float(arr.mean()),
        ocr_text=ocr_text,
        symbol_hint=parsed["symbol_hint"],
        timeframe_hint=parsed["timeframe_hint"],
        price_hint=parsed["price_hint"],
        session_hint=parsed["session_hint"],
        kill_zone_hint=parsed["kill_zone_hint"],
        smc_hint=parsed["smc_hint"],
        indicator_hint=parsed["indicator_hint"],
    )


def choose_latest_screenshot(inbox_dir: str | Path) -> Optional[Path]:
    path = Path(inbox_dir)
    if not path.exists():
        return None
    files = sorted(
        [p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None

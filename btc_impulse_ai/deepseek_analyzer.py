
import os
import json
from pathlib import Path
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv


class DeepSeekAnalyzer:
    def __init__(self, config_path: str = "assets_config.json"):
        config_file = Path(config_path)
        load_dotenv()
        load_dotenv(dotenv_path=config_file.with_name(".env"), override=False)

        with open(config_file, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
        self.model = self.config["deepseek"]["model"]
        self.assets = self.config["assets"]
        self.client = None
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.config["deepseek"]["api_base"],
                timeout=30.0,
                max_retries=1,
            )

    def get_status(self) -> Dict:
        if not self.api_key:
            return {
                "mode": "fallback",
                "ok": False,
                "message": "DEEPSEEK_API_KEY не задан",
                "model": self.model,
                "base_url": self.config["deepseek"]["api_base"],
            }
        return {
            "mode": "online",
            "ok": True,
            "message": "Ключ найден, API готов к вызову",
            "model": self.model,
            "base_url": self.config["deepseek"]["api_base"],
        }

    def test_connection(self) -> Dict:
        status = self.get_status()
        if not status["ok"] or self.client is None:
            return status
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Reply with OK"}],
                temperature=0.0,
                max_tokens=5,
            )
            text = (response.choices[0].message.content or "").strip()
            return {
                "mode": "online",
                "ok": True,
                "message": f"Подключение успешно: {text or 'OK'}",
                "model": self.model,
                "base_url": self.config["deepseek"]["api_base"],
            }
        except Exception as e:
            return {
                "mode": "fallback",
                "ok": False,
                "message": f"Connection error: {e}",
                "model": self.model,
                "base_url": self.config["deepseek"]["api_base"],
            }

    def _local_fallback_analysis(self, asset_key: str, price_data: Dict, news_data: List[Dict], error_text: str | None = None) -> str:
        asset = self.assets[asset_key]
        latest_close = float(price_data.get("latest_close", 0.0) or 0.0)
        price_change = float(price_data.get("price_change", price_data.get("price_change_pct", 0.0)) or 0.0)
        rsi = float(price_data.get("rsi", 50.0) or 50.0)
        atr = float(price_data.get("atr", 0.0) or 0.0)
        macd = price_data.get("macd")
        vwap = price_data.get("vwap_d")

        trend_text = "нейтральный"
        if price_change > 1.0:
            trend_text = "краткосрочно бычий"
        elif price_change < -1.0:
            trend_text = "краткосрочно медвежий"

        momentum_text = "нейтральный"
        if rsi >= 60:
            momentum_text = "бычий импульс"
        elif rsi <= 40:
            momentum_text = "медвежий импульс"

        risk_text = "умеренная"
        if atr > 0 and latest_close > 0:
            atr_pct = atr / latest_close * 100.0
            if atr_pct >= 2.0:
                risk_text = "высокая"
            elif atr_pct <= 0.8:
                risk_text = "низкая"
        else:
            atr_pct = 0.0

        news_count = len(news_data or [])
        header = "## DeepSeek временно недоступен, показан локальный fallback-анализ"
        if error_text:
            header += f"\n\nПричина: `{error_text}`"

        extra = []
        if macd is not None:
            extra.append(f"- `MACD`: `{float(macd):.4f}`")
        if vwap is not None:
            extra.append(f"- `VWAP`: `{float(vwap):.2f}`")

        return "\n".join(
            [
                header,
                "",
                f"### {asset['name']}",
                f"- Цена: `{latest_close:.2f}`",
                f"- Изменение: `{price_change:+.2f}%`",
                f"- RSI: `{rsi:.2f}`",
                f"- ATR: `{atr:.2f}` ({atr_pct:.2f}% от цены)",
                *extra,
                "",
                "### Вывод",
                f"- Текущий фон: `{trend_text}`",
                f"- Импульс: `{momentum_text}`",
                f"- Волатильность: `{risk_text}`",
                f"- Новостной фон: найдено `{news_count}` материалов",
                "",
                "### Базовые рекомендации",
                "- Если цена выше VWAP/дневного открытия и RSI держится выше 50, приоритет у long continuation",
                "- Если цена теряет VWAP и RSI ниже 50, приоритет у short continuation",
                "- На высокой волатильности лучше уменьшать размер позиции и ждать подтверждение по структуре",
            ]
        )

    def analyze_market(self, asset_key: str, price_data: Dict, news_data: List[Dict]) -> str:
        asset = self.assets[asset_key]

        if not self.api_key or self.client is None:
            return self._local_fallback_analysis(asset_key, price_data, news_data, "DEEPSEEK_API_KEY не задан")

        prompt = f"""
        Ты профессиональный аналитик финансовых рынков. Проанализируй текущую ситуацию для {asset['name']}.
        
        Данные о цене:
        {json.dumps(price_data, indent=2, ensure_ascii=False)}
        
        Последние новости:
        {json.dumps(news_data[:5], indent=2, ensure_ascii=False)}
        
        Дай комплексный анализ:
        1. Текущая рыночная ситуация
        2. Ключевые факторы, влияющие на цену
        3. Анализ новостей и их влияние
        4. Прогноз на ближайшую перспективу
        5. Рекомендации (для информации, не финансовый совет)
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Ты эксперт по финансовому анализу и торговле."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return self._local_fallback_analysis(asset_key, price_data, news_data, str(e))

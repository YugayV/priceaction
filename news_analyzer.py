
import json
from datetime import datetime
from typing import List, Dict
from pathlib import Path


class NewsAnalyzer:
    def __init__(self, config_path: str = "assets_config.json"):
        if Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
            self.assets = self.config["assets"]
        else:
            # Fallback if config not found
            self.assets = {
                "bitcoin": {"name": "Bitcoin", "keywords": ["bitcoin", "btc", "crypto"]},
                "gold": {"name": "Gold", "keywords": ["gold", "xau", "precious metal"]},
                "eurusd": {"name": "EUR/USD", "keywords": ["eurusd", "euro", "dollar", "forex"]},
            }
        self.config = {"news": {"max_articles": 10, "language": "en"}}

    def get_news_for_asset(self, asset_key: str, max_articles: int = None) -> List[Dict]:
        if max_articles is None:
            max_articles = self.config["news"]["max_articles"]

        asset = self.assets.get(asset_key, {"name": asset_key, "keywords": [asset_key]})

        # Fallback - mock news if library fails
        mock_news = [
            {
                "title": f"{asset['name']} Market Update: Latest Price Movements",
                "url": f"https://example.com/news/{asset_key}-1",
                "source": "Market News",
                "published_at": datetime.now().isoformat(),
                "summary": f"Latest updates on {asset['name']} price action and market analysis.",
                "keywords": asset['keywords']
            },
            {
                "title": f"Economic Factors Affecting {asset['name']}",
                "url": f"https://example.com/news/{asset_key}-2",
                "source": "Financial Times",
                "published_at": datetime.now().isoformat(),
                "summary": f"Analysis of key economic indicators influencing {asset['name']}.",
                "keywords": asset['keywords']
            },
            {
                "title": f"Technical Analysis: {asset['name']} Outlook",
                "url": f"https://example.com/news/{asset_key}-3",
                "source": "Trading View",
                "published_at": datetime.now().isoformat(),
                "summary": f"Technical analysis and trading signals for {asset['name']}.",
                "keywords": asset['keywords']
            },
        ]

        return mock_news[:max_articles]

    def get_all_assets_news(self) -> Dict[str, List[Dict]]:
        all_news = {}
        for asset_key in self.assets.keys():
            all_news[asset_key] = self.get_news_for_asset(asset_key)
        return all_news


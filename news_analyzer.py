
import json
from datetime import datetime
from typing import List, Dict
from gnewsclient import gnewsclient
import requests
from newspaper import Article
from pathlib import Path


class NewsAnalyzer:
    def __init__(self, config_path: str = "assets_config.json"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.assets = self.config["assets"]
        
    def get_news_for_asset(self, asset_key: str, max_articles: int = None) -> List[Dict]:
        if max_articles is None:
            max_articles = self.config["news"]["max_articles"]
            
        asset = self.assets[asset_key]
        keywords = " OR ".join(asset["keywords"])
        
        client = gnewsclient.NewsClient(
            language=self.config["news"]["language"],
            topic=f"{asset['name']} market",
            max_results=max_articles
        )
        
        articles = []
        try:
            news_list = client.get_news()
            
            for item in news_list:
                article_data = {
                    "title": item["title"],
                    "url": item["link"],
                    "source": item["media"],
                    "published_at": datetime.now().isoformat(),
                    "summary": ""
                }
                
                try:
                    article = Article(item["link"])
                    article.download()
                    article.parse()
                    article.nlp()
                    article_data["summary"] = article.summary
                    article_data["keywords"] = article.keywords
                except Exception:
                    pass
                
                articles.append(article_data)
        except Exception as e:
            print(f"Error fetching news: {e}")
            
        return articles

    def get_all_assets_news(self) -> Dict[str, List[Dict]]:
        all_news = {}
        for asset_key in self.assets.keys():
            all_news[asset_key] = self.get_news_for_asset(asset_key)
        return all_news

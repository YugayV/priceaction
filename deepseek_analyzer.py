
import os
import json
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv


class DeepSeekAnalyzer:
    def __init__(self, config_path: str = "assets_config.json"):
        load_dotenv()
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=self.config["deepseek"]["api_base"]
        )
        self.model = self.config["deepseek"]["model"]
        self.assets = self.config["assets"]
        
    def analyze_market(self, asset_key: str, price_data: Dict, news_data: List[Dict]) -> str:
        asset = self.assets[asset_key]
        
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
            return f"Ошибка анализа: {str(e)}"

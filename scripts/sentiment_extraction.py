from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import os
from dotenv import load_dotenv

load_dotenv()

finnhub = os.getenv('FINNHUB_KEY')


analyser = SentimentIntensityAnalyzer()



def get_news(ticker, start_date, end_date):
    response = requests.get(
        "https://finnhub.io/api/v1/company-news",
        params={
            "symbol": ticker,
            "from": start_date,
            "to": end_date,
            "token": finnhub,
        },
        timeout=15,
    )

    if not response.ok:
        raise RuntimeError(
            f"Finnhub error {response.status_code}: {response.text}"
        )

    return response.json()

def get_sentiment(headlines):
    scores = [
        analyser.polarity_scores(text)["compound"]
        for text in headlines
        if text and text.strip()
    ]

    if not scores:
        return {
            "article_count": 0,
            "sentiment_score": None,
        }

    return {
        "article_count": len(scores),
        "sentiment_score": (sum(scores) / len(scores)) * 100,
    }

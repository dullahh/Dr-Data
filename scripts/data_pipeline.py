from data_pull import get_assets, get_sector_info, extract_atm_calls, get_yields
from sentiment_extraction import get_sentiment, get_news

from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd


def fetch_data(yield_series="DGS10", yield_name="10Y_Yield", days_window=100, yf_period="100d", sector_ticker="XLK",
                        vix_ticker="^VIX", asset_1="NVDA", asset_2="AMD", sentiment_ticker = "AAPL", start_date="2026-10-20", end_date="2026-10-27",
                        stock_attributes=['Open', 'High', 'Low', 'Close', 'Volume'],
                        options_attributes=['strike', 'impliedVolatility', 'volume', 'openInterest']):
    
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        dispatches = {
            executor.submit(get_yields, yield_series, days_window, yield_name): "yields",
            executor.submit(get_sector_info, sector_ticker, yf_period, stock_attributes): "sector",
            executor.submit(get_sector_info, vix_ticker, yf_period, ['Close']): "vix",
            executor.submit(get_assets, asset_1, asset_2, yf_period, stock_attributes): "assets",
            executor.submit(extract_atm_calls, asset_2, yf_period, options_attributes): "atm_calls",
            executor.submit(get_news, sentiment_ticker, start_date, end_date): "news"
        }

        extractions = {}

        for dispatch in as_completed(dispatches):
            name = dispatches[dispatch]

            try:
                extractions[name] = dispatch.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to fetch dataset '{name}'"
                ) from exc

    """
    To be as they're tucked under the same key originally
    """
    extractions["asset_1"], extractions["asset_2"] = extractions.pop("assets")
    return extractions

def iv_and_sentiment_summarisation(datasets):
    mean_iv = datasets["atm_calls"]["impliedVolatility"].mean()

    headlines = [
        article["headline"]
        for article in datasets["news"]
        if article.get("headline")
    ]

    sentiment_info = get_sentiment(headlines)

    return {
        "mean_iv": mean_iv,
        "sentiment_score": sentiment_info.get("sentiment_score"),
        "headline_count": len(headlines),
    }

"""
Aligns all data to be under the same umbrella in terms of time format and timezone
"""
def normalise_daily_index(data):
    data = data.copy()
    data.index = pd.to_datetime(data.index)

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    data.index = data.index.normalize()
    return data.sort_index()


def build_raw_dataframe(datasets):
    parts = [normalise_daily_index(datasets["yields"])]

    for key, prefix in [
        ("sector", "Sector_"),
        ("vix", "VIX_"),
        ("asset_1", "Asset_1_"),
        ("asset_2", "Asset_2_"),
    ]:
        frame = normalise_daily_index(datasets[key])
        parts.append(frame.add_prefix(prefix))

    df_raw = pd.concat(parts, axis=1).sort_index()

    target_dates = normalise_daily_index(datasets["asset_2"]).index
    df_raw = df_raw.reindex(target_dates)
    df_raw.index.name = "Date"

    summary = iv_and_sentiment_summarisation(datasets)
    df_raw["impliedVolatility"] = summary["mean_iv"]
    df_raw["bullish_%"] = summary["sentiment_score"]

    return df_raw
    

def bn_df_construction(df_raw):
    df_raw = df_raw.sort_index()
    df_bn = pd.DataFrame(index=df_raw.index)

    df_bn["Yield_10Y"] = df_raw["10Y_Yield"].diff() * 100

    df_bn["Sentiment"] = df_raw["bullish_%"]
    df_bn["IV_Change"] = df_raw["impliedVolatility"].diff()

    columns = {
        "XLK_Return": "Sector_Close",
        "VIX_Return": "VIX_Close",
        "Asset_1_Return": "Asset_1_Close",
        "Asset_2_Return": "Asset_2_Close",
    }

    for feature, source in columns.items():
        df_bn[feature] = df_raw[source].pct_change(fill_method=None)

    return df_bn

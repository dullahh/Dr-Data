from fredapi import Fred
import pandas as pd
import numpy as np

import os
from dotenv import load_dotenv
import yfinance as yf


load_dotenv()

fred = Fred(api_key=os.getenv('FRED_API_KEY'))

def get_yields(series, time_window, name):
    fred_table = fred.get_series(series, observation_start=pd.Timestamp.now() - pd.DateOffset(days=time_window))
    fred_table.name = name
    return fred_table



def get_sector_info(ticker, time_window, attributes):
    sector_ticker = yf.Ticker(ticker)

    sector_df = sector_ticker.history(period=time_window)

    sector_df.index = sector_df.index.tz_localize(None) #CONSISTENCY

    selected = sector_df[attributes].copy()

    return selected

def extract_atm_calls(ticker, time_window, attributes):
    focal_ticker = yf.Ticker(ticker)

    nearest_expiry = focal_ticker.options[0]
    opt_chain = focal_ticker.option_chain(nearest_expiry)

    calls = opt_chain.calls

    curr_price = focal_ticker.history(period=time_window)['Close'].iloc[-1]
    atm_calls = calls.iloc[(calls['strike'] - curr_price).abs().argsort()[:5]]

    return atm_calls[attributes]

def get_assets(ticker_1, ticker_2, time_window, attributes):
    assets = yf.download([ticker_1, ticker_2], period=time_window, group_by="ticker")

    ticker_1_attributes = assets[ticker_1][attributes].copy()
    ticker_1_attributes.index = ticker_1_attributes.index.tz_localize(None)
    
    ticker_2_attributes = assets[ticker_2][attributes].copy()
    ticker_2_attributes.index = ticker_2_attributes.index.tz_localize(None)
    

    
    

    return ticker_1_attributes, ticker_2_attributes

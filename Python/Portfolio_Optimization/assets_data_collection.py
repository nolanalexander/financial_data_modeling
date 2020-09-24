# Author: Nolan Alexander

import pandas as pd
import datetime
import pandas_datareader as pdr #pip install yfinance
import yfinance as yf #pip install yfinance
import datetime as dt
import numpy as np

# Reads in asset data from yahoo finance and creates a 
# DataFrame with the Adj Close of the assets
def read_in_assets_data(assets, start_date, end_date, save_as_csv, data_filename):
    yf.pdr_override()
    portfolio_adj_close = pd.DataFrame()
    first_data = pdr.get_data_yahoo(assets[0], start_date, end_date)
    first_data.index.name = 'Date'
    first_data.reset_index(inplace=True)
    portfolio_adj_close['Date'] = first_data['Date']
    for asset in assets:
        cur_asset_data = pdr.get_data_yahoo(asset, start_date, end_date)
        cur_asset_data.reset_index(inplace=True)
        portfolio_adj_close[asset] = cur_asset_data['Adj Close']
    portfolio_adj_close.index = portfolio_adj_close['Date']
    portfolio_adj_close = portfolio_adj_close.drop('Date', axis = 1)
    if(save_as_csv):
        print('The assets data has been saved to your Data folder')
        portfolio_adj_close.to_csv(data_filename,index=False)
    return portfolio_adj_close
    
# Reads in the current price of assets and their outstanding shares
# from yahoo finance and returns a list of lists
def read_in_cur_price_and_outstanding_shares(assets):
    today = dt.date.today()
    last_week = today - dt.timedelta(days=7)
    past_week_adj_close = read_in_assets_data(assets, last_week, today, False, "")
    cur_value_adj_close = np.array(past_week_adj_close.iloc[-1])
    shares_outstanding = np.array([])
    for asset in assets:
        asset_data = yf.Ticker(asset)
        shares_outstanding = np.append(shares_outstanding, asset_data.info.get('sharesOutstanding'))
    return [cur_value_adj_close, shares_outstanding]

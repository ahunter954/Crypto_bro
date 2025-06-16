# utils/binance_data.py

from binance.client import Client
import pandas as pd
import os
from datetime import datetime

def get_binance_klines(symbol='BTCUSDT', interval='1h', start_str='1 Jan, 2022', save=True, save_dir='data'):
    client = Client()
    klines = client.get_historical_klines(symbol, interval, start_str)

    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    if save:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/{symbol.lower()}_{interval}.csv"
        df.to_csv(filename)
        print(f"[✓] Données sauvegardées dans {filename}")

    return df

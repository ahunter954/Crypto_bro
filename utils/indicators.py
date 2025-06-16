# utils/indicators.py

import pandas as pd
import ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['ema_fast'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()
    df['ema_slow'] = ta.trend.EMAIndicator(close=df['close'], window=26).ema_indicator()
    df['macd'] = ta.trend.MACD(close=df['close']).macd()
    df['stoch'] = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close']).stoch()

    df = df.dropna()
    return df

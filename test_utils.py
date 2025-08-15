# test_utils.py

from utils.binance_data import get_binance_klines
from utils.indicators import add_indicators

# Télécharger les données (ex: 1h BTC/USDT depuis mars 2025)
df = get_binance_klines(symbol='BTCUSDT', interval='1h', start_str='5 Mar, 2025')

print(">>> Données brutes téléchargées :")
print(df.head())

# Ajouter des indicateurs techniques
df = add_indicators(df)

print("\n>>> Données avec indicateurs :")
print(df.head())

# Sauvegarde
df.to_csv("data/btcusdt_1h_with_indicators.csv")

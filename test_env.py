# test_env.py

import pandas as pd
from envs.trading_env import TradingEnv
from utils.indicators import add_indicators

# Charger les données déjà préparées
df = pd.read_csv("data/btcusdt_1h_with_indicators.csv", index_col=0)
df = add_indicators(df)  # Pour s'assurer que tous les indicateurs sont bien là

# Initialiser l'environnement
env = TradingEnv(df=df, window_size=50)

obs, _ = env.reset()
env.render()

# Tester quelques actions manuelles
for step in range(10):
    action = step % 3  # Alterne entre hold (0), buy (1), sell (2)
    obs, reward, done, _, _ = env.step(action)
    print(f"Action: {action} | Reward: {reward:.2f}")
    env.render()

    if done:
        print("✅ Épisode terminé")
        break

import pandas as pd
import matplotlib.pyplot as plt
from envs.trading_env import TradingEnv
from utils.indicators import add_indicators
from stable_baselines3 import PPO
import matplotlib
matplotlib.use('Agg')

# Charger les données
df = pd.read_csv("data/btcusdt_1h_with_indicators.csv", index_col=0)
df = add_indicators(df)

# Environnement
env = TradingEnv(df, window_size=50)
obs, _ = env.reset()
model = PPO.load("models/best_model/best_model")

# Tracking
timestamps = df.index[env.window_size:]
portfolio_values = []
btc_prices = []
buy_signals = []
sell_signals = []

btc_hold_values = []
initial_price = df["close"].iloc[env.window_size]
initial_balance = env.initial_balance

# Pour le log CSV
log = []

for i in range(env.window_size, len(df) - 1):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)

    price = df["close"].iloc[i]
    btc_prices.append(price)
    portfolio_values.append(env.total_value)
    btc_hold_values.append((initial_balance / initial_price) * price)

    ts = timestamps[i - env.window_size]
    if action == 1:
        buy_signals.append((ts, env.total_value))
    elif action == 2:
        sell_signals.append((ts, env.total_value))

    log.append({
        "date": ts,
        "price": price,
        "balance": env.balance,
        "crypto_held": env.crypto_held,
        "total_value": env.total_value,
        "action": action
    })

# Enregistrer le log CSV
log_df = pd.DataFrame(log)
log_df.to_csv("backtest_log.csv", index=False)

# Visualisation
plt.figure(figsize=(16, 7))
plt.plot(timestamps[:len(portfolio_values)], portfolio_values, label="PPO Portfolio", linewidth=2)
plt.plot(timestamps[:len(btc_hold_values)], btc_hold_values, label="Buy & Hold BTC", linestyle="--")
plt.plot(timestamps[:len(btc_prices)], btc_prices, label="BTC Price", alpha=0.3)

if buy_signals:
    x, y = zip(*buy_signals)
    plt.scatter(x, y, marker="^", color="green", label="Buy", zorder=5)
if sell_signals:
    x, y = zip(*sell_signals)
    plt.scatter(x, y, marker="v", color="red", label="Sell", zorder=5)

plt.title("Performance de l'agent PPO vs Buy & Hold")
plt.xlabel("Temps")
plt.ylabel("Valeur du portefeuille ($)")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("backtest_result.png")

print("✅ Backtest terminé : graphique et log sauvegardés.")

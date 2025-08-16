import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from utils.indicators import add_indicators
from envs.trading_env import TradingEnv
import os

# Importation des paramètres depuis le fichier de configuration
from config import (
    WINDOW_SIZE, EPISODE_MAX_STEPS, DATA_PATH, MODEL_DIR
)

# Charger les données
df = pd.read_csv(DATA_PATH, index_col=0)
df = add_indicators(df)

# Initialisez l'environnement en passant les paramètres du fichier de configuration
env = TradingEnv(df=df, window_size=WINDOW_SIZE, max_steps=EPISODE_MAX_STEPS)

# Charger le modèle
model_path = os.path.join(MODEL_DIR, "best_model.zip")
if not os.path.exists(model_path):
    print(f"Erreur : Le fichier {model_path} n'existe pas. Veuillez vous assurer que l'entraînement a bien eu lieu.")
    exit()

print(f"Chargement du modèle : {model_path}")
model = PPO.load(model_path)

# Tracking
timestamps = df.index[WINDOW_SIZE:]
portfolio_values = []
btc_prices = []
buy_signals = []
sell_signals = []

btc_hold_values = []
initial_price = df["close"].iloc[WINDOW_SIZE]
initial_balance = env.initial_balance

# Pour le log CSV
log = []
done = False

# Boucle principale de backtest
obs, _ = env.reset()
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)

    # Assurez-vous que l'indice ne dépasse pas la taille du DataFrame
    current_index = env.current_step - 1
    if current_index >= len(df):
        break

    price = df["close"].iloc[current_index]
    btc_prices.append(price)
    portfolio_values.append(env.total_value)
    btc_hold_values.append((initial_balance / initial_price) * price)

    ts = timestamps[current_index - WINDOW_SIZE]
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
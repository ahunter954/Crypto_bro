import pandas as pd
import numpy as np
from math import sqrt

log = pd.read_csv("backtest_log.csv", parse_dates=["date"])

# Recalcule portfolio_value si nécessaire
if "portfolio_value" not in log.columns:
    log["portfolio_value"] = log["balance"] + log["crypto_held"] * log["price"]

# Rendement total
initial_value = log["portfolio_value"].iloc[0]
final_value = log["portfolio_value"].iloc[-1]
return_pct = (final_value / initial_value - 1) * 100

# Sharpe Ratio
log["returns"] = log["portfolio_value"].pct_change().fillna(0)
sharpe = (log["returns"].mean() / log["returns"].std()) * sqrt(252) if log["returns"].std() != 0 else 0

# Max Drawdown
cum_returns = (1 + log["returns"]).cumprod()
rolling_max = cum_returns.cummax()
drawdown = (cum_returns - rolling_max) / rolling_max
max_drawdown = drawdown.min() * 100

# Map des actions
action_map = {0: "hold", 1: "buy", 2: "sell"}
log["action"] = log["action"].map(action_map)

# Construction des trades
trades = []
entry = None
for i, row in log.iterrows():
    if row["action"] == "buy" and entry is None:
        entry = row
    elif row["action"] == "sell" and entry is not None:
        pnl = row["portfolio_value"] - entry["portfolio_value"]
        trades.append({
            "entry_time": entry["date"],
            "exit_time": row["date"],
            "entry_value": entry["portfolio_value"],
            "exit_value": row["portfolio_value"],
            "pnl": pnl,
            "pnl_pct": (pnl / entry["portfolio_value"]) * 100
        })
        entry = None

trades_df = pd.DataFrame(trades)

# Calculs conditionnels
if not trades_df.empty:
    win_trades = trades_df[trades_df["pnl"] > 0]
    loss_trades = trades_df[trades_df["pnl"] < 0]
    gain_loss_ratio = abs(win_trades["pnl"].mean() / loss_trades["pnl"].mean()) if not loss_trades.empty else np.nan
else:
    gain_loss_ratio = np.nan

# Résumé
print("📈 Résumé de performance :\n")
print(f"Rendement total      : {return_pct:.2f}%")
print(f"Sharpe Ratio         : {sharpe:.2f}")
print(f"Max Drawdown         : {max_drawdown:.2f}%")
print(f"Ratio gain/perte     : {gain_loss_ratio:.2f}")
print(f"Nombre de trades     : {len(trades_df)}")

# Export
trades_df.to_csv("trade_history.csv", index=False)
print("\n📦 Historique des trades sauvegardé → trade_history.csv")

import os
import glob
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from utils.indicators import add_indicators
from envs.trading_env import TradingEnv
from gymnasium.wrappers import TimeLimit

# Charger les données
df = pd.read_csv("data/btcusdt_1h_with_indicators.csv", index_col=0)
df = add_indicators(df)

# Environnement
window_size = 50
train_env = TradingEnv(df=df, window_size=window_size)
train_env = Monitor(train_env)
train_env = TimeLimit(train_env, max_episode_steps=10000)

# Répertoires
model_dir = "models"
checkpoint_dir = os.path.join(model_dir, "checkpoints")
log_dir = "logs/ppo_tensorboard"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Callbacks
eval_callback = EvalCallback(
    train_env,
    best_model_save_path=model_dir,
    log_path=log_dir,
    eval_freq=1000,
    deterministic=True,
    render=False,
    verbose=1  # ✅ Ajoute cette ligne
)

checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path=checkpoint_dir,
    name_prefix="ppo_checkpoint"
)

# 🔁 Reprise depuis le dernier checkpoint s'il existe
checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "ppo_checkpoint_*.zip")))
if checkpoints:
    latest_checkpoint = checkpoints[-1]
    print(f"🔁 Chargement du checkpoint : {latest_checkpoint}")
    model = PPO.load(latest_checkpoint, env=train_env)
else:
    print("✨ Initialisation d’un nouveau modèle PPO.")
    model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=log_dir)

# 🚀 Entraînement
print("🚀 Entraînement lancé...")
model.learn(
    total_timesteps=200_000,
    callback=[eval_callback, checkpoint_callback]
)

# Sauvegarde finale
model.save(os.path.join(model_dir, "ppo_trading_model"))
print("✅ Modèle entraîné et sauvegardé.")

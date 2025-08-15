import os
import glob
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from utils.indicators import add_indicators
from envs.trading_env import TradingEnv
from gymnasium.wrappers import TimeLimit

# Charger les données
df = pd.read_csv("data/btcusdt_1h_with_indicators.csv", index_col=0)
df = add_indicators(df)

# Diviser les données pour l'entraînement et l'évaluation
train_size = int(len(df) * 0.8)
train_df = df[:train_size]
eval_df = df[train_size:]

# Définir le nombre d'étapes maximum pour les épisodes
episode_max_steps = 50
window_size = 25

# Environnement d'entraînement
train_env = TradingEnv(df=train_df, window_size=window_size, max_steps=episode_max_steps)
train_env = Monitor(train_env)

# Environnement d'évaluation
eval_env = TradingEnv(df=eval_df, window_size=window_size, max_steps=episode_max_steps)
eval_env = Monitor(eval_env)
# eval_env = TimeLimit(eval_env, max_episode_steps=50) # Inutile, EvalCallback s'en charge

# Répertoires
model_dir = "models"
checkpoint_dir = os.path.join(model_dir, "checkpoints")
log_dir = "logs/ppo_tensorboard"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Callbacks
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=model_dir,
    log_path=log_dir,
    eval_freq=10000,
    deterministic=True,
    render=False,
    verbose=1
)

checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path=checkpoint_dir,
    name_prefix="ppo_checkpoint"
)

# Reprise depuis le dernier checkpoint s'il existe
checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "ppo_checkpoint_*.zip")))
if checkpoints:
    latest_checkpoint = checkpoints[-1]
    print(f"🔁 Chargement du checkpoint : {latest_checkpoint}")
    model = PPO.load(latest_checkpoint, env=train_env)
else:
    print("✨ Initialisation d’un nouveau modèle PPO.")
    model = PPO("MlpPolicy", train_env, verbose=0, tensorboard_log=log_dir, ent_coef=0.01)

# Entraînement avec une valeur pour un test
print("🚀 Entraînement lancé...")
model.learn(
    total_timesteps=200000,
    callback=[checkpoint_callback, eval_callback]
)

# Sauvegarde finale
model.save(os.path.join(model_dir, "ppo_trading_model"))
print("✅ Modèle entraîné et sauvegardé.")
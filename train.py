import os
import glob
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from utils.indicators import add_indicators
from envs.trading_env import TradingEnv

# Importation des paramètres depuis le fichier de configuration
from config import (
    WINDOW_SIZE, EPISODE_MAX_STEPS, TOTAL_TIMESTEPS,
    EVAL_FREQ, SAVE_FREQ, ENTROPY_COEF,
    DATA_PATH, MODEL_DIR, CHECKPOINT_DIR, LOG_DIR, VERBOSE
)

# Charger les données
df = pd.read_csv(DATA_PATH, index_col=0)
df = add_indicators(df)

# Diviser les données pour l'entraînement et l'évaluation
train_size = int(len(df) * 0.8)
train_df = df[:train_size]
eval_df = df[train_size:]

# Environnement d'entraînement
train_env = TradingEnv(df=train_df, window_size=WINDOW_SIZE, max_steps=EPISODE_MAX_STEPS)
train_env = Monitor(train_env)

# Environnement d'évaluation
eval_env = TradingEnv(df=eval_df, window_size=WINDOW_SIZE, max_steps=EPISODE_MAX_STEPS)
eval_env = Monitor(eval_env)

# Création des répertoires
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Callbacks
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_DIR,
    log_path=LOG_DIR,
    eval_freq=EVAL_FREQ,
    deterministic=True,
    render=False,
    verbose=VERBOSE
)

checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path=CHECKPOINT_DIR,
    name_prefix="ppo_checkpoint"
)

# Reprise depuis le dernier checkpoint s'il existe
checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "ppo_checkpoint_*.zip")))
if checkpoints:
    latest_checkpoint = checkpoints[-1]
    print(f"🔁 Chargement du checkpoint : {latest_checkpoint}")
    model = PPO.load(latest_checkpoint, env=train_env)
else:
    print("✨ Initialisation d’un nouveau modèle PPO.")
    model = PPO("MlpPolicy", train_env, verbose=VERBOSE, tensorboard_log=LOG_DIR, ent_coef=ENTROPY_COEF)

# Entraînement
print("🚀 Entraînement lancé...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_callback, eval_callback]
)

# Sauvegarde finale
model.save(os.path.join(MODEL_DIR, "ppo_trading_model"))
print("✅ Modèle entraîné et sauvegardé.")
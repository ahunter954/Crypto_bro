# Fichier: config.py

# Paramètres de l'environnement de trading
WINDOW_SIZE = 25
EPISODE_MAX_STEPS = 168 # Une semaine (7 jours * 24 heures)
INITIAL_BALANCE = 10000.0

# Paramètres d'entraînement de PPO
TOTAL_TIMESTEPS = 2500000
EVAL_FREQ = 10000
SAVE_FREQ = 500000
ENTROPY_COEF = 0.01

# Chemins des fichiers
DATA_PATH = "data/btcusdt_1h_with_indicators.csv"
MODEL_DIR = "models"
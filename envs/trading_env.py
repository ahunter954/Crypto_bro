import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, df: pd.DataFrame, window_size: int, max_steps: int):
        super().__init__()
        print("✅ Environnement initialisé")
        self.df = df
        self.window_size = window_size
        self.prices = self.df['close'].values
        self.signal_features = self.df.values
        self.initial_balance = 10000
        self.max_steps = max_steps # Ajout de la définition de max_steps

        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(window_size, self.signal_features.shape[1]), 
            dtype=np.float32
        )

    def _get_observation(self):
        return self.signal_features[self.current_step - self.window_size : self.current_step, :]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #print(f"🔄 Début du reset. Données de trading: {len(self.df)} lignes")
        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.total_value = self.initial_balance
        self.current_step = self.window_size
        self.done = False
        self.trades = []

        observation = self._get_observation()
        info = {}
        #print("✅ Reset terminé. Début du nouvel épisode.")
        return observation, info

def step(self, action: int):
        self.current_step += 1
        
        # S'assurer que l'indice ne dépasse pas la taille des données
        if self.current_step >= len(self.prices):
            self.current_step = len(self.prices) - 1
            self.done = True
        
        price = self.prices[self.current_step - 1]
        
        # 1. Calculer la récompense initiale basée sur l'action (plus forte pénalité)
        reward = 0.0
        if action == 1: # Buy
            if self.balance > 0:
                self.crypto_held += self.balance / price
                self.balance = 0.0
                reward = 0.01 # Récompense pour avoir pris une action d'achat
            else:
                reward = -0.01 # Pénalité si l'action n'est pas possible
        elif action == 2: # Sell
            if self.crypto_held > 0:
                self.balance += self.crypto_held * price
                self.crypto_held = 0.0
                reward = 0.01 # Récompense pour avoir pris une action de vente
            else:
                reward = -0.01 # Pénalité si l'action n'est pas possible
        elif action == 0: # Hold
            reward = -0.05 # Forte pénalité pour l'inaction
        
        # 2. Mettre à jour la valeur totale du portefeuille
        new_total_value = self.balance + self.crypto_held * price
        
        # 3. Ajouter la récompense/pénalité basée sur la performance
        performance_reward = (new_total_value - self.total_value) / self.total_value
        reward += performance_reward * 100 # Multipliez pour un impact plus grand
        
        self.total_value = new_total_value
        
        # 4. Définir la fin de l'épisode
        self.done = self.current_step >= self.max_steps or self.total_value <= 0
        truncated = False
        
        # Logique de débogage
        # print(f"🐛 DEBUG: Étape {self.current_step}, action={action}, reward={reward:.4f}, done={self.done}")

        observation = self._get_observation()
        info = {}
        
        return observation, reward, self.done, truncated, info
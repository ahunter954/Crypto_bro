import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

# Importation des paramètres depuis le fichier de configuration
from config import (
    WINDOW_SIZE, EPISODE_MAX_STEPS, INITIAL_BALANCE
)

class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        
        self.df = df
        self.prices = self.df["close"].values
        self.features = self.df.drop(columns=["open", "high", "low", "close"]).values
        self.window_size = WINDOW_SIZE
        self.max_steps = EPISODE_MAX_STEPS
        
        # Action space: Buy, Hold, Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space
        num_features = self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, num_features + 3), dtype=np.float32
        )
        
        self.initial_balance = INITIAL_BALANCE
        
    def _get_observation(self):
        start = self.current_step - self.window_size
        end = self.current_step
        
        window_prices = self.prices[start:end]
        window_features = self.features[start:end]
        
        # Normalisation des prix
        normalized_prices = (window_prices - np.mean(window_prices)) / (np.std(window_prices) + 1e-8)
        
        # Concaténation des observations
        obs = np.concatenate([
            normalized_prices[:, np.newaxis],
            window_features,
            # Ajouter le solde et la crypto détenue pour chaque pas de temps dans la fenêtre
            np.full((self.window_size, 1), self.balance),
            np.full((self.window_size, 1), self.crypto_held)
        ], axis=1)
        
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.total_value = self.initial_balance
        self.done = False
        self.truncated = False
        
        return self._get_observation(), {}

    def step(self, action: int):
        self.current_step += 1
        
        # S'assurer que l'indice ne dépasse pas la taille des données
        if self.current_step >= len(self.prices):
            self.current_step = len(self.prices) - 1
            self.done = True
        
        price = self.prices[self.current_step - 1]
        
        # 1. Calculer la récompense initiale basée sur l'action
        reward = 0.0
        if action == 1: # Buy
            if self.balance > 0:
                self.crypto_held += self.balance / price
                self.balance = 0.0
                reward = 0.01
            else:
                reward = -0.01
        elif action == 2: # Sell
            if self.crypto_held > 0:
                self.balance += self.crypto_held * price
                self.crypto_held = 0.0
                reward = 0.01
            else:
                reward = -0.01
        elif action == 0: # Hold
            reward = -0.05
        
        # 2. Mettre à jour la valeur totale du portefeuille
        new_total_value = self.balance + self.crypto_held * price
        
        # 3. Ajouter la récompense/pénalité basée sur la performance
        performance_reward = (new_total_value - self.total_value) / self.total_value
        reward += performance_reward * 100
        
        self.total_value = new_total_value
        
        # 4. Définir la fin de l'épisode
        self.done = self.current_step >= self.max_steps or self.total_value <= 0
        truncated = False
        
        observation = self._get_observation()
        info = {}
        
        return observation, reward, self.done, truncated, info
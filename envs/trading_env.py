# envs/trading_env.py

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, window_size: int = 50, initial_balance: float = 10000):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance

        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell

        obs_shape = (window_size, df.shape[1] + 2)  # market features + [balance, position]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        self._reset_state()

    def _reset_state(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.total_value = self.balance
        self.trades = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), {}

    def _get_observation(self):
        frame = self.df.iloc[self.current_step - self.window_size:self.current_step].copy()
        frame = frame.values
        norm_frame = frame / (np.max(np.abs(frame), axis=0) + 1e-8)
        obs = np.hstack([
            norm_frame,
            np.full((self.window_size, 1), self.balance / 100000),
            np.full((self.window_size, 1), self.crypto_held)
        ])
        return obs.astype(np.float32)

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'close']

        prev_value = self.balance + self.crypto_held * current_price

        if action == 1:  # Buy
            if self.balance > 0:
                self.crypto_held += self.balance / current_price
                self.trades.append(('buy', self.current_step, current_price))
                self.balance = 0

        elif action == 2:  # Sell
            if self.crypto_held > 0:
                self.balance += self.crypto_held * current_price
                self.trades.append(('sell', self.current_step, current_price))
                self.crypto_held = 0

        self.current_step += 1

        done = self.current_step >= len(self.df) - 1

        current_value = self.balance + self.crypto_held * current_price
        reward = current_value - prev_value

        self.total_value = current_value

        return self._get_observation(), reward, done, False, {}

    def render(self, mode='human'):
        print(f'Step: {self.current_step} | Balance: {self.balance:.2f} | Crypto: {self.crypto_held:.4f} | Total: {self.total_value:.2f}')

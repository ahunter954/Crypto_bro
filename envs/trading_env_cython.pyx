# trading_env_cython.pyx

import numpy as np
cimport numpy as np
from libc.math cimport fabs
import gymnasium as gym
from gymnasium import spaces

cdef class TradingEnvCython(gym.Env):

    def __init__(self, np.ndarray[np.float64_t, ndim=2] data, int window_size=50, double initial_balance=10000):
        super().__init__()

        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance

        self.action_space = spaces.Discrete(3)
        obs_shape = (window_size, data.shape[1] + 2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        self._reset_state()

    cpdef _reset_state(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.total_value = self.balance
        self.trades = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), {}

    cpdef _get_observation(self):
        cdef int start_idx = self.current_step - self.window_size
        cdef int end_idx = self.current_step
        cdef np.ndarray[np.float64_t, ndim=2] frame = self.data[start_idx:end_idx, :]

        # Normalisation simple sur numpy pur
        cdef np.ndarray[np.float64_t, ndim=1] max_abs = np.max(np.abs(frame), axis=0) + 1e-8
        norm_frame = frame / max_abs

        cdef np.ndarray[np.float64_t, ndim=2] balance_col = np.full((self.window_size, 1), self.balance / 100000)
        cdef np.ndarray[np.float64_t, ndim=2] crypto_col = np.full((self.window_size, 1), self.crypto_held)

        obs = np.hstack([norm_frame, balance_col, crypto_col])
        return obs.astype(np.float32)

    def step(self, int action):
        cdef double current_price = self.data[self.current_step, -1]  # close price assumed last column
        cdef double prev_value = self.balance + self.crypto_held * current_price

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
        done = self.current_step >= self.data.shape[0] - 1

        cdef double current_value = self.balance + self.crypto_held * current_price
        reward = current_value - prev_value
        self.total_value = current_value

        return self._get_observation(), reward, done, False, {}

    def render(self, mode='human'):
        print(f'Step: {self.current_step} | Balance: {self.balance:.2f} | Crypto: {self.crypto_held:.4f} | Total: {self.total_value:.2f}')

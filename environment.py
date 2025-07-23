import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DiabetesEnv(gym.Env):
    def __init__(self):
        super(DiabetesEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=300, shape=(1,), dtype=np.float32)  # Glucose level
        self.action_space = spaces.Discrete(5)  # Insulin doses: 0, 2, 4, 6, 8 units
        self.state = 180  # Start with high glucose
        self.goal = 100   # Ideal glucose

    def reset(self, seed=None):
        self.state = np.random.randint(150, 200)
        return np.array([self.state], dtype=np.float32), {}

    def step(self, action):
        insulin = action * 2  # Convert action to dose
        change = np.random.randint(-5, 5) - insulin  # Natural fluctuation + insulin effect
        self.state += change

        reward = -abs(self.state - self.goal) / 100  # Closer to goal = better
        terminated = bool(self.state < 60 or self.state > 300)  # ✅ Force to bool
        truncated = False

        return np.array([self.state], dtype=np.float32), reward, terminated, truncated, {}

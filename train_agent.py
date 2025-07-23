from stable_baselines3 import PPO
from environment import DiabetesEnv
from stable_baselines3.common.env_checker import check_env

# Create environment
env = DiabetesEnv()

# Optional: Check if environment is valid
check_env(env)

# Create RL model (PPO agent with MLP policy)
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=10000)

# Save the trained model
model.save("diabetes_agent")

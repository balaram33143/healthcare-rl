from stable_baselines3 import PPO
from environment import DiabetesEnv

# Load the environment and model
env = DiabetesEnv()
model = PPO.load("diabetes_agent")

obs, _ = env.reset()
total_reward = 0

print("👨‍⚕️ Testing Agent...\n")

for step in range(20):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)
    total_reward += reward
    print(f"Step {step+1}: Glucose = {obs[0]:.2f}, Reward = {reward:.2f}, Action (dose) = {action*2} units")
    
    if done:
        print("\n❗ Episode ended early due to unsafe glucose level.")
        break

print(f"\n✅ Total Reward: {total_reward:.2f}")

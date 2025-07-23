from stable_baselines3 import PPO
from environment import DiabetesEnv
import matplotlib.pyplot as plt

# Load environment and model
env = DiabetesEnv()
model = PPO.load("diabetes_agent")

obs, _ = env.reset()
total_reward = 0

glucose_levels = []
actions = []

print("👨‍⚕️ Testing Agent...\n")

for step in range(20):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)
    glucose_levels.append(obs[0])
    actions.append(action * 2)  # Convert action to insulin units
    total_reward += reward
    print(f"Step {step+1}: Glucose = {obs[0]:.2f}, Reward = {reward:.2f}, Action = {action*2} units")

    if done:
        print("\n❗ Episode ended early due to unsafe glucose level.")
        break

print(f"\n✅ Total Reward: {total_reward:.2f}")

# 🧪 Plot glucose over time
plt.figure(figsize=(10, 5))
plt.plot(glucose_levels, marker='o', label="Glucose Level")
plt.plot(actions, marker='x', linestyle='--', label="Insulin Dose (units)")
plt.axhline(y=100, color='green', linestyle='--', label='Target Level')
plt.title("Patient Glucose Level Over Time")
plt.xlabel("Time Step")
plt.ylabel("Glucose / Insulin")
plt.legend()
plt.grid()
plt.show()

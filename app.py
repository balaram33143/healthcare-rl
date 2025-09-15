import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from environment import DiabetesEnv  # still using environment.py for the env class

# === Load environment and trained model ===
env = DiabetesEnv()
model = PPO.load("diabetes_agent")

obs, info = env.reset()
total_reward = 0.0

glucose_levels = []
actions = []
rewards = []

print("👨‍⚕️ Testing PPO Agent...\n")

# run up to 20 steps
for step in range(1, 21):
    action, _ = model.predict(obs)  # choose action from agent
    obs, reward, done, truncated, info = env.step(action)

    # record data for plotting
    glucose_levels.append(float(obs[0]))
    actions.append(float(action * 2))  # convert action to insulin units
    rewards.append(float(reward))
    total_reward += reward

    print(f"Step {step:02d}: "
          f"Glucose={obs[0]:.2f}, "
          f"Reward={reward:.2f}, "
          f"Insulin={action*2:.1f} units")

    if done:
        print("\n❗ Episode ended early due to unsafe glucose level.")
        break

print(f"\n✅ Total Reward over episode: {total_reward:.2f}")

# convert lists to numpy for convenience
glucose_levels = np.array(glucose_levels)
actions = np.array(actions)
rewards = np.array(rewards)

# === Plot glucose level and insulin dose ===
plt.figure(figsize=(10, 5))
plt.plot(glucose_levels, marker='o', label="Glucose Level")
plt.plot(actions, marker='x', linestyle='--', label="Insulin Dose (units)")
plt.axhline(y=100, color='green', linestyle='--', label='Target Level')
plt.title("Patient Glucose Level & Insulin Dose Over Time")
plt.xlabel("Time Step")
plt.ylabel("Glucose / Insulin")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot reward per step ===
plt.figure(figsize=(10, 4))
plt.bar(np.arange(1, len(rewards) + 1), rewards, color='skyblue')
plt.title(f"Reward per Step (Total={total_reward:.2f})")
plt.xlabel("Time Step")
plt.ylabel("Reward")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

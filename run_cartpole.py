import gymnasium as gym
import numpy as np

print("Hello World! from test.py")
print("random number: ", np.random.rand())


env = gym.make("CartPole-v1")

observation, info = env.reset(seed=42)
for _ in range(10):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print("observation: ", observation)

    if terminated or truncated:
        observation, info = env.reset()
env.close()

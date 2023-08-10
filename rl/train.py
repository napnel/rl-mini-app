# import ray
from ray.rllib.algorithms import ppo

from rl.tictactoe_env import TicTacToeEnv

print("Hello World! from test.py")


# ray.init()
# config = DQNConfig()
# algo = config.resources(num_gpus=0).environment(TicTacToeEnv).build()
def train(max_episodes=10):
    config = ppo.PPOConfig().environment(env=TicTacToeEnv, env_config={})
    algo = config.build()

    for i in range(max_episodes):
        result = algo.train()
        print("episode reward mean: ", result["episode_reward_mean"])

    return algo


def evaluate(algo):
    env = TicTacToeEnv()
    observation, info = env.reset()
    terminated = False

    while not terminated:
        # action = env.action_space.sample()
        action = algo.compute_single_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        print("observation: ", observation)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


algo = train()
algo.export_policy_model("policy_model", onnx=18)
evaluate(algo)

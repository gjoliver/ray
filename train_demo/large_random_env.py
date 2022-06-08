import gym

import ray
from ray import tune
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


# Note(jungong): Change these values to change the size of the training batches.
OBS_SIZE = 5000
EPISODE_LEN = 100


class DeterministicRandomEnv(RandomEnv):
    def __init__(self, config=None):
        config = config or {}
        config.update({"observation_space": gym.spaces.Box(-1.0, 1.0, (OBS_SIZE,))})
        super().__init__(config=config)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # Fixed length episodes.
        done = self.steps >= EPISODE_LEN
        return obs, reward, done, info


def main():
    ray.init()

    config = {
        "env": DeterministicRandomEnv,
        "num_gpus": 0,
        "num_workers": 2,
        "num_envs_per_worker": 2,
        "framework": "torch",
        "batch_mode": "complete_episodes",
        "rollout_fragment_length": 1,  # Let each sample contain just 1 episode.
    }

    stop = {
        "time_total_s": 300,
    }

    results = tune.run("PPO", config=config, stop=stop, verbose=2)

    ray.shutdown()


if __name__ == "__main__":
    main()

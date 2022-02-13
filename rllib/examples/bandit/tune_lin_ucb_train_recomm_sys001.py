"""Example of using LinUCB on a RecSim environment. """

from matplotlib import pyplot as plt
import pandas as pd
import time

import ray
from ray import tune
from ray.rllib.examples.env.recommender_system_envs import RecommSys001
from ray.rllib.env.wrappers.recsim import MultiDiscreteToDiscreteActionWrapper, \
    RecSimObservationBanditWrapper


if __name__ == "__main__":
    ray.init(local_mode=True)

    tune.register_env("my_env",
                      lambda config: RecSimObservationBanditWrapper(
                          MultiDiscreteToDiscreteActionWrapper(
                              RecommSys001(**config))))

    config = {
        # Use our RLlib in-house "RecommSys001".
        "env": "my_env",
        "env_config": {
            "num_categories": 20,
            "num_docs_to_select_from": 10,
            "slate_size": 1,
            "num_docs_in_db": 100,
            "num_users_in_db": 1,
        },
        "evaluation_interval": 20,
        "evaluation_duration": 100,
        "evaluation_duration_unit": "episodes",

        "simple_optimizer": True,
        "timesteps_per_iteration": 1,
    }

    stop = {
        "timesteps_total": 300,
    }

    start_time = time.time()
    analysis = tune.run(
        "BanditLinUCB",
        config=config,
        stop=stop,
        num_samples=1,
        checkpoint_at_end=False,
        verbose=2,
    )

    print("The trials took", time.time() - start_time, "seconds\n")

    # Analyze cumulative regrets of the trials
    frame = pd.DataFrame()
    for key, df in analysis.trial_dataframes.items():
        frame = frame.append(df, ignore_index=True)
    x = frame.groupby("agent_timesteps_total")["episode_reward_mean"].aggregate(
        ["mean", "max", "min", "std"]
    )

    plt.plot(x["mean"])
    plt.fill_between(
        x.index, x["mean"] - x["std"], x["mean"] + x["std"], color="b", alpha=0.2
    )
    plt.title("Episode reward mean")
    plt.xlabel("Training steps")
    plt.show()

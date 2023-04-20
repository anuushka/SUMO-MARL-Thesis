import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import pandas as pd
import ray
from ray.rllib.agents.a3c.a3c import A3CTrainer
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env
from gym import spaces
import numpy as np
import sumo_rl
import traci


if __name__ == '__main__':
    ray.init()

    register_env("4x4grid", lambda _: PettingZooEnv(sumo_rl.env(net_file='bagatoiruu/cleared.net.xml',
                                                    route_file='bagatoiruu/cleared.rou.xml',  
                                                    out_csv_name='outputs/a3c/test1',
                                                    use_gui=True,
                                                    num_seconds=1000)))

    trainer = A3CTrainer(env="4x4grid", config={
        "multiagent": {
            "policies": {
                '0': (A3CTFPolicy, spaces.Box(low=np.zeros(11), high=np.ones(11)), spaces.Discrete(2), {})
            },
            "policy_mapping_fn": (lambda id: '0')  # Traffic lights are always controlled by this policy
        },
        "lr": 0.001,
        "no_done_at_end": True
    })
    while True:
        print(trainer.train())  # distributed training step
import argparse
import os
import sys
import pandas as pd
from collections import deque
import numpy as np



if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy


if __name__ == '__main__':

    alpha = 0.1
    gamma = 0.99
    decay = 1
    runs = 5

    env = SumoEnvironment(net_file='bagatoiruu/cleared.net.xml',
                          route_file='bagatoiruu/cleared.rou.xml',
                          use_gui=True,
                          num_seconds=100,
                          min_green=8,
                          delta_time=5)

    initial_states = env.reset()
    ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts], ts),
                                 state_space=env.observation_space,
                                 action_space=env.action_space,
                                 alpha=alpha,
                                 gamma=gamma,
                                 exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay)) 
                                 for ts in env.ts_ids}
    scores = []                        # episode болгоны оноог хадгалах
    scores_window = deque(maxlen=100)
    for run in range(1, runs+1):
        if run != 1:
            initial_states = env.reset()
            for ts in initial_states.keys():
                ql_agents[ts].state = env.encode(initial_states[ts], ts)
        
        score = 0.0
        infos = []
        done = {'__all__': False}
        while not done['__all__']:
            actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

            s, r, done, info = env.step(action=actions)
            for agent_id in s.keys():
                ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])
                score += r[agent_id]

        scores_window.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            run, np.mean(scores_window)), end="")
        if run % 1 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                run, np.mean(scores_window)))
            f = open("score_ql.txt", "a")
            f.write(str(np.mean(scores_window))+"\n")
            f.close()
        env.save_csv('outputs/ql/test1', run)
        env.close()



#rewards oorchilson
import os
import torch
import numpy as np
import sumo_rl
import random
from dqn_agent import Agent
from collections import deque

def dqn(n_episodes=1500, max_t=1000, eps_start=1.0, eps_end=0.001, eps_decay=0.995):
    scores = []                        # episode болгоны оноог хадгалах
    scores_window = deque(maxlen=100)  # сүүлийн 100 оноо
    eps = eps_start
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)           
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores


if __name__ == '__main__':

    alpha = 0.1
    gamma = 0.995
    decay = 1
    runs = 100 #100000
    simulation_time = 3600 # seconds 7200
    eps_start=1.0
    eps_end=0.001
    eps_decay=0.95
    random_trip_number = random.randint(simulation_time, 4*simulation_time) # cars

    env = sumo_rl.env(net_file='bagatoiruu/cleared.net.xml',
                      route_file='bagatoiruu/cleared.rou.xml',
                      use_gui=True,
                      min_green=8,
                      delta_time=5,
                      num_seconds=1000)
    env.reset()
    initial_states = {ts: env.observe(ts) for ts in env.agents}
    dqn_agents = {ts: Agent(
        state_size=env.observation_space(ts).shape[0],
        action_size=env.action_space(ts).n,
        alpha=alpha,
        gamma=gamma,
        seed=10) for ts in env.agents}
    eps = eps_start
    scores = []                        # episode болгоны оноог хадгалах
    scores_window = deque(maxlen=100)  # сүүлийн 100 оноо
    for run in range(1, runs+1):
        os.system('python experiments/randomTrips.py -n bagatoiruu/cleared.net.xml -r bagatoiruu/cleared.rou.xml -e %i -L --insertion-rate 5200'%random_trip_number)
        env._route = 'bagatoiruu/cleared.rou.xml'
        env.reset()
        rewards = 0
        alpha = 0.01
        score = 0.0
        for agent in env.agent_iter():
            s, r, done, info = env.last()
            if dqn_agents[agent].act is not None:
                state = env.observe(agent)
                action = dqn_agents[agent].act(state, eps) if not done else None #oorchloh heregtei
                env.step(action)
                next_state = np.array(env.observe(agent), dtype=np.float32)
                reward = env.rewards[agent] + alpha*rewards #oorhcilson
                rewards += reward
                done = env.dones[agent]
                dqn_agents[agent].step(state, action, reward, next_state, done)
                score += reward
                # print(eps)
                if done:
                    break
        eps = max(eps_end, eps_decay*eps) #eps works ok
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)     

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            run, score), end="")
        if run % 1 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                run, score))
        if score >= 0.01:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                run-100, score))
            torch.save(dqn_agents[agent].qnetwork_local.state_dict(), 'checkpoint_'+agent+'.pth')
        env.unwrapped.env.save_csv('outputs/1000/test1', run) #TODO check checkpoints
        env.close()
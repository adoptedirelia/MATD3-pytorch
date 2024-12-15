import numpy as np
import torch
import os
from maddpg.maddpg import MADDPG
from ddpg.ddpg import DDPG
from ppo.ppo import PPO
from td3.td3 import TD3
from matd3.matd3 import MATD3

class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id

        if args.alg == 'MADDPG':
            self.policy = MADDPG(args, agent_id)
        elif args.alg == 'PPO':
            self.policy = PPO(args, agent_id)
        elif args.alg == 'DDPG':
            self.policy = DDPG(args, agent_id)
        elif args.alg == 'TD3':
            self.policy = TD3(args, agent_id)
        elif args.alg == 'MATD3':
            self.policy = MATD3(args, agent_id)

    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            if self.args.alg == 'PPO':
                inputs = torch.tensor(o, dtype=torch.float32)
                pi, sigma = self.policy.actor_network(inputs)
                u = torch.normal(pi, sigma).detach().numpy()
                u = np.clip(u, -self.args.high_action, self.args.high_action)
            else:
                inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
                pi = self.policy.actor_network(inputs).squeeze(0)
                # print('{} : {}'.format(self.name, pi))
                u = pi.cpu().numpy()
                noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
                u += noise
                u = np.clip(u, -self.args.high_action, self.args.high_action)
        return u.copy()

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)


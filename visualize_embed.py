import argparse
import numpy as np
import torch
from env import BaseEnv
from model.MLP import MLP
from agent import Agent
from replay_buffer import ReplayBuffer
from utils import *

parser = argparse.ArgumentParser()

# Training Options
parser.add_argument('--epoch_num', default=5000, type=int)
parser.add_argument('--max_step', default=50, type=int)
parser.add_argument('--batch_size', default=20, type=int)

# Model Options
parser.add_argument('--model_path', default='./model.pt', type=str)
parser.add_argument('--embedding_dim', default=32, type=int)
parser.add_argument('--s_a_hidden_size', default=(64, 64))
parser.add_argument('--s_hidden_size', default=(64, 64))
parser.add_argument('--s_a_lr', default=1e-4)
parser.add_argument('--s_lr', default=1e-4)

# Environment Options
parser.add_argument('--env_size', default=(5, 5))
parser.add_argument('--goal_pos', default=(5, 5))


def visualize(args):
    env = BaseEnv(size=args.env_size)
    f_s_a = MLP(env.state_size + env.action_size, args.s_a_hidden_size, args.embedding_dim)
    f_s = MLP(env.state_size, args.s_hidden_size, args.embedding_dim)
    agent = Agent(env_size=args.env_size)

    checkpoint = torch.load(args.model_path)
    f_s_a.load_state_dict(checkpoint['s_a_state_dict'])
    f_s.load_state_dict(checkpoint['s_state_dict'])
    vis_map = np.zeros(args.env_size)
    goal = args.goal_pos
    for i in vis_map.shape[0]:
        for j in vis_map.shape[1]:
            if (i+1, j+1) == goal:
                vis_map[i, j] = -1
            else:
                state_feature = agent.get_state_feature((i+1, j+1))
                goal_feature = agent.get_state_feature(goal)
                best_action, min_dist = agent.get_best_action(state_feature, f_s_a, f_s, goal_feature)
                vis_map[i, j] = best_action

    for i in vis_map.shape[0]:
        for j in vis_map.shape[1]:
            



if __name__ == '__main__':
    args = parser.parse_args()
    visualize(args)


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
parser.add_argument('--embedding_dim', default=32, type=int)
parser.add_argument('--s_a_hidden_size', default=(64, 64))
parser.add_argument('--s_hidden_size', default=(64, 64))
parser.add_argument('--s_a_lr', default=1e-4)
parser.add_argument('--s_lr', default=1e-4)

# Environment Options
parser.add_argument('--env_size', default=(5, 5))


def main(args):
    env = BaseEnv(size=args.env_size)
    f_s_a = MLP(env.state_size + env.action_size, args.s_a_hidden_size, args.embedding_dim)
    f_s = MLP(env.state_size, args.s_hidden_size, args.embedding_dim)
    buffer = ReplayBuffer(args, env)
    agent = Agent(env_size=args.env_size)

    s_a_optimizer = torch.optim.Adam(f_s_a.parameters(), lr=args.s_a_lr)
    s_optimizer = torch.optim.Adam(f_s.parameters(), lr=args.s_a_lr)

    for epoch in range(args.epoch_num):
        start_position = np.random.randint(1, 6, size=2)
        ns, r, terminate = env.reset(start_pos=start_position)
        for step in range(args.max_step):
            s = ns
            s_feature = agent.get_state_feature(s)
            action, raw_action = agent.get_action(s_feature, f_s_a, f_s)
            ns, r, terminate = env.step(action)
            ns_feature = agent.get_state_feature(ns)
            vec_action = vectorize_action(raw_action)
            buffer.add((s_feature, vec_action, ns_feature))
            # print(s, s_feature, action, raw_action, ns, ns_feature, vec_action)

        s_a_batch, s_batch = buffer.get_batch_data()
        s_a_embed = f_s_a(s_a_batch)
        s_embed = f_s(s_batch)
        loss = torch.mean(torch.norm((s_a_embed - s_embed), dim=1))
        s_a_optimizer.zero_grad()
        s_optimizer.zero_grad()
        loss.backward()
        s_a_optimizer.step()
        s_optimizer.step()
        print("epoch number: {}/{}, loss: {}".format(epoch, args.epoch_num, loss))
        if epoch == args.epoch_num - 1:
            print(s_a_embed, s_embed)

    save_model("./model.pt", f_s_a, f_s, args.epoch_num, loss, s_a_optimizer, s_optimizer)
    print("model saved.")


        # print("Finish optimization for epoch: {}".format(epoch))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)


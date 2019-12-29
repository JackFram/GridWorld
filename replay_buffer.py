import numpy as np
import math


class ReplayBuffer:
    def __init__(self, args, env):
        self.buffer = []
        self.batch_size = args.batch_size
        self.state_size = env.state_size
        self.action_size = env.action_size

    def add(self, item):
        self.buffer.append(item)

    def remove(self, index):
        self.buffer.pop(index)

    def get_batch_data(self):
        index = np.random.randint(len(self.buffer), size=self.batch_size).tolist()
        index_relabel = np.random.randint(len(self.buffer), size=self.batch_size).tolist()
        s_a_feature = np.zeros((0, self.state_size + self.action_size))
        s_feature = np.zeros((0, self.state_size))
        g_feature = np.zeros((0, self.state_size))
        for ind in index:
            s_a = np.concatenate((self.buffer[ind][0], self.buffer[ind][1]), axis=0).reshape((1, -1))
            s = np.array(self.buffer[ind][2]).reshape((1, -1))
            s_a_feature = np.concatenate((s_a_feature, s_a), axis=0)
            s_feature = np.concatenate((s_feature, s), axis=0)
        for ind in index_relabel:
            s = np.array(self.buffer[ind][2]).reshape((1, -1))
            g_feature = np.concatenate((g_feature, s), axis=0)
        return s_a_feature, s_feature, g_feature

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]





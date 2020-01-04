import numpy as np
import random
import math


class ReplayBuffer:
    def __init__(self, args, env):
        self._buffer = []
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.state_size = env.state_size
        self.action_size = env.action_size

    def add(self, item):
        if item[3] is None or any(item[2] != item[3]):
            if len(self._buffer) < self.buffer_size:
                self._buffer.append(item)
                # self._error.append(item[-1])
            else:
                # replace_index = self._error.index(min(self._error))
                replace_index = random.randint(0, len(self._buffer) - 1)
                self._buffer[replace_index] = item
                # self._error[replace_index] = item[-1]

    def remove(self, index):
        self._buffer.pop(index)

    def get_batch_data(self):
        weights = list(map(lambda x: math.exp(x[-1]), self._buffer))
        denom = sum(weights)
        dist = list(map(lambda x: x/denom, weights))
        index = random.choices(range(len(self._buffer)), weights=dist, k=self.batch_size)
        # print(index)
        # exit()
        # batch_data = random.choices(self._buffer, weights=dist, k=self.batch_size)
        # batch_data = random.choices(self._buffer, k=self.batch_size)
        batch_1, batch_2 = {}, {}
        batch_1["sa"] = np.zeros((0, self.state_size + self.action_size))
        batch_1["ns"] = np.zeros((0, self.state_size))
        batch_2["sa"] = np.zeros((0, self.state_size + self.action_size))
        batch_2["ns"] = np.zeros((0, self.state_size))
        batch_2["g"] = np.zeros((0, self.state_size))

        for ind in index:
            instance = self._buffer[ind]
            if instance[3] is None:
                s_a = np.concatenate((instance[0], instance[1]), axis=0).reshape((1, -1))
                s = np.array(instance[2]).reshape((1, -1))
                batch_1["sa"] = np.concatenate((batch_1["sa"], s_a), axis=0)
                batch_1["ns"] = np.concatenate((batch_1["ns"], s), axis=0)
            else:
                s_a = np.concatenate((instance[0], instance[1]), axis=0).reshape((1, -1))
                s = np.array(instance[2]).reshape((1, -1))
                g = np.array(instance[3]).reshape((1, -1))
                batch_2["sa"] = np.concatenate((batch_2["sa"], s_a), axis=0)
                batch_2["ns"] = np.concatenate((batch_2["ns"], s), axis=0)
                batch_2["g"] = np.concatenate((batch_2["g"], g), axis=0)

        for ind in sorted(set(index), reverse=True):
            self._buffer.pop(ind)

        return batch_1, batch_2, index

    def update_error(self, index, error):
        assert len(index) == len(error)
        for i in range(len(index)):
            self._buffer[index[i]][-1] = error[i]

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, index):
        return self._buffer[index]





import numpy as np
from utils import vectorize_action


class Agent:
    def __init__(self, env_size=(5, 5)):
        self.env_size = env_size

    def get_action(self, state_feature, f_s_a, f_s):
        # action = select(state_feature)
        action = np.random.randint(4, size=1)
        if action == 0:
            ret = 'up'
        elif action == 1:
            ret = 'down'
        elif action == 2:
            ret = 'left'
        elif action == 3:
            ret = 'right'
        return ret, action

    def get_best_action(self, state_feature, f_s_a, f_s, goal_feature):
        min_dist = 1e10
        best_action = None

        for action in range(4):
            vec_action = vectorize_action(action)
            embed_s_a = f_s_a.predict(np.concatenate((state_feature, vec_action), axis=0))
            embed_s = f_s.predict(goal_feature)
            dist = np.linalg.norm(embed_s_a - embed_s)
            if dist < min_dist:
                min_dist = dist
                best_action = action

        return best_action, min_dist

    def get_best_actions(self, state_feature, f_s_a, f_s, goal_feature):
        assert state_feature.shape[0] == goal_feature.shape[0]
        batch_size = state_feature.shape[0]
        ns_best_action = np.zeros((0, 4))
        for i in range(batch_size):
            best_action, min_dist = self.get_best_action(state_feature[i], f_s_a, f_s, goal_feature[i])
            vec_action = vectorize_action(best_action).reshape(1, -1)
            ns_best_action = np.concatenate((ns_best_action, vec_action), axis=0)
        return ns_best_action

    def get_state_feature(self, state):
        state_feature = np.zeros((self.env_size[0]*self.env_size[1]))
        assert 1 <= state[0] <= self.env_size[0] and 1 <= state[1] <= self.env_size[1]
        state_feature[(state[0]-1)*self.env_size[1] + state[1] - 1] = 1
        return state_feature

    def get_states_feature(self, states):
        batch_size = len(states)
        ret_val = np.zeros((batch_size, self.env_size[0]*self.env_size[1]))
        for idx, state in enumerate(states):
            state_feature = np.zeros((self.env_size[0]*self.env_size[1]))
            assert 1 <= state[0] <= self.env_size[0] and 1 <= state[1] <= self.env_size[1]
            state_feature[(state[0]-1)*self.env_size[1] + state[1] - 1] = 1
            ret_val[idx, :] = state_feature
        return ret_val


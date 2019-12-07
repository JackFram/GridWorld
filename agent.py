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


    def get_state_feature(self, state):
        state_feature = np.zeros((self.env_size[0]*self.env_size[1]))
        assert 1 <= state[0] <= self.env_size[0] and 1 <= state[1] <= self.env_size[1]
        state_feature[(state[0]-1)*self.env_size[1] + state[1] - 1] = 1
        return state_feature
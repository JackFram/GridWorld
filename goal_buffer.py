import numpy as np
import math


class GoalBuffer:
    def __init__(self):
        '''
        _goal_space: store goals
        goal_counter: counts goal
        dist: sample distribution
        '''
        self._goal_space = []
        self.goal2ind = {}
        self._goal_counter = []
        self.dist = []

    def store(self, goal):
        '''

        :param goal: the goal to store
        :return:
        '''
        if goal in self.goal2ind:
            self._goal_counter[self.goal2ind[goal]] += 1
        else:
            self._goal_space.append(goal)
            self.goal2ind[goal] = len(self._goal_counter)
            self._goal_counter.append(1)

    def generate_dist(self):
        '''
        generate sample distribution
        :return:
        '''
        weights = map(lambda x: math.exp(-x), self._goal_counter)
        denom = sum(weights)
        self.dist = map(lambda x: x/denom, weights)

    def sample_batch_goal(self, size):
        ret = np.random.choice(self._goal_space, size=size, replace=True, p=self.dist)
        return ret

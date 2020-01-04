import numpy as np
import random
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
        weights = list(map(lambda x: math.exp(-x), self._goal_counter))
        # weights = list(map(lambda x: math.exp(-x), self._goal_counter))
        denom = sum(weights)
        self.dist = list(map(lambda x: x/denom, weights))

    def sample_batch_goal(self, size, with_weights=True):
        min_ = min(self._goal_counter)
        new_goal_counter = list(map(lambda x: x - min_ + 1, self._goal_counter))
        # # print(self._goal_space, list(self._goal_counter))
        # # print(self.dist)
        self._goal_counter = new_goal_counter
        if with_weights:
            self.generate_dist()
            ret = random.choices(self._goal_space, weights=self.dist, k=size)
        else:
            ret = random.choices(self._goal_space, k=size)
        return ret

    def goal_visualize(self):
        print(self._goal_space, self._goal_counter, self.dist)

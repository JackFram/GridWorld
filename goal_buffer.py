import numpy as np


class GoalBuffer:
    def __init__(self):
        raise NotImplementedError

    def sample_goal(self):
        raise NotImplementedError

    def sample_batch_goal(self, size):
        raise NotImplementedError

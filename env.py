class BaseEnv:
    def __init__(self, size=(5, 5), start_pos=(1, 1), goal=(5, 5)):
        self.size = size
        self.pos = start_pos
        self.goal = goal

    def reset(self, size=(5, 5), start_pos=(1, 1), goal=(5, 5)):
        self.size = size
        self.pos = start_pos
        self.goal = goal
        return self.pos, 0, False

    def step(self, action):
        terminate = False
        prev_s = self.pos
        if action == 'up':
            self.pos = (self.pos[0], min(self.size[1], self.pos[1]+1))
        elif action == 'down':
            self.pos = (self.pos[0], max(1, self.pos[1] - 1))
        elif action == 'left':
            self.pos = (max(1, self.pos[0] - 1), self.pos[1])
        elif action == 'right':
            self.pos = (min(self.size[0], self.pos[0] + 1), self.pos[1])
        next_s = self.pos
        reward = self.get_reward(prev_s, action, next_s)
        if self.pos == self.goal:
            terminate = True
        return next_s, reward, terminate

    def get_reward(self, prev_s, action, next_s):
        return 0

    @property
    def action_size(self):
        return 4

    @property
    def state_size(self):
        return self.size[0] * self.size[1]




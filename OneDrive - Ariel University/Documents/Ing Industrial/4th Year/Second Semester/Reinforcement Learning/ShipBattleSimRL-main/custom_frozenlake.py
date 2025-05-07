import gymnasium as gym
import numpy as np
import random

def generate_random_desc(size=4):
    desc = [['F' for _ in range(size)] for _ in range(size)]
    desc[0][0] = 'S'
    desc[size - 1][size - 1] = 'G'

    holes = random.sample([(i, j) for i in range(size) for j in range(size)
                           if (i, j) not in [(0, 0), (size - 1, size - 1)]],
                          k=random.randint(1, 3))
    for r, c in holes:
        desc[r][c] = 'H'

    falafels = random.sample([(i, j) for i in range(size) for j in range(size)
                              if desc[i][j] == 'F'], k=random.randint(1, 2))
    for r, c in falafels:
        desc[r][c] = 'F'

    return np.array(desc), falafels

class CustomFrozenLakeWrapper(gym.Wrapper):
    def __init__(self, env, falafel_positions, step_penalty=-1,
                 falafel_reward=5, goal_reward=10):
        super().__init__(env)
        self.size = int(np.sqrt(env.observation_space.n))
        self.init_falafels = [r*self.size + c for r, c in falafel_positions]
        self.falafel_states = self.init_falafels.copy()
        self.last_state = None
        self.step_penalty   = step_penalty
        self.falafel_reward = falafel_reward
        self.goal_reward    = goal_reward

    def reset(self, **kwargs):
        self.falafel_states = self.init_falafels.copy()
        self.last_state = None
        return self.env.reset(**kwargs)

    def step(self, action):
        s, base_r, term, trunc, info = self.env.step(action)
        reward = self.step_penalty

        if self.last_state == s:
            reward -= 2  # penalización por estancarse
        self.last_state = s

        if s in self.falafel_states:
            reward += self.falafel_reward
            self.falafel_states.remove(s)

        if base_r == 1.0:
            reward += self.goal_reward
        elif term:
            reward -= 3  # si termina mal

        return s, reward, term, trunc, info

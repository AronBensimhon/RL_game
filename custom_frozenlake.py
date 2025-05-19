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
                              if desc[i][j] == 'F'],
                             k=min(2, len([(i, j) for i in range(size) for j in range(size) if desc[i][j] == 'F'])))
    for r, c in falafels:
        desc[r][c] = 'R'

    return np.array(desc), falafels

class CustomFrozenLakeWrapper(gym.Wrapper):
    def __init__(self, env, falafel_positions, step_penalty=-1,
                 falafel_reward=5, goal_reward=15, death_penalty=-10,
                 stuck_penalty=-2, loop_penalty=-4):
        super().__init__(env)
        self.size = int(np.sqrt(env.observation_space.n))
        self.init_falafels = [r*self.size + c for r, c in falafel_positions]
        self.falafel_states = self.init_falafels.copy()
        self.step_penalty = step_penalty
        self.falafel_reward = falafel_reward
        self.goal_reward = goal_reward
        self.death_penalty = death_penalty
        self.stuck_penalty = stuck_penalty
        self.last_state = None
        self.loop_penalty = loop_penalty
        self.position_history = []
        self.state_visit_counts = dict()

    def reset(self, **kwargs):
        self.falafel_states = self.init_falafels.copy()
        self.last_state = None
        self.position_history = []
        self.state_visit_counts = dict()
        return self.env.reset(**kwargs)

    def step(self, action):
        state, base_reward, done, truncated, info = self.env.step(action)
        reward = self.step_penalty

        # Initialize visit count if new state
        if state not in self.state_visit_counts:
            self.state_visit_counts[state] = 0
        self.state_visit_counts[state] += 1

        if self.state_visit_counts[state] == 1:
            reward += 0.5

        # Loop penalty grows exponentially with repeat visits
        if state in self.position_history:
            loop_penalty = -1 * self.state_visit_counts[state]
            reward += loop_penalty
            
        # Track last few positions
        self.position_history.append(state)
        if len(self.position_history) > 2:
            self.position_history.pop(0)

        # Stuck penalty (if didn't move)
        if self.last_state == state:
            reward += self.stuck_penalty
        self.last_state = state

        # Falafel collection
        if state in self.falafel_states:
            reward += self.falafel_reward
            self.falafel_states.remove(state)

        # Goal or fall
        if base_reward == 1:
            reward += self.goal_reward
        elif done:
            reward += self.death_penalty

        return state, reward, done, truncated, info

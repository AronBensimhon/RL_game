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
                 stuck_penalty=-2, loop_penalty=-4,
                 wind_probability=0.2, wind_bias=[0.25, 0.25, 0.25, 0.25]):
        super().__init__(env)
        self.size = int(np.sqrt(env.observation_space.n))
        self.init_falafels = [r * self.size + c for r, c in falafel_positions]
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

        # Wind parameters for introducing stochasticity
        # wind_probability: float (0 to 1), the chance that wind affects the chosen action.
        #                   0 means no wind, 1 means wind always affects the action.
        self.wind_probability = wind_probability
        # wind_bias: list/array of 4 floats (summing to 1), probabilities for wind direction.
        #            Corresponds to actions [UP, DOWN, LEFT, RIGHT].
        #            e.g., [0.1, 0.1, 0.4, 0.4] would make wind more likely to blow LEFT or RIGHT.
        #            NOTE: For correct application in `apply_wind`, this bias should map to
        #            action indices [0, 1, 2, 3] which typically mean [LEFT, DOWN, RIGHT, UP].
        self.wind_bias = wind_bias


    def apply_wind(self, action: int) -> tuple[int, bool]:
        """
        Applies a stochastic wind effect to the agent's intended action.

        With a probability of `self.wind_probability`, the wind takes over and
        selects a new action based on `self.wind_bias`. Otherwise, the original
        action is returned.

        The actions are assumed to be mapped as follows (standard for FrozenLake):
        0: LEFT
        1: DOWN
        2: RIGHT
        3: UP
        `self.wind_bias` should be a list of 4 probabilities corresponding to these actions.

        Args:
            action (int): The agent's intended action.

        Returns:
            tuple[int, bool]: A tuple containing:
                - resulting_action (int): The action to be taken (original or wind-modified).
                - wind_interfered_flag (bool): True if wind actively chose a new action, False otherwise.
        """
        wind_interfered_flag = False
        resulting_action = action

        if random.random() < self.wind_probability:
            # Wind is active, choose a new action based on wind_bias
            # The population [0, 1, 2, 3] corresponds to LEFT, DOWN, RIGHT, UP
            wind_action = random.choices(population=[0, 1, 2, 3], weights=self.wind_bias, k=1)[0]
            resulting_action = wind_action
            wind_interfered_flag = True

        return resulting_action, wind_interfered_flag

    def reset(self, **kwargs):
        self.falafel_states = self.init_falafels.copy()
        self.last_state = None
        self.position_history = []
        self.state_visit_counts = dict()
        return self.env.reset(**kwargs)

    def step(self, action):
        # Apply wind effect to the action
        effective_action, wind_interfered = self.apply_wind(action)

        # Pass the effective_action (potentially modified by wind) to the environment
        state, base_reward, done, truncated, info = self.env.step(effective_action)

        # Update info dictionary with wind effect details
        if wind_interfered:
            info['wind_active'] = True
            info['wind_direction'] = effective_action
        else:
            info['wind_active'] = False

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
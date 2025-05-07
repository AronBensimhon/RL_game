
import gymnasium as gym      # ← falta esta línea
import numpy as np

class CustomFrozenLakeWrapper(gym.Wrapper):
    def __init__(self, env,
                 falafel_positions=[(1, 2)],
                 step_penalty=-1,
                 falafel_reward=5,
                 goal_reward=10):
        super().__init__(env)
        self.size = int(np.sqrt(env.observation_space.n))
        self.init_falafels = [r*self.size + c for r, c in falafel_positions]
        self.falafel_states = self.init_falafels.copy()

        self.step_penalty   = step_penalty
        self.falafel_reward = falafel_reward
        self.goal_reward    = goal_reward

    # ——— resetea falafels al comenzar episodio ———
    def reset(self, **kwargs):
        self.falafel_states = self.init_falafels.copy()
        return self.env.reset(**kwargs)

    def step(self, action):
        s, base_r, term, trunc, info = self.env.step(action)
        reward = self.step_penalty

        # Falafel: solo la primera vez
        if s in self.falafel_states:
            reward += self.falafel_reward
            self.falafel_states.remove(s)          # ← desaparece

        if base_r == 1.0:                          # meta
            reward += self.goal_reward

        return s, reward, term, trunc, info
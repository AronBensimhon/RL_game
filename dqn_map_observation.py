import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os
import copy
import gymnasium as gym
from custom_frozenlake import CustomFrozenLakeWrapper, generate_random_desc

# Parámetros generales
GRID_SIZE = 4
NUM_ACTIONS = 4
EPISODES = 5000
GAMMA = 0.99
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.99
BATCH_SIZE = 64
MEMORY_SIZE = 10000

# Codificación del entorno
TILE_ENCODING = {
    'S': [0, 0, 0, 0, 1],
    'F': [0, 0, 0, 0, 1],
    'G': [0, 0, 0, 1, 0],
    'H': [0, 1, 0, 0, 0],
    'R': [0, 0, 1, 0, 0]
}


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(112, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.net(x)


def encode_observation(map_grid, agent_pos, visit_map=None):
    state = np.zeros((GRID_SIZE, GRID_SIZE, 7), dtype=np.float32)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            state[i, j, :5] = TILE_ENCODING[map_grid[i][j]]
    state[agent_pos[0], agent_pos[1], 5] = 1.0
    if visit_map is not None:
        state[:, :, 6] = visit_map
    return torch.tensor(state).unsqueeze(0)


def create_env():
    desc, falafel_positions = generate_random_desc(size=GRID_SIZE)
    env = gym.make("FrozenLake-v1", desc=desc, is_slippery=False)
    wrapped_env = CustomFrozenLakeWrapper(
        env,
        falafel_positions=falafel_positions,
        step_penalty=-1,
        falafel_reward=3,
        goal_reward=10,
        death_penalty=-10,
        stuck_penalty=-2,
        loop_penalty=-4
    )
    return wrapped_env


def train_dqn():
    policy_net = DQN()
    target_net = DQN()
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = EPSILON_START
    rewards = []
    goals_reached = 0
    falls = 0
    falafels_eaten = 0

    env = create_env()
    for ep in range(EPISODES):
        if ep % 5 == 0:
            env = create_env()
        else:
            env = copy.deepcopy(env)
        state, _ = env.reset()
        total_reward = 0
        done = False

        visit_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        for step in range(100):
            row, col = divmod(state, GRID_SIZE)
            visit_map[row][col] = 1.0
            obs = encode_observation(env.unwrapped.desc.astype(str), (row, col), visit_map)

            if random.random() < epsilon:
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                with torch.no_grad():
                    action = policy_net(obs).argmax().item()

            next_state, reward, done, _, _ = env.step(action)
            next_row, next_col = divmod(next_state, GRID_SIZE)
            next_obs = encode_observation(env.unwrapped.desc.astype(str), (next_row, next_col), visit_map)

            memory.append((obs, action, reward, next_obs, done))
            state = next_state
            total_reward += reward

            if done:
                desc = env.unwrapped.desc.astype(str)
                row, col = divmod(state, GRID_SIZE)
                if desc[row][col] == 'G':
                    goals_reached += 1
                elif desc[row][col] == 'H':
                    falls += 1

                for f in env.init_falafels:
                    if f not in env.falafel_states:
                        falafels_eaten += 1
                break

            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                obs_b, act_b, rew_b, next_b, done_b = zip(*batch)
                obs_b = torch.cat(obs_b)
                next_b = torch.cat(next_b)
                act_b = torch.tensor(act_b)
                rew_b = torch.tensor(rew_b, dtype=torch.float32)
                done_b = torch.tensor(done_b, dtype=torch.float32)

                q_vals = policy_net(obs_b).gather(1, act_b.unsqueeze(1)).squeeze()
                next_q_vals = target_net(next_b).max(1)[0]
                expected_q = rew_b + GAMMA * next_q_vals * (1 - done_b)

                loss = nn.functional.mse_loss(q_vals, expected_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if ep % 20 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        rewards.append(total_reward)

        if ep % 100 == 0:
            print(f"Episode {ep} | Reward: {total_reward:.2f} | ε={epsilon:.3f}")
            print(f"🏁 Goals reached: {goals_reached} | 💥 Falls: {falls} | 🧆 Falafels eaten: {falafels_eaten}")
            goals_reached = 0
            falls = 0
            falafels_eaten = 0

    return rewards, policy_net


# --- EJECUCIÓN PRINCIPAL ---
print("🧪 Verificando si entra al __main__...")

if __name__ == "__main__":
    print("Running main")
    history, model = train_dqn()

    # 📈 Graficar recompensas
    plt.plot(history)
    plt.title("Total Reward per Episode (DQN)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.show()

    # 💾 Guardar el modelo
    save_path = os.path.join(os.getcwd(), "dqn_model.pt")
    print(f"⚙️  Intentando guardar modelo en: {save_path}")
    torch.save(model.state_dict(), save_path)
    print("✅ Modelo guardado como 'dqn_model.pt'")

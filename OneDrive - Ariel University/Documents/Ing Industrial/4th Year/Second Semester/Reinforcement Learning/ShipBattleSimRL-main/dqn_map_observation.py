import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os

# Parámetros generales
GRID_SIZE = 4
NUM_ACTIONS = 4
EPISODES = 800
GAMMA = 0.99
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
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
            nn.Linear(GRID_SIZE * GRID_SIZE * 6, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_ACTIONS)
        )
    def forward(self, x):
        return self.net(x)

def encode_observation(map_grid, agent_pos):
    state = np.zeros((GRID_SIZE, GRID_SIZE, 6), dtype=np.float32)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            state[i, j, :5] = TILE_ENCODING[map_grid[i][j]]
    state[agent_pos[0], agent_pos[1], 5] = 1.0
    return torch.tensor(state).unsqueeze(0)

def generate_map():
    grid = [['F' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    grid[0][0] = 'S'
    grid[-1][-1] = 'G'
    for _ in range(3):
        i, j = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
        if (i, j) not in [(0, 0), (GRID_SIZE - 1, GRID_SIZE - 1)]:
            grid[i][j] = 'H'
    for _ in range(2):
        i, j = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
        if grid[i][j] == 'F':
            grid[i][j] = 'R'
    return grid

def move(agent, action):
    r, c = agent
    if action == 0 and c > 0: c -= 1
    elif action == 1 and r < GRID_SIZE - 1: r += 1
    elif action == 2 and c < GRID_SIZE - 1: c += 1
    elif action == 3 and r > 0: r -= 1
    return r, c

def get_reward(tile, done):
    if tile == 'H': return -10, True
    if tile == 'R': return 3, False
    if tile == 'G': return 10, True
    return -1, done

def train_dqn():
    policy_net = DQN()
    target_net = DQN()
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = EPSILON_START
    rewards = []

    for ep in range(EPISODES):
        grid = generate_map()
        agent = (0, 0)
        total_reward = 0
        done = False
        visited_positions = []

        for step in range(100):
            obs = encode_observation(grid, agent)
            if random.random() < epsilon:
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                with torch.no_grad():
                    action = policy_net(obs).argmax().item()

            new_agent = move(agent, action)
            tile = grid[new_agent[0]][new_agent[1]]
            reward, done = get_reward(tile, new_agent == (GRID_SIZE - 1, GRID_SIZE - 1) or tile in 'HG')

            # Penalización por estancarse
            if new_agent == agent:
                reward -= 2

            # Penalización por ciclo
            if new_agent in visited_positions:
                reward -= 4
            visited_positions.append(new_agent)
            if len(visited_positions) > 6:
                visited_positions.pop(0)

            next_obs = encode_observation(grid, new_agent)
            memory.append((obs, action, reward, next_obs, done))
            agent = new_agent
            total_reward += reward

            if done:
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
            print(f"Episode {ep} | Reward: {total_reward} | ε={epsilon:.3f}")

    return rewards, policy_net

# --- EJECUCIÓN PRINCIPAL ---
print("🧪 Verificando si entra al __main__...")

if __name__ == "__main__":
    print("✅ Dentro de __main__")
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

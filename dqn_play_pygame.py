import pygame
import torch
import torch.nn as nn
import numpy as np
import os
import time
import random
import gymnasium as gym
from custom_frozenlake import CustomFrozenLakeWrapper, generate_random_desc

# Parámetros
GRID_SIZE = 4
TILE_SIZE = 120
WINDOW_SIZE = GRID_SIZE * TILE_SIZE
FPS = 2
USE_RANDOM_MAP = True
MAX_STEPS = 50


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


TILE_ENCODING = {
    'S': [0, 0, 0, 0, 1],
    'F': [0, 0, 0, 0, 1],
    'G': [0, 0, 0, 1, 0],
    'H': [0, 1, 0, 0, 0],
    'R': [0, 0, 1, 0, 0]
}


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
        loop_penalty=-4,
        # Wind parameters: wind_probability, wind_bias
        wind_probability=0.2,
        wind_bias=[0.25, 0.25, 0.25, 0.25]
    )
    return wrapped_env


# Pygame setup
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("DQN Agent on Frozen Map")
clock = pygame.time.Clock()
FONT = pygame.font.SysFont("Arial Rounded MT Bold", 24)
BIGFONT = pygame.font.SysFont("Arial Rounded MT Bold", 48)

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
RUNNER_IMG = pygame.transform.smoothscale(pygame.image.load(os.path.join(ASSETS, "runner.png")).convert_alpha(),
                                          (TILE_SIZE, TILE_SIZE))
FALAF_IMG = pygame.transform.smoothscale(pygame.image.load(os.path.join(ASSETS, "falafel.png")).convert_alpha(),
                                         (TILE_SIZE, TILE_SIZE))
MATKOT_IMG = pygame.transform.smoothscale(pygame.image.load(os.path.join(ASSETS, "MATKOT.webp")).convert_alpha(),
                                          (TILE_SIZE, TILE_SIZE))
BEACH_IMG = pygame.transform.smoothscale(pygame.image.load(os.path.join(ASSETS, "SEA.jpg")).convert_alpha(),
                                         (TILE_SIZE, TILE_SIZE))
SAND_TILE = pygame.transform.smoothscale(pygame.image.load(os.path.join(ASSETS, "sand.png")).convert(),
                                         (TILE_SIZE, TILE_SIZE))

# Cargar modelo entrenado
model = DQN()
model.load_state_dict(torch.load("dqn_model.pt"))
model.eval()


# Función de visualización
def draw(grid, agent, episode, score, message="", env=None):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            tile = grid[i][j]
            idx = i * GRID_SIZE + j
            # קודם כל לצייר את הרקע או האובייקט
            if idx in env.falafel_states:
                screen.blit(FALAF_IMG, rect)
            elif tile == 'H':
                screen.blit(MATKOT_IMG, rect)
            elif tile == 'G':
                screen.blit(BEACH_IMG, rect)
            else:
                screen.blit(SAND_TILE, rect)

            if (i, j) == agent:
                screen.blit(RUNNER_IMG, rect)

    label = FONT.render(f"Episode: {episode}   Score: {score}", True, (0, 0, 0))
    screen.blit(label, (WINDOW_SIZE - label.get_width() - 10, 10))

    if message:
        msg = BIGFONT.render(message, True, (0, 0, 0))
        screen.blit(msg, (WINDOW_SIZE // 2 - msg.get_width() // 2, WINDOW_SIZE // 2 - msg.get_height() // 2))

    pygame.display.flip()


# Estado inicial
episode = 1
score = 0
message = ""
message_timer = 0
env = create_env()
state, _ = env.reset()
visit_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
step_count = 0

# Loop principal
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    row, col = divmod(state, GRID_SIZE)
    visit_map[row][col] += 1
    obs = encode_observation(env.unwrapped.desc.astype(str), (row, col), visit_map)

    with torch.no_grad():
        action = model(obs).argmax().item()

    next_state, reward, done, _, _ = env.step(action)
    score += reward
    state = next_state
    step_count += 1

    # Mensaje temporal
    if message_timer > 0 and pygame.time.get_ticks() - message_timer > 1000:
        message = ""

    desc = env.unwrapped.desc.astype(str)
    row, col = divmod(state, GRID_SIZE)
    draw(desc, (row, col), episode, score, message, env)

    if done or step_count >= MAX_STEPS:
        row, col = divmod(state, GRID_SIZE)
        desc = env.unwrapped.desc.astype(str)
        if desc[row][col] == 'G':
            message = "You reached the beach!"
        elif desc[row][col] == 'H':
            message = "Matkot trap!"
        else:
            message = "⏹️ Max steps reached"

        # Reset para nuevo episodio
        episode += 1
        score = 0
        env = create_env()
        state, _ = env.reset()
        visit_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        collected_rewards = set()
        position_history = []
        step_count = 0
        message = ""
        message_timer = 0

    clock.tick(FPS)

pygame.quit()

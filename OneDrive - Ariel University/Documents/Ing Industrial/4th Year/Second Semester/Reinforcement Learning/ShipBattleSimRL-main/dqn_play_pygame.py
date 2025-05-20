import pygame
import torch
import torch.nn as nn
import numpy as np
import os
import time
import random

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
            nn.Linear(GRID_SIZE * GRID_SIZE * 6, 128),
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


def encode_observation(grid, agent_pos):
    obs = np.zeros((GRID_SIZE, GRID_SIZE, 6), dtype=np.float32)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            obs[i, j, :5] = TILE_ENCODING[grid[i][j]]
    obs[agent_pos[0], agent_pos[1], 5] = 1.0
    return torch.tensor(obs).unsqueeze(0)


def move(agent, action):
    r, c = agent
    if action == 0 and c > 0:
        c -= 1
    elif action == 1 and r < GRID_SIZE - 1:
        r += 1
    elif action == 2 and c < GRID_SIZE - 1:
        c += 1
    elif action == 3 and r > 0:
        r -= 1
    return r, c


def generate_map():
    grid = [['F' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    grid[0][0] = 'S'
    grid[-1][-1] = 'G'

    if USE_RANDOM_MAP:
        # Generar posiciones aleatorias para hoyos
        holes = random.sample([(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)
                               if (i, j) not in [(0, 0), (GRID_SIZE - 1, GRID_SIZE - 1)]], 3)
        for r, c in holes:
            grid[r][c] = 'H'

        # Generar posiciones aleatorias para falafel
        empty = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if grid[i][j] == 'F']
        falafels = random.sample(empty, 2)
        for r, c in falafels:
            grid[r][c] = 'R'
    else:
        grid[1][2] = 'H'
        grid[2][0] = 'R'
        grid[3][1] = 'H'

    return grid


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
FALAF_IMG = pygame.transform.smoothscale(pygame.image.load(os.path.join(ASSETS, "falafel.jpeg")).convert_alpha(),
                                         (TILE_SIZE, TILE_SIZE))
MATKOT_IMG = pygame.transform.smoothscale(pygame.image.load(os.path.join(ASSETS, "MATKOT.jpeg")).convert_alpha(),
                                          (TILE_SIZE, TILE_SIZE))
BEACH_IMG = pygame.transform.smoothscale(pygame.image.load(os.path.join(ASSETS, "SEA.png")).convert_alpha(),
                                         (TILE_SIZE, TILE_SIZE))
SAND_TILE = pygame.transform.smoothscale(pygame.image.load(os.path.join(ASSETS, "sand.jpeg")).convert(),
                                         (TILE_SIZE, TILE_SIZE))

# Cargar modelo entrenado
model = DQN()
model.load_state_dict(torch.load("dqn_model.pt"))
model.eval()


# Función de visualización
def draw(grid, agent, episode, score, message=""):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rect = pygame.Rect(j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            tile = grid[i][j]
            if tile == 'H':
                screen.blit(MATKOT_IMG, rect)
            elif tile == 'G':
                screen.blit(BEACH_IMG, rect)
            elif tile == 'R':
                screen.blit(FALAF_IMG, rect)
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
grid = generate_map()
agent = (0, 0)
collected_rewards = set()
position_history = []
step_count = 0

# Loop principal
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    obs = encode_observation(grid, agent)
    with torch.no_grad():
        action = model(obs).argmax().item()

    next_agent = move(agent, action)
    tile = grid[next_agent[0]][next_agent[1]]
    reward = -1
    message = ""

    if tile == 'R' and next_agent not in collected_rewards:
        reward = 3
        collected_rewards.add(next_agent)
        grid[next_agent[0]][next_agent[1]] = 'F'
        message = "Falafel +3!"
        message_timer = pygame.time.get_ticks()
    elif tile == 'G':
        reward = 10
    elif tile == 'H':
        reward = -10

    if next_agent == agent:
        reward -= 2

    # Penalizar bucles de movimiento
    if next_agent in position_history:
        reward -= 4
    position_history.append(next_agent)
    if len(position_history) > 6:
        position_history.pop(0)

    score += reward
    print(f"{agent} → {next_agent} | Acción: {action} | Recompensa: {reward}")
    agent = next_agent
    step_count += 1

    # Mensaje temporal
    if message_timer > 0 and pygame.time.get_ticks() - message_timer > 1000:
        message = ""

    draw(grid, agent, episode, score, message)

    if tile in ['H', 'G'] or step_count >= MAX_STEPS:
        if tile == 'G':
            message = "You reached the beach!"
            print(f"🏁 Episodio {episode} terminado – {message} | Score: {score}")
        elif tile == 'H':
            message = " Matkot Trap!"
            print(f"💥 Episodio {episode} terminado – {message} | Score: {score}")
        else:
            message = "⏹️ Max steps reached"
            print(f"⏹️ Episodio {episode} agotado | Score: {score}")

        draw(grid, agent, episode, score, message)
        pygame.display.flip()
        time.sleep(2)

        # Reset para nuevo episodio
        episode += 1
        score = 0
        grid = generate_map()
        agent = (0, 0)
        collected_rewards = set()
        position_history = []
        step_count = 0
        message = ""
        message_timer = 0

    clock.tick(FPS)

pygame.quit()

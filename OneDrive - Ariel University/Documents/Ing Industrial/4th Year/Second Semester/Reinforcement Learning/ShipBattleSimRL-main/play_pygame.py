import os, sys, pygame, numpy as np, random
import gymnasium as gym
from custom_frozenlake import CustomFrozenLakeWrapper

# ─────────── parámetros ────────────
BOARD, WINDOW, FPS = 4, 480, 10
TILE = WINDOW // BOARD

pygame.init()
FONT = pygame.font.SysFont("Arial Rounded MT Bold", 22)
ASSETS = os.path.join(os.path.dirname(__file__), "assets")

# --- cargar imágenes raw ---
RUNNER_RAW  = pygame.image.load(os.path.join(ASSETS, "runner.png"))
FALAF_RAW   = pygame.image.load(os.path.join(ASSETS, "falafel.jpeg"))
MATKOT_RAW  = pygame.image.load(os.path.join(ASSETS, "MATKOT.jpeg"))
BEACH_RAW   = pygame.image.load(os.path.join(ASSETS, "SEA.png"))
SAND_RAW    = pygame.image.load(os.path.join(ASSETS, "sand.jpeg"))

# --- crear ventana + convertir/escala ---
screen = pygame.display.set_mode((WINDOW, WINDOW))
pygame.display.set_caption("Runner to Beach")

def cvt(img): return pygame.transform.smoothscale(img.convert_alpha(), (TILE, TILE))
RUNNER_IMG  = cvt(RUNNER_RAW)
FALAF_IMG   = cvt(FALAF_RAW)
MATKOT_IMG  = cvt(MATKOT_RAW)
BEACH_IMG   = cvt(BEACH_RAW)
SAND_TILE   = pygame.transform.smoothscale(SAND_RAW.convert(), (120, 120))

# ---------- función para generar mapa dinámico ----------
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

# ---------- inicializar entorno ----------
def create_env():
    desc, falafel_positions = generate_random_desc()
    print("🗺 New Map:")
    for row in desc:
        print(" ".join(row))
    return CustomFrozenLakeWrapper(
        gym.make("FrozenLake-v1", desc=desc, is_slippery=False),  # ← viento manual
        falafel_positions=falafel_positions,
        step_penalty=-1,
        falafel_reward=5,
        goal_reward=10)

env = create_env()
state, _ = env.reset()
episode, total = 1, 0
message = ""
message_time = 0

key2action = {pygame.K_LEFT:0, pygame.K_DOWN:1, pygame.K_RIGHT:2, pygame.K_UP:3}
clock = pygame.time.Clock()

# ---------- funciones de dibujado ----------
def draw_background():
    tw, th = SAND_TILE.get_size()
    for y in range(0, WINDOW, th):
        for x in range(0, WINDOW, tw):
            screen.blit(SAND_TILE, (x, y))

def draw_board():
    desc = [[c.decode() if isinstance(c, bytes) else str(c) for c in row] for row in env.unwrapped.desc]
    for r in range(BOARD):
        for c in range(BOARD):
            idx  = r * BOARD + c
            cell = pygame.Rect(c*TILE, r*TILE, TILE, TILE)

            if desc[r][c] == "H":
                screen.blit(MATKOT_IMG, cell)
            elif desc[r][c] == "G":
                screen.blit(BEACH_IMG, cell)

            if idx in env.falafel_states:
                screen.blit(FALAF_IMG, cell)
            if idx == state:
                screen.blit(RUNNER_IMG, cell)

            pygame.draw.rect(screen, (0,0,0,40), cell, 1)

def draw_ui():
    txt = FONT.render(f"Episode: {episode}   Score: {total}", True, (0,0,0))
    screen.blit(txt, (WINDOW - txt.get_width() - 10, 10))

def draw_message():
    if message and pygame.time.get_ticks() - message_time < 1500:
        msg_surface = FONT.render(message, True, (255, 255, 255))
        msg_rect = msg_surface.get_rect(center=(WINDOW // 2, WINDOW // 2))

        bg_rect = pygame.Rect(msg_rect.x - 10, msg_rect.y - 5,
                              msg_rect.width + 20, msg_rect.height + 10)
        msg_bg = pygame.Surface((bg_rect.width, bg_rect.height))
        msg_bg.set_alpha(200)
        msg_bg.fill((0, 0, 0))

        screen.blit(msg_bg, (bg_rect.x, bg_rect.y))
        screen.blit(msg_surface, msg_rect)

def redraw():
    draw_background()
    draw_board()
    draw_ui()
    draw_message()
    pygame.display.flip()

# ---------- bucle principal ----------
running = True
while running:
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                running = False
            elif ev.key in key2action:
                action = key2action[ev.key]

                # viento artificial con 20% de probabilidad
                if random.random() < 0.2:
                    action = random.choice([a for a in range(4) if a != action])
                    message = "Strong wind!"
                    message_time = pygame.time.get_ticks()

                prev_state = state
                s, r, term, trunc, _ = env.step(action)
                state, total = s, total + r

                # mostrar falafel si corresponde
                if r >= 4:
                    message = "Falafel +5!"
                    message_time = pygame.time.get_ticks()

                if term or trunc:
                    desc = env.unwrapped.desc.astype("U")
                    row, col = divmod(state, BOARD)
                    if desc[row][col] == "G":
                        message = "You reached the beach!"
                    else:
                        message = "Matkot trap!"
                    message_time = pygame.time.get_ticks()

                    redraw()
                    pygame.time.delay(1500)

                    print(f"🏁 Episode {episode} finished • Score: {total}")
                    episode += 1
                    env = create_env()
                    state, _ = env.reset()
                    total = 0

    redraw()
    clock.tick(FPS)

pygame.quit()
sys.exit()
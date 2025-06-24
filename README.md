# Run to the Beach – Reinforcement Learning Final Project

This project is a custom Deep Q-Network (DQN) implementation built to solve a dynamic and partially stochastic gridworld inspired by OpenAI's Frozen Lake environment.

Our environment adds real-world complexity with randomized maps, wind disturbances, penalties for repetition, and optional rewards.

---

## Project Overview

- **Environment**: 4×4 grid with 5 tile types: Start, Sand, Goal (Beach), Falafel (Reward), and Matkot (Trap).
- **Agent**: Learns to navigate efficiently using Deep Q-Network with experience replay and target networks.
- **Objective**: Reach the beach safely, collecting rewards while avoiding traps and wind-induced deviations.
- **Training**: 50,000 episodes using PyTorch and visualized with `pygame`.

---

## Environment Snapshot

<img width="221" alt="Screenshot 2025-06-13 134151" src="https://github.com/user-attachments/assets/ae548e89-36c8-41c5-baad-7797992a99a9" />

---

## Project Structure

```bash
.
├── dqn_map_observation.py     # DQN agent and training logic
├── custom_frozenlake.py       # Environment definition
├── dqn_play_pygame.py         # Visual simulator with Pygame
├── requirements.txt           # Python dependencies
├── assets/                    # Visuals and figures
└── README.md                  # Project documentation
```

## Installation & Running
### 1. Clone this repository:

```bash
git clone https://github.com/AronBensimhon/RL_game.git
cd RL_game
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Train the agent
```bash
python dqn_map_observation.py
```
This will create dqn_model.pt

### 4. Run the visualization
```bash
python dqn_play_pygame.py
```
Note: The custom environment logic is defined in `custom_frozenlake.py`, which is imported automatically by the training and visualization scripts. You do not need to run it directly.



## Features

- Falafel tile: **+3** reward, disappears when collected.
- Matkot trap: **–10** penalty, ends the episode immediately.
- Goal tile: **+10** reward, ends the episode.
- Step penalty: **–1** per step to encourage efficient navigation.
- Repeated visit penalty: **–k** where *k* is the number of times the tile has been visited in the current episode.
- Stuck penalty: **–2** if the agent attempts an action but does not change position.
- Wind: 20% probability of wind randomly changing the agent’s intended action.
- Episode termination: Ends after reaching Goal, Trap, or after a maximum number of steps (default 100 for training, 50 for visualization).
- Reward plot: Generated during training to show performance over episodes.




## Authors

- Aron Bensimhon
- Oriana Felszer
- Eden Shmuel

Affiliation: Industrial Engineering and Management, Ariel University.








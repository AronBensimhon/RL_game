# DQN RunToTheBeach Project

This project implements a Deep Q-Network (DQN) agent trained to navigate a custom FrozenLake-style grid world using Python, PyTorch, and Pygame.

## 🧠 Project Overview

- **Environment**: 4×4 grid with:
  - 🟢 Start tile (S)
  - 🎯 Goal tile (G)
  - 💥 Hazard tiles (matkot – H)
  - 🌮 Reward tiles (falafel – R)
  - 🏖️ Safe tiles (F)
- **Agent**: Learns to reach the beach (goal) while avoiding traps and collecting falafel
- **Model**: DQN with:
  - Input: 4×4×6 tensor (multi-channel map encoding)
  - Network: Flatten → Linear(128) → ReLU → Linear(4)
  - Epsilon-greedy exploration, target network, replay buffer

## 📂 Files

| File | Description |
|------|-------------|
| `dqn_map_observation.py` | Trains the agent using DQN and saves the model as `dqn_model.pt` |
| `dqn_play_pygame.py`     | Loads the trained model and visualizes the agent in Pygame |
| `assets/`                | Contains images for the tiles (runner, falafel, matkot, sea, sand) |

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install torch pygame matplotlib 
```


### 2. Train the agent
```bash
python dqn_map_observation.py
```

This will create dqn_model.pt

### 3. Run the visualization
```bash
python dqn_play_pygame.py
```

## 🎮 Features
- 🧱 Falafel gives +3 and disappears when collected

- 💥 Matkot trap ends the episode and penalizes the agent

- 🏁 Goal tile gives +10 and ends the episode

- 🔁 Automatic reset each episode (or after 50 steps)

- 📈 Total reward graph shown during training

- ✨ On-screen messages: "Falafel +3!", "Matkot trap!"






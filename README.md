# Run to the Beach 🏖️ – Reinforcement Learning Final Project

This project is a custom Deep Q-Network (DQN) implementation built to solve a dynamic and partially stochastic gridworld inspired by OpenAI's Frozen Lake environment.

Our environment adds real-world complexity with randomized maps, wind disturbances, penalties for repetition, and optional rewards.

---

## 🔍 Project Overview

- **Environment**: 4×4 grid with 5 tile types: Start, Sand, Goal (Beach), Falafel (Reward), and Matkot (Trap).
- **Agent**: Learns to navigate efficiently using Deep Q-Network with experience replay and target networks.
- **Objective**: Reach the beach safely, collecting rewards while avoiding traps and wind-induced deviations.
- **Training**: 50,000 episodes using PyTorch and visualized with `pygame`.

---

## 📸 Environment Snapshot

<img width="221" alt="Screenshot 2025-06-13 134151" src="https://github.com/user-attachments/assets/ae548e89-36c8-41c5-baad-7797992a99a9" />

---

## 📁 Project Structure

```bash
.
├── dqn_map_observation.py     # DQN agent and training logic
├── custom_frozenlake.py       # Environment definition
├── dqn_play_pygame.py         # Visual simulator with Pygame
├── requirements.txt           # Python dependencies
├── RL/                        # Visuals and figures (e.g. reward plots)
└── README.md                  # Project documentation
```

## 📦  Installation & Running
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



## 🎮 Features

- 🧱 Falafel tile gives **+3** points and disappears when collected
- 💥 Matkot trap gives **–10** and ends the episode immediately
- 🏁 Goal tile gives **+10** and ends the episode
- 🚶 Step penalty of **–1** to encourage efficient navigation
- 🔁 Repeated visits penalized by **–n**, where *n* is the number of times the tile was visited
- 🧍 Stuck penalty of **–2** when the agent fails to move
- 🌬️ 20% wind probability that randomly changes agent’s direction
- 🔁 Environment resets after each episode or 50 steps
- 📈 Reward plot generated during training




## 👨‍💻 Authors

- Aron Bensimhon  
- Oriana Felszer  
- Eden Shmuel

  
Industrial Engineering and Management, Ariel University








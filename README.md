# Reinforcement Learning for Autonomous Car Navigation

## Test Output GIFs
Here are some video from during the training of our agent trying to reach its goal.

<div style="display: flex; gap: 10px;">
  <img src="https://github.com/amiraliaali/cross_street/blob/main/readme_media/training_time.gif" width="900" height="500" />
</div>

And here  some video from during the test time of our agent trying to reach its goal.

<div style="display: flex; gap: 10px;">
  <img src="https://github.com/amiraliaali/cross_street/blob/main/readme_media/test_time.gif" width="700" height="575" />
</div>

## Overview
This project implements a reinforcement learning approach using Proximal Policy Optimization (PPO) to train an autonomous agent (a free-moving car) to navigate a space with obstacles and reach a predefined goal spot without collisions. The agent learns through trial and error, improving its navigation strategy over time.

## Features
- Implemented using **PyTorch** for deep reinforcement learning.
- Uses a **custom-built environment** for training the car agent.
- **PPO-based Actor-Critic architecture** for policy optimization.
- **Model checkpointing** to save the best-performing model.
- **Dynamic learning rate scheduling** for better training efficiency.
- **Training performance visualization** through loss and return graphs.

## Environment Details
The environment consists of:
- A car that starts from a random location.
- Randomly placed obstacles at predefined locations.
- A randomly chosen parking spot (goal) that the car needs to reach without collisions.
- Discrete action space defining possible movements.
- Reward function guiding the agent towards the goal.

## Project Structure
```
📂 project_root
├── output_videos/  # Videos from during the training and test time
├── readme_media/   # All the media used to be shown in the readme, like result videos etc.
├── scripts/
  ├── environment.py      # Custom environment for car navigation
  ├── replay_memory.py    # A script for the replay memory
  ├── car.py              # Class definition for the car
  ├── car_park_UI.py      # For running the training
  ├── car_park_a2c.py     # Training script for a simple actor2critic
  └── car_park_ppo.py     # Training script for PPO
└── training_output # A plot for the loss and rewards, plus three optimal weights 

```

## Running the Training
Using vscode, simply run the task ```Car Park UI```.

## Model Architecture
The PPO model consists of:
- **Actor Network**: Predicts action probabilities.
- **Critic Network**: Estimates state values for advantage calculation.
- **Layer Normalization & ReLU activations** to improve training stability.

## Results
Here is the plot for the reward and the loss through time.

<div style="display: flex; gap: 10px;">
  <img src="https://github.com/amiraliaali/self_car_park/blob/main/training_output/loss_return_after_46000.png" width="800" height="700" />
</div>

All the result videos can be found under ```readme_media```. Including high quality videos.

## Future Improvements
- Experiment with different reward shaping strategies.
- Extend the environment with dynamic obstacles.
- Compare PPO with other RL algorithms like DQN or SAC.

## Author
Amir Ali Aali
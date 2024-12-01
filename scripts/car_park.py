import numpy as np
import random
from torch import nn as nn
import torch
from replay_memory import ReplayMemory
from torch.optim import AdamW
from tqdm import tqdm
import copy
import torch.nn.functional as F
import cv2 as cv
import environment as env


class CarPark:
    def __init__(self):
        self.num_actions = len(env.ACTIONS_MAPPING)
        self.states_dim = len(env.get_current_state())

        self.q_network = self.generate_network()
        self.target_q_network = copy.deepcopy(self.q_network).eval()

    def generate_network(self):
        return nn.Sequential(
            nn.Linear(self.states_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, self.num_actions))

    def policy(self, state, epsilon=0.):
        if torch.rand(1) < epsilon:
            # Create a 1D tensor with a single element
            return torch.randint(self.num_actions, (1,))
        else:
            av = self.q_network(state).detach()
            # Ensure the output is 1D by setting keepdim=False
            return torch.argmax(av, dim=-1, keepdim=False).unsqueeze(0)


    def next_step(self, action):
        done, reward, next_state = env.execute_move(action)

        return next_state, reward, done

    def step(self, state, action):
        state = state.numpy().flatten()
        state = tuple(int(x) for x in state)
        action = action.item()
        next_state, reward, done = self.next_step(action)
        
        # Ensure next_state has shape [6]
        next_state = torch.from_numpy(np.array(next_state)).float().squeeze()
        
        # Make reward and done 1D tensors
        reward = torch.tensor(reward).float().unsqueeze(0)  # Shape [1]
        done = torch.tensor(done).float().unsqueeze(0)      # Shape [1]
        
        return next_state, reward, done


    def get_current_state(self):
        return torch.from_numpy(np.array(env.get_current_state()))

    def reset(self):
        env.reset()
        return torch.from_numpy(np.array(env.get_current_state(), dtype=np.float32))

    def train_deep_sarsa(self, episodes, alpha=0.0001, batch_size=64, gamma=0.99, epsilon=0.33, update_every=10):
        optim = AdamW(self.q_network.parameters(), lr=alpha)
        memory = ReplayMemory()
        stats = {"MSE Loss": [], "Returns": []}

        progress_bar = tqdm(range(1, episodes+1), desc="Training", leave=True)

        for episode in progress_bar:
            state = self.reset()
            state = torch.FloatTensor(state)  
            done = False
            ep_return = 0

            while not done:
                action = self.policy(state, epsilon)
                next_state, reward, done = self.step(state, action)
                memory.insert([state, action, reward, done, next_state])

                if memory.can_sample(batch_size):
                    state_b, action_b, reward_b, done_b, next_state_b = memory.sample(batch_size)

                    if action_b.dim() == 1:
                        action_b = action_b.unsqueeze(1)

                    # Compute Q-value estimates for the current state-action pairs
                    qsa_b = self.q_network(state_b).gather(1, action_b)

                    # Compute the target Q-value estimates for the next state-action pairs
                    next_action_b = self.policy(next_state_b, epsilon)

                    if next_action_b.dim() == 1:
                        next_action_b = next_action_b.unsqueeze(1) 
                    
                    next_qsa_b = self.target_q_network(next_state_b).gather(1, next_action_b)

                    # Convert `done_b` to Boolean tensor if it's not already
                    done_b = done_b.bool()

                    # Compute the target Q-value using the done mask
                    target_b = reward_b + (~done_b).float() * gamma * next_qsa_b

                    # Select only the first element of each row to reshape target_b to [32, 1]
                    target_b = target_b[:, :1]

                    # Calculate the loss between predicted and target Q-values
                    loss = F.mse_loss(qsa_b, target_b)

                    # Backpropagate and update the network weights
                    self.q_network.zero_grad()
                    loss.backward()
                    optim.step()

                    # Log the loss for monitoring
                    stats["MSE Loss"].append(loss.item())

                
                state = next_state
                ep_return += reward.item()

            stats["Returns"].append(ep_return)

             # Update the progress bar description with the average reward
            if episode % update_every == 0:
                avg_return = np.mean(stats["Returns"][-update_every:])
                avg_loss = np.mean(stats["MSE Loss"][-update_every:]) if stats["MSE Loss"] else 0
                progress_bar.set_description(f"Training (Avg Reward: {avg_return:.2f}, Avg Loss: {avg_loss:.4f})")

            if episode % 50 == 0:
                self.target_q_network.load_state_dict(self.q_network.state_dict())
        return stats




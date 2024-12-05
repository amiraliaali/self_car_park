import numpy as np
from torch import nn as nn
import torch
from replay_memory import ReplayMemory
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn.functional as F
import environment as env

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Shared base layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),  # Outputs probabilities for actions
        )
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1),  # Outputs state value
        )

    def forward(self, state):
        base = self.shared(state)
        action_probs = self.actor(base)
        state_value = self.critic(base)
        return action_probs, state_value


class CarPark:
    def __init__(self):
        self.environment_inst = env.Environment()
        self.num_actions = len(env.ACTIONS_MAPPING)
        self.states_dim = len(self.environment_inst.get_current_state())
        self.model = ActorCritic(self.states_dim, self.num_actions)
        self.optimizer = AdamW(self.model.parameters(), lr=0.001)

        self.replay_memory = ReplayMemory(capacity=10000)  # Replay buffer
        self.batch_size = 32  # Mini-batch size

    def save_model(self, path="/Users/amiraliaali/Documents/Coding/RL/cross_street/actor_critic.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}.")

    def load_model(self, path="/Users/amiraliaali/Documents/Coding/RL/cross_street/actor_critic.pth"):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f"Model loaded from {path}.")

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            action_probs, _ = self.model(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)

    def train_actor_critic(self, episodes, gamma=0.99, update_every=5):
        stats = {"Loss": [], "Returns": []}
        progress_bar = tqdm(range(1, episodes + 1), desc="Training", leave=True)

        for episode in progress_bar:
            state = self.environment_inst.env_reset()
            done = False
            ep_return = 0

            while not done:
                action, log_prob = self.select_action(state)
                done, reward, next_state = self.environment_inst.execute_move(action)

                # Push experience to replay memory
                self.replay_memory.push((state, action, log_prob, reward, next_state, done))

                state = next_state
                ep_return += reward

                # Train if replay memory has enough samples
                if len(self.replay_memory) >= self.batch_size:
                    self.train_batch(gamma)

            stats["Returns"].append(ep_return)

            # Update progress bar
            if episode % update_every == 0:
                avg_return = np.mean(stats["Returns"][-update_every:])
                progress_bar.set_description(f"Training (Avg Reward: {avg_return:.2f})")

        return stats

    def train_batch(self, gamma):
        # Sample a mini-batch from replay memory
        batch = self.replay_memory.sample(self.batch_size)
        states, actions, log_probs, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute values and advantages
        action_probs, state_values = self.model(states)
        _, next_state_values = self.model(next_states)
        next_state_values = next_state_values.detach()

        # Compute targets for the critic
        targets = rewards + gamma * next_state_values.squeeze() * (1 - dones)

        # Compute advantages for the actor
        advantages = targets - state_values.squeeze()

        # Actor loss (policy gradient)
        action_log_probs = torch.stack(log_probs)
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=1).mean()  # Entropy loss
        actor_loss = -torch.sum(action_log_probs * advantages.detach()) - 0.01 * entropy  # Add entropy term


        # Critic loss (value function)
        critic_loss = F.mse_loss(state_values.squeeze(), targets)

        loss = actor_loss + critic_loss

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


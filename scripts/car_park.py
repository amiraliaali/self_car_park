import numpy as np
from torch import nn
import torch
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn.functional as F
import environment as env

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor_base = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        actor_base = self.actor_base(state)
        action_probs = self.actor(actor_base)
        state_value = self.critic(state)
        return action_probs, state_value


class CarPark:
    def __init__(self):
        self.environment_inst = env.Environment()
        self.num_actions = len(env.ACTIONS_MAPPING)
        self.states_dim = len(self.environment_inst.get_current_state())
        self.model = ActorCritic(self.states_dim, self.num_actions)
        self.optimizer = AdamW(self.model.parameters(), lr=0.0001)

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

                # Train directly on the current step
                loss = self.train_step(state, action, log_prob, reward, next_state, done, gamma)

                state = next_state
                ep_return += reward

            stats["Returns"].append(ep_return)

            # Update progress bar
            if episode % update_every == 0:
                avg_return = np.mean(stats["Returns"][-update_every:])
                progress_bar.set_description(f"Training (Avg Reward: {avg_return:.2f})")

        return stats

    def train_step(self, state, action, log_prob, reward, next_state, done, gamma):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([done])

        # Forward pass
        _, state_value = self.model(state)
        _, next_state_value = self.model(next_state)

        # Compute target and advantage
        target = reward + gamma * next_state_value.squeeze() * (1 - done)
        advantage = target - state_value.squeeze()

        # Actor loss
        entropy = -torch.sum(log_prob * torch.log(log_prob + 1e-10))
        actor_loss = -log_prob * advantage - 0.04 * entropy

        # Critic loss
        critic_loss = F.mse_loss(state_value.squeeze(), target)

        loss = actor_loss + 0.5 * critic_loss

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

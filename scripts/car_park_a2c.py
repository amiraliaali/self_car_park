import numpy as np
from torch import nn
import torch
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn.functional as F
import environment as env
import matplotlib.pyplot as plt

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
        self.optimizer = AdamW(self.model.parameters(), lr=0.001)
        self.highest_reward = 0

    def save_model(self, path="/Users/amiraliaali/Documents/Coding/RL/cross_street/training_output/actor_critic.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}.")

    def load_model(self, path="/Users/amiraliaali/Documents/Coding/RL/cross_street/training_output/best_actor_critic.pth"):
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

    def train(self, episodes, gamma=0.99, update_every=5, save_best_model=True):
        stats = {"Loss": [], "Returns": []}
        progress_bar = tqdm(range(1, episodes + 1), desc="Training", leave=True)

        for episode in progress_bar:
            state = self.environment_inst.env_reset()
            done = False
            ep_return = 0
            current_ep_loss = []
            while not done:
                action, log_prob = self.select_action(state)
                done, reward, next_state = self.environment_inst.execute_move(action)

                # Train directly on the current step
                loss = self.train_step(state, action, log_prob, reward, next_state, done, gamma)
                current_ep_loss.append(loss)

                state = next_state
                ep_return += reward
            stats["Loss"].append(np.mean(current_ep_loss))
            if save_best_model:
                if ep_return > self.highest_reward:
                    self.highest_reward = ep_return
                    self.save_model("/Users/amiraliaali/Documents/Coding/RL/cross_street/training_output/best_actor_critic.pth")
            
            stats["Returns"].append(ep_return)

            # Update progress bar
            if episode % update_every == 0:
                avg_return = np.mean(stats["Returns"][-update_every:])
                progress_bar.set_description(f"Training (Avg Reward: {avg_return:.2f})")
            
            if episode % 500 == 0:
                self.plot_stats(stats, episode)
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
        entropy = -torch.exp(log_prob) * log_prob
        entropy = entropy.sum()
        actor_loss = -log_prob * advantage - 0.02 * entropy

        # Critic loss
        critic_loss = F.mse_loss(state_value.squeeze(1), target)

        loss = actor_loss + 1 * critic_loss

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        return loss.item()

    def plot_stats(self, stats, episode_num):
        # Create subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Plot Losses
        axs[0].plot(stats["Loss"], label="Loss")
        axs[0].set_title("Loss Over Time")
        axs[0].set_xlabel("Training Step")
        axs[0].set_ylabel("Loss")
        axs[0].legend()

        # Plot Returns
        axs[1].plot(stats["Returns"], label="Returns")
        axs[1].set_title("Returns Over Episodes")
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Returns")
        axs[1].legend()

        plt.tight_layout()
        fig.savefig(f"/Users/amiraliaali/Documents/Coding/RL/cross_street/training_output/loss_return_after_{episode_num}.png")
        plt.close(fig)

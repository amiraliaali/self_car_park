import numpy as np
from torch import nn
import torch
from torch.optim import AdamW, Adam, lr_scheduler
from tqdm import tqdm
import torch.nn.functional as F
import environment as env
import matplotlib.pyplot as plt
import math

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOActorCritic, self).__init__()
        self.actor_base = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
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
        self.state_dim = len(self.environment_inst.get_current_state())
        self.model = PPOActorCritic(self.state_dim, self.num_actions)
        self.optimizer = AdamW(self.model.parameters(), lr=0.001)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.75)
        self.highest_reward = 0
        self.eps_clip = 0.3

    def save_model(self, path="/Users/amiraliaali/Documents/Coding/RL/cross_street/ppo_model.pth"):
        torch.save(self.model.state_dict(), path)
        # print(f"Model saved to {path}.")

    def load_model(self, path="/Users/amiraliaali/Documents/Coding/RL/cross_street/ppo_model.pth"):
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

    def train(self, episodes, batch_size=4, gamma=0.98, update_every=5, save_best_model=True, ppo_epochs=4):
        stats = {"Loss": [], "Returns": []}
        progress_bar = tqdm(range(1, episodes + 1), desc="Training", leave=True)
        least_loss = math.inf

        memory = []

        for episode in progress_bar:
            if episode % 7500 == 0:
                self.scheduler.step()
                print(f"Updated learning rate: {self.optimizer.param_groups[0]['lr']}")

            for i in range(batch_size):
                state = self.environment_inst.env_reset(episode_num=episode, iteration_num=i)
                done = False
                ep_return = 0
                parked = False

                while not done:
                    action, log_prob = self.select_action(state)
                    done, reward, next_state, parked = self.environment_inst.execute_move(action)

                    memory.append((state, action, log_prob, reward, next_state, done))
                    state = next_state
                    ep_return += reward

                if save_best_model and parked and episode % 2500 == 0:
                    self.save_model(f"/Users/amiraliaali/Documents/Coding/RL/cross_street/training_output/ppo_model_{episode}_{i}.pth")

            mean_loss = self.update(memory, gamma, ppo_epochs)
            stats["Loss"].append(mean_loss)

            if mean_loss < least_loss:
                least_loss = mean_loss
                self.save_model(f"/Users/amiraliaali/Documents/Coding/RL/cross_street/ppo_model_least_loss.pth")

            memory = []


            stats["Returns"].append(ep_return)

            # Update progress bar
            if episode % update_every == 0:
                avg_return = np.mean(stats["Returns"][-update_every:])
                progress_bar.set_description(f"Training (Avg Reward: {avg_return:.2f})")

            if episode % 1000 == 0:
                self.plot_stats(stats, episode)

        return stats

    def update(self, memory, gamma, ppo_epochs):
        states, actions, log_probs, rewards, next_states, dones = zip(*memory)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        log_probs = torch.cat(log_probs)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        discounted_rewards = []
        running_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            running_reward = reward + gamma * running_reward * (1 - done)
            discounted_rewards.insert(0, running_reward)

        discounted_rewards = torch.FloatTensor(discounted_rewards).detach()

        current_ep_loss = []
        for _ in range(ppo_epochs):
            action_probs, state_values = self.model(states)
            dist = torch.distributions.Categorical(action_probs)

            # Compute ratios and advantages
            new_log_probs = dist.log_prob(actions.squeeze())
            ratios = torch.exp(new_log_probs - log_probs)

            advantages = discounted_rewards - state_values.squeeze().detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Compute PPO losses
            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()

            critic_loss = F.mse_loss(state_values.squeeze(), discounted_rewards)

            entropy_loss = dist.entropy().mean()
            loss = actor_loss + 0.2 * critic_loss - 0.04 * entropy_loss
            current_ep_loss.append(loss.item())

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return np.mean(current_ep_loss)
    
    def plot_stats(self, stats, episode_num):
        # Create subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Plot Losses
        axs[0].plot(range(len(stats["Loss"])), stats["Loss"], label="Loss", color="blue", linewidth=1.5)
        axs[0].set_title("Loss Over Time")
        axs[0].set_xlabel("Training Step")
        axs[0].set_ylabel("Loss")
        axs[0].legend()

        # Plot Returns
        axs[1].plot(range(len(stats["Returns"])), stats["Returns"], label="Returns", color="green", linewidth=1.5)
        axs[1].set_title("Returns Over Episodes")
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Returns")
        axs[1].legend()

        # Adjust layout
        plt.tight_layout()

        # Save figure
        output_path = f"/Users/amiraliaali/Documents/Coding/RL/cross_street/training_output/loss_return_after_{episode_num}.png"
        fig.savefig(output_path)
        plt.close(fig)

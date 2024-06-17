import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


device = torch.device( "cpu")


class Policy_Network(nn.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_space_dims, 21)
        self.fc2 = nn.Linear(21, 50)
        self.mean = nn.Linear(50, action_space_dims)
        self.log_std = nn.Parameter(torch.zeros(action_space_dims))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        return mean, self.log_std.exp()


class Agent:
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        self.policy_network = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.0003)
        self.gamma = 0.99

    def sample_action(self, state: np.ndarray) -> tuple[float, torch.Tensor]:
        state = torch.tensor(state, dtype=torch.float32)
        mean, std = self.policy_network(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def update(self, rewards, log_probs):
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() +0.00001)
        
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
    def save_model(self, path):
        torch.save(self.policy_network.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        self.policy_network.load_state_dict(torch.load(path, map_location=device))
        self.policy_network.to(device)
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    env = gym.make("InvertedPendulum-v4", render_mode="human")
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)
    
    
    
    rewards_per_episode = []
    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.shape[0]
    agent = Agent(obs_space_dims, action_space_dims)
    agent.load_model("problem2_first.pth")
    
    total_num_episodes = 5000
    
    for episode in range(total_num_episodes):
        obs, info = wrapped_env.reset()
        done = False
        rewards = []
        log_probs = []
        
        while not done:
            action, log_prob = agent.sample_action(obs)
            obs, reward, terminated, truncated, _ = wrapped_env.step([action])
            done = terminated or truncated
            rewards.append(reward)
            log_probs.append(log_prob)
        
        rewards_per_episode.append(sum(rewards))
        agent.update(rewards, log_probs)
        
       
        avg_reward = np.mean(rewards_per_episode[-1:])
        print(f"Episode {episode}: Reward: {avg_reward:.2f}")
    # # 保存模型
    # agent.save_model("inve.pth")
    
    # 绘制奖励图
    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.show()
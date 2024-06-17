import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import time 


# policy_network_final_18_two.pth很成功

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyWrapper(gym.Wrapper):
    def __init__(self, env_name='InvertedPendulum-v4', render_mode='human'):
        env = gym.make(env_name, render_mode=render_mode)
        super(MyWrapper, self).__init__(env)
        self.env = env
        self.step_n = 0
        self.max_torque = 3.0
  
    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        # 设置初始状态为下垂位置
        state[1] = np.pi 
        self.env.state = state 
        self.env.unwrapped.state = state 
        self.step_n = 0
     
        return state, info
    
    def step(self, action):
        u = np.clip(action, -self.max_torque, self.max_torque)[0]
        state, reward, terminated, truncated, info = self.env.step([u]) 

        reward = 0  
        return state, reward, terminated, truncated, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()

class Policy_Network(nn.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_space_dims, 128)
        self.fc2 = nn.Linear(128, 64)
        self.mean = nn.Linear(64, action_space_dims)
        self.log_std = nn.Parameter(torch.zeros(action_space_dims))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std

class Agent:
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        self.policy_network = Policy_Network(obs_space_dims, action_space_dims).to(device)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=3e-4)
    
    def sample_action(self, state: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        state = torch.tensor(state, dtype=torch.float32).to(device)
        mean, std = self.policy_network(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().cpu().numpy(), log_prob
    
    def update(self, rewards, log_probs):
     
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

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
        self.policy_network.load_state_dict(torch.load(path))
        self.policy_network.to(device)
        print(f"Model loaded from {path}")

if __name__ == "__main__":
    env = MyWrapper('InvertedPendulum-v4', render_mode="human")
    # env = gym.wrappers.RecordEpisodeStatistics(env, 200)
    # print("Max steps per episode:", env.spec.max_episode_steps)

    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.shape[0]
    agent = Agent(obs_space_dims, action_space_dims)
    agent.load_model("policy_network_final_21_two.pth")
    
    
    total_num_episodes = 1000
    max_steps_per_episode = 10000

    for episode in range(total_num_episodes):
        obs, info = env.reset(seed=episode)
       
        done = False
        
        rewards = []
        log_probs = []
        success = 0
        while not done:
            action, log_prob = agent.sample_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            over = False
            u = np.clip(action, -3, 3)[0]
            # print(obs[0])
            
            if abs(obs[1]) <0.2:
                success = 1
             
            #0.1*obs[2]**2 + 0.001 * u**2+
            
            # # 奖励函数
            # reward = - (state[1]**2 + 0.1 * state[3]**2 + 0.001 * u**2)
            # angle = state[1]
            # if abs(angle) < 0.1:
            #     reward += 1
                
            if success== 1 and abs(obs[1]) < 0.1:
                reward = 2
                # reward -=  ( 0.2* (obs[0]-2)**2)
                # reward = reward- ((obs[0]-2)**2 + 0.1 * obs[2]**2 + 0.001 * u**2)
                
            rewards.append(reward)
            log_probs.append(log_prob)
            
                
            # if abs(obs[0]) > 3.5:
            #     over = True
            
            terminated = False
            if success == 1:
                terminated = bool( (np.abs(obs[1]) > 0.2))
           
                
            done = len(rewards) >= max_steps_per_episode or over or terminated

        agent.update(rewards, log_probs)
      
        avg_reward = np.sum(rewards)
        print(f"Episode {episode}: Average Reward: {avg_reward:.2f}")

   
    # 保存模型
    agent.save_model("policy_network_final_22_two.pth")

    # 载入模型
    # agent.load_model("policy_network_final.pth")

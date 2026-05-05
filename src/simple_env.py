"""
Simple Chip Placement Environment for AI Agent Demo
Author: Johir Hossain
Course: CSC59866-E
Date: April 2026

This is a simplified environment where an agent places macros on a grid.
The goal is to minimize wirelength (reward = negative wirelength).

Version History:
- Week 2: Basic environment + random agent
- Week 3: Added wirelength reward + Q-learning
- Week 4: Added Deep Q-Network (DQN) with PyTorch
"""

import numpy as np
import random

# ============================================
# Environment Class (All Weeks)
# ============================================
class SimpleChipEnv:
    """
    A 2D grid environment for chip macro placement.
    
    State: grid_size x grid_size grid with 0=empty, 1=occupied
    Action: (x, y) coordinates to place next macro
    Reward: -wirelength (Manhattan distance between consecutive macros)
    """
    
    def __init__(self, grid_size=8, num_macros=5):
        self.grid_size = grid_size
        self.num_macros = num_macros
        self.reset()
    
    def calculate_wirelength(self, positions):
        """Calculate total Manhattan distance between consecutive macros."""
        if len(positions) < 2:
            return 0
        total = 0
        for i in range(len(positions) - 1):
            x1, y1 = positions[i]
            x2, y2 = positions[i + 1]
            total += abs(x1 - x2) + abs(y1 - y2)
        return total
    
    def reset(self):
        """Reset environment to initial state."""
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.macros_placed = []
        self.step_count = 0
        return self.grid.copy()
    
    def step(self, action):
        """
        Take action (place macro) and return next_state, reward, done.
        
        Args:
            action: tuple (x, y) coordinates OR integer action index
        
        Returns:
            next_state: updated grid
            reward: negative wirelength
            done: True if all macros placed
        """
        # Handle both tuple and integer actions
        if isinstance(action, int):
            x = action // self.grid_size
            y = action % self.grid_size
        else:
            x, y = action
        
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size and self.grid[x, y] == 0:
            self.grid[x, y] = 1
            self.macros_placed.append((x, y))
            self.step_count += 1
            
            wirelength = self.calculate_wirelength(self.macros_placed)
            reward = -wirelength
            
            done = len(self.macros_placed) >= self.num_macros
            return self.grid.copy(), reward, done
        else:
            return self.grid.copy(), -10, False
    
    def render(self):
        """Print current grid to console."""
        print("Current Grid:")
        for row in self.grid:
            print(' '.join(['X' if cell == 1 else '.' for cell in row]))
        print(f"Macros placed: {len(self.macros_placed)}/{self.num_macros}")
        if len(self.macros_placed) >= 2:
            print(f"Wirelength: {self.calculate_wirelength(self.macros_placed)}")
        print()


# ============================================
# Random Agent (Week 2)
# ============================================
class RandomAgent:
    """A random agent that places macros anywhere on the grid."""
    
    def __init__(self, env):
        self.env = env
    
    def get_action(self):
        """Return a random (x,y) position."""
        x = np.random.randint(0, self.env.grid_size)
        y = np.random.randint(0, self.env.grid_size)
        return (x, y)


# ============================================
# Q-Learning Agent (Week 3)
# ============================================
class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration."""
    
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.3):
        self.env = env
        self.lr = learning_rate      # How fast to learn
        self.gamma = discount_factor  # How much to care about future rewards
        self.epsilon = exploration_rate  # How often to try random actions
        self.q_table = {}
    
    def get_state_key(self):
        """Convert current macro positions to a hashable key."""
        return tuple(sorted(self.env.macros_placed))
    
    def get_action(self):
        """Choose action using epsilon-greedy strategy."""
        state_key = self.get_state_key()
        
        # Explore: take random action
        if random.random() < self.epsilon:
            x = random.randint(0, self.env.grid_size - 1)
            y = random.randint(0, self.env.grid_size - 1)
            return (x, y)
        
        # Exploit: take best known action
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        if len(self.q_table[state_key]) == 0:
            x = random.randint(0, self.env.grid_size - 1)
            y = random.randint(0, self.env.grid_size - 1)
            return (x, y)
        
        best_action = max(self.q_table[state_key], key=self.q_table[state_key].get)
        return best_action
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-table using Bellman equation."""
        state_key = tuple(sorted(state))
        next_state_key = tuple(sorted(next_state))
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action not in self.q_table[state_key]:
            self.q_table[state_key][action] = 0
        
        # Calculate max future Q-value
        max_future_q = 0
        if next_state_key in self.q_table and len(self.q_table[next_state_key]) > 0:
            max_future_q = max(self.q_table[next_state_key].values())
        
        # Bellman equation: Q(s,a) = Q(s,a) + lr * [r + gamma * maxQ(s') - Q(s,a)]
        old_q = self.q_table[state_key][action]
        new_q = old_q + self.lr * (reward + self.gamma * max_future_q - old_q)
        self.q_table[state_key][action] = new_q


# ============================================
# Deep Q-Network (DQN) Agent (Week 4)
# ============================================
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    """Neural network for DQN agent."""
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """Deep Q-Network agent with experience replay and target network."""
    
    def __init__(self, env, learning_rate=0.001, discount_factor=0.9, 
                 exploration_rate=1.0, exploration_decay=0.995, 
                 exploration_min=0.01, memory_size=2000, batch_size=32):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = exploration_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        
        self.input_dim = env.grid_size * env.grid_size
        self.output_dim = env.grid_size * env.grid_size
        
        self.policy_net = DQN(self.input_dim, self.output_dim)
        self.target_net = DQN(self.input_dim, self.output_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def get_state_tensor(self, grid):
        """Convert grid to tensor."""
        return torch.FloatTensor(grid.flatten()).unsqueeze(0)
    
    def get_action(self, grid):
        """Choose action using epsilon-greedy."""
        if random.random() < self.epsilon:
            x = random.randint(0, self.env.grid_size - 1)
            y = random.randint(0, self.env.grid_size - 1)
            return x * self.env.grid_size + y
        
        state_tensor = self.get_state_tensor(grid)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train on random batch from memory."""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in batch:
            state_tensor = self.get_state_tensor(state)
            next_state_tensor = self.get_state_tensor(next_state)
            
            current_q = self.policy_net(state_tensor)[0][action]
            
            if done:
                target_q = reward
            else:
                with torch.no_grad():
                    next_q = self.target_net(next_state_tensor).max()
                target_q = reward + self.gamma * next_q
            
            loss = nn.MSELoss()(current_q, torch.tensor(target_q))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Copy policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())


# ============================================
# Demo Code
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("AI Chip Design Agent - Complete Demo")
    print("Week 2: Random Agent | Week 3: Q-Learning | Week 4: DQN")
    print("=" * 60)
    
    # Test Random Agent
    print("\n--- Testing Random Agent ---")
    env = SimpleChipEnv(grid_size=6, num_macros=4)
    agent = RandomAgent(env)
    state = env.reset()
    total_reward = 0
    for step in range(4):
        action = agent.get_action()
        next_state, reward, done = env.step(action)
        total_reward += reward
        print(f"Step {step+1}: Placed at {action}, Reward: {reward}")
    print(f"Random Agent Total Reward: {total_reward}")
    
    # Test DQN Agent (short test)
    print("\n--- Testing DQN Agent (10 episodes) ---")
    env_dqn = SimpleChipEnv(grid_size=6, num_macros=4)
    dqn_agent = DQNAgent(env_dqn)
    rewards = []
    for ep in range(10):
        state = env_dqn.reset()
        total = 0
        done = False
        while not done:
            action = dqn_agent.get_action(state)
            next_state, reward, done = env_dqn.step(action)
            total += reward
            state = next_state
        rewards.append(total)
    print(f"DQN Average Reward (10 eps): {np.mean(rewards):.2f}")
    
    print("\n" + "=" * 60)
    print("Demo Complete! Check GitHub for full code.")
    print("=" * 60)

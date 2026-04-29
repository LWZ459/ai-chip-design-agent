"""
Simple Chip Placement Environment for AI Agent Demo
Author: Johir Hossain
Course: CSC59866-E
Date: April 2026

This is a simplified environment where an agent places macros on a grid.
The goal is to minimize wirelength (reward = negative wirelength).
"""

import numpy as np

class SimpleChipEnv:
    """
    A 2D grid environment for chip macro placement.
    
    State: 10x10 grid with 0=empty, 1=occupied
    Action: (x, y) coordinates to place next macro
    Reward: Negative wirelength (shorter wires = higher reward)
    """
    
    def __init__(self, grid_size=10, num_macros=5):
        self.grid_size = grid_size
        self.num_macros = num_macros
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state."""
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.macros_placed = []
        self.step_count = 0
        return self.grid
    
    def calculate_wirelength(self, positions):
        """
        Calculate total Manhattan distance between consecutive macros.
        Manhattan distance = |x1 - x2| + |y1 - y2|
        """
        if len(positions) < 2:
            return 0
        
        total = 0
        for i in range(len(positions) - 1):
            x1, y1 = positions[i]
            x2, y2 = positions[i + 1]
            distance = abs(x1 - x2) + abs(y1 - y2)
            total += distance
        
        return total
    
    def step(self, action):
        """
        Take an action (place macro at position) and return next state, reward, done.
        
        Args:
            action: tuple (x, y) coordinates
        
        Returns:
            next_state: updated grid
            reward: negative wirelength (shorter = better)
            done: True if all macros placed
        """
        x, y = action
        
        # Check if position is valid and empty
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size and self.grid[x, y] == 0:
            self.grid[x, y] = 1
            self.macros_placed.append((x, y))
            self.step_count += 1
            
            # Calculate wirelength reward
            wirelength = self.calculate_wirelength(self.macros_placed)
            reward = -wirelength  # Shorter wirelength = higher reward (less negative)
            
            # Check if done (all macros placed)
            done = len(self.macros_placed) >= self.num_macros
            
            return self.grid, reward, done
        else:
            # Invalid placement: large penalty
            return self.grid, -10, False
    
    def render(self):
        """Print the current grid to console."""
        print("Current Grid:")
        for row in self.grid:
            print(' '.join(['X' if cell == 1 else '.' for cell in row]))
        print(f"Macros placed: {len(self.macros_placed)}/{self.num_macros}")
        if len(self.macros_placed) >= 2:
            print(f"Wirelength: {self.calculate_wirelength(self.macros_placed)}")
        print()


class RandomAgent:
    """A random agent that places macros anywhere on the grid."""
    
    def __init__(self, env):
        self.env = env
    
    def get_action(self):
        """Return a random valid (x,y) position."""
        x = np.random.randint(0, self.env.grid_size)
        y = np.random.randint(0, self.env.grid_size)
        return (x, y)


# Demo code
if __name__ == "__main__":
    print("=" * 50)
    print("AI Chip Design Agent - Prototype Demo with Wirelength Reward")
    print("=" * 50)
    
    # Create environment
    env = SimpleChipEnv(grid_size=8, num_macros=5)
    agent = RandomAgent(env)
    
    # Run one episode
    state = env.reset()
    env.render()
    
    total_reward = 0
    step = 0
    
    while True:
        action = agent.get_action()
        next_state, reward, done = env.step(action)
        total_reward += reward
        step += 1
        
        print(f"Step {step}: Placed macro at {action}")
        print(f"Reward for this step: {reward}")
        
        if done:
            break
    
    print(f"\n{'='*50}")
    print(f"Episode Complete!")
    print(f"Total steps: {step}")
    print(f"Total reward: {total_reward}")
    print(f"Positions placed: {env.macros_placed}")
    print(f"Final wirelength: {env.calculate_wirelength(env.macros_placed)}")
    print("=" * 50)

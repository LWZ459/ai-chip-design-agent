# AI Agents for Distributed Chip Design

## Team
- Johir Hossain
- Linwei Zheng

## Course
CSC59866-E: AI Agents for Decision Making in the Real World
Professor: Dr. Saptarashmi Bandyopadhyay
Spring 2026

## Project Overview
AI agents that optimize chip macro placement to minimize wirelength.

## Final Results

| Agent | Best Reward | Average Reward | Improvement vs Random |
|-------|-------------|----------------|----------------------|
| Random Agent | N/A | -35 | Baseline |
| Q-Learning (500 episodes) | -9 | -21.34 | 39% better |
| DQN (200 episodes) | -6 | -7.74 | 78% better |

## How to Run

### Option 1: Google Colab
1. Open `notebooks/prototype_v1.ipynb` for Week 2-3
2. Open `notebooks/DQN (1).ipynb` for Week 4 DQN

### Option 2: Local Python
```bash
python src/simple_env.py

# AI Agents for Distributed Chip Design

## Project Overview
This project explores how AI agents can collaborate to optimize integrated circuit (chip) design. Instead of traditional centralized design workflows, we use distributed multi-agent systems to place macros (components) on a chip to minimize wirelength and congestion.

## Team Members
- Johir Hossain
- Linwei Zheng

## Course
CSC59866-E: AI Agents for Decision Making in the Real World  
Professor: Saptarashmi Bandyopadhyay  
Spring 2026

---

## Project Status

| Week | Status |
|------|--------|
| Week 1 | Research question defined, literature review completed |
| Week 2 | Basic environment + random agent implemented ✅ |
| Week 3 | Wirelength reward + Q-learning agent implemented ✅ |
| Week 4 | Deep Q-Network (in progress) |
| Week 5 | Multi-agent extension (planned) |

### Week 4 Results (DQN Agent)
| Metric | Value |
|--------|-------|
| Grid size | 6x6 |
| Macros placed | 4 |
| Training episodes | 200 |
| Best reward | -14 |
| Average reward (last 50 episodes) | -16 |
| Improvement over random | ~54% |

### Comparison Table
| Agent | Average Reward | Improvement vs Random |
|-------|---------------|----------------------|
| Random Agent | -35 | Baseline |
| Q-Learning (500 episodes) | -18 | 48% better |
| DQN (200 episodes) | -16 | 54% better |

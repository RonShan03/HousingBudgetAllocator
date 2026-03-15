# HousingBudgetAllocator
Reinforcement Learning for Dynamic Housing Budget Allocation

A research project exploring how Deep Reinforcement Learning (DRL) can optimize government housing budget allocation to reduce eviction rates and improve housing stability.

Inspired by the FinRL framework for financial portfolio management, this project investigates whether reinforcement learning techniques used in algorithmic trading can be adapted to support data-driven public policy decisions in housing assistance programs.

Overview

Automated decision systems powered by reinforcement learning have successfully been applied to financial portfolio optimization, robotics, and resource allocation problems. Financial RL frameworks such as FinRL treat asset allocation as a sequential decision-making problem where an agent dynamically allocates capital to maximize long-term returns.

This project applies the same idea to housing policy.

Instead of allocating capital across financial assets, an RL agent allocates a government housing budget across housing initiatives with the objective of improving housing stability and minimizing eviction rates.

The agent learns from historical housing and economic data to determine how budget allocation decisions impact eviction outcomes over time.

Project Objectives

Adapt reinforcement learning methods designed for financial portfolio optimization to housing policy allocation

Build a simulation environment representing economic and housing conditions

Train RL agents to dynamically allocate housing assistance budgets

Evaluate whether learned policies can reduce eviction rates relative to baseline allocation strategies

Problem Formulation

The problem is modeled as a Markov Decision Process (MDP).

State Space

The environment state represents the economic and housing conditions of the community.

Example features include:

Eviction rate

Median rental cost

Consumer Price Index (CPI)

Government housing assistance budget

Unemployment rate

Cost of living indicators

These variables capture factors affecting housing instability.

Action Space

At each timestep, the RL agent allocates portions of the available housing budget across multiple housing initiatives.

Example initiatives:

Rental subsidies

Eviction prevention programs

Affordable housing construction

Emergency housing support

Actions represent percentage allocations of the total housing budget, similar to portfolio weights in financial RL.

Reward Function

The reward function measures housing stability outcomes.

Primary objective:

Minimize eviction rates

Example reward formulation:

reward = -(eviction_rate_next_period)

This encourages the agent to discover allocation strategies that reduce housing instability over time.

Data Sources

The project uses publicly available datasets describing housing markets, economic conditions, and eviction statistics.

Data sources include:

U.S. Department of Housing and Urban Development (HUD)

U.S. Census Bureau

NYC Open Data Housing Datasets

NYC Housing Preservation and Development

New York City Council eviction records

City capital budget datasets

Housing cost and affordability metrics

These datasets provide historical signals used to construct the RL environment state.

Methodology

The project adapts the FinRL architecture to create a custom reinforcement learning environment for housing policy simulation.

Key components:

Data preprocessing and feature engineering

Construction of a custom RL environment

Training RL agents using policy optimization algorithms

Evaluation against baseline allocation strategies

The following RL algorithms are explored:

Deep Q Networks (DQN)

Proximal Policy Optimization (PPO)

System Architecture
Housing & Economic Data
        │
        ▼
Data Processing Pipeline
        │
        ▼
State Representation
        │
        ▼
Custom RL Environment
        │
        ▼
RL Agent (PPO / DQN)
        │
        ▼
Budget Allocation Policy
        │
        ▼
Eviction Outcome Simulation
Repository Structure
housing-rl-budget-allocation/

data/
    eviction_data.csv
    housing_costs.csv
    economic_indicators.csv

env/
    housing_env.py

models/
    trained_agents/

scripts/
    train_agent.py
    evaluate_agent.py

notebooks/
    exploratory_analysis.ipynb
    training_experiments.ipynb

results/
    policy_visualizations
    evaluation_metrics

README.md
requirements.txt
Installation

Clone the repository:

git clone https://github.com/yourusername/housing-rl-budget-allocation.git
cd housing-rl-budget-allocation

Install dependencies:

pip install -r requirements.txt

Key libraries include:

Python

PyTorch

Stable-Baselines3

Gymnasium

Pandas

NumPy

Matplotlib

Training the RL Agent

To train the reinforcement learning agent:

python scripts/train_agent.py

Training involves:

constructing the environment

loading housing datasets

training an RL policy using PPO or DQN

Evaluating the Agent

Evaluation compares the RL policy against baseline allocation strategies such as:

equal budget distribution

fixed policy allocation

Run evaluation:

python scripts/evaluate_agent.py

Metrics include:

eviction rate over time

budget efficiency

policy stability

Expected Outcomes

The trained RL agent should learn allocation strategies that:

adapt to economic changes

prioritize housing interventions that reduce eviction risk

outperform static budget allocation strategies

Computational Resources

Large-scale data processing and RL training are supported by:

Wulver, NJIT's high-performance computing cluster.

This infrastructure allows efficient experimentation with large datasets and RL models.

Limitations

Several challenges remain:

Housing datasets may contain measurement bias

Government program definitions change over time

Simulating policy outcomes is inherently complex

Future work may incorporate:

causal inference methods

additional outcome metrics beyond eviction rates

richer simulation environments

References

Liu, X. et al.
FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance
NeurIPS 2020 Deep RL Workshop

Authors

Rohan Shanbhag
Michael Mittleman

#!/usr/bin/env python3
"""
Housing Allocation Environment for FinRL Housing project.

Gym environment simulating housing budget allocation decisions.
"""

import gym
import numpy as np
import pandas as pd
from gym import spaces
from typing import Dict, List, Tuple, Any
import os

class HousingAllocationEnv(gym.Env):
    """
    Gym environment for housing budget allocation RL.

    State space: current housing metrics + budget status
    Action space: allocation percentages across housing programs
    Reward: reduction in eviction rates + equity considerations
    """

    def __init__(self,
                 data_path='data/processed/housing_economic_data_processed.csv',
                 initial_budget=1000000,  # $1M annual budget
                 n_programs=4,
                 program_names=None,
                 transaction_cost=0.05,  # 5% administrative cost
                 max_steps=12):  # Monthly decisions

        super(HousingAllocationEnv, self).__init__()

        # Load data
        self.data_path = data_path
        self.load_data()

        # Environment parameters
        self.initial_budget = initial_budget
        self.n_programs = n_programs
        self.program_names = program_names or [
            'rental_subsidies',
            'eviction_prevention',
            'affordable_housing',
            'emergency_support'
        ]
        self.transaction_cost = transaction_cost
        self.max_steps = max_steps

        # State space dimensions
        self.n_features = len(self.feature_cols) + 1  # +1 for current budget
        self.state_dim = self.n_features

        # Define spaces
        # State: [eviction_rate, rent, cpi, unemployment, affordability, displacement_risk, equity, budget]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # Action: allocation percentages for each program (must sum to 1)
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_programs,),
            dtype=np.float32
        )

        # Initialize episode
        self.reset()

    def load_data(self):
        """Load processed housing data."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        self.feature_cols = [
            'eviction_rate', 'median_gross_rent', 'cpi',
            'unemployment_rate', 'affordability_index',
            'displacement_risk', 'equity_index'
        ]

        # Normalize features
        self.feature_scaler = StandardScaler()
        self.df[self.feature_cols] = self.feature_scaler.fit_transform(self.df[self.feature_cols])

        # Convert to array
        self.feature_array = self.df[self.feature_cols].values
        self.n_timesteps = len(self.df)

        print(f"Loaded {self.n_timesteps} timesteps of housing data")

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = 0
        self.current_budget = self.initial_budget
        self.total_budget_used = 0

        # Start with first timestep data
        state_features = self.feature_array[0]
        self.current_state = np.concatenate([state_features, [self.current_budget / self.initial_budget]])

        return self.current_state.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Allocation percentages for each program (shape: n_programs,)

        Returns:
            next_state, reward, done, info
        """
        # Validate action
        action = np.clip(action, 0, 1)
        action_sum = np.sum(action)
        if action_sum > 0:
            action = action / action_sum  # Normalize to sum to 1
        else:
            action = np.ones(self.n_programs) / self.n_programs  # Equal allocation if all zero

        # Calculate budget allocation
        allocation_amounts = action * self.current_budget
        admin_cost = allocation_amounts * self.transaction_cost
        net_allocation = allocation_amounts - admin_cost

        # Update budget
        self.total_budget_used += np.sum(allocation_amounts)
        self.current_budget -= np.sum(allocation_amounts)

        # Simulate impact on housing metrics
        next_step = min(self.current_step + 1, self.n_timesteps - 1)
        next_features = self.feature_array[next_step].copy()

        # Calculate reward based on program effectiveness
        reward = self.calculate_reward(action, net_allocation, next_features)

        # Update state
        self.current_step = next_step
        budget_ratio = self.current_budget / self.initial_budget
        self.current_state = np.concatenate([next_features, [budget_ratio]])

        # Check if episode done
        done = self.current_step >= self.max_steps - 1 or self.current_budget <= 0

        # Info dict
        info = {
            'allocations': dict(zip(self.program_names, allocation_amounts)),
            'net_allocations': dict(zip(self.program_names, net_allocation)),
            'budget_remaining': self.current_budget,
            'step': self.current_step
        }

        return self.current_state.astype(np.float32), reward, done, info

    def calculate_reward(self, action: np.ndarray, net_allocation: np.ndarray,
                        next_features: np.ndarray) -> float:
        """
        Calculate reward based on housing outcomes.

        Reward components:
        - Eviction reduction (40%)
        - Affordability improvement (20%)
        - Equity improvement (20%)
        - Efficiency (cost per outcome) (10%)
        - Budget utilization penalty (10%)
        """
        # Program effectiveness weights (hypothetical)
        effectiveness = {
            'rental_subsidies': [0.8, 0.9, 0.3, 0.7],      # [eviction, affordability, equity, efficiency]
            'eviction_prevention': [0.9, 0.4, 0.8, 0.8],
            'affordable_housing': [0.6, 0.8, 0.6, 0.5],
            'emergency_support': [0.7, 0.5, 0.9, 0.6]
        }

        # Calculate weighted outcomes
        eviction_impact = 0
        affordability_impact = 0
        equity_impact = 0
        efficiency_score = 0

        for i, program in enumerate(self.program_names):
            eff = effectiveness[program]
            alloc_ratio = net_allocation[i] / self.initial_budget

            eviction_impact += eff[0] * alloc_ratio
            affordability_impact += eff[1] * alloc_ratio
            equity_impact += eff[2] * alloc_ratio
            efficiency_score += eff[3] * alloc_ratio

        # Composite reward
        reward = (
            0.4 * eviction_impact +           # Maximize eviction prevention
            0.2 * affordability_impact +      # Improve affordability
            0.2 * equity_impact +             # Promote equity
            0.1 * efficiency_score -          # Efficiency bonus
            0.1 * abs(1 - self.total_budget_used / self.initial_budget)  # Budget utilization penalty
        )

        return reward

    def render(self, mode='human'):
        """Render environment state."""
        print(f"Step: {self.current_step}")
        print(f"Budget remaining: ${self.current_budget:,.0f}")
        print(f"Current eviction rate: {self.current_state[0]:.3f}")
        print(f"Affordability index: {self.current_state[4]:.3f}")

    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        if seed is not None:
            np.random.seed(seed)
        return [seed]

# Import here to avoid circular imports
from sklearn.preprocessing import StandardScaler
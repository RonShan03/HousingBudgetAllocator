#!/usr/bin/env python3
"""
Generate presentation-ready visualizations and summary tables for the HousingBudgetAllocator agent.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import DDPG
from finrl_housing.meta.env_housing.env_allocation import HousingAllocationEnv
from scripts.train_agent import evaluate_agent, evaluate_baseline


def ensure_results_dir(path='results/demo'):
    os.makedirs(path, exist_ok=True)
    return path


def plot_data_trends(processed_csv='data/processed/housing_economic_data_processed.csv', out_dir='results/demo'):
    df = pd.read_csv(processed_csv)

    plt.style.use('default')

    fig, ax = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

    ax[0].plot(df['year'], df['eviction_rate'], marker='o', color='tab:red')
    ax[0].set_title('Eviction Rate Over Time')
    ax[0].set_ylabel('Eviction Rate (per 1000)')

    ax[1].plot(df['year'], df['median_gross_rent'], marker='o', color='tab:blue', label='Median Gross Rent')
    ax[1].plot(df['year'], df['median_household_income'], marker='o', color='tab:green', label='Median Household Income')
    ax[1].set_title('Rent and Income Trends')
    ax[1].set_ylabel('USD')
    ax[1].legend(loc='best')

    ax[2].plot(df['year'], df['cpi'], marker='o', color='tab:purple')
    ax[2].set_title('CPI Over Time (Synthetic/BLS trend)')
    ax[2].set_ylabel('CPI')
    ax[2].set_xlabel('Year')

    plt.tight_layout()
    path = os.path.join(out_dir, 'data_trends.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def plot_agent_vs_baseline(model_path='models/ddpg_housing_final.zip', out_dir='results/demo'):
    env = HousingAllocationEnv()

    # Evaluate trained agent
    rewards, _ = evaluate_agent(model_path, n_episodes=10)

    # Evaluate baseline
    baseline_rewards = evaluate_baseline(env, n_episodes=10)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(['Agent', 'Baseline'],
                   [np.mean(rewards), np.mean(baseline_rewards)],
                   yerr=[np.std(rewards), np.std(baseline_rewards)],
                   capsize=8, alpha=0.8)

    ax.set_title('Agent vs Baseline Average Reward')
    ax.set_ylabel('Average Episode Reward')
    ax.set_ylim(min(np.mean(rewards), np.mean(baseline_rewards)) - 0.5,
                max(np.mean(rewards), np.mean(baseline_rewards)) + 0.5)

    for bar, value in zip(bars, [np.mean(rewards), np.mean(baseline_rewards)]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f'{value:.3f}',
                ha='center', va='bottom')

    path = os.path.join(out_dir, 'agent_vs_baseline_demo.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def show_allocation_trajectory(model_path='models/ddpg_housing_final.zip', out_dir='results/demo', n_steps=12):
    model = DDPG.load(model_path)
    env = HousingAllocationEnv()

    obs, _ = env.reset()
    actions = []
    eviction_rates = [env.current_state[0]]  # first state eviction rate
    budget = [env.current_budget]

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action.flatten())
        obs, reward, terminated, truncated, info = env.step(action)
        eviction_rates.append(obs[0])
        budget.append(env.current_budget)
        if terminated or truncated:
            break

    actions = np.array(actions)
    years = list(range(len(actions)))

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    axs[0].plot(years, eviction_rates[:-1], marker='o')
    axs[0].set_title('Agent-driven Eviction Rate Trajectory by Step')
    axs[0].set_ylabel('Eviction Rate')

    axs[1].plot(years, budget[:-1], marker='o', color='tab:orange')
    axs[1].set_title('Remaining Budget Trajectory')
    axs[1].set_ylabel('Remaining Budget')

    for i in range(actions.shape[1]):
        axs[2].plot(years, actions[:, i], marker='o', label=f'Program {i + 1}')
    axs[2].set_title('Action Allocation Ratios by Step')
    axs[2].set_ylabel('Allocation Fraction')
    axs[2].set_xlabel('Timestep')
    axs[2].legend(loc='best')

    plt.tight_layout()
    path = os.path.join(out_dir, 'agent_allocation_trajectory.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def main():
    out_dir = ensure_results_dir('results/demo')
    plot_data_trends(out_dir=out_dir)
    plot_agent_vs_baseline(out_dir=out_dir)
    show_allocation_trajectory(out_dir=out_dir)

    print('\nDemo assets prepared in results/demo/')


if __name__ == '__main__':
    main()

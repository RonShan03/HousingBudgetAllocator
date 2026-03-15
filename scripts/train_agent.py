#!/usr/bin/env python3
"""
Training script for Housing Budget Allocation RL agent.

Uses PPO from stable-baselines3 to train the housing allocation policy.
"""

import os
import sys
sys.path.append(os.path.dirname(__file__) + '/..')  # Add project root to path
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.results_plotter import plot_results
import matplotlib.pyplot as plt

from finrl_housing.meta.env_housing.env_allocation import HousingAllocationEnv

class TrainingCallback(BaseCallback):
    """Callback for logging training progress."""

    def __init__(self, check_freq: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Save model
            self.model.save(os.path.join(self.save_path, f"ddpg_housing_{self.n_calls}"))
            print(f"Saved model at step {self.n_calls}")
        return True

def train_agent(total_timesteps=10000, model_save_path='models/'):
    """Train the PPO agent on the housing allocation environment."""

    # Create directories
    os.makedirs(model_save_path, exist_ok=True)
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Create environment
    env = HousingAllocationEnv()

    # Create DDPG agent
    model = DDPG(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=100,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "episode"),
        gradient_steps=1,
        action_noise=None,
        replay_buffer_class=None,
        replay_buffer_kwargs=None,
        optimize_memory_usage=False,
        tensorboard_log=log_dir,
        policy_kwargs=None,
        verbose=1,
        seed=42,
        device="auto"
    )

    # Training callback
    callback = TrainingCallback(check_freq=1000, save_path=model_save_path)

    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    # Save final model
    final_model_path = os.path.join(model_save_path, "ddpg_housing_final")
    model.save(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")

    return model

def evaluate_agent(model_path, n_episodes=10):
    """Evaluate trained agent performance."""

    # Load model
    model = DDPG.load(model_path)

    # Create environment
    env = HousingAllocationEnv()

    rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1

        rewards.append(episode_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.3f}, Steps = {steps}")

    print("\nEvaluation Results:")
    print(f"Average Reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f}")

    return rewards, episode_lengths

def plot_training_results(log_dir="logs/"):
    """Plot training results."""
    plot_results([log_dir], num_timesteps=None, x_axis="timesteps", task_name="Housing Allocation")
    plt.show()

if __name__ == "__main__":
    # Train agent
    model = train_agent(total_timesteps=5000)  # Small training for demo

    # Evaluate
    model_path = "models/ddpg_housing_final.zip"
    if os.path.exists(model_path):
        evaluate_agent(model_path, n_episodes=5)

    print("Training and evaluation complete!")
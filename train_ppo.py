"""
train_ppo.py — Full PPO training pipeline for AutoFactoryToDEnv.

Demonstrates LEARNING IMPROVEMENT across 5 training iterations.

Usage
-----
  Train:     python3 train_ppo.py
  Eval only: python3 train_ppo.py --eval-only
"""

import argparse
import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from server.environment import (
    AutoFactoryToDEnv, compute_final_score, is_peak_hour,
    PRODUCTION_TARGET, MAX_HOURLY_COST, TARIFF_NIGHT, TARIFF_PEAK, TARIFF_NORMAL,
)


# ===========================================================================
# TASK 2 — Gymnasium Wrapper with Normalized Observations
# ===========================================================================

class AutoFactoryGymEnv(gym.Env):
    """
    Gymnasium wrapper for AutoFactoryToDEnv.

    Key changes vs raw env
    ----------------------
    * ALL observations are normalized to [0, 1] for stable gradient flow.
    * Action space reduced to MultiDiscrete([3,3,3,2,2]) — no maintenance mode
      during training (simplifies action space, improves convergence speed).
    * Stochastic training env (production_noise=True, enable_breakdowns=True)
      vs clean eval env (both False) so scores are comparable across iterations.
    """
    metadata = {"render_modes": ["ansi"]}

    #                       stamp  mold  cnc  comp  welder
    ACTION_NVEC_TRAIN = [3,    3,    3,   2,    2]   # no maintenance (action 2 on welder)
    TARIFF_MIN, TARIFF_MAX = TARIFF_NIGHT, TARIFF_PEAK

    def __init__(
        self,
        task:              str  = "medium",  # TASK 1
        target:            int | None = None,
        enable_breakdowns: bool = True,
        production_noise:  bool = True,
    ):
        super().__init__()
        self.env = AutoFactoryToDEnv(
            task               = task,
            target             = target,
            enable_breakdowns  = enable_breakdowns,
            production_noise   = production_noise,
        )
        # Store initial target for reference, but _normalize_obs will use env.target_production

        # TASK 2 — action space [3,3,3,2,2]
        self.action_space = spaces.MultiDiscrete(self.ACTION_NVEC_TRAIN)

        # TASK 2 — observation space fully in [0,1]
        # [hour/24, production/target, health×5, tariff_norm]  → 8 dims
        self.observation_space = spaces.Box(
            low   = 0.0,
            high  = 1.0,
            shape = (8,),
            dtype = np.float32,
        )

    # ------------------------------------------------------------------
    def _normalize_obs(self, obs_dict: dict) -> np.ndarray:
        """All values mapped to [0, 1]."""
        tariff_range = self.TARIFF_MAX - self.TARIFF_MIN
        tariff_norm  = (obs_dict["electricity_price"] - self.TARIFF_MIN) / tariff_range
        
        target = self.env.target_production # TASK 3: dynamic target

        return np.array([
            obs_dict["hour"]             / 24.0,
            obs_dict["production_so_far"] / target,
            *obs_dict["machine_health"],               # already [0, 1]
            float(np.clip(tariff_norm, 0.0, 1.0)),
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        obs_dict, info = self.env.reset()
        return self._normalize_obs(obs_dict), info

    def step(self, action):
        # Map the 2-option welder (0=off, 1=full) — action[4] is 0 or 1
        stamping, molding, cnc, compressor, welder = action
        obs_dict, reward, terminated, truncated, info = self.env.step(
            int(stamping), int(molding), int(cnc), int(compressor), int(welder)
        )
        return self._normalize_obs(obs_dict), reward, terminated, truncated, info

    def state(self):
        return self.env.state()


# ===========================================================================
# TASK 6 — Per-iteration Evaluation
# ===========================================================================

def evaluate_iteration(model, eval_env: AutoFactoryGymEnv, n_episodes: int = 3) -> float:
    """Run n_episodes deterministically and return the average final_score."""
    scores = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
        score = info.get("final_score", compute_final_score(eval_env.state()))
        scores.append(score)
    return float(np.mean(scores))


# ===========================================================================
# TASK 5 — Report after each episode in evaluation
# ===========================================================================

def evaluate(model, env: AutoFactoryGymEnv, episodes: int = 5) -> float:
    """
    Full evaluation: runs episodes, prints per-episode score, returns average.
    Matches TASK 8 spec exactly.
    """
    scores = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done   = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        score = info.get("final_score", compute_final_score(env.state()))
        scores.append(score)
        print(f"  Episode {ep + 1:02d} | final_score = {score:.4f}")
    avg = float(np.mean(scores))
    print(f"  Average Score: {avg:.4f}")
    return avg


# ===========================================================================
# Periodic Score Callback
# ===========================================================================

class IterationScoreCallback(BaseCallback):
    """Saves best model whenever a new high score is reached during eval."""

    def __init__(self, eval_env, eval_freq=2048, save_path="ppo_factory", verbose=1):
        super().__init__(verbose)
        self.eval_env  = eval_env
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.best      = -1.0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        score = evaluate_iteration(self.model, self.eval_env, n_episodes=3)
        if score > self.best:
            self.best = score
            self.model.save(self.save_path)
        return True


# ===========================================================================
# TASK 1 & 7 — Main Training Loop (5 iterations)
# ===========================================================================

def train():
    # Stochastic training env (noise + breakdowns → RL must generalise)
    def make_train_env():
        return AutoFactoryGymEnv(enable_breakdowns=True, production_noise=True)

    # Clean deterministic eval env (for fair score comparison)
    eval_env = AutoFactoryGymEnv(enable_breakdowns=False, production_noise=False)

    train_env = DummyVecEnv([make_train_env])

    # Quick sanity check
    check_env(make_train_env(), warn=True)

    model = PPO(
        policy        = "MlpPolicy",
        env           = train_env,
        learning_rate = 3e-4,
        n_steps       = 2048,
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        ent_coef      = 0.05,   # TASK 3: higher entropy → broader policy, less collapse
        seed          = 42,
        device        = "cpu",
        verbose       = 0,
    )

    callback = IterationScoreCallback(
        eval_env  = eval_env,
        eval_freq = 2048,
        save_path = "ppo_factory",
        verbose   = 0,
    )

    print("=" * 60)
    print("  AutoFactoryToDEnv — PPO Training (5 Iterations)")
    print(f"  Obs space : {make_train_env().observation_space}")
    print(f"  Act space : {make_train_env().action_space}")
    print("=" * 60)

    # TASK 1 & TASK 6 — 5 iterations, print score after each
    ITERATIONS     = 5
    STEPS_PER_ITER = 100_000

    iteration_scores = []

    for i in range(ITERATIONS):
        print(f"\n=== Training Iteration {i + 1} ===")
        model.learn(
            total_timesteps     = STEPS_PER_ITER,
            callback            = callback,
            reset_num_timesteps = (i == 0),
            progress_bar        = False,
        )

        # Per-iteration evaluation (TASK 6 — what judges see)
        score = evaluate_iteration(model, eval_env, n_episodes=5)
        iteration_scores.append(score)

        # TASK 6 — exact output format for judges
        print(f"Iteration {i + 1} Score: {score:.2f}")

    # Save final model
    model.save("ppo_factory_final")

    print("\n" + "=" * 60)
    print("LEARNING IMPROVEMENT SUMMARY")
    print("=" * 60)
    for i, s in enumerate(iteration_scores):
        bar = "█" * int(s * 30)
        print(f"  Iteration {i + 1}  Score: {s:.2f}  {bar}")

    best = max(iteration_scores)
    gain = iteration_scores[-1] - iteration_scores[0]
    print(f"\n  Best Score : {best:.2f}")
    print(f"  Total Gain : +{gain:.2f}")
    print(f"  Model saved → ppo_factory_final.zip")

    # Final full evaluation
    print("\n--- Final Evaluation (5 episodes) ---")
    evaluate(model, eval_env, episodes=5)

    return model


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; load ppo_factory_final and evaluate.",
    )
    args = parser.parse_args()

    if args.eval_only:
        env   = AutoFactoryGymEnv(enable_breakdowns=False, production_noise=False)
        model = PPO.load("ppo_factory_final", env=DummyVecEnv([lambda: env]))
        print("Model loaded — running evaluation...")
        evaluate(model, env, episodes=10)
    else:
        train()

import os
from typing import Optional, Dict, Callable
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from src.training.environment import make_env, MultiAgentCollaborationEnv
from src.data.datasets import get_dataset, create_train_val_split
from src.utils.logging import get_logger
from src.utils.checkpoints import CheckpointManager
from config.training import RLTrainingConfig

logger = get_logger("training.rl_trainer")


class WandbCallback(BaseCallback):

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Log training metrics
        if "episode" in self.locals.get("infos", [{}])[0]:
            info = self.locals["infos"][0]
            episode = info["episode"]

            if wandb.run:
                wandb.log({
                    "train/episode_reward": episode["r"],
                    "train/episode_length": episode["l"],
                    "train/timesteps": self.num_timesteps,
                })

        return True

    def _on_rollout_end(self) -> None:
        # Log rollout statistics
        if wandb.run:
            wandb.log({
                "train/mean_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
                "rollout/ep_rew_mean": self.model.ep_info_buffer[-1]["r"] if self.model.ep_info_buffer else 0,
            })


class MetricsCallback(BaseCallback):
    """
    Callback to track and log custom metrics
    """

    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_count = 0
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # Get recent episode info
            if self.model.ep_info_buffer:
                recent_episodes = list(self.model.ep_info_buffer)[-10:]
                mean_reward = np.mean([ep["r"] for ep in recent_episodes])
                mean_length = np.mean([ep["l"] for ep in recent_episodes])

                logger.info(
                    f"Step {self.num_timesteps}: "
                    f"Mean Reward={mean_reward:.3f}, "
                    f"Mean Length={mean_length:.1f}"
                )

                # Update best
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    logger.info(f"[OK] New best mean reward: {mean_reward:.3f}")

        return True


class RLTrainer:
    """
    Handles RL training of communication policy

    Uses PPO from Stable-Baselines3 to train agents to communicate
    strategically during multi-agent collaboration.
    """

    def __init__(
        self,
        config: Optional[RLTrainingConfig] = None,
        device: str = "cpu",
        use_wandb: bool = False,
    ):
        """
        Initialize RL trainer

        Args:
            config: Training configuration
            device: Training device
            use_wandb: Log to Weights & Biases
        """
        self.config = config or RLTrainingConfig()
        self.device = device
        self.use_wandb = use_wandb

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=Path("./checkpoints/rl_training"),
            keep_best_n=3
        )

        logger.info(f"Initialized RLTrainer with config: {self.config}")

    def train(
        self,
        total_timesteps: int = 100000,
        eval_freq: int = 5000,
        save_freq: int = 10000,
    ) -> PPO:
        """
        Train policy using PPO

        Args:
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
            save_freq: Checkpoint save frequency

        Returns:
            Trained PPO model
        """
        logger.info("=" * 80)
        logger.info("STARTING RL TRAINING (PPO)")
        logger.info("=" * 80)
        logger.info(f"Total timesteps: {total_timesteps}")
        logger.info(f"Eval frequency: {eval_freq}")
        logger.info(f"Save frequency: {save_freq}")
        logger.info(f"PPO config: lr={self.config.learning_rate}, "
                   f"n_steps={self.config.n_steps}, "
                   f"batch_size={self.config.batch_size}")
        logger.info("=" * 80)

        # 1. Load dataset
        logger.info("\n[1/4] Loading training dataset...")
        dataset = get_dataset(
            "alpaca",
            split="train",
            max_samples=self.config.num_training_tasks
        )
        logger.info(f"[OK] Loaded {len(dataset)} training tasks")

        # Split train/val
        train_dataset, val_dataset = create_train_val_split(
            dataset,
            val_ratio=0.1,
            seed=42
        )
        logger.info(f"[OK] Split: {len(train_dataset)} train, {len(val_dataset)} val")

        # 2. Create environments
        logger.info("\n[2/4] Creating training environments...")

        def make_train_env():
            env = make_env(
                num_agents=3,
                max_rounds=5,
                dataset=list(train_dataset),
                config=self.config,
                device=self.device
            )
            return Monitor(env)

        def make_eval_env():
            env = make_env(
                num_agents=3,
                max_rounds=5,
                dataset=list(val_dataset),
                config=self.config,
                device=self.device
            )
            return Monitor(env)

        # Create vectorized environments
        n_envs = self.config.n_envs
        if n_envs > 1:
            train_env = SubprocVecEnv([make_train_env for _ in range(n_envs)])
            eval_env = DummyVecEnv([make_eval_env])
        else:
            train_env = DummyVecEnv([make_train_env])
            eval_env = DummyVecEnv([make_eval_env])

        logger.info(f"[OK] Created {n_envs} training environments")

        # 3. Create PPO model
        logger.info("\n[3/4] Creating PPO model...")

        model = PPO(
            policy="MultiInputPolicy",  # For Dict observation space
            env=train_env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            tensorboard_log=f"./logs/tensorboard/{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            device=self.device,
            verbose=1,
        )

        logger.info(f"[OK] Created PPO model")
        logger.info(f"Policy parameters: {sum(p.numel() for p in model.policy.parameters())}")

        # 4. Setup callbacks
        logger.info("\n[4/4] Setting up callbacks...")

        callbacks = []

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=str(self.checkpoint_manager.save_dir),
            name_prefix="ppo_policy",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        callbacks.append(checkpoint_callback)

        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.checkpoint_manager.save_dir / "best"),
            log_path=str(self.checkpoint_manager.save_dir / "eval_logs"),
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=5,
        )
        callbacks.append(eval_callback)

        # Metrics callback
        metrics_callback = MetricsCallback(log_freq=100)
        callbacks.append(metrics_callback)

        # WandB callback
        if self.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=f"rl_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    **vars(self.config),
                    "total_timesteps": total_timesteps,
                }
            )
            wandb_callback = WandbCallback()
            callbacks.append(wandb_callback)

        logger.info(f"[OK] Set up {len(callbacks)} callbacks")

        # 5. Train!
        logger.info("\n" + "=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)

        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                log_interval=10,
                progress_bar=True,
            )

            logger.info("\n" + "=" * 80)
            logger.info("TRAINING COMPLETE")
            logger.info("=" * 80)

        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user")

        finally:
            # Save final model
            final_path = self.checkpoint_manager.save_dir / "final_model"
            model.save(final_path)
            logger.info(f"[OK] Saved final model to {final_path}")

            # Close environments
            train_env.close()
            eval_env.close()

            if self.use_wandb:
                wandb.finish()

        return model


def train_policy(
    total_timesteps: int = 100000,
    num_tasks: int = 500,
    learning_rate: float = 3e-4,
    device: str = "cpu",
    use_wandb: bool = False,
    save_dir: Optional[Path] = None,
) -> PPO:
    """
    Complete RL training pipeline

    Args:
        total_timesteps: Total training timesteps
        num_tasks: Number of training tasks
        learning_rate: Learning rate
        device: Training device
        use_wandb: Log to wandb
        save_dir: Save directory

    Returns:
        Trained PPO model
    """
    logger.info("=" * 80)
    logger.info("RL TRAINING PIPELINE")
    logger.info("=" * 80)

    # Create config
    config = RLTrainingConfig(
        num_training_tasks=num_tasks,
        learning_rate=learning_rate,
    )

    # Create trainer
    trainer = RLTrainer(config, device, use_wandb)

    # Train
    model = trainer.train(
        total_timesteps=total_timesteps,
        eval_freq=5000,
        save_freq=10000,
    )

    logger.info("\n[OK] RL training complete!")

    return model


if __name__ == "__main__":
    # Test RL trainer setup
    print("=" * 80)
    print("TESTING RL TRAINER SETUP")
    print("=" * 80)

    # Create config
    print("\n[1/2] Creating config...")
    config = RLTrainingConfig(
        num_training_tasks=10,
        n_envs=1,
        n_steps=128,
    )
    print(f"[OK] Config created")
    print(f"  Training tasks: {config.num_training_tasks}")
    print(f"  N envs: {config.n_envs}")
    print(f"  N steps: {config.n_steps}")

    # Create trainer
    print("\n[2/2] Creating trainer...")
    trainer = RLTrainer(config, device="cpu", use_wandb=False)
    print(f"[OK] Trainer created")

    print("\n[OK] RL trainer setup working!")
    print("NOTE: Full training requires running train_policy() with real environment")

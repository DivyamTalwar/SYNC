from typing import Optional
from pydantic import BaseModel, Field


class CKMPretrainingConfig(BaseModel):
    learning_rate: float = Field(default=5e-5, description="Learning rate")
    batch_size: int = Field(default=32, description="Batch size")
    num_epochs: int = Field(default=15, description="Number of training epochs")
    warmup_steps: int = Field(default=1000, description="Number of warmup steps")
    max_grad_norm: float = Field(default=1.0, description="Max gradient norm for clipping")
    weight_decay: float = Field(default=0.01, description="Weight decay")

    num_dialogue_turns: int = Field(default=1_000_000, description="Number of dialogue turns for training")
    mask_probability: float = Field(default=0.15, description="Probability of masking utterances")
    max_sequence_length: int = Field(default=512, description="Maximum sequence length")

    adam_beta1: float = Field(default=0.9, description="Adam beta1")
    adam_beta2: float = Field(default=0.999, description="Adam beta2")
    adam_epsilon: float = Field(default=1e-8, description="Adam epsilon")

    scheduler_type: str = Field(default="linear", description="LR scheduler type")

    save_every: int = Field(default=1000, description="Save checkpoint every N steps")
    eval_every: int = Field(default=500, description="Evaluate every N steps")
    keep_best_n: int = Field(default=3, description="Keep best N checkpoints")

    use_fp16: bool = Field(default=True, description="Use mixed precision training")

    log_every: int = Field(default=100, description="Log metrics every N steps")
    wandb_project: str = Field(default="sync-ckm-pretraining", description="W&B project name")


class RLTrainingConfig(BaseModel):
    learning_rate: float = Field(default=3e-4, description="Learning rate")
    n_steps: int = Field(default=2048, description="Number of steps per update")
    batch_size: int = Field(default=64, description="Batch size")
    n_epochs: int = Field(default=10, description="Number of epochs per update")
    gamma: float = Field(default=0.99, description="Discount factor")
    gae_lambda: float = Field(default=0.95, description="GAE lambda")
    clip_range: float = Field(default=0.2, description="PPO clip range")
    clip_range_vf: Optional[float] = Field(default=None, description="Value function clip range")
    ent_coef: float = Field(default=0.01, description="Entropy coefficient")
    vf_coef: float = Field(default=0.5, description="Value function coefficient")
    max_grad_norm: float = Field(default=0.5, description="Max gradient norm")

    # Training
    total_timesteps: int = Field(default=5_000_000, description="Total training timesteps")
    num_envs: int = Field(default=8, description="Number of parallel environments")

    # Network architecture
    normalize_advantage: bool = Field(default=True, description="Normalize advantages")
    use_sde: bool = Field(default=False, description="Use state-dependent exploration")
    sde_sample_freq: int = Field(default=-1, description="SDE sample frequency")

    # Checkpointing
    save_freq: int = Field(default=10_000, description="Save checkpoint every N steps")
    eval_freq: int = Field(default=5_000, description="Evaluate every N steps")
    n_eval_episodes: int = Field(default=10, description="Number of evaluation episodes")

    # Logging
    tensorboard_log: str = Field(default="./logs/tensorboard", description="Tensorboard log directory")
    wandb_project: str = Field(default="sync-rl-training", description="W&B project name")
    verbose: int = Field(default=1, description="Verbosity level")


class RewardConfig(BaseModel):
    """Reward function configuration"""
    task_success_reward: float = Field(default=1.0, description="Reward for task success")
    task_failure_penalty: float = Field(default=-0.1, description="Penalty for task failure")
    token_penalty_weight: float = Field(default=0.001, description="Weight for token usage penalty")
    gap_reduction_reward: float = Field(default=0.05, description="Weight for gap reduction reward")
    redundancy_penalty: float = Field(default=-0.02, description="Penalty for redundant communication")
    convergence_bonus: float = Field(default=0.1, description="Bonus for early convergence")

    # Normalization
    normalize_rewards: bool = Field(default=True, description="Normalize rewards")
    reward_scale: float = Field(default=1.0, description="Reward scaling factor")


class TrainingConfig(BaseModel):
    """Complete training configuration"""
    ckm_pretraining: CKMPretrainingConfig = Field(default_factory=CKMPretrainingConfig)
    rl_training: RLTrainingConfig = Field(default_factory=RLTrainingConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)

    # General
    seed: int = Field(default=42, description="Random seed")
    use_cuda: bool = Field(default=True, description="Use CUDA if available")
    num_workers: int = Field(default=4, description="Number of data loading workers")


# Default training configuration
training_config = TrainingConfig()

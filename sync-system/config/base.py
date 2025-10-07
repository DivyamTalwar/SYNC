import os
from typing import Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class APIConfig(BaseModel):
    openrouter_api_key: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    cohere_api_key: str = Field(default_factory=lambda: os.getenv("COHERE_API_KEY", ""))
    anthropic_api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    wandb_api_key: str = Field(default_factory=lambda: os.getenv("WANDB_API_KEY", ""))

    primary_model: str = Field(default="anthropic/claude-sonnet-3.7")
    aggregator_model: str = Field(default="anthropic/claude-sonnet-4.0")
    embedding_model: str = Field(default="embed-english-v3.0")


class DatabaseConfig(BaseModel):
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="sync_system")
    postgres_user: str = Field(default="sync_user")
    postgres_password: str = Field(default="")

    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: Optional[str] = Field(default=None)

    @property
    def postgres_url(self) -> str:
        """Get PostgreSQL connection URL"""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


class ModelConfig(BaseModel):
    state_dim: int = Field(default=256, description="Agent state vector dimension")

    # CKM configuration
    ckm_dim: int = Field(default=128, description="CKM output dimension")
    ckm_num_layers: int = Field(default=2, description="Number of Transformer layers in CKM")
    ckm_num_heads: int = Field(default=8, description="Number of attention heads in CKM")
    ckm_hidden_dim: int = Field(default=512, description="Hidden dimension in CKM")
    ckm_dropout: float = Field(default=0.1, description="Dropout rate in CKM")

    # Gap detector configuration
    gap_dim: int = Field(default=64, description="Gap vector dimension")
    gap_num_heads: int = Field(default=4, description="Number of attention heads in gap detector")
    gap_hidden_dim: int = Field(default=256, description="Hidden dimension in gap detector")

    # Policy network configuration
    policy_num_layers: int = Field(default=4, description="Number of Transformer layers in policy")
    policy_hidden_dim: int = Field(default=256, description="Hidden dimension in policy")
    policy_num_heads: int = Field(default=8, description="Number of attention heads in policy")
    policy_dropout: float = Field(default=0.1, description="Dropout rate in policy")

    # Action space
    num_objectives: int = Field(default=10, description="Number of communication objectives")

    # Device configuration
    device: str = Field(default="cuda", description="Device to run models on (cuda/cpu)")


class SystemConfig(BaseModel):
    """System-level configuration"""
    num_agents: int = Field(default=6, description="Number of agents in the system")
    max_communication_rounds: int = Field(default=5, description="Maximum communication rounds")
    convergence_threshold: float = Field(default=0.01, description="Convergence detection threshold")
    max_history_length: int = Field(default=10, description="Maximum dialogue history length")

    # Environment
    environment: Literal["development", "staging", "production"] = Field(default="development")
    log_level: str = Field(default="INFO")
    log_format: Literal["json", "text"] = Field(default="text")


class SYNCConfig(BaseModel):
    """Main SYNC system configuration"""
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_env(cls) -> "SYNCConfig":
        """Load configuration from environment variables"""
        return cls(
            api=APIConfig(
                openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
                cohere_api_key=os.getenv("COHERE_API_KEY", ""),
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
                wandb_api_key=os.getenv("WANDB_API_KEY", ""),
                primary_model=os.getenv("PRIMARY_MODEL", "anthropic/claude-sonnet-3.7"),
                aggregator_model=os.getenv("AGGREGATOR_MODEL", "anthropic/claude-sonnet-4.0"),
                embedding_model=os.getenv("EMBEDDING_MODEL", "embed-english-v3.0"),
            ),
            database=DatabaseConfig(
                postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
                postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
                postgres_db=os.getenv("POSTGRES_DB", "sync_system"),
                postgres_user=os.getenv("POSTGRES_USER", "sync_user"),
                postgres_password=os.getenv("POSTGRES_PASSWORD", ""),
                redis_host=os.getenv("REDIS_HOST", "localhost"),
                redis_port=int(os.getenv("REDIS_PORT", "6379")),
                redis_db=int(os.getenv("REDIS_DB", "0")),
                redis_password=os.getenv("REDIS_PASSWORD"),
            ),
            model=ModelConfig(
                device=os.getenv("DEVICE", "cuda"),
            ),
            system=SystemConfig(
                num_agents=int(os.getenv("NUM_AGENTS", "6")),
                max_communication_rounds=int(os.getenv("MAX_COMMUNICATION_ROUNDS", "5")),
                environment=os.getenv("ENVIRONMENT", "development"),
                log_level=os.getenv("LOG_LEVEL", "INFO"),
            ),
        )


# Global configuration instance
config = SYNCConfig.from_env()

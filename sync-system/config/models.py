from typing import List
from pydantic import BaseModel, Field
from enum import Enum


class CommunicationObjective(str, Enum):
    REQUEST_CLARIFICATION = "request_clarification"
    PROPOSE_REFINEMENT = "propose_refinement"
    HIGHLIGHT_DISCREPANCY = "highlight_discrepancy"
    CHALLENGE_ASSUMPTION = "challenge_assumption"
    PROVIDE_EVIDENCE = "provide_evidence"
    SYNTHESIZE_PERSPECTIVES = "synthesize_perspectives"
    REQUEST_ELABORATION = "request_elaboration"
    SUGGEST_ALTERNATIVE = "suggest_alternative"
    CONFIRM_UNDERSTANDING = "confirm_understanding"
    SIGNAL_AGREEMENT = "signal_agreement"


class GapType(str, Enum):
    METHOD_DIVERGENCE = "method_divergence"
    FACTUAL_DISAGREEMENT = "factual_disagreement"
    GOAL_MISALIGNMENT = "goal_misalignment"
    CONFIDENCE_DISCREPANCY = "confidence_discrepancy"


class CKMArchitectureConfig(BaseModel):
    input_dim: int = Field(default=768, description="Input embedding dimension")
    hidden_dim: int = Field(default=512, description="Hidden dimension")
    output_dim: int = Field(default=128, description="Output CKM dimension")
    num_layers: int = Field(default=2, description="Number of Transformer layers")
    num_heads: int = Field(default=8, description="Number of attention heads")
    dropout: float = Field(default=0.1, description="Dropout rate")
    max_seq_length: int = Field(default=512, description="Maximum sequence length")

    # GRU configuration
    gru_hidden_dim: int = Field(default=128, description="GRU hidden dimension")
    gru_num_layers: int = Field(default=1, description="Number of GRU layers")

    # History configuration
    history_window: int = Field(default=5, description="Number of recent utterances to consider")


class GapDetectorConfig(BaseModel):
    """Gap detector architecture configuration"""
    state_dim: int = Field(default=256, description="Agent state dimension")
    ckm_dim: int = Field(default=128, description="CKM dimension")
    gap_dim: int = Field(default=64, description="Gap vector dimension")
    num_attention_heads: int = Field(default=4, description="Number of cross-attention heads")
    hidden_dim: int = Field(default=256, description="MLP hidden dimension")
    dropout: float = Field(default=0.1, description="Dropout rate")

    # Gap types
    gap_types: List[str] = Field(
        default_factory=lambda: [gt.value for gt in GapType],
        description="Types of gaps to detect"
    )


class PolicyNetworkConfig(BaseModel):
    """Communication policy network architecture configuration"""
    input_dim: int = Field(default=512, description="Input dimension (state + CKMs + gaps)")
    hidden_dim: int = Field(default=256, description="Hidden dimension")
    num_layers: int = Field(default=4, description="Number of Transformer layers")
    num_heads: int = Field(default=8, description="Number of attention heads")
    dropout: float = Field(default=0.1, description="Dropout rate")

    # Action space
    num_objectives: int = Field(default=10, description="Number of communication objectives")
    objectives: List[str] = Field(
        default_factory=lambda: [obj.value for obj in CommunicationObjective],
        description="Communication objectives"
    )

    # Temperature for action sampling
    temperature: float = Field(default=1.0, description="Temperature for action sampling")
    deterministic_inference: bool = Field(default=False, description="Use deterministic actions at inference")


class StateEncoderConfig(BaseModel):
    """State encoder configuration"""
    embedding_model: str = Field(default="sentence-transformers/all-mpnet-base-v2", description="Sentence embedding model")
    embedding_dim: int = Field(default=768, description="Embedding dimension from model")
    state_dim: int = Field(default=256, description="Final state vector dimension")
    pooling_strategy: str = Field(default="mean", description="Pooling strategy (mean, max, cls)")

    # Projection
    use_projection: bool = Field(default=True, description="Use projection layer")
    projection_hidden_dim: int = Field(default=512, description="Projection hidden dimension")


class ModelArchitectureConfig(BaseModel):
    """Complete model architecture configuration"""
    ckm: CKMArchitectureConfig = Field(default_factory=CKMArchitectureConfig)
    gap_detector: GapDetectorConfig = Field(default_factory=GapDetectorConfig)
    policy: PolicyNetworkConfig = Field(default_factory=PolicyNetworkConfig)
    state_encoder: StateEncoderConfig = Field(default_factory=StateEncoderConfig)


# Default model configuration
model_architecture_config = ModelArchitectureConfig()

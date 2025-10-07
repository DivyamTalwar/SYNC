import torch
import torch.nn as nn
from typing import Optional, Dict, List
from enum import Enum
from dataclasses import dataclass
from src.utils.logging import get_logger

logger = get_logger("gap_detector")


class GapType(str, Enum):
    """Types of cognitive gaps between agents"""
    METHOD_DIVERGENCE = "method_divergence"
    FACTUAL_DISAGREEMENT = "factual_disagreement"
    GOAL_MISALIGNMENT = "goal_misalignment"
    CONFIDENCE_DISCREPANCY = "confidence_discrepancy"


@dataclass
class GapAnalysis:
    source_agent_id: int
    target_agent_id: int
    gap_vector: torch.Tensor  # Shape: (64,)
    gap_magnitude: float
    gap_types: Dict[str, float]


class MultiHeadCrossAttention(nn.Module):

    def __init__(
        self,
        query_dim: int = 256,  # Φᵢ dimension
        kv_dim: int = 128,     # zᵢⱼ dimension
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize multi-head cross-attention

        Args:
            query_dim: Query dimension (agent state)
            kv_dim: Key/value dimension (CKM)
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"

        self.query_dim = query_dim
        self.kv_dim = kv_dim

        # Projection layers
        self.query_proj = nn.Linear(query_dim, query_dim)
        self.key_proj = nn.Linear(kv_dim, query_dim)
        self.value_proj = nn.Linear(kv_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        logger.info(f"Initialized MultiHeadCrossAttention: {num_heads} heads")

    def forward(
        self,
        query: torch.Tensor,  # (B x query_dim) - agent state
        key_value: torch.Tensor,  # (B x kv_dim) - CKM state
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through cross-attention

        Args:
            query: Agent's own state (B x query_dim)
            key_value: Model of collaborator (B x kv_dim)
            mask: Optional attention mask

        Returns:
            Attended features (B x query_dim)
        """
        B = query.shape[0]

        # Expand dimensions for attention: (B x 1 x dim)
        query = query.unsqueeze(1)  # (B x 1 x query_dim)
        key_value = key_value.unsqueeze(1)  # (B x 1 x kv_dim)

        # Project
        Q = self.query_proj(query)  # (B x 1 x query_dim)
        K = self.key_proj(key_value)  # (B x 1 x query_dim)
        V = self.value_proj(key_value)  # (B x 1 x query_dim)

        # Reshape for multi-head attention
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B x num_heads x 1 x head_dim)
        K = K.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B x num_heads x 1 x 1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        attended = torch.matmul(attn_weights, V)  # (B x num_heads x 1 x head_dim)

        # Reshape and project
        attended = attended.transpose(1, 2).contiguous().view(B, 1, self.query_dim)  # (B x 1 x query_dim)
        output = self.out_proj(attended)  # (B x 1 x query_dim)

        return output.squeeze(1)  # (B x query_dim)


class GapDetector(nn.Module):
    """
    Cognitive Gap Detector

    Analyzes the difference between an agent's own state and their
    model of a collaborator's state using cross-attention + MLP.
    """

    def __init__(
        self,
        state_dim: int = 256,
        ckm_dim: int = 128,
        gap_dim: int = 64,
        num_attention_heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Initialize gap detector

        Args:
            state_dim: Agent state dimension (Φᵢ)
            ckm_dim: CKM dimension (zᵢⱼ)
            gap_dim: Output gap vector dimension
            num_attention_heads: Number of cross-attention heads
            hidden_dim: MLP hidden dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.state_dim = state_dim
        self.ckm_dim = ckm_dim
        self.gap_dim = gap_dim

        # Cross-attention between state and CKM
        self.cross_attention = MultiHeadCrossAttention(
            query_dim=state_dim,
            kv_dim=ckm_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        # MLP for gap computation
        self.gap_mlp = nn.Sequential(
            nn.Linear(state_dim * 2 + ckm_dim, hidden_dim),  # Concat state, CKM, attended
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, gap_dim),
        )

        # Gap type classification heads
        num_gap_types = len(GapType)
        self.gap_type_classifier = nn.Sequential(
            nn.Linear(gap_dim, gap_dim),
            nn.ReLU(),
            nn.Linear(gap_dim, num_gap_types),
            nn.Sigmoid(),  # Multi-label classification
        )

        logger.info(f"Initialized GapDetector: {state_dim} + {ckm_dim} -> {gap_dim}")

    def forward(
        self,
        own_state: torch.Tensor,  # (B x state_dim)
        ckm_state: torch.Tensor,  # (B x ckm_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Detect cognitive gaps

        Args:
            own_state: Agent's own state vector (B x state_dim)
            ckm_state: Agent's model of collaborator (B x ckm_dim)

        Returns:
            gap_vector: Gap vector (B x gap_dim)
            gap_type_scores: Gap type scores (B x num_gap_types)
        """
        # Cross-attention: attend from own state to CKM
        attended = self.cross_attention(own_state, ckm_state)  # (B x state_dim)

        # Concatenate features
        combined = torch.cat([own_state, ckm_state, attended], dim=-1)  # (B x (state_dim*2 + ckm_dim))

        # Compute gap vector
        gap_vector = self.gap_mlp(combined)  # (B x gap_dim)

        # Classify gap types
        gap_type_scores = self.gap_type_classifier(gap_vector)  # (B x num_gap_types)

        return gap_vector, gap_type_scores

    def analyze_gap(
        self,
        own_state: torch.Tensor,
        ckm_state: torch.Tensor,
        source_agent_id: int,
        target_agent_id: int,
    ) -> GapAnalysis:
        """
        Perform full gap analysis

        Args:
            own_state: Agent's own state (1 x state_dim)
            ckm_state: Agent's model of collaborator (1 x ckm_dim)
            source_agent_id: ID of agent performing analysis
            target_agent_id: ID of collaborator being analyzed

        Returns:
            GapAnalysis object
        """
        with torch.no_grad():
            gap_vector, gap_type_scores = self.forward(own_state, ckm_state)

            # Compute gap magnitude
            gap_magnitude = torch.norm(gap_vector, p=2).item()

            # Extract gap type scores
            gap_types = {}
            for i, gap_type in enumerate(GapType):
                gap_types[gap_type.value] = gap_type_scores[0, i].item()

            return GapAnalysis(
                source_agent_id=source_agent_id,
                target_agent_id=target_agent_id,
                gap_vector=gap_vector.squeeze(0),
                gap_magnitude=gap_magnitude,
                gap_types=gap_types,
            )


class GapManager:
    """Manages gap computations for all agent pairs"""

    def __init__(self, num_agents: int, gap_dim: int = 64, device: str = "cpu"):
        """
        Initialize gap manager

        Args:
            num_agents: Number of agents
            gap_dim: Dimension of gap vectors
            device: Device to store tensors on
        """
        self.num_agents = num_agents
        self.gap_dim = gap_dim
        self.device = device

        # Gap analyses: agent i's gap analysis with agent j
        self.gap_analyses: Dict[tuple[int, int], GapAnalysis] = {}

        logger.info(f"Initialized GapManager for {num_agents} agents")

    def store_gap_analysis(self, gap_analysis: GapAnalysis) -> None:
        """Store a gap analysis"""
        key = (gap_analysis.source_agent_id, gap_analysis.target_agent_id)
        self.gap_analyses[key] = gap_analysis

    def get_gap_analysis(
        self,
        source_agent_id: int,
        target_agent_id: int,
    ) -> Optional[GapAnalysis]:
        """Get gap analysis for a specific pair"""
        return self.gap_analyses.get((source_agent_id, target_agent_id))

    def get_all_gaps_for_agent(self, agent_id: int) -> List[GapAnalysis]:
        """Get all gap analyses performed by an agent"""
        return [
            gap for (src, _), gap in self.gap_analyses.items()
            if src == agent_id
        ]

    def get_gap_vectors_for_agent(self, agent_id: int) -> torch.Tensor:
        """Get all gap vectors for an agent as a tensor"""
        gaps = self.get_all_gaps_for_agent(agent_id)
        if not gaps:
            return torch.zeros(0, self.gap_dim, device=self.device)
        vectors = [gap.gap_vector for gap in gaps]
        return torch.stack(vectors)

    def get_total_gap_magnitude(self, agent_id: int) -> float:
        """Get total gap magnitude for an agent across all collaborators"""
        gaps = self.get_all_gaps_for_agent(agent_id)
        return sum(gap.gap_magnitude for gap in gaps)

    def reset(self) -> None:
        """Reset all gap analyses"""
        self.gap_analyses.clear()
        logger.info("Reset gap manager")

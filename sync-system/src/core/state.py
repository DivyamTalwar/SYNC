import torch
import torch.nn as nn
from typing import List, Optional
from dataclasses import dataclass
from src.utils.logging import get_logger

logger = get_logger("state")


@dataclass
class AgentState:
    agent_id: int
    state_vector: torch.Tensor  # Shape: (256,)
    reasoning_trace: str
    query: str
    step: int
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            "agent_id": self.agent_id,
            "state_vector": self.state_vector.cpu().numpy().tolist(),
            "reasoning_trace": self.reasoning_trace,
            "query": self.query,
            "step": self.step,
            "metadata": self.metadata,
        }


class StateEncoder(nn.Module):

    def __init__(
        self,
        embedding_dim: int = 768,
        state_dim: int = 256,
        use_projection: bool = True,
        projection_hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        """
        Initialize state encoder

        Args:
            embedding_dim: Dimension of input embeddings (from sentence-transformers)
            state_dim: Output state dimension (256)
            use_projection: Whether to use projection layer
            projection_hidden_dim: Hidden dimension for projection
            dropout: Dropout rate
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.use_projection = use_projection

        if use_projection:
            # Project embeddings to state dimension
            self.projection = nn.Sequential(
                nn.Linear(embedding_dim, projection_hidden_dim),
                nn.LayerNorm(projection_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(projection_hidden_dim, state_dim),
                nn.LayerNorm(state_dim),
            )
        else:
            # Simple linear projection
            assert embedding_dim == state_dim, "embedding_dim must equal state_dim when use_projection=False"
            self.projection = nn.Identity()

        logger.info(f"Initialized StateEncoder: {embedding_dim} -> {state_dim}")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode embeddings into state vectors

        Args:
            embeddings: Input embeddings (B x embedding_dim)

        Returns:
            State vectors (B x state_dim)
        """
        state_vectors = self.projection(embeddings)
        return state_vectors

    def encode_from_text(
        self,
        texts: List[str],
        embedding_model,
    ) -> torch.Tensor:
        """
        Encode texts into state vectors using embedding model

        Args:
            texts: List of text strings
            embedding_model: Sentence embedding model

        Returns:
            State vectors (B x state_dim)
        """
        # Get embeddings from sentence-transformers model
        with torch.no_grad():
            embeddings = embedding_model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
            )

        # Project to state space
        state_vectors = self.forward(embeddings)
        return state_vectors


class StateManager:
    """Manages agent states during multi-agent collaboration"""

    def __init__(self, num_agents: int, state_dim: int = 256):
        """
        Initialize state manager

        Args:
            num_agents: Number of agents
            state_dim: Dimension of state vectors
        """
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.states: dict[int, AgentState] = {}
        self.state_history: dict[int, List[AgentState]] = {i: [] for i in range(num_agents)}

    def update_state(self, state: AgentState) -> None:
        """Update an agent's state"""
        self.states[state.agent_id] = state
        self.state_history[state.agent_id].append(state)
        logger.debug(f"Updated state for agent {state.agent_id} at step {state.step}")

    def get_state(self, agent_id: int) -> Optional[AgentState]:
        """Get current state for an agent"""
        return self.states.get(agent_id)

    def get_all_states(self) -> List[AgentState]:
        """Get all current agent states"""
        return list(self.states.values())

    def get_state_vectors(self) -> torch.Tensor:
        """Get state vectors for all agents as a tensor"""
        vectors = [state.state_vector for state in self.states.values()]
        return torch.stack(vectors) if vectors else torch.zeros(0, self.state_dim)

    def get_state_history(self, agent_id: int) -> List[AgentState]:
        """Get state history for an agent"""
        return self.state_history.get(agent_id, [])

    def reset(self) -> None:
        """Reset all states"""
        self.states.clear()
        self.state_history = {i: [] for i in range(self.num_agents)}
        logger.info("Reset state manager")


def compute_state_similarity(
    state1: torch.Tensor,
    state2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine similarity between two state vectors

    Args:
        state1: First state vector (D,) or (B x D)
        state2: Second state vector (D,) or (B x D)

    Returns:
        Cosine similarity
    """
    return torch.nn.functional.cosine_similarity(state1, state2, dim=-1)


def compute_state_distance(
    state1: torch.Tensor,
    state2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Euclidean distance between two state vectors

    Args:
        state1: First state vector (D,) or (B x D)
        state2: Second state vector (D,) or (B x D)

    Returns:
        Euclidean distance
    """
    return torch.norm(state1 - state2, p=2, dim=-1)

import torch
import torch.nn as nn
from typing import Optional, List, Tuple
from dataclasses import dataclass
from src.utils.logging import get_logger

logger = get_logger("ckm")


@dataclass
class CKMState:
    source_agent_id: int
    target_agent_id: int
    cognitive_state: torch.Tensor 
    last_update_step: int


class CKMTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_length: int = 512,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_length, hidden_dim)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        logger.info(f"Initialized CKMTransformer: {num_layers} layers, {num_heads} heads")

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through Transformer

        Args:
            embeddings: Input embeddings (B x seq_len x input_dim)
            mask: Optional attention mask (B x seq_len)

        Returns:
            Encoded representations (B x seq_len x hidden_dim)
        """
        B, seq_len, _ = embeddings.shape

        # Project input
        x = self.input_projection(embeddings)  # (B x seq_len x hidden_dim)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Apply Transformer
        # Create attention mask if needed
        if mask is not None:
            # Convert mask to attention mask format
            attention_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        else:
            attention_mask = None

        x = self.transformer(x, src_key_padding_mask=attention_mask)

        # Layer norm
        x = self.layer_norm(x)

        return x


class CKMGRU(nn.Module):
    """
    GRU for temporal updates of CKM state

    Updates cognitive state based on new messages: z(t+1) = GRU(z(t), msg)
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        """
        Initialize CKM GRU

        Args:
            input_dim: Input dimension (from Transformer)
            hidden_dim: Hidden dimension (CKM output dimension, 128)
            num_layers: Number of GRU layers
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        logger.info(f"Initialized CKMGRU: {input_dim} -> {hidden_dim}")

    def forward(
        self,
        input_seq: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GRU

        Args:
            input_seq: Input sequence (B x seq_len x input_dim)
            hidden_state: Previous hidden state (num_layers x B x hidden_dim)

        Returns:
            output: GRU outputs (B x seq_len x hidden_dim)
            hidden: Final hidden state (num_layers x B x hidden_dim)
        """
        output, hidden = self.gru(input_seq, hidden_state)
        return output, hidden


class CollaboratorKnowledgeModel(nn.Module):
    """
    Complete CKM: Transformer + GRU for modeling collaborator cognitive states

    Predicts and updates models of other agents' understanding based on
    their utterances and behavior.
    """

    def __init__(
        self,
        input_dim: int = 768,
        transformer_hidden_dim: int = 512,
        output_dim: int = 128,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        num_gru_layers: int = 1,
        dropout: float = 0.1,
        max_seq_length: int = 512,
    ):
        """
        Initialize Collaborator Knowledge Model

        Args:
            input_dim: Input embedding dimension
            transformer_hidden_dim: Transformer hidden dimension
            output_dim: Output CKM dimension (128)
            num_transformer_layers: Number of Transformer layers
            num_heads: Number of attention heads
            num_gru_layers: Number of GRU layers
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Transformer encoder
        self.transformer = CKMTransformer(
            input_dim=input_dim,
            hidden_dim=transformer_hidden_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_length=max_seq_length,
        )

        # GRU for temporal updates
        self.gru = CKMGRU(
            input_dim=transformer_hidden_dim,
            hidden_dim=output_dim,
            num_layers=num_gru_layers,
            dropout=dropout,
        )

        # Initialize hidden state
        self.register_buffer(
            "initial_hidden",
            torch.zeros(num_gru_layers, 1, output_dim)
        )

        logger.info(f"Initialized CollaboratorKnowledgeModel: {input_dim} -> {output_dim}")

    def forward(
        self,
        utterance_embeddings: torch.Tensor,
        previous_ckm_state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: update CKM based on utterances

        Args:
            utterance_embeddings: Embeddings of recent utterances (B x seq_len x input_dim)
            previous_ckm_state: Previous CKM hidden state (num_layers x B x output_dim)
            mask: Optional attention mask

        Returns:
            ckm_vector: Updated CKM vector (B x output_dim)
            hidden_state: Updated hidden state (num_layers x B x output_dim)
        """
        B = utterance_embeddings.shape[0]

        # Encode utterances with Transformer
        encoded = self.transformer(utterance_embeddings, mask)  # (B x seq_len x transformer_hidden)

        # Update with GRU
        if previous_ckm_state is None:
            previous_ckm_state = self.initial_hidden.expand(-1, B, -1).contiguous()

        gru_output, hidden_state = self.gru(encoded, previous_ckm_state)

        # Take last output as CKM vector
        ckm_vector = gru_output[:, -1, :]  # (B x output_dim)

        return ckm_vector, hidden_state

    def predict_utterance(
        self,
        context_embeddings: torch.Tensor,
        ckm_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pre-training task: predict masked utterance

        Args:
            context_embeddings: Context utterances (B x seq_len x input_dim)
            ckm_state: Current CKM state (B x output_dim)

        Returns:
            Predicted utterance embedding (B x input_dim)
        """
        # Encode context
        encoded = self.transformer(context_embeddings)  # (B x seq_len x transformer_hidden)

        # Pool encoded representations (mean pooling)
        pooled = encoded.mean(dim=1)  # (B x transformer_hidden)

        # Combine with CKM state for prediction
        # This is a simple approach; can be made more sophisticated
        combined = torch.cat([pooled, ckm_state], dim=-1)  # (B x (transformer_hidden + output_dim))

        # Project to input_dim for prediction
        # Note: In practice, you'd add a prediction head
        # For now, this is a placeholder
        return pooled  # Simplified


class CKMManager:
    """Manages CKM states for all agent pairs"""

    def __init__(self, num_agents: int, ckm_dim: int = 128, device: str = "cpu"):
        """
        Initialize CKM manager

        Args:
            num_agents: Number of agents
            ckm_dim: Dimension of CKM vectors
            device: Device to store tensors on
        """
        self.num_agents = num_agents
        self.ckm_dim = ckm_dim
        self.device = device

        # CKM states: agent i's model of agent j
        # ckm_states[i][j] = agent i's model of agent j
        self.ckm_states: dict[int, dict[int, CKMState]] = {
            i: {} for i in range(num_agents)
        }

        # GRU hidden states for each pair
        self.hidden_states: dict[tuple, torch.Tensor] = {}

        self._initialize_ckm_states()

    def _initialize_ckm_states(self) -> None:
        """Initialize CKM states for all pairs"""
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j:
                    # Initialize with zero vector
                    initial_state = CKMState(
                        source_agent_id=i,
                        target_agent_id=j,
                        cognitive_state=torch.zeros(self.ckm_dim, device=self.device),
                        last_update_step=0,
                    )
                    self.ckm_states[i][j] = initial_state

        logger.info(f"Initialized CKM states for {self.num_agents} agents")

    def update_ckm(
        self,
        source_agent_id: int,
        target_agent_id: int,
        new_ckm_vector: torch.Tensor,
        step: int,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> None:
        """Update CKM state for a specific agent pair"""
        self.ckm_states[source_agent_id][target_agent_id] = CKMState(
            source_agent_id=source_agent_id,
            target_agent_id=target_agent_id,
            cognitive_state=new_ckm_vector.detach(),
            last_update_step=step,
        )

        if hidden_state is not None:
            self.hidden_states[(source_agent_id, target_agent_id)] = hidden_state.detach()

    def get_ckm(self, source_agent_id: int, target_agent_id: int) -> Optional[CKMState]:
        """Get CKM state for a specific pair"""
        return self.ckm_states[source_agent_id].get(target_agent_id)

    def get_all_ckms_for_agent(self, agent_id: int) -> dict[int, CKMState]:
        """Get all CKMs maintained by a specific agent"""
        return self.ckm_states[agent_id]

    def get_ckm_vectors_for_agent(self, agent_id: int) -> torch.Tensor:
        """Get all CKM vectors for an agent as a tensor"""
        ckms = self.get_all_ckms_for_agent(agent_id)
        vectors = [ckm.cognitive_state for ckm in ckms.values()]
        return torch.stack(vectors) if vectors else torch.zeros(0, self.ckm_dim, device=self.device)

    def reset(self) -> None:
        """Reset all CKM states"""
        self._initialize_ckm_states()
        self.hidden_states.clear()
        logger.info("Reset CKM manager")

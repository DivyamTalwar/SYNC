import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
from src.utils.logging import get_logger

logger = get_logger("policy")


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


@dataclass
class CommunicationAction:
    source_agent_id: int
    target_agent_id: int
    objective: CommunicationObjective
    priority: float
    confidence: float


class PolicyTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize policy Transformer

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_layers: Number of Transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

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

        logger.info(f"Initialized PolicyTransformer: {num_layers} layers, {num_heads} heads")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (B x seq_len x input_dim) or (B x input_dim)

        Returns:
            Encoded representation (B x hidden_dim)
        """
        # Handle single vector input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B x 1 x input_dim)

        # Project
        x = self.input_proj(x)  # (B x seq_len x hidden_dim)

        # Transform
        x = self.transformer(x)  # (B x seq_len x hidden_dim)

        # Pool (mean pooling)
        x = x.mean(dim=1)  # (B x hidden_dim)

        # Layer norm
        x = self.layer_norm(x)

        return x


class CommunicationPolicyNetwork(nn.Module):
    """
    PPO-trainable policy network for strategic communication

    Takes agent state, all CKM states, all gap vectors, and dialogue history
    to select optimal communication actions.
    """

    def __init__(
        self,
        state_dim: int = 256,
        ckm_dim: int = 128,
        gap_dim: int = 64,
        num_agents: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_objectives: int = 10,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        """
        Initialize communication policy network

        Args:
            state_dim: Agent state dimension
            ckm_dim: CKM dimension
            gap_dim: Gap vector dimension
            num_agents: Number of agents in system
            hidden_dim: Hidden dimension
            num_layers: Number of Transformer layers
            num_heads: Number of attention heads
            num_objectives: Number of communication objectives
            dropout: Dropout rate
            temperature: Temperature for action sampling
        """
        super().__init__()

        self.state_dim = state_dim
        self.ckm_dim = ckm_dim
        self.gap_dim = gap_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.num_objectives = num_objectives
        self.temperature = temperature

        # Calculate input dimension
        # Own state + (N-1) CKMs + (N-1) gaps
        self.input_dim = state_dim + (num_agents - 1) * (ckm_dim + gap_dim)

        # Policy Transformer
        self.policy_transformer = PolicyTransformer(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Action heads

        # 1. Objective head: what to communicate
        self.objective_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_objectives),
        )

        # 2. Target head: to whom to communicate (N-1 other agents + broadcast)
        num_targets = num_agents  # N-1 agents + 1 broadcast option
        self.target_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_targets),
        )

        # 3. Priority head: how important is this communication
        self.priority_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # [0, 1]
        )

        # Value head for PPO (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        logger.info(
            f"Initialized CommunicationPolicyNetwork: "
            f"{self.input_dim} -> {num_objectives} objectives, {num_targets} targets"
        )

    def forward(
        self,
        own_state: torch.Tensor,  # (B x state_dim)
        ckm_states: torch.Tensor,  # (B x (N-1) x ckm_dim)
        gap_vectors: torch.Tensor,  # (B x (N-1) x gap_dim)
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through policy network

        Args:
            own_state: Agent's own state
            ckm_states: All CKM states for other agents
            gap_vectors: All gap vectors for other agents

        Returns:
            Dictionary containing:
                - objective_logits: Logits for objective selection
                - target_logits: Logits for target selection
                - priority: Priority score
                - value: State value estimate (for PPO)
        """
        B = own_state.shape[0]

        # Flatten CKMs and gaps
        ckm_flat = ckm_states.view(B, -1)  # (B x (N-1)*ckm_dim)
        gap_flat = gap_vectors.view(B, -1)  # (B x (N-1)*gap_dim)

        # Concatenate all features
        policy_input = torch.cat([own_state, ckm_flat, gap_flat], dim=-1)  # (B x input_dim)

        # Encode with Transformer
        encoded = self.policy_transformer(policy_input)  # (B x hidden_dim)

        # Compute action logits
        objective_logits = self.objective_head(encoded)  # (B x num_objectives)
        target_logits = self.target_head(encoded)  # (B x num_targets)
        priority = self.priority_head(encoded)  # (B x 1)

        # Compute value estimate
        value = self.value_head(encoded)  # (B x 1)

        return {
            "objective_logits": objective_logits,
            "target_logits": target_logits,
            "priority": priority.squeeze(-1),
            "value": value.squeeze(-1),
            "encoded_state": encoded,
        }

    def select_action(
        self,
        own_state: torch.Tensor,
        ckm_states: torch.Tensor,
        gap_vectors: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[CommunicationAction, Dict[str, torch.Tensor]]:
        """
        Select a communication action

        Args:
            own_state: Agent's own state (1 x state_dim)
            ckm_states: All CKM states (1 x (N-1) x ckm_dim)
            gap_vectors: All gap vectors (1 x (N-1) x gap_dim)
            deterministic: If True, select argmax; else sample

        Returns:
            action: Selected CommunicationAction
            action_info: Dictionary with action probabilities and log_probs
        """
        outputs = self.forward(own_state, ckm_states, gap_vectors)

        # Apply temperature
        objective_logits = outputs["objective_logits"] / self.temperature
        target_logits = outputs["target_logits"] / self.temperature

        # Compute probabilities
        objective_probs = torch.softmax(objective_logits, dim=-1)
        target_probs = torch.softmax(target_logits, dim=-1)

        if deterministic:
            # Argmax selection
            objective_idx = objective_probs.argmax(dim=-1).item()
            target_idx = target_probs.argmax(dim=-1).item()
        else:
            # Sample from distribution
            objective_dist = torch.distributions.Categorical(objective_probs)
            target_dist = torch.distributions.Categorical(target_probs)

            objective_idx = objective_dist.sample().item()
            target_idx = target_dist.sample().item()

            # Store log probabilities for PPO
            outputs["objective_log_prob"] = objective_dist.log_prob(torch.tensor([objective_idx]))
            outputs["target_log_prob"] = target_dist.log_prob(torch.tensor([target_idx]))

        # Get objective
        objectives = list(CommunicationObjective)
        objective = objectives[objective_idx]

        # Get target agent ID
        # Last index is broadcast (-1), others are agent IDs
        if target_idx == self.num_agents - 1:
            target_agent_id = -1  # Broadcast
        else:
            target_agent_id = target_idx

        # Get priority and confidence
        priority = outputs["priority"].item()
        confidence = max(objective_probs[0, objective_idx].item(), target_probs[0, target_idx].item())

        action = CommunicationAction(
            source_agent_id=-1,  # To be filled by agent
            target_agent_id=target_agent_id,
            objective=objective,
            priority=priority,
            confidence=confidence,
        )

        action_info = {
            "objective_probs": objective_probs,
            "target_probs": target_probs,
            "value": outputs["value"],
        }

        if not deterministic:
            action_info["objective_log_prob"] = outputs["objective_log_prob"]
            action_info["target_log_prob"] = outputs["target_log_prob"]

        return action, action_info

    def evaluate_actions(
        self,
        own_state: torch.Tensor,
        ckm_states: torch.Tensor,
        gap_vectors: torch.Tensor,
        objective_actions: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate actions for PPO training

        Args:
            own_state: Agent states (B x state_dim)
            ckm_states: CKM states (B x (N-1) x ckm_dim)
            gap_vectors: Gap vectors (B x (N-1) x gap_dim)
            objective_actions: Objective actions taken (B,)
            target_actions: Target actions taken (B,)

        Returns:
            Dictionary with log_probs, values, and entropy
        """
        outputs = self.forward(own_state, ckm_states, gap_vectors)

        # Compute probabilities
        objective_probs = torch.softmax(outputs["objective_logits"], dim=-1)
        target_probs = torch.softmax(outputs["target_logits"], dim=-1)

        # Create distributions
        objective_dist = torch.distributions.Categorical(objective_probs)
        target_dist = torch.distributions.Categorical(target_probs)

        # Compute log probabilities
        objective_log_probs = objective_dist.log_prob(objective_actions)
        target_log_probs = target_dist.log_prob(target_actions)

        # Combined log probability
        action_log_probs = objective_log_probs + target_log_probs

        # Compute entropy for exploration bonus
        objective_entropy = objective_dist.entropy()
        target_entropy = target_dist.entropy()
        entropy = objective_entropy + target_entropy

        return {
            "action_log_probs": action_log_probs,
            "values": outputs["value"],
            "entropy": entropy,
            "objective_log_probs": objective_log_probs,
            "target_log_probs": target_log_probs,
        }


def create_policy_input(
    agent_id: int,
    state_manager,
    ckm_manager,
    gap_manager,
    num_agents: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper function to create policy network inputs

    Args:
        agent_id: Agent ID
        state_manager: StateManager instance
        ckm_manager: CKMManager instance
        gap_manager: GapManager instance
        num_agents: Total number of agents

    Returns:
        own_state, ckm_states, gap_vectors
    """
    # Get own state
    own_state = state_manager.get_state(agent_id).state_vector.unsqueeze(0)

    # Get all CKM states for other agents
    ckm_states_list = []
    gap_vectors_list = []

    for other_id in range(num_agents):
        if other_id != agent_id:
            ckm = ckm_manager.get_ckm(agent_id, other_id)
            gap = gap_manager.get_gap_analysis(agent_id, other_id)

            if ckm is not None:
                ckm_states_list.append(ckm.cognitive_state)
            if gap is not None:
                gap_vectors_list.append(gap.gap_vector)

    # Stack into tensors
    ckm_states = torch.stack(ckm_states_list).unsqueeze(0)  # (1 x (N-1) x ckm_dim)
    gap_vectors = torch.stack(gap_vectors_list).unsqueeze(0)  # (1 x (N-1) x gap_dim)

    return own_state, ckm_states, gap_vectors

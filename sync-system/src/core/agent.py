import torch
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import asyncio

from src.core.state import StateEncoder, AgentState, compute_state_similarity
from src.core.ckm import CollaboratorKnowledgeModel, CKMState
from src.core.gap_detector import GapDetector, GapAnalysis
from src.core.policy import CommunicationPolicyNetwork, CommunicationAction
from src.llm.openrouter import OpenRouterClient
from src.llm.embeddings import CohereEmbeddingsClient
from src.llm.message_gen import MessageParams, generate_message_prompt
from src.utils.logging import get_logger

logger = get_logger("agent")


@dataclass
class Message:
    sender_id: int
    receiver_id: int 
    content: str
    step: int
    objective: str
    metadata: Optional[Dict] = None


class Agent:

    def __init__(
        self,
        agent_id: int,
        num_agents: int,
        state_encoder: StateEncoder,
        ckm_model: CollaboratorKnowledgeModel,
        gap_detector: GapDetector,
        policy_network: CommunicationPolicyNetwork,
        llm_client: Optional[OpenRouterClient] = None,
        embedding_client: Optional[CohereEmbeddingsClient] = None,
        role: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize agent

        Args:
            agent_id: Unique agent ID
            num_agents: Total number of agents in system
            state_encoder: State encoder network
            ckm_model: CKM network
            gap_detector: Gap detector network
            policy_network: Policy network
            llm_client: OpenRouter client
            embedding_client: Cohere embeddings client
            role: Optional role description
            device: Device to run on
        """
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.role = role or f"Agent {agent_id}"
        self.device = device

        # Neural components
        self.state_encoder = state_encoder
        self.ckm_model = ckm_model
        self.gap_detector = gap_detector
        self.policy_network = policy_network

        # LLM clients
        self.llm_client = llm_client
        self.embedding_client = embedding_client

        # Internal state
        self.current_state: Optional[AgentState] = None
        self.current_step = 0

        # CKM states for other agents
        self.ckm_states: Dict[int, CKMState] = {}
        self.ckm_hidden_states: Dict[int, torch.Tensor] = {}

        # Gap analyses
        self.gap_analyses: Dict[int, GapAnalysis] = {}

        # Message history
        self.messages_sent: List[Message] = []
        self.messages_received: List[Message] = []

        # Dialogue history (for CKM updates)
        self.dialogue_history: List[Dict] = []

        logger.info(f"Initialized Agent {agent_id} with role: {self.role}")

    async def process_query(self, query: str, context: Optional[str] = None) -> str:
        """
        Generate reasoning for a query

        Args:
            query: Task or query
            context: Optional context

        Returns:
            Reasoning text
        """
        if self.llm_client is None:
            logger.warning("No LLM client available")
            return f"[Agent {self.agent_id}] Processing: {query}"

        try:
            reasoning = await self.llm_client.generate_reasoning(
                query=query,
                context=context,
                agent_role=self.role,
            )
            logger.debug(f"Agent {self.agent_id} generated reasoning ({len(reasoning)} chars)")
            return reasoning

        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return f"[Agent {self.agent_id}] Error processing query"

    def encode_state(self, reasoning_text: str, query: str) -> AgentState:
        """
        Encode reasoning into state vector

        Args:
            reasoning_text: Agent's reasoning
            query: Current query

        Returns:
            AgentState object
        """
        # Get embedding
        if self.embedding_client:
            embedding = self.embedding_client.embed_for_reasoning(reasoning_text)
        else:
            # Fallback to random (for testing)
            embedding = torch.randn(768)

        # Encode to state
        embedding = embedding.unsqueeze(0).to(self.device)
        state_vector = self.state_encoder(embedding).squeeze(0)

        # Create state object
        state = AgentState(
            agent_id=self.agent_id,
            state_vector=state_vector,
            reasoning_trace=reasoning_text,
            query=query,
            step=self.current_step,
        )

        self.current_state = state
        logger.debug(f"Agent {self.agent_id} encoded state: {state_vector.shape}")

        return state

    def update_ckm(
        self,
        target_agent_id: int,
        messages: List[str],
    ) -> CKMState:
        """
        Update CKM for a specific agent based on their messages

        Args:
            target_agent_id: ID of agent being modeled
            messages: Recent messages from that agent

        Returns:
            Updated CKM state
        """
        if not messages:
            # Initialize with zero state if no messages
            ckm_vector = torch.zeros(self.ckm_model.output_dim, device=self.device)
            ckm_state = CKMState(
                source_agent_id=self.agent_id,
                target_agent_id=target_agent_id,
                cognitive_state=ckm_vector,
                last_update_step=self.current_step,
            )
            self.ckm_states[target_agent_id] = ckm_state
            return ckm_state

        # Embed messages
        if self.embedding_client:
            embeddings = torch.stack([
                self.embedding_client.embed_for_message(msg) for msg in messages
            ]).unsqueeze(0)  # (1 x seq_len x 768)
        else:
            # Fallback
            embeddings = torch.randn(1, len(messages), 768)

        embeddings = embeddings.to(self.device)

        # Get previous hidden state
        prev_hidden = self.ckm_hidden_states.get(target_agent_id)

        # Update CKM
        ckm_vector, hidden_state = self.ckm_model(embeddings, prev_hidden)

        # Store
        ckm_state = CKMState(
            source_agent_id=self.agent_id,
            target_agent_id=target_agent_id,
            cognitive_state=ckm_vector.squeeze(0),
            last_update_step=self.current_step,
        )

        self.ckm_states[target_agent_id] = ckm_state
        self.ckm_hidden_states[target_agent_id] = hidden_state

        logger.debug(f"Agent {self.agent_id} updated CKM for Agent {target_agent_id}")

        return ckm_state

    def detect_gap(self, target_agent_id: int) -> GapAnalysis:
        """
        Detect cognitive gap with a specific agent

        Args:
            target_agent_id: ID of collaborator

        Returns:
            Gap analysis
        """
        if self.current_state is None:
            raise ValueError("Agent state not initialized")

        ckm_state = self.ckm_states.get(target_agent_id)
        if ckm_state is None:
            raise ValueError(f"No CKM state for agent {target_agent_id}")

        # Detect gap
        gap_analysis = self.gap_detector.analyze_gap(
            own_state=self.current_state.state_vector.unsqueeze(0),
            ckm_state=ckm_state.cognitive_state.unsqueeze(0),
            source_agent_id=self.agent_id,
            target_agent_id=target_agent_id,
        )

        self.gap_analyses[target_agent_id] = gap_analysis

        logger.debug(
            f"Agent {self.agent_id} detected gap with Agent {target_agent_id}: "
            f"magnitude={gap_analysis.gap_magnitude:.3f}"
        )

        return gap_analysis

    def select_communication_action(self, deterministic: bool = False) -> CommunicationAction:
        """
        Select a communication action using policy network

        Args:
            deterministic: Use deterministic action selection

        Returns:
            Communication action
        """
        if self.current_state is None:
            raise ValueError("Agent state not initialized")

        # Prepare inputs for policy network
        own_state = self.current_state.state_vector.unsqueeze(0)

        # Get all CKM states
        ckm_list = []
        gap_list = []

        for other_id in range(self.num_agents):
            if other_id != self.agent_id:
                ckm = self.ckm_states.get(other_id)
                gap = self.gap_analyses.get(other_id)

                if ckm is not None:
                    ckm_list.append(ckm.cognitive_state)
                else:
                    ckm_list.append(torch.zeros(self.ckm_model.output_dim, device=self.device))

                if gap is not None:
                    gap_list.append(gap.gap_vector)
                else:
                    gap_list.append(torch.zeros(self.gap_detector.gap_dim, device=self.device))

        ckm_states = torch.stack(ckm_list).unsqueeze(0)  # (1 x (N-1) x ckm_dim)
        gap_vectors = torch.stack(gap_list).unsqueeze(0)  # (1 x (N-1) x gap_dim)

        # Select action
        action, info = self.policy_network.select_action(
            own_state, ckm_states, gap_vectors, deterministic=deterministic
        )

        action.source_agent_id = self.agent_id

        logger.debug(
            f"Agent {self.agent_id} selected action: {action.objective.value} "
            f"-> Agent {action.target_agent_id}"
        )

        return action

    async def generate_message(
        self,
        action: CommunicationAction,
    ) -> Message:
        """
        Generate message content for a communication action

        Args:
            action: Communication action

        Returns:
            Message object
        """
        if self.current_state is None:
            raise ValueError("Agent state not initialized")

        # Prepare gap info if available
        gap_info = None
        if action.target_agent_id >= 0 and action.target_agent_id in self.gap_analyses:
            gap_analysis = self.gap_analyses[action.target_agent_id]
            gap_info = gap_analysis.gap_types

        # Generate message using LLM
        if self.llm_client:
            try:
                message_content = await self.llm_client.generate_message(
                    objective=action.objective.value,
                    own_reasoning=self.current_state.reasoning_trace,
                    target_gap_info=str(gap_info) if gap_info else None,
                )
            except Exception as e:
                logger.error(f"Error generating message: {e}")
                message_content = f"[{action.objective.value}] {self.current_state.reasoning_trace[:100]}..."
        else:
            message_content = f"[{action.objective.value}] Message from Agent {self.agent_id}"

        # Create message
        message = Message(
            sender_id=self.agent_id,
            receiver_id=action.target_agent_id,
            content=message_content,
            step=self.current_step,
            objective=action.objective.value,
            metadata={"priority": action.priority, "confidence": action.confidence},
        )

        self.messages_sent.append(message)
        logger.debug(f"Agent {self.agent_id} generated message: {message_content[:100]}...")

        return message

    def receive_message(self, message: Message):
        """
        Receive and process a message

        Args:
            message: Incoming message
        """
        self.messages_received.append(message)
        self.dialogue_history.append({
            "sender": message.sender_id,
            "content": message.content,
            "step": message.step,
        })

        logger.debug(f"Agent {self.agent_id} received message from Agent {message.sender_id}")

    def increment_step(self):
        """Increment communication step"""
        self.current_step += 1

    def reset(self):
        """Reset agent state"""
        self.current_state = None
        self.current_step = 0
        self.ckm_states.clear()
        self.ckm_hidden_states.clear()
        self.gap_analyses.clear()
        self.messages_sent.clear()
        self.messages_received.clear()
        self.dialogue_history.clear()

        logger.info(f"Agent {self.agent_id} reset")

    def get_state_summary(self) -> Dict:
        """Get summary of agent's current state"""
        return {
            "agent_id": self.agent_id,
            "step": self.current_step,
            "has_state": self.current_state is not None,
            "num_ckms": len(self.ckm_states),
            "num_gaps": len(self.gap_analyses),
            "messages_sent": len(self.messages_sent),
            "messages_received": len(self.messages_received),
        }

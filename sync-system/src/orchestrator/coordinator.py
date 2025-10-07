import asyncio
from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime

from src.core.agent import Agent, Message
from src.core.state import StateEncoder, StateManager
from src.core.ckm import CollaboratorKnowledgeModel, CKMManager
from src.core.gap_detector import GapDetector, GapManager
from src.core.policy import CommunicationPolicyNetwork
from src.llm.openrouter import OpenRouterClient
from src.llm.embeddings import CohereEmbeddingsClient
from src.orchestrator.communication import CommunicationChannel
from src.orchestrator.convergence import ConvergenceDetector, ConvergenceMetrics
from src.orchestrator.aggregator import ResponseAggregator, AgregatedResponse
from src.utils.logging import get_logger

logger = get_logger("coordinator")


@dataclass
class CollaborationResult:
    query: str
    final_response: AgregatedResponse
    convergence_metrics: List[ConvergenceMetrics]
    total_rounds: int
    total_messages: int
    dialogue_history: str
    computation_time: float
    success: bool


class MultiAgentCoordinator:
    def __init__(
        self,
        num_agents: int = 3,
        max_rounds: int = 5,
        device: str = "cpu",
    ):

        self.num_agents = num_agents
        self.max_rounds = max_rounds
        self.device = device

        logger.info("Initializing shared neural components...")

        self.state_encoder = StateEncoder(
            embedding_dim=1024, 
            state_dim=256,
        ).to(device)

        self.ckm_model = CollaboratorKnowledgeModel(
            input_dim=1024,
            transformer_hidden_dim=512,
            output_dim=128,
        ).to(device)

        self.gap_detector = GapDetector(
            state_dim=256,
            ckm_dim=128,
            gap_dim=64,
        ).to(device)

        self.policy_network = CommunicationPolicyNetwork(
            state_dim=256,
            ckm_dim=128,
            gap_dim=64,
            num_agents=num_agents,
        ).to(device)

        self.llm_client = OpenRouterClient()
        self.embedding_client = CohereEmbeddingsClient()

        self.communication_channel = CommunicationChannel(num_agents)
        self.convergence_detector = ConvergenceDetector(max_rounds=max_rounds)
        self.aggregator = ResponseAggregator(self.llm_client)

        self.state_manager = StateManager(num_agents, state_dim=256)
        self.ckm_manager = CKMManager(num_agents, ckm_dim=128, device=device)
        self.gap_manager = GapManager(num_agents, gap_dim=64, device=device)

        self.agents: List[Agent] = []
        self._create_agents()

        logger.info(f"Initialized MultiAgentCoordinator with {num_agents} agents")

    def _create_agents(self):
        agent_roles = [
            "Analytical Reasoner",
            "Creative Thinker",
            "Critical Evaluator",
            "Practical Synthesizer",
            "Detail-Oriented Specialist",
            "Big-Picture Strategist",
        ]

        for i in range(self.num_agents):
            role = agent_roles[i] if i < len(agent_roles) else f"Agent {i}"

            agent = Agent(
                agent_id=i,
                num_agents=self.num_agents,
                state_encoder=self.state_encoder,
                ckm_model=self.ckm_model,
                gap_detector=self.gap_detector,
                policy_network=self.policy_network,
                llm_client=self.llm_client,
                embedding_client=self.embedding_client,
                role=role,
                device=self.device,
            )

            self.agents.append(agent)
            logger.info(f"Created {role} (Agent {i})")

    async def collaborate(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> CollaborationResult:
        start_time = datetime.now()
        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING MULTI-AGENT COLLABORATION")
        logger.info(f"Query: {query}")
        logger.info(f"Agents: {self.num_agents}")
        logger.info(f"{'='*80}\n")

        try:
            logger.info("PHASE 1: Initial agent reasoning...")
            await self._initial_reasoning_phase(query, context)

            logger.info("PHASE 2: Multi-round collaboration...")
            convergence_metrics = await self._collaboration_rounds()

            logger.info("PHASE 3: Response aggregation...")
            final_response = await self._aggregation_phase(query)

            computation_time = (datetime.now() - start_time).total_seconds()
            dialogue_history = self.communication_channel.format_dialogue_history()
            stats = self.communication_channel.get_statistics()

            result = CollaborationResult(
                query=query,
                final_response=final_response,
                convergence_metrics=convergence_metrics,
                total_rounds=stats["total_rounds"],
                total_messages=stats["total_messages"],
                dialogue_history=dialogue_history,
                computation_time=computation_time,
                success=True,
            )

            logger.info(f"\n{'='*80}")
            logger.info(f"COLLABORATION COMPLETE!")
            logger.info(f"Rounds: {result.total_rounds}")
            logger.info(f"Messages: {result.total_messages}")
            logger.info(f"Time: {computation_time:.2f}s")
            logger.info(f"Confidence: {final_response.confidence:.2%}")
            logger.info(f"Consensus: {final_response.consensus_level:.2%}")
            logger.info(f"{'='*80}\n")

            return result

        except Exception as e:
            logger.error(f"Collaboration failed: {e}", exc_info=True)
            raise

    async def _initial_reasoning_phase(
        self,
        query: str,
        context: Optional[str] = None,
    ):
        tasks = []

        for agent in self.agents:
            task = self._agent_initial_reasoning(agent, query, context)
            tasks.append(task)

        await asyncio.gather(*tasks)

        logger.info(f"All {self.num_agents} agents completed initial reasoning")

    async def _agent_initial_reasoning(
        self,
        agent: Agent,
        query: str,
        context: Optional[str] = None,
    ):
        reasoning = await agent.process_query(query, context)

        state = agent.encode_state(reasoning, query)
        self.state_manager.update_state(state)

        logger.debug(f"Agent {agent.agent_id} completed initial reasoning")

    async def _collaboration_rounds(self) -> List[ConvergenceMetrics]:
        convergence_metrics = []
        round_num = 0

        while round_num < self.max_rounds:
            logger.info(f"\n--- ROUND {round_num} ---")

            await self._update_all_ckms()

            self._detect_all_gaps()

            actions = self._select_all_actions()

            await self._generate_and_send_messages(actions, round_num)

            self._deliver_messages()

            self.communication_channel.complete_round(round_num)

            metrics = self._check_convergence(round_num)
            convergence_metrics.append(metrics)

            logger.info(
                f"Round {round_num}: "
                f"Score={metrics.convergence_score:.3f}, "
                f"Converged={metrics.is_converged} ({metrics.reason})"
            )

            if metrics.is_converged:
                logger.info(f"Converged after {round_num + 1} rounds!")
                break

            round_num += 1

        return convergence_metrics

    async def _update_all_ckms(self):
        for agent in self.agents:
            for other_agent in self.agents:
                if agent.agent_id != other_agent.agent_id:
                    messages_from_other = [
                        msg.content for msg in self.communication_channel.get_messages_from_agent(other_agent.agent_id)
                    ]

                    if messages_from_other:
                        ckm_state = agent.update_ckm(
                            target_agent_id=other_agent.agent_id,
                            messages=messages_from_other[-5:], 
                        )

                        self.ckm_manager.update_ckm(
                            source_agent_id=agent.agent_id,
                            target_agent_id=other_agent.agent_id,
                            new_ckm_vector=ckm_state.cognitive_state,
                            step=agent.current_step,
                        )

    def _detect_all_gaps(self):
        for agent in self.agents:
            for other_agent in self.agents:
                if agent.agent_id != other_agent.agent_id:
                    if other_agent.agent_id in agent.ckm_states:
                        gap_analysis = agent.detect_gap(other_agent.agent_id)

                        self.gap_manager.store_gap_analysis(gap_analysis)

    def _select_all_actions(self) -> List:
        actions = []

        for agent in self.agents:
            action = agent.select_communication_action(deterministic=False)
            actions.append((agent, action))

        return actions

    async def _generate_and_send_messages(self, actions, round_num):
        tasks = []

        for agent, action in actions:
            task = self._agent_generate_message(agent, action, round_num)
            tasks.append(task)

        messages = await asyncio.gather(*tasks)

        for message in messages:
            if message:
                self.communication_channel.send_message(message)

    async def _agent_generate_message(self, agent, action, round_num):
        try:
            message = await agent.generate_message(action)
            message.step = round_num
            return message
        except Exception as e:
            logger.error(f"Agent {agent.agent_id} message generation failed: {e}")
            return None

    def _deliver_messages(self):
        for agent in self.agents:
            messages = self.communication_channel.get_messages_for_agent(
                agent.agent_id, clear=True
            )

            for message in messages:
                agent.receive_message(message)

    def _check_convergence(self, round_num) -> ConvergenceMetrics:
        agent_states = [
            self.state_manager.get_state(i) for i in range(self.num_agents)
        ]

        gap_magnitudes = [
            gap.gap_magnitude
            for gap in self.gap_manager.gap_analyses.values()
        ]

        messages = [
            msg for msg in self.communication_channel.all_messages
            if msg.step == round_num
        ]

        metrics = self.convergence_detector.check_convergence(
            agent_states=agent_states,
            gap_magnitudes=gap_magnitudes,
            messages=messages,
            round_number=round_num,
        )

        return metrics

    async def _aggregation_phase(self, query: str) -> AgregatedResponse:
        agent_states = [
            self.state_manager.get_state(i) for i in range(self.num_agents)
        ]

        dialogue_history = self.communication_channel.format_dialogue_history(
            max_chars_per_message=200
        )

        final_response = await self.aggregator.aggregate(
            query=query,
            agent_states=agent_states,
            messages=self.communication_channel.all_messages,
            dialogue_history=dialogue_history,
        )

        return final_response

    def reset(self):
        self.communication_channel.reset()
        self.convergence_detector.reset()
        self.state_manager.reset()
        self.ckm_manager.reset()
        self.gap_manager.reset()

        for agent in self.agents:
            agent.reset()

        logger.info("Reset coordinator")

    async def close(self):
        await self.llm_client.close()
        await self.aggregator.close()
        logger.info("Closed coordinator")

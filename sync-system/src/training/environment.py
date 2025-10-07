from typing import Dict, List, Optional, Tuple, Any
import asyncio

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from src.core.agent import Agent, Message
from src.core.state import AgentState, StateManager
from src.core.ckm import CKMManager
from src.core.gap_detector import GapManager
from src.orchestrator.coordinator import MultiAgentCoordinator
from src.data.datasets import TrainingExample
from src.training.rewards import RewardComputer, RewardComponents
from src.utils.logging import get_logger
from config.training import RLTrainingConfig

logger = get_logger("training.environment")


class MultiAgentCollaborationEnv(gym.Env):

    def __init__(
        self,
        num_agents: int = 3,
        max_rounds: int = 5,
        dataset: Optional[List[TrainingExample]] = None,
        config: Optional[RLTrainingConfig] = None,
        device: str = "cpu",
    ):
        super().__init__()

        self.num_agents = num_agents
        self.max_rounds = max_rounds
        self.dataset = dataset or []
        self.config = config or RLTrainingConfig()
        self.device = device

        # Define spaces
        self._define_spaces()

        # Create coordinator (will be used for episodes)
        self.coordinator = None
        self.reward_computer = RewardComputer(self.config.reward_config)

        # Episode state
        self.current_task = None
        self.current_round = 0
        self.episode_messages = []
        self.episode_gaps = []
        self.episode_tokens = 0

        logger.info(
            f"Initialized MultiAgentCollaborationEnv: "
            f"{num_agents} agents, {max_rounds} max rounds"
        )

    def _define_spaces(self):
        """Define observation and action spaces"""
        # Observation space: [own_state, all_CKMs, all_gaps]
        state_dim = 256
        ckm_dim = 128
        gap_dim = 64

        obs_dim = state_dim + (self.num_agents - 1) * ckm_dim + (self.num_agents - 1) * gap_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action space: [objective, target_agent, priority]
        # For Stable-Baselines3, we use a Dict space
        self.action_space = spaces.Dict({
            "objective": spaces.Discrete(10),  # 10 communication objectives
            "target": spaces.Discrete(self.num_agents),  # N agents (includes broadcast)
            "priority": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })

        logger.debug(
            f"Observation space: {obs_dim}-dim, "
            f"Action space: objective(10) x target({self.num_agents}) x priority(1)"
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment for new episode

        Returns:
            (observation, info)
        """
        super().reset(seed=seed)

        # Sample new task
        if self.dataset:
            self.current_task = self.dataset[self.np_random.integers(0, len(self.dataset))]
        else:
            # Use default task
            from src.data.datasets import TrainingExample
            self.current_task = TrainingExample(
                task_id="default",
                query="What is 2+2?",
                difficulty="easy"
            )

        # Reset episode state
        self.current_round = 0
        self.episode_messages = []
        self.episode_gaps = []
        self.episode_tokens = 0

        # Create new coordinator
        if self.coordinator:
            try:
                asyncio.get_event_loop().run_until_complete(self.coordinator.close())
            except:
                pass

        self.coordinator = MultiAgentCoordinator(
            num_agents=self.num_agents,
            max_rounds=self.max_rounds,
            device=self.device
        )

        # Run initial reasoning phase (synchronous wrapper)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            self.coordinator._initial_reasoning_phase(
                self.current_task.query,
                self.current_task.context
            )
        )

        # Get initial observation
        observation = self._get_observation(agent_id=0)  # For agent 0
        info = {
            "task_id": self.current_task.task_id,
            "query": self.current_task.query,
            "round": self.current_round
        }

        return observation, info

    def step(
        self,
        action: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment

        Args:
            action: Dict with objective, target, priority

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Parse action
        objective_idx = int(action["objective"])
        target_idx = int(action["target"])
        priority = float(action["priority"][0])

        # Map to communication action
        from config.models import CommunicationObjective
        objectives = list(CommunicationObjective)
        objective = objectives[objective_idx] if objective_idx < len(objectives) else objectives[0]

        # Generate and send message (for agent 0)
        agent = self.coordinator.agents[0]

        # Create communication action
        from src.core.policy import CommunicationAction
        comm_action = CommunicationAction(
            objective=objective,
            target_agent_id=target_idx - 1 if target_idx > 0 else -1,  # -1 for broadcast
            priority=priority,
            metadata={}
        )

        # Generate message (synchronous wrapper)
        loop = asyncio.get_event_loop()
        try:
            message = loop.run_until_complete(agent.generate_message(comm_action))
            self.coordinator.communication_channel.send_message(message)
            self.episode_messages.append(message)
            self.episode_tokens += len(message.content.split()) * 2  # Rough estimate
        except Exception as e:
            logger.error(f"Error generating message: {e}")

        # Advance round
        self.current_round += 1

        # Update CKMs and detect gaps
        loop.run_until_complete(self.coordinator._update_all_ckms())
        self.coordinator._detect_all_gaps()

        # Store gaps
        for gap_analysis in self.coordinator.gap_manager.gap_analyses.values():
            self.episode_gaps.append(gap_analysis)

        # Check if episode done
        terminated = False
        truncated = False

        if self.current_round >= self.max_rounds:
            truncated = True
        else:
            # Check convergence
            metrics = self.coordinator._check_convergence(self.current_round)
            if metrics.is_converged:
                terminated = True

        # Compute reward
        if terminated or truncated:
            # Episode done - compute final reward
            loop = asyncio.get_event_loop()
            final_response = loop.run_until_complete(
                self.coordinator._aggregation_phase(self.current_task.query)
            )

            agent_states = [
                self.coordinator.state_manager.get_state(i)
                for i in range(self.num_agents)
            ]

            reward_components = self.reward_computer.compute_reward(
                query=self.current_task.query,
                final_response=final_response,
                agent_states=agent_states,
                messages=self.episode_messages,
                gap_analyses=self.episode_gaps,
                rounds_taken=self.current_round,
                tokens_used=self.episode_tokens,
                reference_answer=self.current_task.reference_answer
            )

            reward = reward_components.total_reward
        else:
            # Intermediate step - small shaped reward
            reward = self.reward_computer.compute_shaped_reward(
                "message_sent",
                gap_magnitude=self.episode_gaps[-1].gap_magnitude if self.episode_gaps else 0.0
            )

        # Get next observation
        observation = self._get_observation(agent_id=0)

        # Info
        info = {
            "round": self.current_round,
            "messages_sent": len(self.episode_messages),
            "tokens_used": self.episode_tokens
        }

        if terminated or truncated:
            info["episode_reward"] = reward
            info["final_answer"] = final_response.final_answer if 'final_response' in locals() else ""

        return observation, reward, terminated, truncated, info

    def _get_observation(self, agent_id: int) -> np.ndarray:
        """
        Get observation for an agent

        Args:
            agent_id: Agent ID

        Returns:
            Observation vector
        """
        agent = self.coordinator.agents[agent_id]

        # Own state (256)
        own_state = agent.current_state.state_vector.cpu().numpy()

        # All CKMs for other agents (N-1 x 128)
        ckm_states = []
        for other_id in range(self.num_agents):
            if other_id != agent_id:
                if other_id in agent.ckm_states:
                    ckm_state = agent.ckm_states[other_id].cognitive_state.cpu().numpy()
                else:
                    ckm_state = np.zeros(128)
                ckm_states.append(ckm_state)
        ckm_states = np.concatenate(ckm_states)

        # All gaps for other agents (N-1 x 64)
        gap_states = []
        for other_id in range(self.num_agents):
            if other_id != agent_id:
                key = (agent_id, other_id)
                if key in self.coordinator.gap_manager.gap_analyses:
                    gap_analysis = self.coordinator.gap_manager.gap_analyses[key]
                    gap_state = gap_analysis.gap_vector.cpu().numpy()
                else:
                    gap_state = np.zeros(64)
                gap_states.append(gap_state)
        gap_states = np.concatenate(gap_states)

        # Concatenate
        observation = np.concatenate([own_state, ckm_states, gap_states])

        return observation.astype(np.float32)

    def close(self):
        """Close environment"""
        if self.coordinator:
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.coordinator.close())
            except:
                pass
        logger.info("Environment closed")


def make_env(
    num_agents: int = 3,
    max_rounds: int = 5,
    dataset: Optional[List[TrainingExample]] = None,
    config: Optional[RLTrainingConfig] = None,
    device: str = "cpu",
) -> MultiAgentCollaborationEnv:
    """
    Factory function to create environment

    Args:
        num_agents: Number of agents
        max_rounds: Max rounds
        dataset: Training dataset
        config: RL config
        device: Device

    Returns:
        Environment instance
    """
    return MultiAgentCollaborationEnv(
        num_agents=num_agents,
        max_rounds=max_rounds,
        dataset=dataset,
        config=config,
        device=device
    )


if __name__ == "__main__":
    # Test environment
    print("=" * 80)
    print("TESTING GYM ENVIRONMENT")
    print("=" * 80)

    # Create environment
    print("\n[1/3] Creating environment...")
    env = make_env(num_agents=3, max_rounds=3)
    print(f"[OK] Environment created")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space}")

    # Reset
    print("\n[2/3] Resetting environment...")
    obs, info = env.reset()
    print(f"[OK] Environment reset")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Info: {info}")

    # Sample action and step
    print("\n[3/3] Taking step...")
    action = env.action_space.sample()
    print(f"  Sampled action: objective={action['objective']}, target={action['target']}")

    # NOTE: This will make real API calls - skip for now
    print("[SKIPPED] Step would make real API calls")

    # Close
    env.close()
    print("\n[OK] Gym environment working!")

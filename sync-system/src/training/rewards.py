from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

import torch
import numpy as np

from src.core.agent import Message
from src.core.state import AgentState
from src.core.gap_detector import GapAnalysis
from src.orchestrator.aggregator import AgregatedResponse
from src.utils.logging import get_logger
from config.training import RewardConfig

logger = get_logger("training.rewards")


@dataclass
class RewardComponents:
    task_success: float
    token_penalty: float
    gap_reduction: float
    communication_quality: float
    convergence_bonus: float
    total_reward: float

    tokens_used: int
    rounds_taken: int
    messages_sent: int
    final_gap_magnitude: float


class RewardComputer:

    def __init__(self, config: Optional[RewardConfig] = None):
        """
        Initialize reward computer

        Args:
            config: Reward configuration
        """
        self.config = config or RewardConfig()
        logger.info(f"Initialized RewardComputer with weights: {self.config}")

    def compute_reward(
        self,
        query: str,
        final_response: AgregatedResponse,
        agent_states: List[AgentState],
        messages: List[Message],
        gap_analyses: List[GapAnalysis],
        rounds_taken: int,
        tokens_used: int,
        reference_answer: Optional[str] = None,
    ) -> RewardComponents:
        """
        Compute total reward for a collaboration episode

        Args:
            query: Original task query
            final_response: Aggregated response from agents
            agent_states: Final states of all agents
            messages: All messages exchanged
            gap_analyses: All gap analyses
            rounds_taken: Number of collaboration rounds
            tokens_used: Total tokens used
            reference_answer: Optional ground truth answer

        Returns:
            RewardComponents with breakdown
        """
        # 1. Task Success Reward
        task_success = self._compute_task_success(
            final_response,
            reference_answer,
            query
        )

        # 2. Token Penalty (negative reward for cost)
        token_penalty = -1.0 * self.config.token_penalty * tokens_used

        # 3. Gap Reduction Reward
        gap_reduction = self._compute_gap_reduction(gap_analyses)

        # 4. Communication Quality Reward
        comm_quality = self._compute_communication_quality(messages)

        # 5. Convergence Bonus (bonus for fast convergence)
        convergence_bonus = self._compute_convergence_bonus(
            rounds_taken,
            final_response.consensus_level
        )

        # Total Reward
        total_reward = (
            self.config.task_success_weight * task_success +
            token_penalty +
            self.config.gap_reduction_weight * gap_reduction +
            self.config.communication_quality_weight * comm_quality +
            self.config.convergence_bonus_weight * convergence_bonus
        )

        # Get final gap magnitude
        final_gap_magnitude = (
            np.mean([gap.gap_magnitude for gap in gap_analyses])
            if gap_analyses else 0.0
        )

        components = RewardComponents(
            task_success=task_success,
            token_penalty=token_penalty,
            gap_reduction=gap_reduction,
            communication_quality=comm_quality,
            convergence_bonus=convergence_bonus,
            total_reward=total_reward,
            tokens_used=tokens_used,
            rounds_taken=rounds_taken,
            messages_sent=len(messages),
            final_gap_magnitude=final_gap_magnitude
        )

        logger.debug(
            f"Reward: {total_reward:.3f} = "
            f"Task({task_success:.2f}) + "
            f"Tokens({token_penalty:.2f}) + "
            f"Gap({gap_reduction:.2f}) + "
            f"Comm({comm_quality:.2f}) + "
            f"Conv({convergence_bonus:.2f})"
        )

        return components

    def _compute_task_success(
        self,
        final_response: AgregatedResponse,
        reference_answer: Optional[str],
        query: str
    ) -> float:
        """
        Compute task success reward

        Uses multiple signals:
        - Reference answer match (if available)
        - Confidence score
        - Consensus level
        - Answer quality heuristics

        Returns:
            Reward in [0, 1]
        """
        # Base score from confidence and consensus
        base_score = (final_response.confidence + final_response.consensus_level) / 2.0

        # If reference answer available, check match
        if reference_answer:
            match_score = self._compute_answer_match(
                final_response.final_answer,
                reference_answer
            )
            # Weighted combination
            success_score = 0.6 * match_score + 0.4 * base_score
        else:
            # Use heuristics
            quality_score = self._compute_answer_quality(
                final_response.final_answer,
                query
            )
            success_score = 0.7 * base_score + 0.3 * quality_score

        return float(np.clip(success_score, 0.0, 1.0))

    def _compute_answer_match(self, answer: str, reference: str) -> float:
        """
        Compute how well answer matches reference

        Uses fuzzy matching for flexibility.

        Returns:
            Match score in [0, 1]
        """
        # Normalize
        answer_norm = answer.lower().strip()
        reference_norm = reference.lower().strip()

        # Exact match
        if answer_norm == reference_norm:
            return 1.0

        # Substring match
        if reference_norm in answer_norm or answer_norm in reference_norm:
            return 0.8

        # Token overlap
        answer_tokens = set(answer_norm.split())
        reference_tokens = set(reference_norm.split())

        if len(reference_tokens) == 0:
            return 0.5

        overlap = len(answer_tokens & reference_tokens) / len(reference_tokens)

        return float(overlap)

    def _compute_answer_quality(self, answer: str, query: str) -> float:
        """
        Compute answer quality using heuristics

        Returns:
            Quality score in [0, 1]
        """
        score = 0.5  # Base score

        # Length heuristic (not too short, not too long)
        answer_length = len(answer.split())
        if 20 <= answer_length <= 500:
            score += 0.2
        elif 10 <= answer_length <= 1000:
            score += 0.1

        # Structure heuristic (has punctuation, capitalization)
        if answer[0].isupper() and answer.endswith(('.', '!', '?')):
            score += 0.1

        # Content heuristic (mentions keywords from query)
        query_keywords = set(
            word.lower() for word in query.split()
            if len(word) > 3 and word.isalnum()
        )
        answer_words = set(
            word.lower() for word in re.findall(r'\w+', answer)
        )
        keyword_overlap = len(query_keywords & answer_words) / max(len(query_keywords), 1)
        score += 0.2 * keyword_overlap

        return float(np.clip(score, 0.0, 1.0))

    def _compute_gap_reduction(self, gap_analyses: List[GapAnalysis]) -> float:
        """
        Compute gap reduction reward

        Reward agents for reducing cognitive gaps over time.

        Returns:
            Reward (higher is better)
        """
        if not gap_analyses or len(gap_analyses) < 2:
            return 0.0

        # Get gap magnitudes over time
        gap_magnitudes = [gap.gap_magnitude for gap in gap_analyses]

        # Compute reduction (initial - final)
        initial_gap = np.mean(gap_magnitudes[:len(gap_magnitudes)//3])  # First third
        final_gap = np.mean(gap_magnitudes[-len(gap_magnitudes)//3:])  # Last third

        reduction = initial_gap - final_gap

        # Normalize to [0, 1]
        # Positive reduction is good
        normalized_reduction = np.clip(reduction, 0.0, 1.0)

        return float(normalized_reduction)

    def _compute_communication_quality(self, messages: List[Message]) -> float:
        """
        Compute communication quality reward

        Rewards:
        - Relevant messages
        - Non-redundant messages
        - Appropriate message objectives

        Returns:
            Reward in [0, 1]
        """
        if not messages:
            return 0.0

        scores = []

        # Message diversity (not all the same objective)
        objectives = [msg.objective for msg in messages]
        unique_objectives = len(set(objectives))
        diversity_score = min(unique_objectives / len(objectives), 1.0)
        scores.append(diversity_score)

        # Message length distribution (not too short, not too long)
        lengths = [len(msg.content.split()) for msg in messages]
        avg_length = np.mean(lengths)
        if 10 <= avg_length <= 100:
            length_score = 1.0
        elif 5 <= avg_length <= 200:
            length_score = 0.7
        else:
            length_score = 0.5
        scores.append(length_score)

        # Message relevance (heuristic: messages should have substantive content)
        substantive_count = sum(1 for msg in messages if len(msg.content.split()) >= 5)
        relevance_score = substantive_count / len(messages)
        scores.append(relevance_score)

        return float(np.mean(scores))

    def _compute_convergence_bonus(
        self,
        rounds_taken: int,
        consensus_level: float
    ) -> float:
        """
        Compute convergence bonus

        Bonus for reaching high consensus in fewer rounds.

        Returns:
            Bonus reward
        """
        # Target is high consensus in few rounds
        if consensus_level >= 0.85:
            # Bonus decreases with rounds taken
            if rounds_taken <= 3:
                bonus = 0.5
            elif rounds_taken <= 5:
                bonus = 0.3
            else:
                bonus = 0.1
        else:
            bonus = 0.0

        return float(bonus)

    def compute_shaped_reward(
        self,
        step_type: str,
        **kwargs
    ) -> float:
        """
        Compute shaped reward for intermediate steps

        Allows reward shaping for better learning.

        Args:
            step_type: Type of step (gap_detected, message_sent, etc.)
            **kwargs: Step-specific data

        Returns:
            Shaped reward
        """
        if step_type == "gap_detected":
            # Small penalty for detecting large gaps (encourages alignment)
            gap_magnitude = kwargs.get("gap_magnitude", 0.0)
            return -0.01 * gap_magnitude

        elif step_type == "message_sent":
            # Small reward for sending messages (encourages communication)
            return 0.01

        elif step_type == "convergence_improved":
            # Reward for improving convergence
            improvement = kwargs.get("improvement", 0.0)
            return 0.05 * improvement

        else:
            return 0.0


def compute_episode_reward(
    query: str,
    final_response: AgregatedResponse,
    agent_states: List[AgentState],
    messages: List[Message],
    gap_analyses: List[GapAnalysis],
    rounds_taken: int,
    tokens_used: int,
    reference_answer: Optional[str] = None,
    config: Optional[RewardConfig] = None,
) -> RewardComponents:
    """
    Convenience function to compute episode reward

    Args:
        query: Task query
        final_response: Aggregated response
        agent_states: Final agent states
        messages: All messages
        gap_analyses: All gap analyses
        rounds_taken: Number of rounds
        tokens_used: Total tokens
        reference_answer: Optional reference
        config: Reward config

    Returns:
        RewardComponents
    """
    computer = RewardComputer(config)
    return computer.compute_reward(
        query=query,
        final_response=final_response,
        agent_states=agent_states,
        messages=messages,
        gap_analyses=gap_analyses,
        rounds_taken=rounds_taken,
        tokens_used=tokens_used,
        reference_answer=reference_answer
    )


if __name__ == "__main__":
    # Test reward computation
    print("=" * 80)
    print("TESTING REWARD COMPUTATION")
    print("=" * 80)

    # Mock data
    from src.orchestrator.aggregator import AgregatedResponse

    mock_response = AgregatedResponse(
        final_answer="Machine learning is a subset of AI...",
        confidence=0.9,
        consensus_level=0.85,
        reasoning_summary="Agents agreed on definition",
        conflicts=[],
        agent_contributions={},
        metadata={}
    )

    mock_messages = [
        Message(0, 1, "Let's discuss ML", 0, "request_clarification"),
        Message(1, 0, "I think it's about learning from data", 0, "propose_refinement"),
        Message(0, -1, "I agree with that definition", 1, "signal_agreement"),
    ]

    # Compute reward
    print("\n[1/1] Computing reward...")
    reward_computer = RewardComputer()

    rewards = reward_computer.compute_reward(
        query="What is machine learning?",
        final_response=mock_response,
        agent_states=[],
        messages=mock_messages,
        gap_analyses=[],
        rounds_taken=2,
        tokens_used=500,
        reference_answer="Machine learning is a subset of AI"
    )

    print(f"\n[OK] Reward computed:")
    print(f"  Task Success: {rewards.task_success:.3f}")
    print(f"  Token Penalty: {rewards.token_penalty:.3f}")
    print(f"  Gap Reduction: {rewards.gap_reduction:.3f}")
    print(f"  Comm Quality: {rewards.communication_quality:.3f}")
    print(f"  Convergence Bonus: {rewards.convergence_bonus:.3f}")
    print(f"  TOTAL REWARD: {rewards.total_reward:.3f}")

    print("\n[OK] Reward computation working!")

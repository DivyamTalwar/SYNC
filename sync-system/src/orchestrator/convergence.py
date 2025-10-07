import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from src.core.state import AgentState, compute_state_similarity
from src.core.agent import Message
from src.utils.logging import get_logger

logger = get_logger("convergence")


@dataclass
class ConvergenceMetrics:
    round_number: int
    state_similarity_avg: float
    gap_magnitude_avg: float
    gap_reduction_rate: float
    message_similarity_avg: float
    agreement_signals: int
    total_messages: int
    is_converged: bool
    convergence_score: float
    reason: str


class ConvergenceDetector:

    def __init__(
        self,
        state_similarity_threshold: float = 0.9,
        gap_magnitude_threshold: float = 0.5,
        gap_reduction_threshold: float = 0.05,
        min_rounds: int = 2,
        max_rounds: int = 5,
    ):
        self.state_similarity_threshold = state_similarity_threshold
        self.gap_magnitude_threshold = gap_magnitude_threshold
        self.gap_reduction_threshold = gap_reduction_threshold
        self.min_rounds = min_rounds
        self.max_rounds = max_rounds

        self.metrics_history: List[ConvergenceMetrics] = []
        self.gap_history: List[float] = []

        logger.info(
            f"Initialized ConvergenceDetector: "
            f"similarity>{state_similarity_threshold}, "
            f"gap<{gap_magnitude_threshold}, "
            f"max_rounds={max_rounds}"
        )

    def check_convergence(
        self,
        agent_states: List[AgentState],
        gap_magnitudes: List[float],
        messages: List[Message],
        round_number: int,
    ) -> ConvergenceMetrics:
        state_similarity = self._compute_state_similarity(agent_states)

        gap_magnitude_avg = sum(gap_magnitudes) / len(gap_magnitudes) if gap_magnitudes else 0.0
        gap_reduction_rate = self._compute_gap_reduction(gap_magnitude_avg)

        message_similarity = self._analyze_message_similarity(messages)
        agreement_signals = self._count_agreement_signals(messages)

        convergence_score = self._compute_convergence_score(
            state_similarity=state_similarity,
            gap_magnitude=gap_magnitude_avg,
            gap_reduction=gap_reduction_rate,
            agreement_rate=agreement_signals / len(messages) if messages else 0.0,
        )

        is_converged, reason = self._determine_convergence(
            round_number=round_number,
            state_similarity=state_similarity,
            gap_magnitude=gap_magnitude_avg,
            gap_reduction=gap_reduction_rate,
            convergence_score=convergence_score,
        )

        metrics = ConvergenceMetrics(
            round_number=round_number,
            state_similarity_avg=state_similarity,
            gap_magnitude_avg=gap_magnitude_avg,
            gap_reduction_rate=gap_reduction_rate,
            message_similarity_avg=message_similarity,
            agreement_signals=agreement_signals,
            total_messages=len(messages),
            is_converged=is_converged,
            convergence_score=convergence_score,
            reason=reason,
        )

        self.metrics_history.append(metrics)
        self.gap_history.append(gap_magnitude_avg)

        logger.info(
            f"Round {round_number} convergence: "
            f"score={convergence_score:.3f}, "
            f"similarity={state_similarity:.3f}, "
            f"gap={gap_magnitude_avg:.3f}, "
            f"converged={is_converged} ({reason})"
        )

        return metrics

    def _compute_state_similarity(self, states: List[AgentState]) -> float:
        if len(states) < 2:
            return 1.0

        similarities = []
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                sim = compute_state_similarity(
                    states[i].state_vector,
                    states[j].state_vector
                )
                similarities.append(sim.item())

        return sum(similarities) / len(similarities) if similarities else 0.0

    def _compute_gap_reduction(self, current_gap: float) -> float:
        if len(self.gap_history) == 0:
            return 1.0  

        previous_gap = self.gap_history[-1]

        if previous_gap == 0:
            return 0.0

        reduction = (previous_gap - current_gap) / previous_gap
        return max(0.0, reduction) 

    def _analyze_message_similarity(self, messages: List[Message]) -> float:
        if len(messages) < 2:
            return 1.0

        all_words = set()
        common_words = set()

        for msg in messages:
            words = set(msg.content.lower().split())
            if not all_words:
                all_words = words
            else:
                common_words.update(all_words.intersection(words))
                all_words.update(words)

        if not all_words:
            return 0.0

        similarity = len(common_words) / len(all_words)
        return similarity

    def _count_agreement_signals(self, messages: List[Message]) -> int:
        agreement_keywords = [
            "agree", "confirm", "correct", "yes", "exactly",
            "consensus", "aligned", "same", "match"
        ]

        agreement_objectives = [
            "signal_agreement", "confirm_understanding", "synthesize_perspectives"
        ]

        count = 0

        for msg in messages:
            if msg.objective in agreement_objectives:
                count += 1

            content_lower = msg.content.lower()
            if any(keyword in content_lower for keyword in agreement_keywords):
                count += 1

        return count

    def _compute_convergence_score(
        self,
        state_similarity: float,
        gap_magnitude: float,
        gap_reduction: float,
        agreement_rate: float,
    ) -> float:
        normalized_gap = max(0.0, 1.0 - gap_magnitude)

        weights = {
            "state_similarity": 0.35,
            "gap": 0.35,
            "gap_reduction": 0.15,
            "agreement": 0.15,
        }

        score = (
            weights["state_similarity"] * state_similarity +
            weights["gap"] * normalized_gap +
            weights["gap_reduction"] * gap_reduction +
            weights["agreement"] * agreement_rate
        )

        return min(1.0, max(0.0, score))

    def _determine_convergence(
        self,
        round_number: int,
        state_similarity: float,
        gap_magnitude: float,
        gap_reduction: float,
        convergence_score: float,
    ) -> Tuple[bool, str]:
        if round_number >= self.max_rounds:
            return True, f"max_rounds_reached ({self.max_rounds})"

        if round_number < self.min_rounds:
            return False, f"min_rounds_not_met ({round_number}/{self.min_rounds})"

        if convergence_score >= 0.85:
            return True, f"high_convergence_score ({convergence_score:.3f})"

        if state_similarity >= self.state_similarity_threshold:
            if gap_magnitude <= self.gap_magnitude_threshold:
                return True, "state_similarity_and_low_gaps"

        if gap_reduction <= self.gap_reduction_threshold:
            if round_number >= self.min_rounds + 1:
                return True, "gap_reduction_plateaued"

        return False, f"converging (score={convergence_score:.3f})"

    def get_convergence_trend(self) -> Dict:
        if not self.metrics_history:
            return {}

        recent = self.metrics_history[-3:] 

        return {
            "rounds_elapsed": len(self.metrics_history),
            "current_score": self.metrics_history[-1].convergence_score,
            "score_trend": [m.convergence_score for m in recent],
            "gap_trend": [m.gap_magnitude_avg for m in recent],
            "is_improving": (
                len(recent) >= 2 and
                recent[-1].convergence_score > recent[-2].convergence_score
            ),
        }

    def reset(self) -> None:
        self.metrics_history.clear()
        self.gap_history.clear()
        logger.info("Reset convergence detector")

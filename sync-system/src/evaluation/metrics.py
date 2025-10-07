from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np

from src.orchestrator.coordinator import CollaborationResult
from src.utils.logging import get_logger

logger = get_logger("evaluation.metrics")


@dataclass
class TaskMetrics:
    task_id: str
    success: bool
    confidence: float
    consensus_level: float
    rounds: int
    messages: int
    tokens: int
    computation_time: float
    converged: bool
    final_gap_magnitude: float


@dataclass
class AggregatedMetrics:
    task_success_rate: float
    avg_confidence: float
    avg_consensus: float

    avg_rounds: float
    avg_messages: float
    avg_tokens: float
    avg_computation_time: float

    convergence_rate: float
    avg_convergence_rounds: float

    # Gap metrics
    avg_final_gap: float
    gap_reduction_rate: float

    # Communication metrics
    message_redundancy: float
    avg_messages_per_round: float

    # Quality metrics
    high_confidence_rate: float  # % with confidence > 0.8
    high_consensus_rate: float   # % with consensus > 0.8

    # Metadata
    num_tasks: int
    num_failed: int


class MetricsComputer:
    """
    Computes performance metrics for evaluation

    Tracks metrics across multiple collaborations to measure
    system performance against target benchmarks.
    """

    def __init__(self):
        """Initialize metrics computer"""
        self.task_metrics: List[TaskMetrics] = []
        self.task_results: Dict[str, CollaborationResult] = {}

        logger.info("Initialized MetricsComputer")

    def add_result(self, result: CollaborationResult):
        """
        Add a collaboration result

        Args:
            result: Collaboration result to track
        """
        # Extract task metrics
        task_metric = TaskMetrics(
            task_id=result.query[:50],  # Use query as ID
            success=result.success,
            confidence=result.final_response.confidence,
            consensus_level=result.final_response.consensus_level,
            rounds=result.total_rounds,
            messages=result.total_messages,
            tokens=self._estimate_tokens(result),
            computation_time=result.computation_time,
            converged=any(m.is_converged for m in result.convergence_metrics),
            final_gap_magnitude=self._get_final_gap(result)
        )

        self.task_metrics.append(task_metric)
        self.task_results[task_metric.task_id] = result

        logger.debug(
            f"Added metrics for task: success={task_metric.success}, "
            f"rounds={task_metric.rounds}, confidence={task_metric.confidence:.2f}"
        )

    def compute_aggregated_metrics(self) -> AggregatedMetrics:
        """
        Compute aggregated metrics across all tasks

        Returns:
            AggregatedMetrics with all computed metrics
        """
        if not self.task_metrics:
            logger.warning("No task metrics to aggregate")
            return self._empty_metrics()

        # Success metrics
        task_success_rate = np.mean([m.success for m in self.task_metrics])
        avg_confidence = np.mean([m.confidence for m in self.task_metrics])
        avg_consensus = np.mean([m.consensus_level for m in self.task_metrics])

        # Efficiency metrics
        avg_rounds = np.mean([m.rounds for m in self.task_metrics])
        avg_messages = np.mean([m.messages for m in self.task_metrics])
        avg_tokens = np.mean([m.tokens for m in self.task_metrics])
        avg_computation_time = np.mean([m.computation_time for m in self.task_metrics])

        # Convergence metrics
        convergence_rate = np.mean([m.converged for m in self.task_metrics])
        converged_tasks = [m for m in self.task_metrics if m.converged]
        avg_convergence_rounds = (
            np.mean([m.rounds for m in converged_tasks])
            if converged_tasks else avg_rounds
        )

        # Gap metrics
        avg_final_gap = np.mean([m.final_gap_magnitude for m in self.task_metrics])
        gap_reduction_rate = self._compute_gap_reduction_rate()

        # Communication metrics
        message_redundancy = self._compute_message_redundancy()
        avg_messages_per_round = avg_messages / avg_rounds if avg_rounds > 0 else 0

        # Quality metrics
        high_confidence_rate = np.mean([m.confidence > 0.8 for m in self.task_metrics])
        high_consensus_rate = np.mean([m.consensus_level > 0.8 for m in self.task_metrics])

        # Metadata
        num_tasks = len(self.task_metrics)
        num_failed = sum(1 for m in self.task_metrics if not m.success)

        metrics = AggregatedMetrics(
            task_success_rate=float(task_success_rate),
            avg_confidence=float(avg_confidence),
            avg_consensus=float(avg_consensus),
            avg_rounds=float(avg_rounds),
            avg_messages=float(avg_messages),
            avg_tokens=float(avg_tokens),
            avg_computation_time=float(avg_computation_time),
            convergence_rate=float(convergence_rate),
            avg_convergence_rounds=float(avg_convergence_rounds),
            avg_final_gap=float(avg_final_gap),
            gap_reduction_rate=float(gap_reduction_rate),
            message_redundancy=float(message_redundancy),
            avg_messages_per_round=float(avg_messages_per_round),
            high_confidence_rate=float(high_confidence_rate),
            high_consensus_rate=float(high_consensus_rate),
            num_tasks=num_tasks,
            num_failed=num_failed
        )

        logger.info(
            f"Computed aggregated metrics: "
            f"Success={metrics.task_success_rate:.1%}, "
            f"Avg Rounds={metrics.avg_rounds:.1f}, "
            f"Convergence={metrics.convergence_rate:.1%}"
        )

        return metrics

    def _estimate_tokens(self, result: CollaborationResult) -> int:
        """Estimate tokens used in collaboration"""
        # Rough estimate: count characters in dialogue history
        total_chars = len(result.dialogue_history) if result.dialogue_history else 0
        # Add final answer length
        if result.final_response and result.final_response.final_answer:
            total_chars += len(result.final_response.final_answer)
        tokens = total_chars / 4  # Rough chars-to-tokens ratio
        return int(tokens)

    def _get_final_gap(self, result: CollaborationResult) -> float:
        """Get final gap magnitude from convergence metrics"""
        if result.convergence_metrics:
            return result.convergence_metrics[-1].gap_magnitude_avg
        return 0.0

    def _compute_gap_reduction_rate(self) -> float:
        """Compute average gap reduction rate"""
        reductions = []

        for task_id, result in self.task_results.items():
            if len(result.convergence_metrics) >= 2:
                initial_gap = result.convergence_metrics[0].gap_magnitude_avg
                final_gap = result.convergence_metrics[-1].gap_magnitude_avg

                if initial_gap > 0:
                    reduction = (initial_gap - final_gap) / initial_gap
                    reductions.append(reduction)

        return float(np.mean(reductions)) if reductions else 0.0

    def _compute_message_redundancy(self) -> float:
        """
        Compute message redundancy

        Measures how much message content is repeated/redundant.
        Lower is better.
        """
        redundancy_scores = []

        for result in self.task_results.values():
            messages = result.dialogue_history.split('\n')
            if len(messages) < 2:
                continue

            # Simple redundancy: count repeated phrases
            phrases = set()
            redundant = 0

            for msg in messages:
                msg_phrases = set(msg.lower().split())
                overlap = len(msg_phrases & phrases)
                if overlap > len(msg_phrases) * 0.5:  # >50% overlap
                    redundant += 1
                phrases.update(msg_phrases)

            redundancy = redundant / len(messages) if messages else 0
            redundancy_scores.append(redundancy)

        return float(np.mean(redundancy_scores)) if redundancy_scores else 0.0

    def _empty_metrics(self) -> AggregatedMetrics:
        """Return empty metrics"""
        return AggregatedMetrics(
            task_success_rate=0.0,
            avg_confidence=0.0,
            avg_consensus=0.0,
            avg_rounds=0.0,
            avg_messages=0.0,
            avg_tokens=0.0,
            avg_computation_time=0.0,
            convergence_rate=0.0,
            avg_convergence_rounds=0.0,
            avg_final_gap=0.0,
            gap_reduction_rate=0.0,
            message_redundancy=0.0,
            avg_messages_per_round=0.0,
            high_confidence_rate=0.0,
            high_consensus_rate=0.0,
            num_tasks=0,
            num_failed=0
        )

    def print_metrics(self, metrics: Optional[AggregatedMetrics] = None):
        """
        Print metrics in a nice format

        Args:
            metrics: Metrics to print (computes if None)
        """
        if metrics is None:
            metrics = self.compute_aggregated_metrics()

        print("\n" + "=" * 80)
        print("EVALUATION METRICS")
        print("=" * 80)

        print("\n[SUCCESS METRICS]")
        print(f"  Task Success Rate: {metrics.task_success_rate:.1%}")
        print(f"  Avg Confidence: {metrics.avg_confidence:.2f}")
        print(f"  Avg Consensus: {metrics.avg_consensus:.2f}")
        print(f"  High Confidence Rate: {metrics.high_confidence_rate:.1%}")
        print(f"  High Consensus Rate: {metrics.high_consensus_rate:.1%}")

        print("\n[EFFICIENCY METRICS]")
        print(f"  Avg Rounds: {metrics.avg_rounds:.1f}")
        print(f"  Avg Messages: {metrics.avg_messages:.1f}")
        print(f"  Avg Tokens: {metrics.avg_tokens:.0f}")
        print(f"  Avg Computation Time: {metrics.avg_computation_time:.1f}s")
        print(f"  Avg Messages/Round: {metrics.avg_messages_per_round:.1f}")

        print("\n[CONVERGENCE METRICS]")
        print(f"  Convergence Rate: {metrics.convergence_rate:.1%}")
        print(f"  Avg Convergence Rounds: {metrics.avg_convergence_rounds:.1f}")

        print("\n[GAP METRICS]")
        print(f"  Avg Final Gap: {metrics.avg_final_gap:.3f}")
        print(f"  Gap Reduction Rate: {metrics.gap_reduction_rate:.1%}")

        print("\n[COMMUNICATION METRICS]")
        print(f"  Message Redundancy: {metrics.message_redundancy:.1%}")

        print("\n[SUMMARY]")
        print(f"  Total Tasks: {metrics.num_tasks}")
        print(f"  Failed Tasks: {metrics.num_failed}")

        # Compare to targets
        print("\n[TARGETS vs ACTUAL]")
        print(f"  Success Rate:       Target > 81%  | Actual: {metrics.task_success_rate:.1%}")
        print(f"  Avg Rounds:         Target < 5    | Actual: {metrics.avg_rounds:.1f}")
        print(f"  Message Redundancy: Target < 15%  | Actual: {metrics.message_redundancy:.1%}")
        print(f"  Convergence Rate:   Target > 90%  | Actual: {metrics.convergence_rate:.1%}")

        print("\n" + "=" * 80)

    def reset(self):
        """Reset all metrics"""
        self.task_metrics = []
        self.task_results = {}
        logger.info("Reset metrics")

    def export_to_dict(self) -> Dict:
        """Export metrics to dict"""
        metrics = self.compute_aggregated_metrics()
        return asdict(metrics)


if __name__ == "__main__":
    # Test metrics computation
    print("=" * 80)
    print("TESTING METRICS SYSTEM")
    print("=" * 80)

    # Create mock results
    from src.orchestrator.coordinator import CollaborationResult
    from src.orchestrator.aggregator import AgregatedResponse
    from src.orchestrator.convergence import ConvergenceMetrics

    mock_response = AgregatedResponse(
        final_answer="Test answer",
        confidence=0.9,
        consensus_level=0.85,
        reasoning_summary="Test",
        conflicts=[],
        agent_contributions={},
        metadata={}
    )

    mock_convergence = [
        ConvergenceMetrics(
            round_number=0,
            convergence_score=0.3,
            state_similarity_avg=0.6,
            gap_magnitude_avg=0.8,
            gap_reduction_rate=0.0,
            agreement_signal_count=0,
            is_converged=False,
            reason="Initial round"
        ),
        ConvergenceMetrics(
            round_number=3,
            convergence_score=0.95,
            state_similarity_avg=0.92,
            gap_magnitude_avg=0.25,
            gap_reduction_rate=0.68,
            agreement_signal_count=3,
            is_converged=True,
            reason="High similarity and low gaps"
        )
    ]

    mock_result = CollaborationResult(
        query="What is machine learning?",
        final_response=mock_response,
        convergence_metrics=mock_convergence,
        total_rounds=3,
        total_messages=9,
        dialogue_history="Agent 0: ...\nAgent 1: ...\n",
        computation_time=35.5,
        success=True
    )

    # Compute metrics
    print("\n[1/2] Adding results...")
    computer = MetricsComputer()

    for i in range(10):
        computer.add_result(mock_result)

    print(f"[OK] Added {len(computer.task_metrics)} results")

    # Print metrics
    print("\n[2/2] Computing aggregated metrics...")
    computer.print_metrics()

    print("\n[OK] Metrics system working!")

import asyncio
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import json

from tqdm.asyncio import tqdm_asyncio

from src.orchestrator.coordinator import MultiAgentCoordinator
from src.data.datasets import get_dataset, TrainingExample
from src.evaluation.metrics import MetricsComputer, AggregatedMetrics
from src.utils.logging import get_logger

logger = get_logger("evaluation.benchmarks")


class BenchmarkRunner:
    def __init__(
        self,
        num_agents: int = 3,
        max_rounds: int = 5,
        device: str = "cpu",
    ):
        self.num_agents = num_agents
        self.max_rounds = max_rounds
        self.device = device

        self.metrics_computer = MetricsComputer()

        logger.info(
            f"Initialized BenchmarkRunner: "
            f"{num_agents} agents, {max_rounds} max rounds"
        )

    async def run_alpaca_eval(
        self,
        num_tasks: int = 100,
        save_results: bool = True,
        results_dir: Optional[Path] = None,
    ) -> AggregatedMetrics:
        """
        Run evaluation on AlpacaEval benchmark

        Args:
            num_tasks: Number of tasks to evaluate
            save_results: Save results to file
            results_dir: Directory to save results

        Returns:
            AggregatedMetrics with results
        """
        logger.info("=" * 80)
        logger.info("RUNNING ALPACAEVAL BENCHMARK")
        logger.info("=" * 80)
        logger.info(f"Tasks: {num_tasks}")
        logger.info(f"Agents: {self.num_agents}")
        logger.info(f"Max Rounds: {self.max_rounds}")
        logger.info("=" * 80)

        # Load dataset
        logger.info("\n[1/3] Loading AlpacaEval dataset...")
        dataset = get_dataset("alpaca", split="test", max_samples=num_tasks)
        logger.info(f"[OK] Loaded {len(dataset)} tasks")

        # Create coordinator
        coordinator = MultiAgentCoordinator(
            num_agents=self.num_agents,
            max_rounds=self.max_rounds,
            device=self.device
        )

        # Run evaluation
        logger.info("\n[2/3] Running evaluation...")
        results = []

        # Use tqdm for progress tracking
        for i, example in enumerate(tqdm_asyncio(dataset, desc="Evaluating")):
            try:
                result = await coordinator.collaborate(
                    query=example.query,
                    context=example.context
                )

                results.append(result)
                self.metrics_computer.add_result(result)

                logger.debug(
                    f"Task {i+1}/{len(dataset)}: "
                    f"Success={result.success}, "
                    f"Rounds={result.total_rounds}, "
                    f"Time={result.computation_time:.1f}s"
                )

                # Reset for next task
                coordinator.reset()

            except Exception as e:
                logger.error(f"Task {i+1} failed: {e}")
                continue

        await coordinator.close()

        # Compute metrics
        logger.info("\n[3/3] Computing metrics...")
        metrics = self.metrics_computer.compute_aggregated_metrics()

        # Print metrics
        self.metrics_computer.print_metrics(metrics)

        # Save results
        if save_results:
            self._save_results(
                benchmark="alpacaeval",
                metrics=metrics,
                results=results,
                results_dir=results_dir
            )

        logger.info("\n[OK] AlpacaEval benchmark complete!")

        return metrics

    async def run_custom_benchmark(
        self,
        benchmark_name: str,
        tasks: List[TrainingExample],
        save_results: bool = True,
        results_dir: Optional[Path] = None,
    ) -> AggregatedMetrics:
        """
        Run evaluation on custom benchmark

        Args:
            benchmark_name: Name of benchmark
            tasks: List of tasks to evaluate
            save_results: Save results
            results_dir: Results directory

        Returns:
            AggregatedMetrics
        """
        logger.info("=" * 80)
        logger.info(f"RUNNING CUSTOM BENCHMARK: {benchmark_name}")
        logger.info("=" * 80)
        logger.info(f"Tasks: {len(tasks)}")
        logger.info("=" * 80)

        # Create coordinator
        coordinator = MultiAgentCoordinator(
            num_agents=self.num_agents,
            max_rounds=self.max_rounds,
            device=self.device
        )

        # Run evaluation
        logger.info("\nRunning evaluation...")
        results = []

        for i, task in enumerate(tqdm_asyncio(tasks, desc="Evaluating")):
            try:
                result = await coordinator.collaborate(
                    query=task.query,
                    context=task.context
                )

                results.append(result)
                self.metrics_computer.add_result(result)

                coordinator.reset()

            except Exception as e:
                logger.error(f"Task {i+1} failed: {e}")
                continue

        await coordinator.close()

        # Compute metrics
        metrics = self.metrics_computer.compute_aggregated_metrics()
        self.metrics_computer.print_metrics(metrics)

        # Save results
        if save_results:
            self._save_results(
                benchmark=benchmark_name,
                metrics=metrics,
                results=results,
                results_dir=results_dir
            )

        logger.info(f"\n[OK] {benchmark_name} benchmark complete!")

        return metrics

    def _save_results(
        self,
        benchmark: str,
        metrics: AggregatedMetrics,
        results: List,
        results_dir: Optional[Path] = None,
    ):
        """Save benchmark results"""
        if results_dir is None:
            results_dir = Path("./results/benchmarks")

        results_dir.mkdir(parents=True, exist_ok=True)

        # Create results dict
        results_dict = {
            "benchmark": benchmark,
            "timestamp": datetime.now().isoformat(),
            "num_agents": self.num_agents,
            "max_rounds": self.max_rounds,
            "metrics": self.metrics_computer.export_to_dict(),
            "num_tasks": len(results),
        }

        # Save to JSON
        filename = f"{benchmark}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = results_dir / filename

        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"[OK] Saved results to {filepath}")


async def evaluate_on_alpaca(
    num_tasks: int = 100,
    num_agents: int = 3,
    max_rounds: int = 5,
    device: str = "cpu",
) -> AggregatedMetrics:
    """
    Convenience function to evaluate on AlpacaEval

    Args:
        num_tasks: Number of tasks
        num_agents: Number of agents
        max_rounds: Max rounds
        device: Device

    Returns:
        AggregatedMetrics
    """
    runner = BenchmarkRunner(
        num_agents=num_agents,
        max_rounds=max_rounds,
        device=device
    )

    metrics = await runner.run_alpaca_eval(num_tasks=num_tasks)

    return metrics


if __name__ == "__main__":
    # Test benchmark runner (without real evaluation)
    print("=" * 80)
    print("TESTING BENCHMARK RUNNER SETUP")
    print("=" * 80)

    # Create runner
    print("\n[1/2] Creating benchmark runner...")
    runner = BenchmarkRunner(num_agents=3, max_rounds=5)
    print(f"[OK] Benchmark runner created")

    # Load test dataset
    print("\n[2/2] Loading test dataset...")
    dataset = get_dataset("alpaca", split="test", max_samples=5)
    print(f"[OK] Loaded {len(dataset)} test tasks")

    print("\n[OK] Benchmark runner setup working!")
    print("NOTE: Full evaluation requires running run_alpaca_eval() with real collaboration")

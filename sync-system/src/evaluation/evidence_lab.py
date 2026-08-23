"""Provider-neutral experiment manifests for collaboration baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable


@dataclass(frozen=True)
class ExperimentResult:
    strategy: str
    task_id: str
    seed: int
    quality: float
    cost: float
    latency_seconds: float
    trace_path: str | None = None


class EvidenceLab:
    REQUIRED_BASELINES = ("learned", "random", "independent", "debate", "no_ckm")

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        tasks: Iterable[tuple[str, str]],
        runner: Callable[[str, str, str, int], ExperimentResult],
        *,
        strategies: Iterable[str] = REQUIRED_BASELINES,
        seeds: Iterable[int] = (0, 1, 2),
    ) -> list[ExperimentResult]:
        materialized_tasks = list(tasks)
        results = [
            runner(strategy, task_id, prompt, seed)
            for strategy in strategies
            for seed in seeds
            for task_id, prompt in materialized_tasks
        ]
        raw_path = self.output_dir / "raw_results.jsonl"
        with raw_path.open("w", encoding="utf-8") as stream:
            for result in results:
                stream.write(json.dumps(asdict(result), sort_keys=True) + "\n")
        self._write_summary(results)
        return results

    def _write_summary(self, results: list[ExperimentResult]) -> None:
        grouped: dict[str, list[ExperimentResult]] = {}
        for result in results:
            grouped.setdefault(result.strategy, []).append(result)
        summary = {
            strategy: {
                "runs": len(group),
                "mean_quality": mean(item.quality for item in group),
                "mean_cost": mean(item.cost for item in group),
                "mean_latency_seconds": mean(item.latency_seconds for item in group),
            }
            for strategy, group in sorted(grouped.items())
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

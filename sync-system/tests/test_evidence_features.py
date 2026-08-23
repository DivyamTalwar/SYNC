import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.evaluation.evidence_lab import EvidenceLab, ExperimentResult
from src.observability.trace import CognitiveTrace
from src.orchestrator.budget import AdaptiveBudgetPolicy, BudgetAction, BudgetState


class EvidenceFeatureTests(unittest.TestCase):
    def test_trace_detects_tampering(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            trace = CognitiveTrace(path)
            trace.append("message", {"agent": 1, "text": "evidence"})
            trace.append("decision", {"action": "critique"})
            self.assertTrue(trace.verify())
            path.write_text(path.read_text().replace("evidence", "tampered"))
            self.assertFalse(CognitiveTrace(path).verify())

    def test_budget_policy_stops_before_overspend(self):
        policy = AdaptiveBudgetPolicy()
        state = BudgetState(100, 100, 1, 5, 0.9, 0.9)
        self.assertEqual(policy.decide(state), BudgetAction.STOP)
        state = BudgetState(10, 100, 1, 5, 0.1, 0.1)
        self.assertEqual(policy.decide(state), BudgetAction.SYNTHESIZE)

    def test_evidence_lab_writes_raw_and_summary_receipts(self):
        with TemporaryDirectory() as directory:
            lab = EvidenceLab(Path(directory))

            def runner(strategy, task_id, prompt, seed):
                return ExperimentResult(strategy, task_id, seed, 1.0, 0.1, 0.2)

            results = lab.run([("t1", "prompt")], runner, strategies=("learned", "random"), seeds=(0,))
            self.assertEqual(len(results), 2)
            summary = json.loads((Path(directory) / "summary.json").read_text())
            self.assertEqual(summary["learned"]["runs"], 1)


if __name__ == "__main__":
    unittest.main()

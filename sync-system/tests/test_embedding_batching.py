import unittest

from src.llm.batching import DeduplicatedBatch


class DeduplicatedBatchTests(unittest.TestCase):
    def test_deduplicates_and_restores_original_order(self):
        plan = DeduplicatedBatch.from_texts(["beta", "alpha", "beta", "gamma", "alpha"])
        self.assertEqual(plan.unique_texts, ("beta", "alpha", "gamma"))
        restored = plan.restore([20, 10, 30])
        self.assertEqual(restored, [20, 10, 20, 30, 10])

    def test_rejects_empty_or_invalid_text(self):
        with self.assertRaises(ValueError):
            DeduplicatedBatch.from_texts([])
        with self.assertRaises(ValueError):
            DeduplicatedBatch.from_texts(["valid", "  "])

    def test_restore_requires_one_value_per_unique_text(self):
        plan = DeduplicatedBatch.from_texts(["one", "two"])
        with self.assertRaises(ValueError):
            plan.restore([1])


if __name__ == "__main__":
    unittest.main()

import unittest

from vptd_eae.emnlp_adapter import convert_emnlp_results
from vptd_eae.metrics import evaluate_arguments, evaluate_directional_corrections


class MetricAndAdapterTests(unittest.TestCase):
    def test_native_emnlp_result_is_evaluated(self):
        items = [
            {
                "sentence_id": "s1",
                "event_type": "Conflict||Attack",
                "ground_truth": [
                    {"role": "Attacker", "entity": "Rebels"},
                    {"role": "Target", "entity": "the base"},
                ],
                "output": '[{"role":"Attacker","entity":"Rebels"},{"role":"Target","entity":"the base"}]',
            }
        ]
        gold, predictions = convert_emnlp_results(items)
        metrics = evaluate_arguments(gold, predictions)
        self.assertEqual(metrics["argument_identification"]["f1"], 1.0)
        self.assertEqual(metrics["argument_classification"]["f1"], 1.0)

    def test_directional_reversal_correction(self):
        gold = [{"sample_id": "s1", "arguments": [{"entity": "John", "role": "Giver"}]}]
        baseline = [{"sample_id": "s1", "arguments": [{"entity": "John", "role": "Recipient"}]}]
        temporal = gold
        report = evaluate_directional_corrections(gold, baseline, temporal)
        self.assertEqual(report["by_pair"]["Giver/Recipient"]["correction_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

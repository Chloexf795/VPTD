import unittest

from vptd_eae.calibration import apply_score_threshold, tune_global_threshold


class CalibrationTests(unittest.TestCase):
    def test_dev_threshold_removes_false_positive(self):
        gold = [{"sample_id": "s1", "arguments": [{"entity": "John", "role": "Giver"}]}]
        scored = [
            {
                "sample_id": "s1",
                "arguments": [
                    {"entity": "John", "role": "Giver", "score": 1.0},
                    {"entity": "Mary", "role": "Giver", "score": -0.2},
                ],
            }
        ]
        result = tune_global_threshold(gold, scored)
        predictions = apply_score_threshold(scored, result["threshold"])
        self.assertEqual(result["metrics"]["f1"], 1.0)
        self.assertEqual(len(predictions[0]["arguments"]), 1)


if __name__ == "__main__":
    unittest.main()

"""Check candidate-ranking perspective and tie handling at the report seam."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"trainer"))
import clean_search_ranking as cr


class RankingReportTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"label": {"record_id": 1, "game_id": 5, "S4": [9, 3, 1]},
                      "samples": [None]*3}]

    def test_child_negation(self):
        correct = cr.summarize(self.rows, [-9, -3, -1])
        self.assertEqual(correct["pair_disagreement"], 0)
        self.assertEqual(correct["top1_tie_adjusted"], 1)
        reversed_order = cr.summarize(self.rows, [9, 3, 1])
        self.assertEqual(reversed_order["pair_disagreement"], 1)
        self.assertEqual(reversed_order["top1_tie_adjusted"], 0)

    def test_constant_model_gets_fractional_tie_credit(self):
        metrics = cr.summarize(self.rows, [0, 0, 0])
        self.assertEqual(metrics["pair_disagreement"], .5)
        self.assertAlmostEqual(metrics["top1_tie_adjusted"], 1/3)

    def test_teacher_ties_excluded_from_pairs(self):
        self.rows[0]["label"]["S4"] = [9, 9, 1]
        metrics = cr.summarize(self.rows, [-8, -7, -1])
        self.assertEqual(metrics["pair_disagreement"], 0)
        self.assertEqual(metrics["top1_tie_adjusted"], 1)

    def test_parent_macro_weighting(self):
        self.rows.append({"label": {"record_id": 2, "game_id": 6, "S4": [4, 1]},
                          "samples": [None]*2})
        metrics = cr.summarize(self.rows, [-9, -3, -1, 4, 1])
        self.assertEqual(metrics["pair_disagreement"], .5)

    def test_bad_prediction_count_rejected(self):
        with self.assertRaises(ValueError):
            cr.summarize(self.rows, [0]*4)


if __name__ == "__main__":
    unittest.main()

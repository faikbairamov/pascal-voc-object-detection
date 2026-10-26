"""Regression checks for matching rules migrated from the original notebook."""
import unittest

import numpy as np
import torch

from voc_detection.metrics import calculate_voc_ap, evaluate_category

def make_evaluation_test(predicted_boxes, scores, difficult=False):
    return [{
        "image_id": "evaluation_test",
        "target": {
            "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "difficult": torch.tensor([difficult], dtype=torch.bool),
        },
        "prediction": {
            "boxes": torch.tensor(
                predicted_boxes, dtype=torch.float32
            ).reshape(-1, 4),
            "labels": torch.ones(len(scores), dtype=torch.int64),
            "scores": torch.tensor(scores, dtype=torch.float32),
        },
    }]

class VOCMetricTests(unittest.TestCase):
    def test_ap_boundaries(self):
        self.assertAlmostEqual(calculate_voc_ap(np.array([.5, 1.]), np.array([1., 1.])), 1.)
        self.assertAlmostEqual(calculate_voc_ap(np.array([]), np.array([])), 0.)

    def test_no_detections(self):
        result = evaluate_category(make_evaluation_test([], []), class_id=1)
        self.assertEqual(result["ap"], 0.0)
        self.assertEqual(result["objects"], 1)

    def test_exact_threshold_is_not_a_match(self):
        result = evaluate_category(
            make_evaluation_test([[0, 0, 5, 10]], [.9]), class_id=1
        )
        self.assertEqual(result["ap"], 0.0)


    def test_matching_rules(self):


        perfect = evaluate_category(
            make_evaluation_test([[0, 0, 10, 10]], [0.9]),
            class_id=1,
        )

        assert np.isclose(perfect["ap"], 1.0)

        false_first = evaluate_category(
            make_evaluation_test(
                [[20, 20, 30, 30], [0, 0, 10, 10]],
                [0.9, 0.8],
            ),
            class_id=1,
        )

        assert np.isclose(false_first["ap"], 0.5)

        duplicate = evaluate_category(
            make_evaluation_test(
                [[0, 0, 10, 10], [0, 0, 10, 10]],
                [0.9, 0.8],
            ),
            class_id=1,
        )

        assert np.allclose(duplicate["recall"], [1.0, 1.0])

        assert np.allclose(duplicate["precision"], [1.0, 0.5])

        difficult_record = make_evaluation_test(
            [[0, 0, 10, 10]],
            [0.99],
            difficult=True,
        )

        difficult_record[0]["image_id"] = "difficult_test"

        mixed_records = difficult_record + make_evaluation_test(
            [[0, 0, 10, 10]],
            [0.9],
        )

        difficult_result = evaluate_category(mixed_records, class_id=1)

        assert difficult_result["objects"] == 1

        assert np.isclose(difficult_result["ap"], 1.0)


if __name__ == "__main__":
    unittest.main()

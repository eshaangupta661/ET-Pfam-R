"""
Unit tests for the uncertainty calibration and OOD detection module.
"""
import unittest
import numpy as np
import torch as tr

from src.uncertainty_calibration import (
    ConformalPredictionWrapper,
    compute_random_neighbor_score,
    OODDetector,
)


class TestUncertaintyCalibration(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        tr.manual_seed(42)

        self.num_classes = 10
        self.categories = [f"PF{i:05d}" for i in range(self.num_classes)]
        self.n_cal = 500
        self.n_test = 500

        # Simulate calibrated model probabilities
        # Create noisy ground-truth logits
        self.cal_labels = np.random.randint(0, self.num_classes, size=self.n_cal)
        cal_logits = np.random.randn(self.n_cal, self.num_classes)
        # Boost true class logit to make predictions reasonable
        cal_logits[np.arange(self.n_cal), self.cal_labels] += 3.0
        exp_logits = np.exp(cal_logits)
        self.cal_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        self.test_labels = np.random.randint(0, self.num_classes, size=self.n_test)
        test_logits = np.random.randn(self.n_test, self.num_classes)
        test_logits[np.arange(self.n_test), self.test_labels] += 3.0
        exp_test = np.exp(test_logits)
        self.test_probs = exp_test / exp_test.sum(axis=1, keepdims=True)

    def test_conformal_lac_coverage(self):
        """Test standard LAC conformal prediction achieves target coverage."""
        alpha = 0.10  # 90% target coverage
        cp = ConformalPredictionWrapper(categories=self.categories, method="lac")
        cp.calibrate(self.cal_probs, self.cal_labels)

        metrics = cp.evaluate_coverage(self.test_probs, self.test_labels, alpha=alpha)

        # Empirical coverage should be close to or above 1 - alpha (0.90)
        self.assertGreaterEqual(
            metrics["empirical_coverage"],
            0.87,
            f"Coverage {metrics['empirical_coverage']} below tolerance for target 0.90",
        )
        self.assertGreater(metrics["avg_set_size"], 0.0)
        self.assertEqual(metrics["empty_rate"], 0.0, "Empty sets must not be returned.")

    def test_conformal_aps_coverage(self):
        """Test Adaptive Prediction Sets (APS) method achieves target coverage."""
        alpha = 0.15  # 85% target coverage
        cp = ConformalPredictionWrapper(categories=self.categories, method="aps")
        cp.calibrate(self.cal_probs, self.cal_labels)

        metrics = cp.evaluate_coverage(self.test_probs, self.test_labels, alpha=alpha)

        self.assertGreaterEqual(
            metrics["empirical_coverage"],
            0.82,
            f"Coverage {metrics['empirical_coverage']} below tolerance for target 0.85",
        )

    def test_predict_set_formats(self):
        """Test output format of predict_set with string categories and masks."""
        cp = ConformalPredictionWrapper(categories=self.categories, method="lac")
        cp.calibrate(self.cal_probs, self.cal_labels)

        mask, sets = cp.predict_set(self.test_probs[:5], alpha=0.10)
        self.assertEqual(mask.shape, (5, self.num_classes))
        self.assertEqual(len(sets), 5)
        for s in sets:
            self.assertIsInstance(s, list)
            self.assertGreater(len(s), 0)
            for item in s:
                self.assertIn(item, self.categories)

    def test_random_neighbor_score(self):
        """Test that OOD embeddings receive higher RNS than in-distribution embeddings."""
        emb_dim = 1280
        n_ref = 200

        # In-distribution: clustered around a centroid with small variance
        centroid = tr.randn(1, emb_dim)
        in_dist_ref = centroid + 0.1 * tr.randn(n_ref, emb_dim)
        in_dist_queries = centroid + 0.1 * tr.randn(50, emb_dim)

        # Out-of-distribution: far away orthogonal cluster
        ood_centroid = -centroid + 5.0
        ood_queries = ood_centroid + 0.1 * tr.randn(50, emb_dim)

        in_scores = compute_random_neighbor_score(
            in_dist_queries, in_dist_ref, k=5, metric="cosine"
        )
        ood_scores = compute_random_neighbor_score(
            ood_queries, in_dist_ref, k=5, metric="cosine"
        )

        # In-distribution scores must be strictly lower than OOD scores
        self.assertLess(
            float(np.mean(in_scores)),
            float(np.mean(ood_scores)),
            "In-distribution RNS must be lower than OOD RNS",
        )

    def test_ood_detector_fit_predict(self):
        """Test OODDetector fitting and classification."""
        emb_dim = 128
        in_ref = tr.randn(300, emb_dim)
        in_queries = in_ref[:30] + 0.05 * tr.randn(30, emb_dim)
        ood_queries = tr.randn(30, emb_dim) * 10.0 + 5.0

        detector = OODDetector(k=5, metric="cosine", target_fpr=0.05)
        detector.fit(in_ref)

        is_ood_in, _ = detector.predict(in_queries)
        is_ood_out, _ = detector.predict(ood_queries)

        # Few in-distribution queries should be flagged
        self.assertLessEqual(np.mean(is_ood_in), 0.20)
        # Most OOD queries should be flagged
        self.assertGreater(np.mean(is_ood_out), 0.70)


if __name__ == "__main__":
    unittest.main()

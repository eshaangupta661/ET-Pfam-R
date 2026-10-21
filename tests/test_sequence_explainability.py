"""
Unit tests for the Sequence Explainability and Domain Attribution module.
"""
import unittest
import numpy as np
import torch as tr
import torch.nn as nn

from src.basemodel import BaseModel
from src.sequence_explainability import (
    SaliencyExplainer,
    DomainAttributionMapper,
    DomainAttributionReport,
    explain_prediction,
)


class TestSequenceExplainability(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        tr.manual_seed(42)

        self.seq_len = 60
        self.emb_dim = 1280
        self.num_classes = 5
        self.categories = [f"PF{i:05d}" for i in range(self.num_classes)]

        # Sample protein sequence
        self.sequence = (
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGY"
        )
        self.assertEqual(len(self.sequence), self.seq_len)

        # Create a lightweight BaseModel for testing on CPU
        self.model = BaseModel(
            nclasses=self.num_classes,
            emb_size=self.emb_dim,
            filters=32,
            kernel_size=5,
            num_layers=2,
            device="cpu",
        )
        self.model.eval()

        # Synthetic embedding tensor of shape [1280, L]
        self.embedding = tr.randn(self.emb_dim, self.seq_len)

    def test_saliency_methods(self):
        """Test Vanilla Grad, Input x Grad, and Integrated Gradients."""
        explainer = SaliencyExplainer(self.model, device="cpu")

        # 1. Input x Grad
        scores_ixg = explainer.attribute(
            self.embedding, target_class=1, method="input_x_grad"
        )
        self.assertEqual(scores_ixg.shape, (self.seq_len,))
        self.assertTrue(np.all(scores_ixg >= 0.0))

        # 2. Integrated Gradients
        scores_ig = explainer.attribute(
            self.embedding, target_class=1, method="integrated_gradients", n_steps=10
        )
        self.assertEqual(scores_ig.shape, (self.seq_len,))
        self.assertTrue(np.all(scores_ig >= 0.0))

        # 3. Vanilla Grad
        scores_vg = explainer.attribute(
            self.embedding, target_class=1, method="vanilla_grad"
        )
        self.assertEqual(scores_vg.shape, (self.seq_len,))
        self.assertTrue(np.all(scores_vg >= 0.0))

    def test_domain_attribution_mapping(self):
        """Test mapping residue attributions to a specific Pfam domain region."""
        # Create synthetic attribution profile with high values in domain region [20, 45]
        scores = np.ones(self.seq_len) * 0.1
        scores[19:45] = 0.9  # 1-based [20, 45]

        report = DomainAttributionMapper.map_to_domain(
            scores=scores,
            domain_start=20,
            domain_end=45,
            sequence=self.sequence,
            predicted_pfam="PF00001",
            target_class_idx=1,
            top_k_residues=5,
        )

        self.assertIsInstance(report, DomainAttributionReport)
        self.assertEqual(report.sequence_length, self.seq_len)
        self.assertEqual(report.domain_start, 20)
        self.assertEqual(report.domain_end, 45)

        # Domain region was boosted: enrichment ratio must be > 1.5x
        self.assertGreater(report.domain_enrichment_ratio, 1.5)

        # Top key residues must be within the domain
        self.assertEqual(len(report.key_residues), 5)
        for res in report.key_residues:
            self.assertTrue(res.is_in_domain)
            self.assertTrue(20 <= res.position <= 45)
            self.assertIsNotNone(res.amino_acid)

        # High-scoring region should form a conserved motif
        self.assertGreater(len(report.conserved_motifs), 0)
        top_motif = report.conserved_motifs[0]
        self.assertGreaterEqual(top_motif.length, 3)
        self.assertIsNotNone(top_motif.sequence_slice)

    def test_explain_prediction_end_to_end(self):
        """Test full explain_prediction convenience function."""
        domain = (15, 40)
        report = explain_prediction(
            model=self.model,
            embedding=self.embedding,
            sequence=self.sequence,
            domain_interval=domain,
            categories=self.categories,
            method="input_x_grad",
            top_k_residues=5,
        )

        self.assertIn(report.predicted_pfam, self.categories)
        self.assertEqual(len(report.per_residue_scores), self.seq_len)

        summary_text = report.summary()
        self.assertIn(report.predicted_pfam, summary_text)
        self.assertIn("Top Key Functional Residues", summary_text)


if __name__ == "__main__":
    unittest.main()

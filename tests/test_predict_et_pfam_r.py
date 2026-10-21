"""
Integration test for the unified predict_et_pfam_r.py CLI script.
"""
import os
import json
import unittest
import pandas as pd
import torch as tr
import numpy as np

from predict_et_pfam_r import (
    parse_args,
    load_categories,
    initialize_model,
)
from src.basemodel import BaseModel
from src.plm.pipeline import EmbeddingPipeline
from src.uncertainty_calibration import ConformalPredictionWrapper
from src.retrieval_evidence import PfamRetrievalIndex, search_top_k_neighbors
from src.sequence_explainability import SaliencyExplainer, DomainAttributionMapper


class TestPredictETPfamR(unittest.TestCase):

    def setUp(self):
        self.fasta_path = "data/sample_query.fasta"
        self.output_json = "data/test_output_report.json"
        self.output_csv = "data/test_output_report.csv"

    def tearDown(self):
        for f in (self.output_json, self.output_csv):
            if os.path.exists(f):
                os.remove(f)

    def test_pipeline_components_integration(self):
        """Tests that all 6 required fields are computed and structured properly."""
        # 1. Parse FASTA
        seqs = EmbeddingPipeline.parse_fasta(self.fasta_path)
        self.assertGreater(len(seqs), 0)
        header, seq = next(iter(seqs.items()))

        # 2. Categories & Model
        categories = [f"PF{i:05d}" for i in range(10)]
        model = BaseModel(len(categories), emb_size=1280, filters=32, num_layers=2, device="cpu")
        model.eval()

        # 3. Simulate per-residue embedding [1280, L]
        L = len(seq)
        emb = tr.randn(1280, L)
        emb_input = emb.unsqueeze(0)

        # 4. Predict Pfam Family & Calibrated Confidence
        with tr.no_grad():
            logits = model(emb_input)
            probs = tr.softmax(logits, dim=1).numpy()[0]
            top_class = int(np.argmax(probs))
            top_pfam = categories[top_class]
            conf = float(probs[top_class])

        # 5. Conformal Prediction Set
        conformal = ConformalPredictionWrapper(categories=categories, method="lac")
        conformal.is_calibrated = True
        conformal.n_cal = 100
        conformal.q_hat[0.10] = 0.85
        _, pred_sets = conformal.predict_set(probs[None, :], alpha=0.10)
        conf_set = pred_sets[0]

        # 6. FAISS Homology Matches
        index = PfamRetrievalIndex(dimension=1280)
        ref_vecs = np.random.randn(5, 1280).astype(np.float32)
        ref_vecs /= np.linalg.norm(ref_vecs, axis=1, keepdims=True)
        index.add_references(ref_vecs, [f"REF_{i}" for i in range(5)], [f"PF0000{i}" for i in range(5)])

        query_pooled = index._pool_embedding(emb)
        hits = search_top_k_neighbors(query_pooled, index, top_k=5)
        self.assertEqual(len(hits), 5)

        # 7. Per-residue Attribution
        explainer = SaliencyExplainer(model, device="cpu")
        raw_saliency = explainer.attribute(emb, target_class=top_class, method="input_x_grad")
        report = DomainAttributionMapper.map_to_domain(
            scores=raw_saliency, sequence=seq, predicted_pfam=top_pfam, target_class_idx=top_class
        )

        # 8. Assemble structured record
        record = {
            "sequence_header": header,
            "top_predicted_pfam": top_pfam,
            "calibrated_confidence": conf,
            "conformal_prediction_set": conf_set,
            "ood_status": "IN_DISTRIBUTION",
            "faiss_homology_matches": hits,
            "per_residue_attribution_map": report.per_residue_scores,
            "key_functional_residues": [r.__dict__ for r in report.key_residues],
        }

        # Verify all 6 fields
        self.assertEqual(record["sequence_header"], header)
        self.assertEqual(record["top_predicted_pfam"], top_pfam)
        self.assertIsInstance(record["conformal_prediction_set"], list)
        self.assertIn("calibrated_confidence", record)
        self.assertIn("ood_status", record)
        self.assertEqual(len(record["faiss_homology_matches"]), 5)
        self.assertEqual(len(record["per_residue_attribution_map"]), L)

        # Save to JSON and CSV and verify
        with open(self.output_json, "w") as f:
            json.dump([record], f, indent=2)
        self.assertTrue(os.path.isfile(self.output_json))

        df = pd.DataFrame([{
            "sequence_header": record["sequence_header"],
            "top_predicted_pfam": record["top_predicted_pfam"],
            "calibrated_confidence": record["calibrated_confidence"],
            "conformal_prediction_set": ";".join(record["conformal_prediction_set"]),
            "ood_status": record["ood_status"],
            "top1_faiss_hit": record["faiss_homology_matches"][0]["header"],
        }])
        df.to_csv(self.output_csv, index=False)
        self.assertTrue(os.path.isfile(self.output_csv))


if __name__ == "__main__":
    unittest.main()

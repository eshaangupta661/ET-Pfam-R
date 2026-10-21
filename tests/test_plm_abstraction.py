"""
Unit and integration tests for the Protein Language Model (pLM) abstraction layer.
Validates shape compatibility, tensor formatting, and downstream integration with ET-Pfam.
"""
import os
import pickle
import unittest
import torch as tr
import torch.nn as nn

from src.plm.base import BaseProteinLanguageModel
from src.plm.registry import get_plm, list_available_models, MODEL_REGISTRY
from src.plm.pipeline import EmbeddingPipeline
from src.basemodel import BaseModel


class MockProteinLanguageModel(BaseProteinLanguageModel):
    """
    Mock pLM for testing without downloading multi-gigabyte model weights.
    Simulates transformer output with <cls>, residues, and <eos>.
    """

    def __init__(self, native_dim=1280, target_dim=1280, device="cpu"):
        super().__init__(
            model_name="mock_plm",
            device=device,
            target_dim=target_dim,
        )
        self._embedding_dim = native_dim
        self.setup_dimension_adapter()

    def load_model(self):
        pass

    def _extract_raw_embeddings(self, sequences):
        # Simulates per-residue raw output [L_i, native_dim]
        raw_list = []
        for seq in sequences:
            L = len(seq)
            # Create deterministic tensor based on sequence length
            raw_emb = tr.randn(L, self.embedding_dim)
            raw_list.append(raw_emb)
        return raw_list


class TestPLMAbstraction(unittest.TestCase):

    def setUp(self):
        self.sample_sequence = (
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        )
        self.sample_pid = "TEST_PROT_01"
        self.test_output_dir = "data/test_embeddings/"
        os.makedirs(self.test_output_dir, exist_ok=True)

    def tearDown(self):
        # Clean up test output directory
        if os.path.exists(self.test_output_dir):
            for f in os.listdir(self.test_output_dir):
                os.remove(os.path.join(self.test_output_dir, f))
            os.rmdir(self.test_output_dir)

    def test_sequence_cleaning(self):
        """Test sequence validation and sanitation."""
        raw_seq = "  mktv rqer \n lks* "
        cleaned = BaseProteinLanguageModel.clean_sequence(raw_seq)
        self.assertTrue(cleaned.isupper())
        self.assertNotIn(" ", cleaned)
        self.assertNotIn("\n", cleaned)
        self.assertIn("X", cleaned)  # '*' replaced by 'X'

    def test_native_1280_shape_compatibility(self):
        """
        Verify that a 1280-dim pLM (ESM-2 650M or ESM-1b baseline)
        produces exactly [1280, L] output shape.
        """
        plm = MockProteinLanguageModel(native_dim=1280, target_dim=1280)
        emb = plm.extract_sequence_embedding(self.sample_sequence)

        L = len(self.sample_sequence)
        self.assertEqual(emb.shape, (1280, L), "Output shape must be [1280, L]")
        self.assertEqual(emb.dtype, tr.float32)

    def test_dimension_adaptation(self):
        """
        Verify that a non-1280 pLM (e.g. 640-dim ESM-2 150M)
        is projected to [1280, L] via the dimension adapter.
        """
        plm = MockProteinLanguageModel(native_dim=640, target_dim=1280)
        self.assertIsNotNone(plm.projection)

        emb = plm.extract_sequence_embedding(self.sample_sequence)
        L = len(self.sample_sequence)
        self.assertEqual(emb.shape, (1280, L), "Projected shape must be [1280, L]")

    def test_downstream_classifier_compatibility(self):
        """
        Test that extracted embeddings plug directly into:
        1. Pickle serialization/deserialization as used by PFamDataset.
        2. Window extraction [channels, win_len].
        3. BaseModel CNN forward pass [1, channels, win_len] -> [1, nclasses].
        """
        plm = MockProteinLanguageModel(native_dim=1280, target_dim=1280)
        emb = plm.extract_sequence_embedding(self.sample_sequence)

        # 1. Serialization
        save_path = os.path.join(self.test_output_dir, f"{self.sample_pid}.pk")
        with open(save_path, "wb") as f:
            pickle.dump(emb, f)

        # 2. Deserialization (matching PFamDataset and sliding_window_test)
        loaded_emb = pickle.load(open(save_path, "rb")).squeeze().float()
        self.assertEqual(loaded_emb.shape[0], 1280)
        self.assertEqual(loaded_emb.shape[1], len(self.sample_sequence))

        # 3. Simulate PFamDataset window sampling (win_len = 64)
        win_len = 64
        emb_win = tr.zeros((loaded_emb.shape[0], win_len), dtype=tr.float)
        emb_win[:, :min(loaded_emb.shape[1], win_len)] = loaded_emb[:, :win_len]
        self.assertEqual(emb_win.shape, (1280, 64))

        # 4. Feed into BaseModel CNN
        nclasses = 395
        base_model = BaseModel(
            nclasses=nclasses,
            emb_size=1280,
            filters=64,  # Small filter size for quick CPU test
            num_layers=2,
            device="cpu",
        )
        base_model.eval()

        input_batch = emb_win.unsqueeze(0)  # [1, 1280, 64]
        with tr.no_grad():
            output_logits = base_model(input_batch)

        self.assertEqual(
            output_logits.shape,
            (1, nclasses),
            f"Expected output logits of shape [1, {nclasses}], got {output_logits.shape}",
        )

    def test_pipeline_fasta_processing_and_verification(self):
        """Test EmbeddingPipeline end-to-end with FASTA parsing and verification."""
        fasta_path = os.path.join(self.test_output_dir, "test.fasta")
        with open(fasta_path, "w") as f:
            f.write(f">{self.sample_pid}\n{self.sample_sequence}\n")

        plm = MockProteinLanguageModel(native_dim=1280, target_dim=1280)
        pipeline = EmbeddingPipeline(
            plm=plm,
            output_dir=self.test_output_dir,
            device="cpu",
            target_dim=1280,
            batch_size=2,
        )

        saved = pipeline.process_fasta(fasta_path)
        self.assertIn(self.sample_pid, saved)
        self.assertTrue(os.path.isfile(saved[self.sample_pid]))

        # Run verification routine
        is_valid = pipeline.verify_compatibility(self.sample_pid)
        self.assertTrue(is_valid)

    def test_registry_contents(self):
        """Verify registered model aliases and configurations."""
        models = list_available_models()
        self.assertIn("esm2", models)
        self.assertIn("baseline", models)
        self.assertIn("esm1b", models)
        self.assertEqual(models["esm2"]["native_dim"], 1280)
        self.assertEqual(models["baseline"]["native_dim"], 1280)


if __name__ == "__main__":
    unittest.main()

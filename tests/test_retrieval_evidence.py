"""
Unit tests for the retrieval evidence module using FAISS / cosine similarity.
"""
import os
import unittest
import numpy as np
import torch as tr

from src.retrieval_evidence import (
    PfamRetrievalIndex,
    RetrievalHit,
    search_top_k_neighbors,
)


class TestRetrievalEvidence(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        tr.manual_seed(42)

        self.dim = 1280
        self.num_refs = 20
        self.temp_dir = "data/test_retrieval/"
        os.makedirs(self.temp_dir, exist_ok=True)

        # Generate synthetic reference embeddings
        self.ref_embeddings = np.random.randn(self.num_refs, self.dim).astype(np.float32)
        # Normalize
        self.ref_embeddings /= np.linalg.norm(self.ref_embeddings, axis=1, keepdims=True)

        self.headers = [f"PROT_{i:04d}_SPEC" for i in range(self.num_refs)]
        self.pfam_ids = [f"PF{(i % 5):05d}" for i in range(self.num_refs)]

        self.index = PfamRetrievalIndex(dimension=self.dim)
        self.index.add_references(
            embeddings=self.ref_embeddings,
            headers=self.headers,
            pfam_ids=self.pfam_ids,
        )

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            for f in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, f))
            os.rmdir(self.temp_dir)

    def test_exact_match_retrieval(self):
        """Querying with a reference embedding should return itself as rank 1 with similarity ~1.0."""
        target_idx = 7
        query = self.ref_embeddings[target_idx]

        hits = self.index.search(query, top_k=5)
        self.assertEqual(len(hits), 5)

        # Top hit must be the exact reference entry
        top_hit = hits[0]
        self.assertEqual(top_hit.rank, 1)
        self.assertEqual(top_hit.header, self.headers[target_idx])
        self.assertEqual(top_hit.pfam_id, self.pfam_ids[target_idx])
        self.assertAlmostEqual(top_hit.cosine_similarity, 1.0, places=4)

    def test_sequence_embedding_mean_pooling(self):
        """Testing top-K search with 2D [1280, L] sequence embedding input."""
        L = 80
        target_idx = 3
        base_vector = self.ref_embeddings[target_idx]

        # Construct a 2D [1280, L] sequence embedding whose mean across L is base_vector
        seq_emb = np.repeat(base_vector[:, None], L, axis=1).astype(np.float32)
        # Add small random noise across residues
        noise = np.random.randn(self.dim, L).astype(np.float32) * 0.01
        seq_emb += noise

        hits = self.index.search(seq_emb, top_k=5)
        self.assertEqual(len(hits), 5)
        top_hit = hits[0]
        self.assertEqual(top_hit.header, self.headers[target_idx])
        self.assertEqual(top_hit.pfam_id, self.pfam_ids[target_idx])
        self.assertGreater(top_hit.cosine_similarity, 0.95)

    def test_search_top_k_neighbors_convenience_function(self):
        """Test search_top_k_neighbors returns correct dictionary structure."""
        query = self.ref_embeddings[0]
        results = search_top_k_neighbors(query, self.index, top_k=5)

        self.assertEqual(len(results), 5)
        for r in results:
            self.assertIn("rank", r)
            self.assertIn("header", r)
            self.assertIn("pfam_id", r)
            self.assertIn("cosine_similarity", r)
            self.assertIsInstance(r["cosine_similarity"], float)
            self.assertTrue(-1.0 <= r["cosine_similarity"] <= 1.0)

    def test_batch_search(self):
        """Test batched retrieval."""
        queries = self.ref_embeddings[:3]
        batch_results = self.index.batch_search(queries, top_k=3)

        self.assertEqual(len(batch_results), 3)
        for i in range(3):
            self.assertEqual(batch_results[i][0].header, self.headers[i])
            self.assertAlmostEqual(batch_results[i][0].cosine_similarity, 1.0, places=4)

    def test_save_and_load_index(self):
        """Test index serialization and restoration."""
        idx_file = os.path.join(self.temp_dir, "test.index")
        meta_file = os.path.join(self.temp_dir, "test_meta.json")

        self.index.save(idx_file, meta_file)
        self.assertTrue(os.path.isfile(idx_file))
        self.assertTrue(os.path.isfile(meta_file))

        loaded_index = PfamRetrievalIndex.load(idx_file, meta_file, dimension=self.dim)
        self.assertEqual(loaded_index.size, self.index.size)

        query = self.ref_embeddings[5]
        orig_hits = self.index.search(query, top_k=3)
        loaded_hits = loaded_index.search(query, top_k=3)

        for h1, h2 in zip(orig_hits, loaded_hits):
            self.assertEqual(h1.header, h2.header)
            self.assertEqual(h1.pfam_id, h2.pfam_id)
            self.assertAlmostEqual(h1.cosine_similarity, h2.cosine_similarity, places=4)


if __name__ == "__main__":
    unittest.main()

"""
Retrieval-Augmented Evidence Module for ET-Pfam using FAISS.

Builds a high-performance vector similarity index over target Pfam reference
embeddings and retrieves the top-K nearest neighbors to provide interpretable
homology and family evidence for query protein sequences.
"""
import os
import json
import pickle
from typing import List, Dict, Any, Union, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import torch as tr

# Attempt to import FAISS; provide a PyTorch fallback if faiss is not installed
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@dataclass
class RetrievalHit:
    """Represents a single retrieved Pfam reference entry."""
    rank: int
    header: str               # Sequence header / Protein ID (PID)
    pfam_id: str              # Assigned Pfam family ID (e.g., 'PF12767')
    cosine_similarity: float  # Cosine similarity metric in [-1, 1]
    start: Optional[int] = None
    end: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PfamRetrievalIndex:
    """
    FAISS-based vector index for Pfam reference protein sequence embeddings.

    Uses an Inner Product index on L2-normalized representations (IndexFlatIP)
    to perform exact cosine similarity nearest-neighbor search.

    Features:
        - Exact cosine similarity retrieval via FAISS (or fallback tensor engine).
        - Automatic sequence pooling for ET-Pfam 2D embeddings [1280, L].
        - Support for reference metadata: Sequence headers (PID), Pfam family IDs (PF).
        - Persistence methods: save and load FAISS index and metadata.
    """

    def __init__(
        self,
        dimension: int = 1280,
        use_gpu: bool = False,
    ):
        self.dimension = dimension
        self.use_gpu = use_gpu
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self._raw_vectors: Optional[np.ndarray] = None  # Fallback buffer if no FAISS

        self._init_index()

    def _init_index(self) -> None:
        """Initializes the FAISS IndexFlatIP (or fallback)."""
        if FAISS_AVAILABLE:
            base_index = faiss.IndexFlatIP(self.dimension)
            if self.use_gpu and hasattr(faiss, "StandardGpuResources"):
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, base_index)
            else:
                self.index = base_index
        else:
            self.index = None
            self._raw_vectors = np.empty((0, self.dimension), dtype=np.float32)

    @staticmethod
    def _pool_embedding(emb: Union[tr.Tensor, np.ndarray]) -> np.ndarray:
        """
        Converts query or reference embedding to a 1D vector of shape [D].
        If input is 2D [D, L] (ET-Pfam per-residue representation),
        computes mean pooling across the sequence length L.
        """
        if isinstance(emb, tr.Tensor):
            arr = emb.detach().cpu().float().numpy()
        else:
            arr = np.asarray(emb, dtype=np.float32)

        arr = np.squeeze(arr)

        if arr.ndim == 2:
            # Shape is [D, L] where D is 1280 and L is sequence length
            # If D is the first dimension (D == 1280 or D > L for short domains):
            if arr.shape[0] == 1280 or arr.shape[0] > arr.shape[1]:
                # Mean pool across L (axis 1)
                vector = np.mean(arr, axis=1)
            else:
                # Shape is [L, D], mean pool across L (axis 0)
                vector = np.mean(arr, axis=0)
        elif arr.ndim == 1:
            vector = arr
        else:
            raise ValueError(f"Expected 1D or 2D embedding, got shape {arr.shape}")

        # Normalize to unit length for cosine similarity
        norm = np.linalg.norm(vector)
        if norm > 1e-12:
            vector = vector / norm
        else:
            vector = np.zeros_like(vector)

        return vector.astype(np.float32)

    def add_reference(
        self,
        embedding: Union[tr.Tensor, np.ndarray],
        header: str,
        pfam_id: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Adds a single reference entry to the index."""
        vector = self._pool_embedding(embedding).reshape(1, -1)
        self.add_references(
            embeddings=vector,
            headers=[header],
            pfam_ids=[pfam_id],
            starts=[start] if start is not None else None,
            ends=[end] if end is not None else None,
            extra_metadata=[extra_metadata] if extra_metadata else None,
        )

    def add_references(
        self,
        embeddings: Union[tr.Tensor, np.ndarray, List[Union[tr.Tensor, np.ndarray]]],
        headers: List[str],
        pfam_ids: List[str],
        starts: Optional[List[Optional[int]]] = None,
        ends: Optional[List[Optional[int]]] = None,
        extra_metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> None:
        """
        Batches and adds reference protein sequence embeddings and their metadata to the index.

        Args:
            embeddings: Array of vectors [N, D] or list of [D, L] sequence embeddings.
            headers: Sequence identifiers / UniProt PIDs.
            pfam_ids: Target Pfam family annotations.
            starts: Optional domain start coordinates.
            ends: Optional domain end coordinates.
            extra_metadata: Optional list of arbitrary metadata dictionaries.
        """
        N = len(headers)
        if len(pfam_ids) != N:
            raise ValueError(f"Length mismatch: {N} headers vs {len(pfam_ids)} pfam_ids.")

        # Process and normalize vectors
        if isinstance(embeddings, (list, tuple)):
            vectors = np.array([self._pool_embedding(e) for e in embeddings], dtype=np.float32)
        else:
            if isinstance(embeddings, tr.Tensor):
                raw = embeddings.detach().cpu().float().numpy()
            else:
                raw = np.asarray(embeddings, dtype=np.float32)

            if raw.ndim == 3:
                # [N, D, L] -> mean pool over L
                vectors = np.mean(raw, axis=2)
            elif raw.ndim == 2 and raw.shape[1] == self.dimension:
                vectors = raw
            elif raw.ndim == 2 and raw.shape[0] == self.dimension and N == 1:
                # Single sequence [1280, L]
                vectors = self._pool_embedding(raw).reshape(1, -1)
            else:
                vectors = np.array([self._pool_embedding(v) for v in raw], dtype=np.float32)

            # Normalize rows to unit L2 norm
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms > 1e-12, norms, 1.0)
            vectors = (vectors / norms).astype(np.float32)

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}."
            )

        # Add to FAISS index or fallback array
        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(vectors)
        else:
            if self._raw_vectors is None or len(self._raw_vectors) == 0:
                self._raw_vectors = vectors
            else:
                self._raw_vectors = np.vstack([self._raw_vectors, vectors])

        # Record metadata
        for i in range(N):
            entry = {
                "header": headers[i],
                "pfam_id": pfam_ids[i],
                "start": starts[i] if starts is not None else None,
                "end": ends[i] if ends is not None else None,
            }
            if extra_metadata and extra_metadata[i]:
                entry["metadata"] = extra_metadata[i]
            self.metadata.append(entry)

    def search(
        self,
        query_embedding: Union[tr.Tensor, np.ndarray],
        top_k: int = 5,
    ) -> List[RetrievalHit]:
        """
        Searches the index for the top-K nearest neighbors to the query protein embedding.

        Args:
            query_embedding: Query embedding. Can be:
                             - 1D pooled vector [1280]
                             - 2D sequence embedding [1280, L] (automatically pooled)
            top_k: Number of most similar reference entries to return (default: 5).

        Returns:
            List of RetrievalHit dataclass instances sorted by descending cosine similarity.
        """
        if self.size == 0:
            raise RuntimeError("Index is empty. Add reference embeddings before searching.")

        k_actual = min(top_k, self.size)
        query_vec = self._pool_embedding(query_embedding).reshape(1, -1)

        if FAISS_AVAILABLE and self.index is not None:
            # FAISS search returns (distances/inner products, neighbor indices)
            sims, indices = self.index.search(query_vec, k_actual)
            sims = sims[0]
            indices = indices[0]
        else:
            # Fallback exact dot-product over normalized vectors
            sims_all = np.dot(self._raw_vectors, query_vec.T).squeeze(1)
            indices = np.argsort(-sims_all)[:k_actual]
            sims = sims_all[indices]

        hits: List[RetrievalHit] = []
        for rank, (idx, score) in enumerate(zip(indices, sims), start=1):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            hit = RetrievalHit(
                rank=rank,
                header=meta["header"],
                pfam_id=meta["pfam_id"],
                cosine_similarity=float(np.clip(score, -1.0, 1.0)),
                start=meta.get("start"),
                end=meta.get("end"),
                metadata=meta.get("metadata"),
            )
            hits.append(hit)

        return hits

    def batch_search(
        self,
        query_embeddings: Union[tr.Tensor, np.ndarray, List[Union[tr.Tensor, np.ndarray]]],
        top_k: int = 5,
    ) -> List[List[RetrievalHit]]:
        """Batched top-K nearest neighbor search."""
        if isinstance(query_embeddings, (list, tuple)):
            vectors = np.array([self._pool_embedding(q) for q in query_embeddings], dtype=np.float32)
        else:
            if isinstance(query_embeddings, tr.Tensor):
                raw = query_embeddings.detach().cpu().float().numpy()
            else:
                raw = np.asarray(query_embeddings, dtype=np.float32)
            if raw.ndim == 3:
                vectors = np.mean(raw, axis=2)
            else:
                vectors = np.array([self._pool_embedding(q) for q in raw], dtype=np.float32)

            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms > 1e-12, norms, 1.0)
            vectors = (vectors / norms).astype(np.float32)

        k_actual = min(top_k, self.size)

        if FAISS_AVAILABLE and self.index is not None:
            sims, indices = self.index.search(vectors, k_actual)
        else:
            dot_matrix = np.dot(vectors, self._raw_vectors.T)
            indices = np.argsort(-dot_matrix, axis=1)[:, :k_actual]
            sims = np.take_along_axis(dot_matrix, indices, axis=1)

        batch_hits = []
        for i in range(len(vectors)):
            hits = []
            for rank, (idx, score) in enumerate(zip(indices[i], sims[i]), start=1):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                meta = self.metadata[idx]
                hit = RetrievalHit(
                    rank=rank,
                    header=meta["header"],
                    pfam_id=meta["pfam_id"],
                    cosine_similarity=float(np.clip(score, -1.0, 1.0)),
                    start=meta.get("start"),
                    end=meta.get("end"),
                    metadata=meta.get("metadata"),
                )
                hits.append(hit)
            batch_hits.append(hits)

        return batch_hits

    @property
    def size(self) -> int:
        """Number of indexed reference entries."""
        if FAISS_AVAILABLE and self.index is not None:
            return self.index.ntotal
        return len(self.metadata)

    def save(self, index_file: str, metadata_file: str) -> None:
        """Serializes the FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(index_file)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(metadata_file)), exist_ok=True)

        if FAISS_AVAILABLE and self.index is not None:
            # Transfer GPU index to CPU before saving if necessary
            cpu_index = (
                faiss.index_gpu_to_cpu(self.index)
                if hasattr(faiss, "index_gpu_to_cpu") and "Gpu" in type(self.index).__name__
                else self.index
            )
            faiss.write_index(cpu_index, index_file)
        else:
            with open(index_file, "wb") as f:
                pickle.dump(self._raw_vectors, f)

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        print(f"[PfamRetrievalIndex] Saved {self.size} reference entries to {index_file} and {metadata_file}.")

    @classmethod
    def load(cls, index_file: str, metadata_file: str, dimension: int = 1280) -> "PfamRetrievalIndex":
        """Loads a pre-built FAISS index and metadata from disk."""
        instance = cls(dimension=dimension)

        if not os.path.isfile(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
        if not os.path.isfile(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        if FAISS_AVAILABLE:
            instance.index = faiss.read_index(index_file)
        else:
            with open(index_file, "rb") as f:
                instance._raw_vectors = pickle.load(f)

        with open(metadata_file, "r", encoding="utf-8") as f:
            instance.metadata = json.load(f)

        print(f"[PfamRetrievalIndex] Loaded {instance.size} reference entries from {index_file}.")
        return instance


def search_top_k_neighbors(
    query_embedding: Union[tr.Tensor, np.ndarray],
    index: PfamRetrievalIndex,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Convenience function to search the top-K nearest neighbors.

    Args:
        query_embedding: Protein representation [1280] or [1280, L].
        index: Initialized or loaded PfamRetrievalIndex instance.
        top_k: Number of nearest neighbors to retrieve (default: 5).

    Returns:
        List of dictionaries with keys:
            - 'rank': (1 to top_k)
            - 'header': sequence header / PID
            - 'pfam_id': assigned Pfam family ID
            - 'cosine_similarity': float in [-1, 1]
    """
    hits = index.search(query_embedding, top_k=top_k)
    return [hit.to_dict() for hit in hits]


def build_index_from_directory(
    csv_path: str,
    emb_path: str,
    dimension: int = 1280,
    max_entries: Optional[int] = None,
) -> PfamRetrievalIndex:
    """
    Builds a PfamRetrievalIndex from an ET-Pfam dataset partition CSV (e.g., train.csv)
    and corresponding embedding files (.pk) in emb_path.

    Args:
        csv_path: Path to train.csv or dev.csv (PID, start, end, PF).
        emb_path: Path to directory containing <PID>.pk embeddings.
        dimension: Embedding dimension (default: 1280).
        max_entries: Optional cap on reference entries for testing/memory limits.

    Returns:
        Populated PfamRetrievalIndex ready for nearest neighbor retrieval.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if max_entries:
        df = df.iloc[:max_entries]

    index = PfamRetrievalIndex(dimension=dimension)

    embeddings = []
    headers = []
    pfam_ids = []
    starts = []
    ends = []

    print(f"[RetrievalIndex] Ingesting {len(df)} reference entries from {csv_path}...")
    for _, row in df.iterrows():
        pid = row["PID"]
        pk_file = os.path.join(emb_path, f"{pid}.pk")
        if not os.path.isfile(pk_file):
            continue

        try:
            with open(pk_file, "rb") as f:
                emb = pickle.load(f)
            embeddings.append(emb)
            headers.append(pid)
            pfam_ids.append(row["PF"])
            starts.append(int(row["start"]) if "start" in row else None)
            ends.append(int(row["end"]) if "end" in row else None)
        except Exception as e:
            print(f"Warning: Failed to load {pk_file}: {e}")
            continue

    if embeddings:
        index.add_references(
            embeddings=embeddings,
            headers=headers,
            pfam_ids=pfam_ids,
            starts=starts,
            ends=ends,
        )
        print(f"[RetrievalIndex] Successfully indexed {index.size} reference entries.")
    else:
        print("[RetrievalIndex] Warning: No embedding files were found to index.")

    return index

"""
Retrieval-Augmented Evidence module for ET-Pfam using FAISS.
Re-exports the core implementations from src.retrieval_evidence.
"""
from src.retrieval_evidence import (
    PfamRetrievalIndex,
    RetrievalHit,
    search_top_k_neighbors,
    build_index_from_directory,
)

__all__ = [
    "PfamRetrievalIndex",
    "RetrievalHit",
    "search_top_k_neighbors",
    "build_index_from_directory",
]

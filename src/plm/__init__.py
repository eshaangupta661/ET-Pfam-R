"""
Protein Language Model (pLM) Abstraction Layer for ET-Pfam.
"""
from src.plm.base import BaseProteinLanguageModel
from src.plm.esm import ESM2ProteinLanguageModel, ESM1bProteinLanguageModel
from src.plm.registry import get_plm, register_plm, list_available_models
from src.plm.pipeline import EmbeddingPipeline

__all__ = [
    "BaseProteinLanguageModel",
    "ESM2ProteinLanguageModel",
    "ESM1bProteinLanguageModel",
    "get_plm",
    "register_plm",
    "list_available_models",
    "EmbeddingPipeline",
]

"""
Model registry and factory for Protein Language Models in ET-Pfam.
"""
from typing import Dict, Type, Any, Optional
from src.plm.base import BaseProteinLanguageModel
from src.plm.esm import ESM2ProteinLanguageModel, ESM1bProteinLanguageModel


# Registry of known aliases and model mappings
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ESM-2 Models
    "esm2": {
        "cls": ESM2ProteinLanguageModel,
        "checkpoint": "facebook/esm2_t33_650M_UR50D",
        "native_dim": 1280,
        "description": "ESM-2 650M parameter model (33 layers, 1280 dim) - Recommended for ET-Pfam",
    },
    "esm2_650m": {
        "cls": ESM2ProteinLanguageModel,
        "checkpoint": "facebook/esm2_t33_650M_UR50D",
        "native_dim": 1280,
        "description": "ESM-2 650M parameter model (33 layers, 1280 dim)",
    },
    "esm2_3b": {
        "cls": ESM2ProteinLanguageModel,
        "checkpoint": "facebook/esm2_t36_3B_UR50D",
        "native_dim": 2560,
        "description": "ESM-2 3B parameter model (36 layers, 2560 dim)",
    },
    "esm2_150m": {
        "cls": ESM2ProteinLanguageModel,
        "checkpoint": "facebook/esm2_t30_150M_UR50D",
        "native_dim": 640,
        "description": "ESM-2 150M parameter model (30 layers, 640 dim)",
    },
    "esm2_35m": {
        "cls": ESM2ProteinLanguageModel,
        "checkpoint": "facebook/esm2_t12_35M_UR50D",
        "native_dim": 480,
        "description": "ESM-2 35M parameter model (12 layers, 480 dim)",
    },
    "esm2_8m": {
        "cls": ESM2ProteinLanguageModel,
        "checkpoint": "facebook/esm2_t6_8M_UR50D",
        "native_dim": 320,
        "description": "ESM-2 8M parameter model (6 layers, 320 dim)",
    },
    # Baseline ESM-1b Model
    "baseline": {
        "cls": ESM1bProteinLanguageModel,
        "checkpoint": "facebook/esm1b_t33_650M_UR50S",
        "native_dim": 1280,
        "description": "Baseline ESM-1b 650M parameter model (33 layers, 1280 dim) used in ET-Pfam paper",
    },
    "esm1b": {
        "cls": ESM1bProteinLanguageModel,
        "checkpoint": "facebook/esm1b_t33_650M_UR50S",
        "native_dim": 1280,
        "description": "ESM-1b 650M parameter model (33 layers, 1280 dim)",
    },
}


def register_plm(
    alias: str,
    cls: Type[BaseProteinLanguageModel],
    checkpoint: str,
    native_dim: int,
    description: str = "",
) -> None:
    """Register a new pLM alias dynamically."""
    MODEL_REGISTRY[alias.lower()] = {
        "cls": cls,
        "checkpoint": checkpoint,
        "native_dim": native_dim,
        "description": description,
    }


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """Returns all registered pLM aliases and configurations."""
    return MODEL_REGISTRY.copy()


def get_plm(
    model_name_or_alias: str = "facebook/esm2_t33_650M_UR50D",
    device: str = "cuda",
    target_dim: int = 1280,
    load_weights: bool = True,
    **kwargs: Any,
) -> BaseProteinLanguageModel:
    """
    Factory function to instantiate a Protein Language Model.

    Args:
        model_name_or_alias: Alias (e.g., 'esm2', 'baseline', 'esm2_150m') or
                             HuggingFace model identifier (e.g., 'facebook/esm2_t33_650M_UR50D').
        device: Device to place the model on ('cuda', 'cpu', or device index).
        target_dim: Target channel dimension for ET-Pfam compatibility (default 1280).
        load_weights: If True, calls load_model() immediately.
        **kwargs: Extra parameters passed to the model constructor (max_length, repr_layer, etc.).

    Returns:
        An instance of BaseProteinLanguageModel configured and ready for embedding extraction.
    """
    key = model_name_or_alias.lower()

    if key in MODEL_REGISTRY:
        info = MODEL_REGISTRY[key]
        model_cls = info["cls"]
        checkpoint = info["checkpoint"]
    else:
        # Determine class based on checkpoint string
        checkpoint = model_name_or_alias
        if "esm1" in key or "esm1b" in key:
            model_cls = ESM1bProteinLanguageModel
        else:
            # Default to ESM2 class for any ESM-2 or unrecognized model
            model_cls = ESM2ProteinLanguageModel

    plm_instance = model_cls(
        model_name=checkpoint,
        device=device,
        target_dim=target_dim,
        **kwargs,
    )

    if load_weights:
        plm_instance.load_model()

    return plm_instance

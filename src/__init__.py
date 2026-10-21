"""
ET-Pfam: Ensemble transfer learning for protein family prediction.
"""
from src.basemodel import BaseModel, ResidualLayer
from src.ensemble import EnsembleModel
from src.dataset import PFamDataset
from src.plm import (
    BaseProteinLanguageModel,
    ESM2ProteinLanguageModel,
    ESM1bProteinLanguageModel,
    get_plm,
    EmbeddingPipeline,
)
from src.uncertainty_calibration import (
    ConformalPredictionWrapper,
    compute_random_neighbor_score,
    OODDetector,
)
from src.retrieval_evidence import (
    PfamRetrievalIndex,
    RetrievalHit,
    search_top_k_neighbors,
    build_index_from_directory,
)
from src.sequence_explainability import (
    SaliencyExplainer,
    AttentionExplainer,
    DomainAttributionMapper,
    DomainAttributionReport,
    FunctionalResidue,
    ConservedMotif,
    explain_prediction,
)

__all__ = [
    "BaseModel",
    "ResidualLayer",
    "EnsembleModel",
    "PFamDataset",
    "BaseProteinLanguageModel",
    "ESM2ProteinLanguageModel",
    "ESM1bProteinLanguageModel",
    "get_plm",
    "EmbeddingPipeline",
    "ConformalPredictionWrapper",
    "compute_random_neighbor_score",
    "OODDetector",
    "PfamRetrievalIndex",
    "RetrievalHit",
    "search_top_k_neighbors",
    "build_index_from_directory",
    "SaliencyExplainer",
    "AttentionExplainer",
    "DomainAttributionMapper",
    "DomainAttributionReport",
    "FunctionalResidue",
    "ConservedMotif",
    "explain_prediction",
]




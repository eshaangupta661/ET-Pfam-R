"""
Sequence Explainability and Domain Attribution module for ET-Pfam.
Re-exports the core implementations from src.sequence_explainability.
"""
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
    "SaliencyExplainer",
    "AttentionExplainer",
    "DomainAttributionMapper",
    "DomainAttributionReport",
    "FunctionalResidue",
    "ConservedMotif",
    "explain_prediction",
]

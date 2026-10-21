"""
Base abstraction for Protein Language Models (pLMs) in ET-Pfam.
"""
from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any, Optional
import re
import torch as tr
import torch.nn as nn


# Canonical standard 20 amino acids
STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


class BaseProteinLanguageModel(nn.Module, ABC):
    """
    Abstract base class for all Protein Language Models (pLMs).

    Downstream Shape Compatibility Requirement:
        The ET-Pfam downstream pipeline (PFamDataset, BaseModel CNN, and
        sliding_window_test) expects stored per-residue embeddings of shape:
            [embedding_dim, L]
        where:
            - embedding_dim: Channel dimension (typically 1280, matching BaseModel.emb_size)
            - L: Sequence length in amino acid residues (excluding special tokens like <cls>/<eos>)

        This base class enforces that any extracted embedding tensor matches this
        [D, L] convention and provides optional linear projection if the native pLM
        dimension does not match the target classifier dimension (default 1280).
    """

    def __init__(
        self,
        model_name: str,
        device: Union[str, tr.device] = "cuda" if tr.cuda.is_available() else "cpu",
        target_dim: int = 1280,
        dtype: tr.dtype = tr.float32,
        max_length: Optional[int] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = tr.device(device) if isinstance(device, str) else device
        self.target_dim = target_dim
        self.inference_dtype = dtype
        self.max_length = max_length

        # Model and tokenizer / alphabet references
        self.model: Optional[nn.Module] = None
        self.tokenizer: Any = None
        self._embedding_dim: Optional[int] = None

        # Optional dimension projection layer if native pLM dimension != target_dim
        self.projection: Optional[nn.Linear] = None

    @property
    def embedding_dim(self) -> int:
        """Native hidden dimension of the pLM."""
        if self._embedding_dim is None:
            raise ValueError("Model has not been loaded yet; call `load_model()` first.")
        return self._embedding_dim

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the pLM weights and tokenizer / alphabet onto the specified device.
        Must set `self.model`, `self.tokenizer`, and `self._embedding_dim`.
        """
        pass

    def setup_dimension_adapter(self) -> None:
        """
        Sets up an adapter layer if the native pLM hidden dimension does not
        match `target_dim` (default 1280).
        """
        if self.embedding_dim != self.target_dim:
            print(
                f"[pLM Abstraction] Warning: Native pLM dimension ({self.embedding_dim}) "
                f"differs from target downstream dimension ({self.target_dim}). "
                f"Initializing dimension projection layer: {self.embedding_dim} -> {self.target_dim}."
            )
            self.projection = nn.Linear(self.embedding_dim, self.target_dim).to(
                self.device
            )
        else:
            self.projection = None

    @staticmethod
    def clean_sequence(sequence: str) -> str:
        """
        Validates, strips whitespace, and cleans protein sequences.
        Converts non-standard/ambiguous amino acids to standard tokens or 'X'.
        """
        cleaned = re.sub(r"\s+", "", sequence).upper()
        if not cleaned:
            raise ValueError("Encountered empty protein sequence.")

        # Replace non-standard amino acids with 'X'
        # e.g., U (Selenocysteine), O (Pyrrolysine), B (Asx), Z (Glx), J (Xle)
        cleaned = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", cleaned)
        return cleaned

    @abstractmethod
    def _extract_raw_embeddings(self, sequences: List[str]) -> List[tr.Tensor]:
        """
        Extract raw per-residue embeddings from the pLM for a batch of sequences.

        Args:
            sequences: List of cleaned amino acid strings.

        Returns:
            List of tensors, one per sequence, each of shape [L_i, native_dim],
            where special tokens (<cls>, <eos>, padding) have been stripped.
        """
        pass

    def format_embedding(self, raw_embedding: tr.Tensor) -> tr.Tensor:
        """
        Formats a single raw sequence embedding [L, D] to the downstream
        format [target_dim, L].

        Steps:
            1. Cast to float32 on CPU for storage consistency.
            2. Apply dimension projection if native_dim != target_dim.
            3. Transpose from [L, D] to [D, L].
            4. Verify final tensor shape.
        """
        # Ensure 2D tensor [L, D]
        if raw_embedding.dim() != 2:
            raise ValueError(
                f"Expected 2D embedding tensor [L, D], got shape {raw_embedding.shape}"
            )

        # Apply projection if dimensions differ
        if self.projection is not None:
            raw_embedding = self.projection(raw_embedding.to(self.device))

        # Transpose [L, D] -> [D, L] to match ET-Pfam convention
        formatted = raw_embedding.transpose(0, 1).contiguous().to(tr.float32).cpu()

        if formatted.shape[0] != self.target_dim:
            raise ValueError(
                f"Embedding channel dimension ({formatted.shape[0]}) does not match "
                f"downstream target_dim ({self.target_dim})."
            )

        return formatted

    def extract_sequence_embedding(self, sequence: str) -> tr.Tensor:
        """
        Extract formatted per-residue embedding for a single sequence.

        Returns:
            torch.FloatTensor of shape [target_dim, L]
        """
        return self.extract_batch_embeddings([sequence])[0]

    def extract_batch_embeddings(self, sequences: List[str]) -> List[tr.Tensor]:
        """
        Extract formatted per-residue embeddings for a batch of sequences.

        Args:
            sequences: List of amino acid sequences.

        Returns:
            List of torch.FloatTensor, each of shape [target_dim, L_i].
        """
        cleaned_seqs = [self.clean_sequence(s) for s in sequences]
        raw_list = self._extract_raw_embeddings(cleaned_seqs)

        formatted_list = []
        for raw_emb in raw_list:
            formatted = self.format_embedding(raw_emb)
            formatted_list.append(formatted)

        return formatted_list

    def forward(self, sequences: Union[str, List[str]]) -> Union[tr.Tensor, List[tr.Tensor]]:
        """PyTorch module forward pass."""
        if isinstance(sequences, str):
            return self.extract_sequence_embedding(sequences)
        return self.extract_batch_embeddings(sequences)

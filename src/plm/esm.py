"""
ESM-2 and ESM-1b (Baseline) Protein Language Model implementations.
"""
from typing import List, Union, Optional
import torch as tr
import torch.nn as nn
from src.plm.base import BaseProteinLanguageModel


class ESM2ProteinLanguageModel(BaseProteinLanguageModel):
    """
    ESM-2 Protein Language Model implementation.

    Supported model identifiers (via Hugging Face or fair-esm):
        - 'facebook/esm2_t33_650M_UR50D' (1280-dim, default, exact match for ET-Pfam)
        - 'facebook/esm2_t36_3B_UR50D'   (2560-dim)
        - 'facebook/esm2_t30_150M_UR50D' (640-dim)
        - 'facebook/esm2_t12_35M_UR50D'  (480-dim)
        - 'facebook/esm2_t6_8M_UR50D'    (320-dim)
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: Union[str, tr.device] = "cuda" if tr.cuda.is_available() else "cpu",
        target_dim: int = 1280,
        dtype: tr.dtype = tr.float32,
        repr_layer: Optional[int] = None,
        max_length: Optional[int] = 1024,
        backend: str = "auto",  # 'auto', 'transformers', or 'fair-esm'
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            target_dim=target_dim,
            dtype=dtype,
            max_length=max_length,
        )
        self.repr_layer = repr_layer
        self.backend = backend
        self.batch_converter = None

    def load_model(self) -> None:
        """Loads ESM-2 model and tokenizer via transformers or fair-esm."""
        loaded = False
        backend = self.backend

        if backend in ("auto", "transformers"):
            try:
                from transformers import AutoTokenizer, AutoModel, EsmModel

                print(f"[pLM: ESM-2] Loading {self.model_name} via HuggingFace transformers...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=self.inference_dtype,
                )
                self.model.eval().to(self.device)
                self._embedding_dim = self.model.config.hidden_size
                self.backend = "transformers"
                loaded = True
                print(
                    f"[pLM: ESM-2] Model loaded successfully. "
                    f"Hidden dimension: {self._embedding_dim}, Device: {self.device}."
                )
            except Exception as e:
                if backend == "transformers":
                    raise RuntimeError(
                        f"Failed to load ESM-2 model with transformers: {e}"
                    ) from e
                print(
                    f"[pLM: ESM-2] HuggingFace transformers load failed ({e}), "
                    "falling back to fair-esm..."
                )

        if not loaded and backend in ("auto", "fair-esm"):
            try:
                import esm

                # Normalize model name for fair-esm if needed
                short_name = self.model_name.split("/")[-1]
                if hasattr(esm.pretrained, short_name):
                    loader_fn = getattr(esm.pretrained, short_name)
                    self.model, alphabet = loader_fn()
                    self.batch_converter = alphabet.get_batch_converter()
                    self.model.eval().to(self.device)
                    self._embedding_dim = self.model.embed_dim
                    self.backend = "fair-esm"
                    loaded = True
                    print(
                        f"[pLM: ESM-2] Model loaded via fair-esm. "
                        f"Hidden dimension: {self._embedding_dim}, Device: {self.device}."
                    )
                else:
                    raise ValueError(f"Model {short_name} not found in esm.pretrained.")
            except Exception as e:
                raise RuntimeError(
                    f"Could not load ESM-2 model using either transformers or fair-esm. Error: {e}\n"
                    "Please install transformers (`pip install transformers`) or fair-esm (`pip install fair-esm`)."
                ) from e

        # Set up dimension adapter if embedding_dim != target_dim (1280)
        self.setup_dimension_adapter()

    def _extract_raw_embeddings(self, sequences: List[str]) -> List[tr.Tensor]:
        """
        Extracts raw per-residue embeddings for each sequence, stripping special tokens.
        Returns List of [L_i, D] tensors.
        """
        if self.model is None:
            self.load_model()

        if self.backend == "transformers":
            return self._extract_with_transformers(sequences)
        elif self.backend == "fair-esm":
            return self._extract_with_fair_esm(sequences)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _extract_with_transformers(self, sequences: List[str]) -> List[tr.Tensor]:
        # Tokenize with padding
        tok_kwargs = {
            "return_tensors": "pt",
            "padding": True,
        }
        if self.max_length is not None:
            tok_kwargs["truncation"] = True
            tok_kwargs["max_length"] = self.max_length

        inputs = self.tokenizer(sequences, **tok_kwargs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with tr.no_grad():
            outputs = self.model(**inputs)
            # last_hidden_state: [batch_size, seq_len_with_tokens, hidden_dim]
            last_hidden = outputs.last_hidden_state

        raw_embeddings = []
        for i, seq in enumerate(sequences):
            actual_len = len(seq)
            # Index 0 is <cls>, index 1 to actual_len is the residue sequence
            # Index actual_len + 1 is <eos>
            seq_emb = last_hidden[i, 1 : 1 + actual_len, :]  # [actual_len, D]
            raw_embeddings.append(seq_emb)

        return raw_embeddings

    def _extract_with_fair_esm(self, sequences: List[str]) -> List[tr.Tensor]:
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        if self.max_length is not None and batch_tokens.shape[1] > self.max_length:
            batch_tokens = batch_tokens[:, : self.max_length]

        # Determine target layer (last layer if not specified)
        target_layer = self.repr_layer or self.model.num_layers

        with tr.no_grad():
            results = self.model(
                batch_tokens,
                repr_layers=[target_layer],
                return_contacts=False,
            )
            token_representations = results["representations"][target_layer]

        raw_embeddings = []
        for i, seq in enumerate(sequences):
            actual_len = len(seq)
            # Index 1 : 1 + actual_len strips <cls> and <eos>
            seq_emb = token_representations[i, 1 : 1 + actual_len, :]
            raw_embeddings.append(seq_emb)

        return raw_embeddings


class ESM1bProteinLanguageModel(BaseProteinLanguageModel):
    """
    Baseline ESM-1b Protein Language Model (esm1b_t33_650M_UR50S).

    Used as the original baseline pLM in the ET-Pfam study.
    Native hidden dimension: 1280.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm1b_t33_650M_UR50S",
        device: Union[str, tr.device] = "cuda" if tr.cuda.is_available() else "cpu",
        target_dim: int = 1280,
        dtype: tr.dtype = tr.float32,
        repr_layer: Optional[int] = None,
        max_length: Optional[int] = 1024,
        backend: str = "auto",
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            target_dim=target_dim,
            dtype=dtype,
            max_length=max_length,
        )
        self.repr_layer = repr_layer
        self.backend = backend
        self.batch_converter = None

    def load_model(self) -> None:
        """Loads ESM-1b baseline model and tokenizer."""
        loaded = False
        backend = self.backend

        if backend in ("auto", "transformers"):
            try:
                from transformers import AutoTokenizer, AutoModel

                print(f"[pLM: ESM-1b Baseline] Loading {self.model_name} via HuggingFace transformers...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=self.inference_dtype,
                )
                self.model.eval().to(self.device)
                self._embedding_dim = self.model.config.hidden_size
                self.backend = "transformers"
                loaded = True
                print(
                    f"[pLM: ESM-1b Baseline] Model loaded successfully. "
                    f"Hidden dimension: {self._embedding_dim}, Device: {self.device}."
                )
            except Exception as e:
                if backend == "transformers":
                    raise RuntimeError(
                        f"Failed to load ESM-1b model with transformers: {e}"
                    ) from e
                print(
                    f"[pLM: ESM-1b Baseline] HuggingFace transformers load failed ({e}), "
                    "falling back to fair-esm..."
                )

        if not loaded and backend in ("auto", "fair-esm"):
            try:
                import esm

                short_name = self.model_name.split("/")[-1]
                if short_name == "esm1b_t33_650M_UR50S" or hasattr(esm.pretrained, "esm1b_t33_650M_UR50S"):
                    self.model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
                    self.batch_converter = alphabet.get_batch_converter()
                    self.model.eval().to(self.device)
                    self._embedding_dim = self.model.embed_dim
                    self.backend = "fair-esm"
                    loaded = True
                    print(
                        f"[pLM: ESM-1b Baseline] Model loaded via fair-esm. "
                        f"Hidden dimension: {self._embedding_dim}, Device: {self.device}."
                    )
                else:
                    raise ValueError(f"Model {short_name} not available in esm.pretrained.")
            except Exception as e:
                raise RuntimeError(
                    f"Could not load ESM-1b baseline model. Error: {e}"
                ) from e

        self.setup_dimension_adapter()

    def _extract_raw_embeddings(self, sequences: List[str]) -> List[tr.Tensor]:
        if self.model is None:
            self.load_model()

        if self.backend == "transformers":
            tok_kwargs = {
                "return_tensors": "pt",
                "padding": True,
            }
            if self.max_length is not None:
                tok_kwargs["truncation"] = True
                tok_kwargs["max_length"] = self.max_length

            inputs = self.tokenizer(sequences, **tok_kwargs)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with tr.no_grad():
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state

            raw_embeddings = []
            for i, seq in enumerate(sequences):
                actual_len = len(seq)
                seq_emb = last_hidden[i, 1 : 1 + actual_len, :]
                raw_embeddings.append(seq_emb)
            return raw_embeddings

        elif self.backend == "fair-esm":
            data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)

            if self.max_length is not None and batch_tokens.shape[1] > self.max_length:
                batch_tokens = batch_tokens[:, : self.max_length]

            target_layer = self.repr_layer or 33
            with tr.no_grad():
                results = self.model(
                    batch_tokens,
                    repr_layers=[target_layer],
                    return_contacts=False,
                )
                token_representations = results["representations"][target_layer]

            raw_embeddings = []
            for i, seq in enumerate(sequences):
                actual_len = len(seq)
                seq_emb = token_representations[i, 1 : 1 + actual_len, :]
                raw_embeddings.append(seq_emb)
            return raw_embeddings
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

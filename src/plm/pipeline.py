"""
End-to-end Embedding Generation Pipeline for ET-Pfam.
"""
import os
import pickle
from typing import Dict, List, Tuple, Union, Optional
import torch as tr
from tqdm import tqdm
from src.plm.base import BaseProteinLanguageModel
from src.plm.registry import get_plm


class EmbeddingPipeline:
    """
    High-level embedding generation pipeline that:
        1. Ingests protein sequences from FASTA, CSV, or Python dictionaries.
        2. Batches and feeds sequences through a modular Protein Language Model (pLM).
        3. Formats outputs to exact ET-Pfam tensor specifications ([1280, L]).
        4. Serializes embeddings to <output_dir>/<PID>.pk files.
        5. Verifies compatibility against downstream ET-Pfam classifiers.
    """

    def __init__(
        self,
        plm: Union[BaseProteinLanguageModel, str] = "facebook/esm2_t33_650M_UR50D",
        output_dir: str = "data/embeddings/",
        device: str = "cuda" if tr.cuda.is_available() else "cpu",
        target_dim: int = 1280,
        batch_size: int = 8,
        overwrite: bool = False,
    ):
        if isinstance(plm, str):
            self.plm = get_plm(plm, device=device, target_dim=target_dim)
        else:
            self.plm = plm

        self.output_dir = output_dir
        self.batch_size = batch_size
        self.overwrite = overwrite
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def parse_fasta(fasta_path: str) -> Dict[str, str]:
        """
        Parses a FASTA file into a dictionary of {PID: sequence}.
        Handles standard FASTA format, extracting ID from the header line.
        """
        sequences = {}
        current_id = None
        current_seq = []

        with open(fasta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_id is not None:
                        sequences[current_id] = "".join(current_seq)
                    # Extract PID: first token after '>', handling UniProt style (e.g. >tr|E3QE54|... or >PID)
                    header = line[1:].strip()
                    parts = header.split()
                    raw_id = parts[0]
                    if "|" in raw_id:
                        # UniProt header: >db|UniqueIdentifier|EntryName
                        pipe_parts = raw_id.split("|")
                        current_id = pipe_parts[1] if len(pipe_parts) > 1 else raw_id
                    else:
                        current_id = raw_id
                    current_seq = []
                else:
                    current_seq.append(line)

            if current_id is not None:
                sequences[current_id] = "".join(current_seq)

        print(f"[Pipeline] Parsed {len(sequences)} sequences from {fasta_path}.")
        return sequences

    def process_sequences(
        self,
        sequences: Union[Dict[str, str], List[Tuple[str, str]]],
    ) -> Dict[str, str]:
        """
        Extracts embeddings for a collection of (PID, sequence) items
        and saves each to <output_dir>/<PID>.pk.

        Args:
            sequences: Dictionary of {PID: sequence} or list of (PID, sequence).

        Returns:
            Dictionary mapping PID to saved .pk file path.
        """
        if isinstance(sequences, dict):
            items = list(sequences.items())
        else:
            items = list(sequences)

        # Filter already existing files if not overwrite
        if not self.overwrite:
            filtered_items = []
            for pid, seq in items:
                out_path = os.path.join(self.output_dir, f"{pid}.pk")
                if not os.path.exists(out_path):
                    filtered_items.append((pid, seq))
            skipped = len(items) - len(filtered_items)
            if skipped > 0:
                print(f"[Pipeline] Skipping {skipped} already existing embeddings (overwrite=False).")
            items = filtered_items

        if not items:
            print("[Pipeline] No sequences to process.")
            return {}

        saved_files = {}

        # Process in batches
        for i in tqdm(range(0, len(items), self.batch_size), desc="Extracting embeddings"):
            batch = items[i : i + self.batch_size]
            pids = [item[0] for item in batch]
            seqs = [item[1] for item in batch]

            # Extract formatted embeddings: each is [target_dim, L_i]
            embs = self.plm.extract_batch_embeddings(seqs)

            for pid, emb in zip(pids, embs):
                out_path = os.path.join(self.output_dir, f"{pid}.pk")
                with open(out_path, "wb") as f:
                    pickle.dump(emb, f, protocol=pickle.HIGHEST_PROTOCOL)
                saved_files[pid] = out_path

        print(f"[Pipeline] Successfully generated and saved {len(saved_files)} embeddings.")
        return saved_files

    def process_fasta(self, fasta_path: str) -> Dict[str, str]:
        """Convenience method to process all sequences from a FASTA file."""
        sequences = self.parse_fasta(fasta_path)
        return self.process_sequences(sequences)

    def verify_compatibility(self, sample_pid: str) -> bool:
        """
        Verifies that a generated embedding file adheres to the downstream
        shape, dtype, and indexing conventions required by ET-Pfam:
            - File exists and loads cleanly via pickle.load
            - Tensor is 2D with shape [target_dim, L]
            - Dimension 0 is channel dimension (1280)
            - Slicing along dimension 1 (sequence length) matches window extraction
        """
        file_path = os.path.join(self.output_dir, f"{sample_pid}.pk")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Embedding file not found: {file_path}")

        with open(file_path, "rb") as f:
            emb = pickle.load(f)

        if not isinstance(emb, tr.Tensor):
            raise TypeError(f"Loaded object is not a torch.Tensor, got {type(emb)}")

        emb = emb.squeeze().float()

        if emb.dim() != 2:
            raise ValueError(f"Expected 2D tensor [D, L], got {emb.shape}")

        channels, length = emb.shape[0], emb.shape[1]
        if channels != self.plm.target_dim:
            raise ValueError(
                f"Channel dimension mismatch: got {channels}, expected {self.plm.target_dim}."
            )

        # Simulate PFamDataset window extraction
        test_win_len = 64
        test_window = tr.zeros((channels, test_win_len), dtype=tr.float)
        slice_end = min(length, test_win_len)
        test_window[:, :slice_end] = emb[:, :slice_end]

        # Simulate BaseModel Conv1d forward pass batch [1, channels, test_win_len]
        batch = test_window.unsqueeze(0)
        conv = tr.nn.Conv1d(channels, 1100, kernel_size=9, padding="same")
        out = conv(batch)

        print(f"[Verification PASSED] Protein '{sample_pid}':")
        print(f"  - Shape: [{channels}, {length}] (Channels x Length)")
        print(f"  - Dtype: {emb.dtype}")
        print(f"  - Window slice test shape: {test_window.shape}")
        print(f"  - Conv1d output shape: {out.shape}")
        return True

"""
Sequence Explainability and Domain Attribution Module for ET-Pfam.

Provides:
1. SaliencyExplainer:
   Computes per-residue attribution maps across input sequence embeddings using
   gradient-based methods:
     - Vanilla Saliency (|grad|)
     - Input x Gradient (element-wise product of input and gradient)
     - Integrated Gradients (Sundararajan et al., 2017)
2. AttentionExplainer:
   Extracts and aggregates multi-head self-attention weights from Transformer pLMs
   (e.g., ESM-2 / ESM-1b) across sequence positions.
3. DomainAttributionMapper:
   Maps residue-level attribution scores to annotated or predicted Pfam domains,
   calculating domain enrichment, identifying key functional residues, and
   detecting high-importance conserved functional motifs.
"""
from typing import List, Dict, Tuple, Union, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
import torch as tr
import torch.nn as nn


@dataclass
class FunctionalResidue:
    """Represents a key functional residue identified by attribution scoring."""
    position: int              # 1-based residue position
    amino_acid: Optional[str]  # Amino acid single-letter code if sequence provided
    attribution_score: float   # Normalized importance score in [0, 1]
    is_in_domain: bool         # Whether the residue lies within the Pfam domain
    percentile: float          # Percentile rank across the sequence (0-100%)


@dataclass
class ConservedMotif:
    """Represents a contiguous high-importance sequence segment/motif."""
    start: int                 # 1-based start position
    end: int                   # 1-based end position
    length: int
    mean_score: float
    sequence_slice: Optional[str] = None


@dataclass
class DomainAttributionReport:
    """Comprehensive explainability report for a predicted Pfam domain."""
    predicted_pfam: str
    target_class_idx: int
    sequence_length: int
    domain_start: Optional[int]
    domain_end: Optional[int]
    domain_enrichment_ratio: float  # Mean domain score / Mean background score
    per_residue_scores: List[float] # Normalized [0, 1] scores per position
    key_residues: List[FunctionalResidue]
    conserved_motifs: List[ConservedMotif]
    method: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    def summary(self) -> str:
        lines = [
            "=" * 65,
            f"Pfam Domain Attribution Report: {self.predicted_pfam}",
            "=" * 65,
            f"Method                  : {self.method}",
            f"Sequence Length         : {self.sequence_length} residues",
            f"Pfam Domain Region      : [{self.domain_start}, {self.domain_end}]" if self.domain_start else "Pfam Domain Region      : Full Sequence",
            f"Domain Enrichment Ratio : {self.domain_enrichment_ratio:.2f}x (domain vs background)",
            "-" * 65,
            "Top Key Functional Residues:",
        ]
        for res in self.key_residues[:5]:
            aa = res.amino_acid or "?"
            dom_str = "[Domain]" if res.is_in_domain else "[Flanking]"
            lines.append(
                f"  Pos {res.position:4d} ({aa}) : Score = {res.attribution_score:.4f} "
                f"({res.percentile:5.1f}%ile) {dom_str}"
            )
        if self.conserved_motifs:
            lines.append("-" * 65)
            lines.append("Highlighted Conserved Motifs / High-Attribution Cores:")
            for m in self.conserved_motifs[:3]:
                motif_str = f" '{m.sequence_slice}'" if m.sequence_slice else ""
                lines.append(
                    f"  Region [{m.start:4d} - {m.end:4d}] (len {m.length}): "
                    f"Mean Score = {m.mean_score:.4f}{motif_str}"
                )
        lines.append("=" * 65)
        return "\n".join(lines)


class SaliencyExplainer:
    """
    Gradient-based saliency and attribution explainer for ET-Pfam CNN models (BaseModel).

    Supported Methods:
        - 'input_x_grad': Input * Gradient (efficient, captures magnitude & sign).
        - 'integrated_gradients': Path integral from zero baseline to input (axiomatic, smooth).
        - 'vanilla_grad': Pure gradient magnitude ||grad||_2.
    """

    def __init__(self, model: nn.Module, device: Optional[Union[str, tr.device]] = None):
        self.model = model
        self.device = (
            device
            if device is not None
            else getattr(model, "device", "cuda" if tr.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()

    def attribute(
        self,
        embedding: Union[tr.Tensor, np.ndarray],
        target_class: Optional[int] = None,
        method: str = "input_x_grad",
        n_steps: int = 25,
    ) -> np.ndarray:
        """
        Computes per-residue attribution scores for a sequence embedding.

        Args:
            embedding: Sequence embedding of shape [D, L] or [1, D, L] (e.g. 1280 x L).
            target_class: Target Pfam class index. If None, uses model's predicted argmax.
            method: Attribution method ('input_x_grad', 'integrated_gradients', 'vanilla_grad').
            n_steps: Number of interpolation steps for integrated gradients.

        Returns:
            attribution_scores: 1D numpy array of shape [L] with per-residue scores.
        """
        if isinstance(embedding, np.ndarray):
            emb_tensor = tr.from_numpy(embedding).float()
        else:
            emb_tensor = embedding.clone().detach().float()

        if emb_tensor.dim() == 2:
            # Add batch dimension: [1, D, L]
            emb_tensor = emb_tensor.unsqueeze(0)

        emb_tensor = emb_tensor.to(self.device)

        # Determine target class if not given
        if target_class is None:
            with tr.no_grad():
                logits = self.model(emb_tensor)
                target_class = int(tr.argmax(logits, dim=1).item())

        if method == "input_x_grad":
            scores = self._input_x_grad(emb_tensor, target_class)
        elif method == "integrated_gradients":
            scores = self._integrated_gradients(emb_tensor, target_class, n_steps=n_steps)
        elif method == "vanilla_grad":
            scores = self._vanilla_grad(emb_tensor, target_class)
        else:
            raise ValueError(f"Unknown attribution method: {method}")

        return scores

    def _vanilla_grad(self, x: tr.Tensor, target_class: int) -> np.ndarray:
        x = x.clone().detach().requires_grad_(True)
        out = self.model(x)
        score = out[0, target_class]
        score.backward()
        # L2 norm across channel dimension D -> [L]
        grad_norm = tr.norm(x.grad[0], p=2, dim=0)
        return grad_norm.detach().cpu().numpy()

    def _input_x_grad(self, x: tr.Tensor, target_class: int) -> np.ndarray:
        x = x.clone().detach().requires_grad_(True)
        out = self.model(x)
        score = out[0, target_class]
        score.backward()
        # Element-wise product Input * Grad, take L2 norm across channels -> [L]
        input_x_grad = tr.norm(x[0] * x.grad[0], p=2, dim=0)
        return input_x_grad.detach().cpu().numpy()

    def _integrated_gradients(
        self, x: tr.Tensor, target_class: int, n_steps: int = 25
    ) -> np.ndarray:
        # Zero baseline of identical shape [1, D, L]
        baseline = tr.zeros_like(x)
        diff = x - baseline

        accumulated_grads = tr.zeros_like(x)

        # Reimann approximation of the path integral
        for step in range(1, n_steps + 1):
            alpha = float(step) / n_steps
            interpolated = (baseline + alpha * diff).detach().requires_grad_(True)
            out = self.model(interpolated)
            score = out[0, target_class]
            score.backward()
            accumulated_grads += interpolated.grad

        avg_grads = accumulated_grads / n_steps
        ig = tr.norm(diff[0] * avg_grads[0], p=2, dim=0)
        return ig.detach().cpu().numpy()


class AttentionExplainer:
    """
    Attention-weight extraction for Transformer-based Protein Language Models (ESM-2 / ESM-1b).
    Extracts multi-head attention matrices and aggregates them to per-residue importance scores.
    """

    def __init__(self, plm_model: Any, device: Optional[str] = None):
        self.plm_model = plm_model
        self.device = device or getattr(plm_model, "device", "cpu")

    def attribute_sequence(
        self,
        sequence: str,
        layer_index: int = -1,
        aggregation: str = "mean",
    ) -> np.ndarray:
        """
        Extracts residue attention weights from the underlying Transformer pLM.

        Args:
            sequence: Raw amino acid sequence string.
            layer_index: Layer to extract attention from (-1 for last layer).
            aggregation: Aggregation across attention heads ('mean' or 'max').

        Returns:
            per_residue_attention: 1D numpy array of shape [L].
        """
        raw_model = getattr(self.plm_model, "model", None)
        tokenizer = getattr(self.plm_model, "tokenizer", None)

        if raw_model is None or tokenizer is None:
            # If mock or non-transformer, return uniform attention
            return np.ones(len(sequence), dtype=np.float32) / len(sequence)

        inputs = tokenizer([sequence], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with tr.no_grad():
            outputs = raw_model(**inputs, output_attentions=True)
            # Tuple of [batch_size, num_heads, seq_len, seq_len] per layer
            attentions = outputs.attentions

        layer_att = attentions[layer_index][0]  # [num_heads, total_len, total_len]

        if aggregation == "mean":
            head_agg = tr.mean(layer_att, dim=0)  # [total_len, total_len]
        else:
            head_agg = tr.max(layer_att, dim=0)[0]

        # Slicing out <cls> (index 0) and <eos> (index len+1)
        # Residue positions are 1 to len(sequence)
        L = len(sequence)
        residue_att = head_agg[1 : 1 + L, 1 : 1 + L]

        # Total incoming attention received by each residue position
        col_attention = tr.sum(residue_att, dim=0).cpu().numpy()
        return col_attention.astype(np.float32)


class DomainAttributionMapper:
    """
    Maps raw residue attribution scores to Pfam domain annotations.
    Identifies functional residues, enrichment factors, and conserved core motifs.
    """

    @staticmethod
    def normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Min-max normalizes scores to [0, 1]."""
        min_v = np.min(scores)
        max_v = np.max(scores)
        if max_v - min_v > 1e-12:
            return (scores - min_v) / (max_v - min_v)
        return np.ones_like(scores)

    @classmethod
    def map_to_domain(
        cls,
        scores: np.ndarray,
        domain_start: Optional[int] = None,
        domain_end: Optional[int] = None,
        sequence: Optional[str] = None,
        predicted_pfam: str = "Unknown_Pfam",
        target_class_idx: int = 0,
        top_k_residues: int = 10,
        motif_threshold_percentile: float = 85.0,
        min_motif_length: int = 3,
        method: str = "input_x_grad",
    ) -> DomainAttributionReport:
        """
        Analyzes per-residue attribution with respect to the Pfam domain interval.

        Args:
            scores: Raw 1D array of per-residue attribution values [L].
            domain_start: 1-based start coordinate of Pfam domain (inclusive).
            domain_end: 1-based end coordinate of Pfam domain (inclusive).
            sequence: Optional amino acid sequence string [L].
            predicted_pfam: Pfam family name string.
            target_class_idx: Integer class index.
            top_k_residues: Number of top key residues to highlight.
            motif_threshold_percentile: Score percentile to trigger conserved motif grouping.
            min_motif_length: Minimum contiguous residues to form a conserved motif.
            method: Attribution method string.

        Returns:
            DomainAttributionReport object.
        """
        L = len(scores)
        norm_scores = cls.normalize_scores(scores)

        # Convert coordinates to 0-based indexing for slicing
        if domain_start is not None and domain_end is not None:
            s_idx = max(0, domain_start - 1)
            e_idx = min(L, domain_end)
            domain_slice = norm_scores[s_idx:e_idx]

            # Calculate flanking / background scores
            bg_indices = list(range(0, s_idx)) + list(range(e_idx, L))
            if bg_indices:
                bg_slice = norm_scores[bg_indices]
                mean_domain = float(np.mean(domain_slice))
                mean_bg = float(np.mean(bg_slice))
                enrichment = mean_domain / max(1e-6, mean_bg)
            else:
                enrichment = 1.0
        else:
            s_idx, e_idx = 0, L
            enrichment = 1.0

        # Identify Top-K Functional Residues
        sorted_indices = np.argsort(-norm_scores)
        key_residues = []
        for rank_idx in sorted_indices[:top_k_residues]:
            pos_1based = int(rank_idx + 1)
            aa = sequence[rank_idx] if sequence and rank_idx < len(sequence) else None
            is_in_dom = (domain_start is not None and domain_end is not None) and (
                domain_start <= pos_1based <= domain_end
            )
            pct = float(np.mean(norm_scores <= norm_scores[rank_idx]) * 100.0)

            key_residues.append(
                FunctionalResidue(
                    position=pos_1based,
                    amino_acid=aa,
                    attribution_score=float(norm_scores[rank_idx]),
                    is_in_domain=is_in_dom,
                    percentile=pct,
                )
            )

        # Detect High-Attribution Conserved Motifs (contiguous regions above threshold)
        cutoff = float(np.percentile(norm_scores, motif_threshold_percentile))
        high_mask = norm_scores >= cutoff

        motifs = []
        in_segment = False
        seg_start = 0

        for i in range(L):
            if high_mask[i] and not in_segment:
                in_segment = True
                seg_start = i
            elif not high_mask[i] and in_segment:
                in_segment = False
                seg_len = i - seg_start
                if seg_len >= min_motif_length:
                    mean_sc = float(np.mean(norm_scores[seg_start:i]))
                    seq_slice = sequence[seg_start:i] if sequence else None
                    motifs.append(
                        ConservedMotif(
                            start=seg_start + 1,
                            end=i,
                            length=seg_len,
                            mean_score=mean_sc,
                            sequence_slice=seq_slice,
                        )
                    )

        # Handle segment running to end of sequence
        if in_segment and (L - seg_start) >= min_motif_length:
            mean_sc = float(np.mean(norm_scores[seg_start:L]))
            seq_slice = sequence[seg_start:L] if sequence else None
            motifs.append(
                ConservedMotif(
                    start=seg_start + 1,
                    end=L,
                    length=L - seg_start,
                    mean_score=mean_sc,
                    sequence_slice=seq_slice,
                )
            )

        # Sort motifs by mean score descending
        motifs.sort(key=lambda m: -m.mean_score)

        return DomainAttributionReport(
            predicted_pfam=predicted_pfam,
            target_class_idx=target_class_idx,
            sequence_length=L,
            domain_start=domain_start,
            domain_end=domain_end,
            domain_enrichment_ratio=float(enrichment),
            per_residue_scores=norm_scores.tolist(),
            key_residues=key_residues,
            conserved_motifs=motifs,
            method=method,
        )


def explain_prediction(
    model: nn.Module,
    embedding: Union[tr.Tensor, np.ndarray],
    sequence: Optional[str] = None,
    domain_interval: Optional[Tuple[int, int]] = None,
    target_class: Optional[int] = None,
    categories: Optional[List[str]] = None,
    method: str = "integrated_gradients",
    top_k_residues: int = 10,
) -> DomainAttributionReport:
    """
    Convenience end-to-end function to compute per-residue attributions
    and map them to the Pfam domain.

    Args:
        model: Trained ET-Pfam BaseModel or compatible CNN classifier.
        embedding: Sequence embedding tensor [1280, L] or [1, 1280, L].
        sequence: Optional amino acid sequence string [L].
        domain_interval: Optional (start, end) tuple of 1-based domain coordinates.
        target_class: Target class index. Defaults to model argmax prediction.
        categories: Optional list of Pfam category string names.
        method: Attribution algorithm ('input_x_grad', 'integrated_gradients', 'vanilla_grad').
        top_k_residues: Number of top key residues to list.

    Returns:
        DomainAttributionReport containing full attribution scores, enrichment,
        key functional residues, and conserved functional motifs.
    """
    explainer = SaliencyExplainer(model)
    raw_scores = explainer.attribute(embedding, target_class=target_class, method=method)

    if target_class is None:
        with tr.no_grad():
            emb_t = tr.as_tensor(embedding, dtype=tr.float)
            if emb_t.dim() == 2:
                emb_t = emb_t.unsqueeze(0)
            target_class = int(tr.argmax(model(emb_t.to(explainer.device)), dim=1).item())

    pfam_name = (
        categories[target_class]
        if categories and target_class < len(categories)
        else f"Class_{target_class}"
    )

    domain_start = domain_interval[0] if domain_interval else None
    domain_end = domain_interval[1] if domain_interval else None

    report = DomainAttributionMapper.map_to_domain(
        scores=raw_scores,
        domain_start=domain_start,
        domain_end=domain_end,
        sequence=sequence,
        predicted_pfam=pfam_name,
        target_class_idx=target_class,
        top_k_residues=top_k_residues,
        method=method,
    )

    return report

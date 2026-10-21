"""
Uncertainty Calibration and Out-of-Distribution (OOD) Detection for ET-Pfam.

This module provides:
1. ConformalPredictionWrapper:
   A distribution-free, finite-sample valid conformal prediction wrapper over the
   ensemble model's probability outputs to produce prediction sets at user-defined
   significance levels (e.g. alpha = 0.10 for 90% marginal coverage).
   Supports both:
     - Threshold / Least Ambiguous Set (LAC): non-conformity s_i = 1 - p(y_i | x_i)
     - Adaptive Prediction Sets (APS): cumulative probability sorting
2. Random Neighbor Score (RNS):
   Out-of-Distribution (OOD) score based on distance to nearest/sampled neighbors
   in the embedding representation space (pLM 1280-dim representations) to flag
   novel, twilight-zone, or non-Pfam protein sequences.
"""
from typing import List, Dict, Tuple, Union, Optional
import numpy as np
import torch as tr
import torch.nn.functional as F


class ConformalPredictionWrapper:
    """
    Conformal Prediction Wrapper for ET-Pfam Ensemble and Base Models.

    Guarantees finite-sample distribution-free marginal coverage:
        P(Y_test in C(X_test)) >= 1 - alpha
    under the exchangeability assumption of calibration and test data.

    Parameters:
        categories: List of Pfam family identifiers (strings).
        method: Conformal scoring method:
            - 'lac' / 'score': Least Ambiguous Classification (1 - p_y)
            - 'aps': Adaptive Prediction Sets (cumulative probability mass)
        temperature: Temperature scaling factor for calibration before conformalization.
    """

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        method: str = "lac",
        temperature: float = 1.0,
    ):
        self.categories = categories
        self.method = method.lower()
        self.temperature = max(1e-4, temperature)
        self.q_hat: Dict[float, float] = {}  # Cached cutoffs per alpha
        self.is_calibrated = False
        self.cal_scores: Optional[np.ndarray] = None
        self.n_cal = 0

    def _to_numpy(self, x: Union[tr.Tensor, np.ndarray, List]) -> np.ndarray:
        """Helper to convert inputs to numpy array."""
        if isinstance(x, tr.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, list):
            return np.array(x)
        return np.asarray(x)

    def _get_probabilities(
        self, outputs: Union[tr.Tensor, np.ndarray], apply_softmax: Optional[bool] = None
    ) -> np.ndarray:
        """
        Converts logits or raw scores into valid probabilities summing to 1.
        Applies temperature scaling if provided.
        """
        probs = self._to_numpy(outputs).astype(np.float64)

        # Check if already softmax probabilities (each row sums to ~1 and all >= 0)
        row_sums = probs.sum(axis=1)
        already_probs = (
            np.all(probs >= 0.0)
            and np.allclose(row_sums, 1.0, atol=1e-3)
        )

        if apply_softmax is True or (apply_softmax is None and not already_probs):
            # Scale by temperature
            scaled = probs / self.temperature
            # Stable softmax
            scaled -= np.max(scaled, axis=1, keepdims=True)
            exp_scaled = np.exp(scaled)
            probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)

        return probs

    def calibrate(
        self,
        outputs: Union[tr.Tensor, np.ndarray],
        labels: Union[tr.Tensor, np.ndarray, List[int], List[str]],
        apply_softmax: Optional[bool] = None,
    ) -> None:
        """
        Fits the conformal non-conformity quantile on a held-out calibration set (e.g. dev set).

        Args:
            outputs: Model predictions (logits or probabilities) of shape [N, num_classes].
            labels: True ground-truth class labels. Can be:
                    - 1D integer indices of shape [N]
                    - One-hot / soft-label vectors of shape [N, num_classes]
                    - List of string category names matching self.categories
            apply_softmax: Explicitly force or disable softmax.
        """
        probs = self._get_probabilities(outputs, apply_softmax=apply_softmax)
        n_samples, n_classes = probs.shape

        # Parse ground-truth labels into integer indices
        if isinstance(labels, list) and len(labels) > 0 and isinstance(labels[0], str):
            if self.categories is None:
                raise ValueError("Must provide categories list to decode string labels.")
            label_indices = np.array([self.categories.index(lbl) for lbl in labels])
        else:
            lbl_arr = self._to_numpy(labels)
            if lbl_arr.ndim == 2:
                # One-hot or soft labels: take argmax as reference class
                label_indices = np.argmax(lbl_arr, axis=1)
            else:
                label_indices = lbl_arr.astype(int)

        if len(label_indices) != n_samples:
            raise ValueError(
                f"Sample count mismatch: {n_samples} predictions vs {len(label_indices)} labels."
            )

        self.n_cal = n_samples

        if self.method in ("lac", "score", "threshold"):
            # Non-conformity score: 1 - P(true class)
            true_class_probs = probs[np.arange(n_samples), label_indices]
            self.cal_scores = 1.0 - true_class_probs

        elif self.method in ("aps", "adaptive"):
            # Adaptive Prediction Sets: cumulative sum of sorted probabilities up to the true class
            sorted_indices = np.argsort(-probs, axis=1)
            sorted_probs = np.take_along_axis(probs, sorted_indices, axis=1)
            cum_probs = np.cumsum(sorted_probs, axis=1)

            # Find rank of the true class
            ranks = np.where(sorted_indices == label_indices[:, None])[1]
            self.cal_scores = cum_probs[np.arange(n_samples), ranks]

        else:
            raise ValueError(f"Unknown conformal prediction method: {self.method}")

        self.q_hat.clear()
        self.is_calibrated = True

    def get_quantile(self, alpha: float = 0.10) -> float:
        """
        Computes the conformal quantile cutoff with exact finite-sample correction:
            p_val = ceil((n_cal + 1) * (1 - alpha)) / n_cal
        """
        if not self.is_calibrated or self.cal_scores is None:
            raise RuntimeError("Model must be calibrated before computing quantiles.")

        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Significance level alpha must be in (0, 1), got {alpha}")

        if alpha not in self.q_hat:
            n = self.n_cal
            # Conformal quantile level with finite-sample adjustment
            q_level = np.clip(np.ceil((n + 1) * (1.0 - alpha)) / n, 0.0, 1.0)
            # Higher score = less conforming
            q_val = float(np.quantile(self.cal_scores, q_level, method="higher"))
            self.q_hat[alpha] = q_val

        return self.q_hat[alpha]

    def predict_set(
        self,
        outputs: Union[tr.Tensor, np.ndarray],
        alpha: float = 0.10,
        apply_softmax: Optional[bool] = None,
    ) -> Tuple[np.ndarray, List[List[Union[str, int]]]]:
        """
        Generates conformal prediction sets for test samples at significance level alpha.

        Args:
            outputs: Model predictions (logits or probabilities) of shape [N, num_classes].
            alpha: Significance level (e.g. 0.10 for 90% coverage).
            apply_softmax: Whether to apply softmax.

        Returns:
            mask: Boolean array of shape [N, num_classes] where True indicates inclusion in set.
            sets: List of length N, where each element is a list of predicted family names/indices.
        """
        probs = self._get_probabilities(outputs, apply_softmax=apply_softmax)
        n_samples, n_classes = probs.shape
        q_val = self.get_quantile(alpha)

        if self.method in ("lac", "score", "threshold"):
            # Condition: 1 - p(y|x) <= q_val  <=>  p(y|x) >= 1 - q_val
            threshold = 1.0 - q_val
            mask = probs >= threshold

            # Ensure non-empty prediction set by falling back to argmax if threshold is too strict
            empty_mask = ~np.any(mask, axis=1)
            if np.any(empty_mask):
                top_preds = np.argmax(probs[empty_mask], axis=1)
                mask[empty_mask, top_preds] = True

        elif self.method in ("aps", "adaptive"):
            # Sort probabilities descending
            sorted_indices = np.argsort(-probs, axis=1)
            sorted_probs = np.take_along_axis(probs, sorted_indices, axis=1)
            cum_probs = np.cumsum(sorted_probs, axis=1)

            # Slicing sets up to cutoff q_val
            included = np.zeros_like(probs, dtype=bool)
            # Always include the top-1 prediction
            included[:, 0] = True
            # Include subsequent classes whose previous cumulative sum was < q_val
            if n_classes > 1:
                included[:, 1:] = cum_probs[:, :-1] < q_val

            # Map back to original class index positions
            mask = np.zeros_like(probs, dtype=bool)
            for i in range(n_samples):
                mask[i, sorted_indices[i, included[i]]] = True

        # Convert boolean mask to list of class labels or indices
        sets: List[List[Union[str, int]]] = []
        for i in range(n_samples):
            indices = np.where(mask[i])[0]
            if self.categories is not None:
                sets.append([self.categories[idx] for idx in indices])
            else:
                sets.append(indices.tolist())

        return mask, sets

    def evaluate_coverage(
        self,
        outputs: Union[tr.Tensor, np.ndarray],
        labels: Union[tr.Tensor, np.ndarray, List[int], List[str]],
        alpha: float = 0.10,
        apply_softmax: Optional[bool] = None,
    ) -> Dict[str, float]:
        """
        Evaluates the empirical coverage and efficiency (set size metrics)
        of the conformal prediction sets on a test set.

        Returns:
            Dict containing:
                - empirical_coverage: Fraction of true labels covered
                - target_coverage: 1 - alpha
                - avg_set_size: Mean cardinality of prediction sets
                - median_set_size: Median cardinality
                - singleton_rate: Fraction of prediction sets with exactly 1 class
                - empty_rate: Fraction of empty sets (should be 0)
        """
        mask, _ = self.predict_set(outputs, alpha=alpha, apply_softmax=apply_softmax)
        n_samples = len(mask)

        # Parse ground truth
        if isinstance(labels, list) and len(labels) > 0 and isinstance(labels[0], str):
            label_indices = np.array([self.categories.index(lbl) for lbl in labels])
        else:
            lbl_arr = self._to_numpy(labels)
            label_indices = (
                np.argmax(lbl_arr, axis=1) if lbl_arr.ndim == 2 else lbl_arr.astype(int)
            )

        # Check coverage
        covered = mask[np.arange(n_samples), label_indices]
        set_sizes = mask.sum(axis=1)

        metrics = {
            "empirical_coverage": float(np.mean(covered)),
            "target_coverage": float(1.0 - alpha),
            "avg_set_size": float(np.mean(set_sizes)),
            "median_set_size": float(np.median(set_sizes)),
            "singleton_rate": float(np.mean(set_sizes == 1)),
            "empty_rate": float(np.mean(set_sizes == 0)),
        }
        return metrics


def compute_random_neighbor_score(
    query_embeddings: Union[tr.Tensor, np.ndarray],
    reference_embeddings: Union[tr.Tensor, np.ndarray],
    k: int = 10,
    metric: str = "cosine",
    sample_size: Optional[int] = 1000,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Computes the Random Neighbor Score (RNS) to flag out-of-distribution (OOD) sequences.

    RNS quantifies how atypical a candidate sequence embedding is with respect to the
    known in-distribution representation manifold (e.g. Pfam training/dev embeddings).
    A high RNS value indicates the sequence is isolated or distant from known clusters.

    Algorithm:
        1. Optionally subsamples `sample_size` reference vectors uniformly at random
           to ensure fast, scalable computation over massive protein databases.
        2. Computes pairwise distances (Cosine or Euclidean) from each query to the reference set.
        3. Identifies the k nearest neighbors among the reference pool.
        4. Calculates the mean distance across the k nearest neighbors as the RNS score.

    Args:
        query_embeddings: Query protein representations [M, D] (e.g., mean-pooled or max-pooled pLM).
        reference_embeddings: In-distribution reference representations [N, D].
        k: Number of nearest neighbors to consider (default 10).
        metric: Distance metric ('cosine' or 'euclidean').
        sample_size: If reference set > sample_size, uniformly samples reference points.
        random_seed: Random seed for reference sampling reproducibility.

    Returns:
        rns_scores: 1D numpy array of shape [M] with the RNS score for each query sequence.
    """
    # Convert inputs to torch tensors
    if isinstance(query_embeddings, np.ndarray):
        query = tr.from_numpy(query_embeddings).float()
    else:
        query = query_embeddings.detach().cpu().float()

    if isinstance(reference_embeddings, np.ndarray):
        ref = tr.from_numpy(reference_embeddings).float()
    else:
        ref = reference_embeddings.detach().cpu().float()

    # Flatten if multi-dimensional (e.g. [N, D, L] -> pool across L to [N, D])
    if query.dim() == 3:
        query = query.mean(dim=2)
    if ref.dim() == 3:
        ref = ref.mean(dim=2)

    M, D = query.shape
    N, D_ref = ref.shape

    if D != D_ref:
        raise ValueError(
            f"Embedding dimension mismatch: query is {D}-dim, reference is {D_ref}-dim."
        )

    # Subsample reference if large
    if sample_size is not None and N > sample_size:
        rng = np.random.RandomState(random_seed)
        sampled_indices = rng.choice(N, size=sample_size, replace=False)
        ref = ref[sampled_indices]
        N = sample_size

    k_actual = min(k, N)

    if metric == "cosine":
        # Cosine distance = 1 - cosine_similarity
        query_norm = F.normalize(query, p=2, dim=1)
        ref_norm = F.normalize(ref, p=2, dim=1)
        # Cosine similarities: [M, N]
        sims = tr.mm(query_norm, ref_norm.transpose(0, 1))
        # Distances: [M, N] in range [0, 2]
        dists = 1.0 - sims
    elif metric == "euclidean":
        # Pairwise euclidean distances
        dists = tr.cdist(query, ref, p=2)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Choose 'cosine' or 'euclidean'.")

    # Get top-k smallest distances
    topk_dists, _ = tr.topk(dists, k=k_actual, dim=1, largest=False)
    # RNS is the average distance to the k nearest neighbors
    rns_scores = topk_dists.mean(dim=1).numpy()

    return rns_scores


class OODDetector:
    """
    Out-of-Distribution Detector using Random Neighbor Score (RNS).

    Calibrates an OOD detection threshold on in-distribution calibration embeddings
    at a target False Positive Rate (FPR), e.g., FPR = 0.05 (95% specificity).
    """

    def __init__(
        self,
        k: int = 10,
        metric: str = "cosine",
        target_fpr: float = 0.05,
        sample_size: Optional[int] = 1000,
    ):
        self.k = k
        self.metric = metric
        self.target_fpr = target_fpr
        self.sample_size = sample_size
        self.reference_embeddings: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None

    def fit(
        self,
        reference_embeddings: Union[tr.Tensor, np.ndarray],
        calibration_embeddings: Optional[Union[tr.Tensor, np.ndarray]] = None,
    ) -> float:
        """
        Fits the detector by saving reference embeddings and setting the threshold.

        If `calibration_embeddings` is provided, threshold is set to the (1 - target_fpr)
        quantile of in-distribution RNS scores.
        """
        if isinstance(reference_embeddings, tr.Tensor):
            self.reference_embeddings = reference_embeddings.detach().cpu().numpy()
        else:
            self.reference_embeddings = np.asarray(reference_embeddings)

        cal_set = (
            calibration_embeddings
            if calibration_embeddings is not None
            else self.reference_embeddings
        )
        in_scores = compute_random_neighbor_score(
            cal_set,
            self.reference_embeddings,
            k=self.k,
            metric=self.metric,
            sample_size=self.sample_size,
        )

        # Cutoff at (1 - target_fpr) quantile
        self.threshold = float(np.quantile(in_scores, 1.0 - self.target_fpr))
        return self.threshold

    def predict(
        self, query_embeddings: Union[tr.Tensor, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes RNS scores and flags OOD sequences.

        Returns:
            is_ood: Boolean array of shape [M], True if flagged as Out-of-Distribution.
            scores: RNS scores of shape [M].
        """
        if self.reference_embeddings is None or self.threshold is None:
            raise RuntimeError("OODDetector must be fitted via `fit()` before predicting.")

        scores = compute_random_neighbor_score(
            query_embeddings,
            self.reference_embeddings,
            k=self.k,
            metric=self.metric,
            sample_size=self.sample_size,
        )
        is_ood = scores > self.threshold
        return is_ood, scores

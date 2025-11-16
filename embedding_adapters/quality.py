# embedding_adapters/quality.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union, Sequence, List
import math

import numpy as np
import torch
import torch.nn.functional as F

ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass
class QualityStats:
    mean: torch.Tensor        # (d,)
    inv_cov: torch.Tensor     # (d, d)
    m95: float
    m99: float
    knn_ref: torch.Tensor     # (M, d) normalized
    knn95: float
    knn99: float


class QualityModel:
    """
    Scores how 'in-domain' an embedding is relative to the adapter's
    training distributions in both:
      - source space (e.g. e5-base-v2)
      - target space (e.g. text-embedding-3-small)
    """

    def __init__(self, stats_path: Union[str, Path], device: str = "cpu"):
        stats_path = Path(stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(f"Quality stats not found: {stats_path}")

        data = np.load(stats_path, allow_pickle=False)
        self.device = torch.device(device)

        # Source stats (may or may not be present)
        if all(k in data for k in ["mean_src", "inv_cov_src", "knn_ref_src"]):
            mean_src = torch.from_numpy(data["mean_src"])
            inv_cov_src = torch.from_numpy(data["inv_cov_src"])
            knn_ref_src = torch.from_numpy(data["knn_ref_src"])
            self.src: Optional[QualityStats] = QualityStats(
                mean=mean_src.to(self.device),
                inv_cov=inv_cov_src.to(self.device),
                m95=float(data["m95_src"]),
                m99=float(data["m99_src"]),
                knn_ref=knn_ref_src.to(self.device),
                knn95=float(data["knn95_src"]),
                knn99=float(data["knn99_src"]),
            )
        else:
            self.src = None

        # Target stats (may or may not be present)
        if all(k in data for k in ["mean_tgt", "inv_cov_tgt", "knn_ref_tgt"]):
            mean_tgt = torch.from_numpy(data["mean_tgt"])
            inv_cov_tgt = torch.from_numpy(data["inv_cov_tgt"])
            knn_ref_tgt = torch.from_numpy(data["knn_ref_tgt"])
            self.tgt: Optional[QualityStats] = QualityStats(
                mean=mean_tgt.to(self.device),
                inv_cov=inv_cov_tgt.to(self.device),
                m95=float(data["m95_tgt"]),
                m99=float(data["m99_tgt"]),
                knn_ref=knn_ref_tgt.to(self.device),
                knn95=float(data["knn95_tgt"]),
                knn99=float(data["knn99_tgt"]),
            )
        else:
            self.tgt = None

    def _to_tensor(self, x: ArrayLike) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
        elif isinstance(x, torch.Tensor):
            t = x
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")
        return t.to(self.device).float()

    @staticmethod
    def _confidence_from_range(
        vals: torch.Tensor, p95: float, p99: float, eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Map a distance metric to a smooth confidence in [0, 1].

        We use a logistic curve calibrated so that:
          - distance at p95 -> conf ≈ 0.8
          - distance at p99 -> conf ≈ 0.2

        Below p95, confidence smoothly approaches 1.0.
        Above p99, confidence smoothly approaches 0.0.
        """
        vals = vals.float()

        if p99 <= p95 + eps:
            # degenerate: just treat everything as in-distribution
            return torch.ones_like(vals)

        # target confidences at calibration points
        c95 = 0.8
        c99 = 0.2

        # logit(x) = log(x / (1 - x))
        def logit(x: float) -> float:
            return math.log(x / (1.0 - x))

        A = logit(c95)            # conf when z = 0 (at p95)
        A99 = logit(c99)          # conf when z = 1 (at p99)
        B = A - A99               # ensures conf(z=1) = c99

        # normalized distance between p95 and p99
        z = (vals - p95) / (p99 - p95 + eps)

        # logistic curve
        conf = torch.sigmoid(A - B * z)
        return torch.clamp(conf, 0.0, 1.0)

    def _score_impl(self, X: torch.Tensor, stats: QualityStats) -> Dict[str, np.ndarray]:
        # Mahalanobis
        centered = X - stats.mean  # (N, d)
        tmp = torch.matmul(centered, stats.inv_cov)  # (N, d)
        md2 = (tmp * centered).sum(dim=1)
        md = torch.sqrt(torch.clamp(md2, min=0.0))

        # kNN distance on unit sphere
        Xn = F.normalize(X, dim=1)
        Rn = F.normalize(stats.knn_ref, dim=1)
        sims = torch.matmul(Xn, Rn.T)  # (N, M)
        nn_cos, _ = sims.max(dim=1)
        knn_dist = torch.sqrt(torch.clamp(2.0 - 2.0 * nn_cos, min=0.0))

        # confidence
        conf_maha = self._confidence_from_range(md, stats.m95, stats.m99)
        conf_knn = self._confidence_from_range(knn_dist, stats.knn95, stats.knn99)
        confidence = torch.min(conf_maha, conf_knn)

        return {
            "mahalanobis": md.cpu().numpy(),
            "knn_distance": knn_dist.cpu().numpy(),
            "conf_maha": conf_maha.cpu().numpy(),
            "conf_knn": conf_knn.cpu().numpy(),
            "confidence": confidence.cpu().numpy(),
        }

    @torch.no_grad()
    def score_source(self, x: ArrayLike) -> Dict[str, np.ndarray]:
        """
        Score *source-space* embeddings (e.g. e5-base-v2).
        """
        if self.src is None:
            raise RuntimeError("No source quality stats in this adapter.")
        X = self._to_tensor(x)
        return self._score_impl(X, self.src)

    @torch.no_grad()
    def score_target(self, y: ArrayLike) -> Dict[str, np.ndarray]:
        """
        Score *target-space* embeddings (e.g. text-embedding-3-small).
        """
        if self.tgt is None:
            raise RuntimeError("No target quality stats in this adapter.")
        Y = self._to_tensor(y)
        return self._score_impl(Y, self.tgt)

    # For convenience / backward compatibility:
    def score(self, x: ArrayLike) -> Dict[str, np.ndarray]:
        """
        Alias for score_source(x).
        """
        return self.score_source(x)


# ---------------------------------------------------------------------------
# English interpretation utilities
# ---------------------------------------------------------------------------

def _confidence_label(c: float) -> str:
    if c >= 0.95:
        return "very in-distribution"
    if c >= 0.8:
        return "mostly in-distribution"
    if c >= 0.5:
        return "borderline / somewhat out-of-distribution"
    if c >= 0.2:
        return "clearly out-of-distribution"
    return "strongly out-of-distribution"


def _confidence_explanation(c: float) -> str:
    if c >= 0.95:
        return (
            "This query looks very similar to the adapter's training data. "
            "The adapter is expected to behave reliably here."
        )
    if c >= 0.8:
        return (
            "This query is mostly similar to the training data. "
            "The adapter should generally work well, with only minor risk of degradation."
        )
    if c >= 0.5:
        return (
            "This query is somewhat atypical compared to the training data. "
            "The adapter may still work, but its behavior is less predictable."
        )
    if c >= 0.2:
        return (
            "This query is clearly out-of-distribution. "
            "The adapter's outputs may be noticeably degraded or unreliable."
        )
    return (
        "This query is strongly out-of-distribution. "
        "The adapter is unlikely to behave reliably for this input."
    )


def interpret_quality(
    texts: Sequence[str],
    scores: Dict[str, np.ndarray],
    space_label: str = "source",
) -> str:
    """
    Produce an English-language summary of quality scores for a batch of texts.

    `scores` is the dict returned by QualityModel.score_source/score_target.

    Example usage:

        src_embs = adapter.encode(texts, as_numpy=True, normalize=True, return_source=True)
        qs_src = adapter.score_source(src_embs)
        print(interpret_quality(texts, qs_src, space_label="source"))

    This will print per-example confidence, distances, and a human-readable
    interpretation, plus a batch-level summary.
    """
    conf = scores["confidence"]
    maha = scores["mahalanobis"]
    knn = scores["knn_distance"]

    lines: List[str] = []
    lines.append(f"Adapter quality interpretation ({space_label} space)")
    lines.append(f"Number of examples: {len(texts)}")
    lines.append("")

    for i, (t, c, m, k) in enumerate(zip(texts, conf, maha, knn)):
        c_f = float(c)
        m_f = float(m)
        k_f = float(k)
        label = _confidence_label(c_f)
        expl = _confidence_explanation(c_f)

        lines.append(f"Example {i+1}: {t!r}")
        lines.append(f"  - Confidence: {c_f:.3f}  ({label})")
        lines.append(
            f"  - Distances: Mahalanobis = {m_f:.3f}, "
            f"kNN distance = {k_f:.3f}"
        )
        lines.append(f"  - Interpretation: {expl}")
        lines.append("")

    # Batch-level summary
    conf_mean = float(conf.mean())
    conf_min = float(conf.min())
    conf_max = float(conf.max())

    lines.append("Batch summary:")
    lines.append(
        f"  - Mean confidence: {conf_mean:.3f} "
        f"(range: {conf_min:.3f} – {conf_max:.3f})"
    )

    if conf_mean >= 0.9 and conf_min >= 0.8:
        lines.append(
            "  - Most queries look very similar to the training distribution. "
            "The adapter should be broadly reliable on this batch."
        )
    elif conf_mean >= 0.7:
        lines.append(
            "  - The batch is mostly in-distribution, but some queries are atypical. "
            "Expect generally good behavior with occasional degradation."
        )
    elif conf_mean >= 0.4:
        lines.append(
            "  - Many queries are borderline or out-of-distribution. "
            "Expect mixed reliability and degraded behavior on a non-trivial subset."
        )
    else:
        lines.append(
            "  - The batch is largely out-of-distribution. "
            "The adapter is likely unreliable for many of these inputs."
        )

    return "\n".join(lines)

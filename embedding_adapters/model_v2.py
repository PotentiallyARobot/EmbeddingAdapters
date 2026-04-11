"""Embedding adapter: MLP mapping source → target with quality head.

Architecture:
    source (D_s) → in_norm → hidden layers → last hidden (H)
                                                   ↓
                                     ┌─────────────┼─────────────┐
                                     ↓                           ↓ (detached)
                               final_proj (H→D_t)         quality_head
                                     ↓                     H_detached + D_s → 256 → 1
                               + skip (D_s→D_t)                 ↓
                                     ↓                      sigmoid → [0, 1]
                               out_norm → L2 norm                ↓
                                     ↓                     confidence score
                               embedding (D_t)

Skip connection: always a learned linear projection from source space to
target space. Even when source_dim == target_dim, the two embedding spaces
have fundamentally different geometry, so an identity residual would anchor
the output near the source space and prevent learning the full rotation.

The quality head is a confidence predictor: "given this input, how well
will the adapter replicate the target embedding?"

It is GRADIENT-ISOLATED from the embedding path — the hidden state is
detached before entering the quality head. This means:
  - Quality loss trains ONLY the quality head weights
  - The quality head cannot influence the embedding — it's purely diagnostic
  - At inference it reads the trunk's internal state to predict fidelity
    without needing access to the target embedding

Trained against: actual cosine similarity between adapter output and real
target, mapped to [0, 1]. But that's just the supervision signal — at
inference the quality head has never seen the target, it's making a pure
prediction from the input and hidden state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingAdapter(nn.Module):
    """
    Maps source embeddings to target embedding space.

    Returns (embedding, quality) where:
        embedding: (B, target_dim) — L2-normalized target-space vector
        quality:   (B,) — confidence that embedding faithfully replicates
                   the target, in [0, 1]. 1.0 = perfect replication expected.
    """

    def __init__(self, source_dim: int, target_dim: int,
                 hidden_dim: int = 1024, num_layers: int = 3,
                 dropout: float = 0.1, use_skip: bool = True,
                 quality_hidden: int = 256, activation: str = "gelu"):
        super().__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim

        self.in_norm = nn.LayerNorm(source_dim)

        # Activation selection — SELU is used by Embedding-Converter (Yoon & Arık, ACL 2025)
        # and has self-normalizing properties beneficial for deep MLPs.
        act_map = {"gelu": nn.GELU, "selu": nn.SELU, "relu": nn.ReLU}
        act_cls = act_map.get(activation, nn.GELU)

        # Hidden layers: source_dim → hidden_dim (repeated)
        hidden_layers = []
        in_d = source_dim
        for i in range(num_layers - 1):
            hidden_layers.append(nn.Linear(in_d, hidden_dim))
            # SELU is self-normalizing — LayerNorm is counterproductive with it
            if activation != "selu":
                hidden_layers.append(nn.LayerNorm(hidden_dim))
            hidden_layers.append(act_cls())
            if activation == "selu":
                hidden_layers.append(nn.AlphaDropout(dropout))
            else:
                hidden_layers.append(nn.Dropout(dropout))
            in_d = hidden_dim
        self.hidden = nn.Sequential(*hidden_layers) if hidden_layers else nn.Identity()

        self.hidden_out_dim = hidden_dim if num_layers > 1 else source_dim

        # Final projection to target space
        self.final_proj = nn.Linear(self.hidden_out_dim, target_dim)
        self.out_norm = nn.LayerNorm(target_dim)

        # Skip connection: learned linear projection from input to output space.
        # Even when source_dim == target_dim, we use a learned skip rather than
        # identity — the two spaces have fundamentally different geometry, so an
        # identity residual anchors the output near the source space and prevents
        # the model from learning the full rotation into the target space.
        self.use_skip = use_skip
        self.activation = activation
        if use_skip:
            self.skip = nn.Linear(source_dim, target_dim)

        # Quality head — gradient-isolated from embedding path.
        # Input: detached hidden state + detached normalized input.
        # This ensures quality loss never interferes with embedding training.
        # The head learns to read the trunk's representation to predict
        # "will my embedding path do well on this kind of input?"
        self.quality_head = nn.Sequential(
            nn.Linear(self.hidden_out_dim + source_dim, quality_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(quality_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.final_proj.bias)
        nn.init.normal_(self.final_proj.weight, std=0.01)
        # sigmoid(0) = 0.5 → starts uncertain
        nn.init.zeros_(self.quality_head[-1].bias)
        # SELU benefits from LeCun normal initialization
        if self.activation == "selu":
            for m in self.hidden.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, source_dim) — ideally pre-normalized
        Returns:
            embedding: (B, target_dim) L2-normalized
            quality:   (B,) confidence score in [0, 1]
        """
        h = self.in_norm(x)
        hidden = self.hidden(h)

        # ── Embedding path (receives full gradients from all losses) ──
        out = self.final_proj(hidden)
        if self.use_skip:
            out = out + self.skip(h)
        embedding = F.normalize(self.out_norm(out), dim=-1)

        # ── Quality path (gradient-isolated — only quality loss trains this) ──
        # Detach both inputs so quality head gradients don't flow back into
        # the trunk or affect the embedding. The quality head trains only
        # its own weights to predict replication fidelity from frozen features.
        quality = torch.sigmoid(
            self.quality_head(
                torch.cat([hidden.detach(), h.detach()], dim=-1)
            ).squeeze(-1)
        )

        return embedding, quality

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


class SourceDecoder(nn.Module):
    """
    Training-only decoder: target_dim → source_dim.

    Reconstructs the original source embedding from the adapter's output.
    Forces the adapter to preserve source information in its output space,
    not just replicate the target.

    This module is DISCARDED after training — it adds zero cost at inference.
    It only needs to be "good enough" to verify information retention,
    so it's kept small (2 layers, smaller hidden dim than the adapter).
    """

    def __init__(self, target_dim: int, source_dim: int,
                 hidden_dim: int = 2048, activation: str = "gelu"):
        super().__init__()
        self.target_dim = target_dim
        self.source_dim = source_dim

        act_map = {"gelu": nn.GELU, "selu": nn.SELU, "relu": nn.ReLU}
        act_cls = act_map.get(activation, nn.GELU)

        self.net = nn.Sequential(
            nn.LayerNorm(target_dim),
            nn.Linear(target_dim, hidden_dim),
            act_cls(),
            nn.Linear(hidden_dim, hidden_dim),
            act_cls(),
            nn.Linear(hidden_dim, source_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, target_dim) — adapter output (L2-normalized)
        Returns:
            reconstructed: (B, source_dim) — reconstructed source embedding
        """
        return self.net(pred)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

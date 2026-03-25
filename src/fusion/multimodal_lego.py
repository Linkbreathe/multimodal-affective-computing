"""Multimodal Lego fusion: faithful implementation of the MM-Lego paper.

Paper approach: Perceiver-style modality adapters (LegoBlock) with learnable
latent bottlenecks, cross-attention in Fourier domain, and merge/fuse modes
for combining modality latents.

Also contains CustomTopologyFusion (our original pairwise/hierarchical/gated
cross-attention designs), which remain useful as alternative baselines.
"""
from __future__ import annotations

from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.fusion.base import BaseFusionModule


# ---------------------------------------------------------------------------
# MM-Lego building blocks (faithful to paper)
# ---------------------------------------------------------------------------

class PreNorm(nn.Module):
    """Pre-layer-norm wrapper, optionally normalises context too."""

    def __init__(self, dim: int, fn: nn.Module, context_dim: int | None = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if context_dim is not None else None

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.norm(x)
        if self.norm_context is not None and "context" in kwargs:
            kwargs["context"] = self.norm_context(kwargs["context"])
        return self.fn(x, **kwargs)


class LegoAttention(nn.Module):
    """Multi-head attention used inside LegoBlock.

    Supports both self-attention (context=None) and cross-attention.
    Uses SELU-gated output projection following the paper.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.LeakyReLU(negative_slope=1e-2),
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.heads
        q = self.to_q(x)
        ctx = context if context is not None else x
        k, v = self.to_kv(ctx).chunk(2, dim=-1)

        q, k, v = (
            rearrange(t, "b n (h d) -> (b h) n d", h=h) for t in (q, k, v)
        )

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
        # Temperature-scaled softmax (T=0.5) as in original code
        attn = F.softmax(sim / 0.5, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class LegoFeedForward(nn.Module):
    """Gated feed-forward with SELU activation (as in paper)."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            _SELUGate(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SELUGate(nn.Module):
    """Split tensor in half and gate with SELU (paper's activation)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * F.selu(gates)


class LegoBlock(nn.Module):
    """Perceiver-style modality adapter from the MM-Lego paper.

    Architecture per depth iteration:
    1. FFT on latent (normalise, take real component)
    2. FFT on modality input (real component)
    3. Cross-attention: latent queries modality embedding (in Fourier domain)
    4. Feed-forward on latent
    5. IFFT to bring latent back to spatial domain (except last iteration)

    Args:
        input_dim: dimension of modality embeddings (from frozen encoder)
        latent_channels: number of latent vectors (l_c in paper)
        latent_dim: dimension of each latent vector (l_d in paper)
        depth: number of cross-attention iterations
        heads: number of attention heads
        dim_head: dimension per attention head
        attn_dropout: dropout on attention weights
        ff_dropout: dropout in feed-forward
        frequency_domain: whether to use FFT on latents (core paper idea)
        fourier_dim: FFT dimension (1 = along latent channel axis)
        track_imaginary: whether to track imaginary component for IFFT
        normalise: whether to L2-normalise latent before FFT
    """

    def __init__(
        self,
        input_dim: int,
        latent_channels: int = 64,
        latent_dim: int = 64,
        depth: int = 2,
        heads: int = 8,
        dim_head: int = 64,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        frequency_domain: bool = True,
        fourier_dim: int = 1,
        track_imaginary: bool = True,
        normalise: bool = True,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_dim = latent_dim
        self.depth = depth
        self.frequency_domain = frequency_domain
        self.fourier_dim = fourier_dim
        self.track_imaginary = track_imaginary
        self.normalise = normalise

        # Learnable latent bottleneck
        self.latent = nn.Parameter(torch.randn(latent_channels, latent_dim))

        # Build layers for each depth (weight-shared by default like the paper)
        self.cross_attns = nn.ModuleList()
        self.cross_ffs = nn.ModuleList()
        for _ in range(depth):
            self.cross_attns.append(
                PreNorm(
                    latent_dim,
                    LegoAttention(
                        query_dim=latent_dim,
                        context_dim=input_dim,
                        heads=heads,
                        dim_head=dim_head,
                        dropout=attn_dropout,
                    ),
                    context_dim=input_dim,
                )
            )
            self.cross_ffs.append(
                PreNorm(latent_dim, LegoFeedForward(latent_dim, dropout=ff_dropout))
            )

    def forward(
        self,
        x: torch.Tensor,
        latent_override: torch.Tensor | None = None,
        return_complex: bool = False,
    ) -> torch.Tensor:
        """Run the LegoBlock.

        Args:
            x: modality embedding [B, input_dim] or [B, seq, input_dim]
            latent_override: if provided, use this instead of self.latent
                             (for fuse-stack mode where latent is passed between blocks)
            return_complex: if True, return complex-valued latent (for merge mode)

        Returns:
            latent: [B, latent_channels, latent_dim]
        """
        # Ensure x is 3D [B, seq, dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B = x.shape[0]

        # Initialise latent
        if latent_override is not None:
            latent = latent_override
        else:
            latent = repeat(self.latent, "n d -> b n d", b=B)

        l_imag = None

        for i in range(self.depth):
            # Fourier transform on latent and input
            if self.frequency_domain:
                if self.normalise:
                    latent = latent / (latent.norm(dim=self.fourier_dim, keepdim=True) + 1e-8)
                fft_l = torch.fft.fft(latent, dim=self.fourier_dim)
                l_real = fft_l.real
                l_imag = fft_l.imag
                x_real = torch.fft.fft(x, dim=self.fourier_dim).real
            else:
                l_real = latent
                x_real = x

            # Cross-attention: latent queries modality embedding (real components)
            latent = self.cross_attns[i](l_real, context=x_real) + l_real
            latent = self.cross_ffs[i](latent) + latent

            # Skip IFFT on last iteration (paper keeps head in Fourier domain)
            if i == self.depth - 1:
                break

            # Inverse FFT to bring back to spatial domain for next iteration
            if self.frequency_domain:
                if self.track_imaginary and l_imag is not None:
                    latent = torch.complex(latent, l_imag)
                latent = torch.fft.ifft(latent, dim=self.fourier_dim)
                # Take real part for next iteration input
                if latent.is_complex():
                    latent = latent.real

        # Return
        if return_complex and self.frequency_domain and l_imag is not None:
            return torch.complex(latent, l_imag)
        else:
            if latent.is_complex():
                latent = latent.real
            return latent


# ---------------------------------------------------------------------------
# Merge and Fuse strategies (faithful to paper)
# ---------------------------------------------------------------------------

class MultimodalLegoFusion(BaseFusionModule):
    """MM-Lego fusion: LegoBlock per modality + merge/fuse in Fourier domain.

    Modes:
        merge-sum / merge-product / merge-mean / merge-harmonic:
            Each modality block runs independently, latents merged in Fourier domain.
        fuse-stack:
            Latent is passed sequentially through each modality's block.
        fuse-weave:
            Alternating single-depth cross-attention passes from each block.

    Args:
        d_common: common embedding dimension (input from frozen encoders)
        modality_ids: list of modality identifiers
        mode: fusion mode string (e.g. "merge-sum", "fuse-stack", "fuse-weave")
        latent_channels: number of learnable latent vectors per block
        latent_dim: latent vector dimension (also used as d_out)
        depth: number of cross-attention iterations per LegoBlock
        heads: attention heads
        dim_head: dimension per head
        attn_dropout: attention dropout
        ff_dropout: feed-forward dropout
        frequency_domain: whether to use Fourier-domain processing
        fourier_dim: dimension for FFT (1 = along latent channel axis)
        track_imaginary: track imaginary for IFFT reconstruction
        normalise: L2-normalise latents before FFT
        alpha: weighting for harmonic merge mode
    """

    def __init__(
        self,
        d_common: int = 256,
        modality_ids: list[str] | None = None,
        mode: str = "merge-sum",
        latent_channels: int = 64,
        latent_dim: int = 64,
        depth: int = 2,
        heads: int = 8,
        dim_head: int = 64,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        frequency_domain: bool = True,
        fourier_dim: int = 1,
        track_imaginary: bool = True,
        normalise: bool = True,
        alpha: float = 0.5,
    ) -> None:
        super().__init__(d_common=d_common, d_out=latent_dim)
        modality_ids = modality_ids or ["video", "eye_tracking", "ppg"]
        self.mod_ids = modality_ids
        self.mode = mode
        self.alpha = alpha
        self.latent_channels = latent_channels
        self.latent_dim = latent_dim

        # One LegoBlock per modality
        self.blocks = nn.ModuleDict()
        for mod_id in modality_ids:
            self.blocks[mod_id] = LegoBlock(
                input_dim=d_common,
                latent_channels=latent_channels,
                latent_dim=latent_dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                frequency_domain=frequency_domain,
                fourier_dim=fourier_dim,
                track_imaginary=track_imaginary,
                normalise=normalise,
            )

        # Shared latent for fuse modes (passed between blocks)
        self.shared_latent = nn.Parameter(torch.randn(latent_channels, latent_dim))

        # Output: mean pool over latent channels -> project to latent_dim
        self.output_norm = nn.LayerNorm(latent_dim)
        self.output_proj = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        B = embeddings[0].shape[0]

        if self.mode.startswith("merge"):
            return self._forward_merge(embeddings, modality_ids, B)
        elif self.mode == "fuse-stack":
            return self._forward_fuse_stack(embeddings, modality_ids, B)
        elif self.mode == "fuse-weave":
            return self._forward_fuse_weave(embeddings, modality_ids, B)
        else:
            raise ValueError(
                f"Unknown mode '{self.mode}'. "
                "Use merge-sum, merge-product, merge-mean, merge-harmonic, "
                "fuse-stack, or fuse-weave."
            )

    def _forward_merge(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        B: int,
    ) -> torch.Tensor:
        """LegoMerge: each block runs independently, merge latents in Fourier domain."""
        merge_method = self.mode.split("-", 1)[1]  # sum, product, mean, harmonic

        # Collect latents from each modality block (complex-valued for merge)
        latents = []
        for emb, mod_id in zip(embeddings, modality_ids):
            block = self.blocks[mod_id]
            latent = block(emb, return_complex=True)
            latents.append(latent)

        # Merge in Fourier domain
        if merge_method == "sum":
            merged = sum(latents)
        elif merge_method == "product":
            merged = torch.stack(latents, dim=0).prod(dim=0)
        elif merge_method == "mean":
            merged = torch.stack(latents, dim=0).mean(dim=0)
        elif merge_method == "harmonic":
            # Weighted harmonic mean of magnitudes, average of phases
            # Works for 2+ modalities; generalises the paper's 2-modality version
            if len(latents) == 2:
                l1, l2 = latents[0], latents[1]
                mag1, mag2 = torch.abs(l1), torch.abs(l2)
                phase1, phase2 = torch.angle(l1), torch.angle(l2)
                a = self.alpha
                mag = 2 * a * (1 - a) * mag1 * mag2 / (a * mag2 + (1 - a) * mag1 + 1e-8)
                phase = (phase1 + phase2) / 2
                merged = mag * torch.exp(1j * phase)
            else:
                # Fallback: harmonic mean of magnitudes for N modalities
                mags = [torch.abs(lat) for lat in latents]
                phases = [torch.angle(lat) for lat in latents]
                inv_sum = sum(1.0 / (m + 1e-8) for m in mags)
                mag = len(mags) / (inv_sum + 1e-8)
                phase = sum(phases) / len(phases)
                merged = mag * torch.exp(1j * phase)
        else:
            raise ValueError(f"Unknown merge method: {merge_method}")

        # Take real part and pool
        merged_real = merged.real if merged.is_complex() else merged
        return self._output_head(merged_real)

    def _forward_fuse_stack(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        B: int,
    ) -> torch.Tensor:
        """LegoFuse stack: pass latent sequentially through each modality block."""
        latent = None
        for i, (emb, mod_id) in enumerate(zip(embeddings, modality_ids)):
            block = self.blocks[mod_id]
            if i == 0:
                # First block uses its own learnable latent
                latent = block(emb, return_complex=False)
            else:
                # Subsequent blocks receive the latent from previous block
                latent = block(emb, latent_override=latent, return_complex=False)

        return self._output_head(latent)

    def _forward_fuse_weave(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        B: int,
    ) -> torch.Tensor:
        """LegoFuse weave: alternating single-depth passes from each block.

        Instead of each block doing its full depth, we interleave: for each
        depth step, each block does one cross-attention pass on the shared latent.
        """
        latent = repeat(self.shared_latent, "n d -> b n d", b=B)

        # Ensure embeddings are 3D
        embs = []
        for emb in embeddings:
            if emb.dim() == 2:
                emb = emb.unsqueeze(1)
            embs.append(emb)

        # Get the depth from the first block (all blocks share same depth)
        first_block = next(iter(self.blocks.values()))
        depth = first_block.depth

        for d in range(depth):
            for emb, mod_id in zip(embs, modality_ids):
                block = self.blocks[mod_id]
                # Single-depth pass: use layer d's cross-attn and ff
                cross_attn = block.cross_attns[d]
                cross_ff = block.cross_ffs[d]

                # Apply Fourier transform if enabled
                if block.frequency_domain:
                    if block.normalise:
                        latent_n = latent / (latent.norm(dim=block.fourier_dim, keepdim=True) + 1e-8)
                    else:
                        latent_n = latent
                    l_real = torch.fft.fft(latent_n, dim=block.fourier_dim).real
                    x_real = torch.fft.fft(emb, dim=block.fourier_dim).real
                else:
                    l_real = latent
                    x_real = emb

                # Cross-attention + FF (single depth step)
                latent = cross_attn(l_real, context=x_real) + l_real
                latent = cross_ff(latent) + latent

        return self._output_head(latent)

    def _output_head(self, latent: torch.Tensor) -> torch.Tensor:
        """Mean-pool over latent channels, normalise, project."""
        # latent shape: [B, latent_channels, latent_dim]
        if latent.is_complex():
            latent = latent.real
        pooled = latent.mean(dim=1)  # [B, latent_dim]
        return self.output_proj(self.output_norm(pooled))


# ---------------------------------------------------------------------------
# Custom topology fusion (our original designs, renamed from MultimodalLegoFusion)
# ---------------------------------------------------------------------------

class CrossAttnBlock(nn.Module):
    """Cross-attention between two modalities."""

    def __init__(self, d: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        if query.dim() == 2:
            query = query.unsqueeze(1)
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(1)
        residual = query
        q = self.norm(query)
        out, _ = self.attn(q, key_value, key_value)
        return (residual + out).squeeze(1)


class GatedFusionBlock(nn.Module):
    """Sigmoid-gated element-wise fusion."""

    def __init__(self, d: int, n_inputs: int):
        super().__init__()
        self.gate = nn.Linear(d * n_inputs, n_inputs)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        concat = torch.cat(inputs, dim=-1)
        gates = torch.sigmoid(self.gate(concat))
        stacked = torch.stack(inputs, dim=-1)
        gated = (stacked * gates.unsqueeze(1)).sum(dim=-1)
        return gated


class CustomTopologyFusion(BaseFusionModule):
    """Composable fusion with configurable topology (pairwise/hierarchical/gated).

    These are custom cross-attention designs, NOT from the MM-Lego paper.
    Kept as useful alternative baselines.
    """

    def __init__(
        self,
        d_common: int = 256,
        modality_ids: list[str] | None = None,
        topology: str = "pairwise",
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(d_common=d_common, d_out=d_common)
        modality_ids = modality_ids or ["video", "eye_tracking", "ppg"]
        self.topology = topology
        self.mod_ids = modality_ids
        n_mod = len(modality_ids)

        if topology == "pairwise":
            self.pair_blocks = nn.ModuleDict()
            for a, b in combinations(range(n_mod), 2):
                self.pair_blocks[f"{a}_{b}"] = CrossAttnBlock(d_common, n_heads, dropout)
                self.pair_blocks[f"{b}_{a}"] = CrossAttnBlock(d_common, n_heads, dropout)
            n_outputs = n_mod * (n_mod - 1)
            self.output_mlp = nn.Sequential(
                nn.Linear(d_common * n_outputs, d_common),
                nn.ReLU(),
                nn.Linear(d_common, d_common),
            )

        elif topology == "hierarchical":
            self.stage1 = CrossAttnBlock(d_common, n_heads, dropout)
            self.stage2 = CrossAttnBlock(d_common, n_heads, dropout)
            self.output_mlp = nn.Sequential(
                nn.Linear(d_common * 2, d_common),
                nn.ReLU(),
                nn.Linear(d_common, d_common),
            )

        elif topology == "gated":
            self.self_attns = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_common, n_heads, d_common * 4, dropout,
                    batch_first=True, norm_first=True,
                )
                for _ in range(n_mod)
            ])
            self.cross_blocks = nn.ModuleDict()
            for a, b in combinations(range(n_mod), 2):
                self.cross_blocks[f"{a}_{b}"] = CrossAttnBlock(d_common, n_heads, dropout)
            self.gate = GatedFusionBlock(d_common, n_mod)
            self.output_mlp = nn.Linear(d_common, d_common)

    def forward(
        self,
        embeddings: list[torch.Tensor],
        modality_ids: list[str],
        masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        embs = [e.squeeze(1) if e.dim() == 3 and e.shape[1] == 1 else e for e in embeddings]
        if embs[0].dim() == 3:
            embs = [e.mean(dim=1) for e in embs]

        if self.topology == "pairwise":
            outputs = []
            for (a, b) in combinations(range(len(embs)), 2):
                outputs.append(self.pair_blocks[f"{a}_{b}"](embs[a], embs[b]))
                outputs.append(self.pair_blocks[f"{b}_{a}"](embs[b], embs[a]))
            return self.output_mlp(torch.cat(outputs, dim=-1))

        elif self.topology == "hierarchical":
            physio_fused = self.stage1(embs[1], embs[2])
            combined = self.stage2(embs[0], physio_fused)
            return self.output_mlp(torch.cat([combined, physio_fused], dim=-1))

        elif self.topology == "gated":
            refined = [sa(e.unsqueeze(1)).squeeze(1) for sa, e in zip(self.self_attns, embs)]
            for (a, b) in combinations(range(len(refined)), 2):
                crossed = self.cross_blocks[f"{a}_{b}"](refined[a], refined[b])
                refined[a] = refined[a] + crossed
            gated = self.gate(refined)
            return self.output_mlp(gated)

        raise ValueError(f"Unknown topology: {self.topology}")

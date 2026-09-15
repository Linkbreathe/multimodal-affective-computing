"""Configuration-driven fusion construction and shared projection/pooling."""

import torch

from mac.encoders.registry import ModalityRegistry
from mac.fusion.projector import ModalityProjector


def build_fusion_model(cfg: dict, registry: ModalityRegistry):
    """Build a fusion model + projector from config."""
    fusion_type = cfg.get("fusion_type", "early")
    d_common = cfg["fusion"]["d_common"]
    dropout = cfg["fusion"].get("dropout", 0.1)
    enabled = registry.get_enabled_modalities()
    embed_dims = registry.get_all_embed_dims()

    # Build projector
    projector = ModalityProjector(embed_dims=embed_dims, d_common=d_common)

    # Build fusion module
    if fusion_type == "early":
        from mac.fusion.early import EarlyFusion
        fusion = EarlyFusion(d_common=d_common, num_modalities=len(enabled), dropout=dropout)

    elif fusion_type == "mid":
        from mac.fusion.mid import MidFusion
        fusion = MidFusion(d_common=d_common, modality_ids=enabled, dropout=dropout)

    elif fusion_type == "late":
        from mac.fusion.late import LateFusion
        mode = cfg["fusion"].get("mode", "weighted")
        fusion = LateFusion(
            d_common=d_common,
            num_modalities=len(enabled),
            modality_ids=enabled,
            mode=mode,
            dropout=dropout,
        )

    elif fusion_type == "perceiver_io":
        from mac.fusion.perceiver_io import PerceiverIOFusion
        pcfg = cfg["fusion"].get("perceiver", {})
        fusion = PerceiverIOFusion(
            d_common=d_common,
            n_latents=pcfg.get("n_latents", 32),
            d_latent=pcfg.get("d_latent", d_common),
            n_layers=pcfg.get("n_layers", 2),
            n_heads=pcfg.get("n_heads", 4),
            dropout=pcfg.get("dropout", dropout),
        )

    elif fusion_type == "qformer":
        from mac.fusion.qformer import QFormerFusion
        qcfg = cfg["fusion"].get("qformer", {})
        fusion = QFormerFusion(
            d_common=d_common,
            n_queries=qcfg.get("n_queries", 16),
            d_query=qcfg.get("d_query", d_common),
            n_layers=qcfg.get("n_layers", 6),
            n_heads=qcfg.get("n_heads", 4),
            cross_attn_freq=qcfg.get("cross_attn_freq", 2),
            dropout=qcfg.get("dropout", dropout),
        )

    elif fusion_type == "healnet":
        from mac.fusion.healnet import HEALNetFusion
        hcfg = cfg["fusion"].get("healnet", {})
        fusion = HEALNetFusion(
            d_common=d_common,
            memory_size=hcfg.get("memory_size", 16),
            n_layers=hcfg.get("n_layers", 2),
            n_heads=hcfg.get("n_heads", 4),
            num_modalities=len(enabled),
            dropout=hcfg.get("dropout", dropout),
        )

    elif fusion_type == "multimodal_lego":
        from mac.fusion.multimodal_lego import MultimodalLegoFusion
        lcfg = cfg["fusion"].get("lego", {})
        fusion = MultimodalLegoFusion(
            d_common=d_common,
            modality_ids=enabled,
            mode=lcfg.get("mode", "merge-sum"),
            latent_channels=lcfg.get("latent_channels", 64),
            latent_dim=lcfg.get("latent_dim", 64),
            depth=lcfg.get("depth", 2),
            heads=lcfg.get("heads", 8),
            dim_head=lcfg.get("dim_head", 64),
            attn_dropout=lcfg.get("attn_dropout", 0.0),
            ff_dropout=lcfg.get("ff_dropout", 0.0),
            frequency_domain=lcfg.get("frequency_domain", True),
            fourier_dim=lcfg.get("fourier_dim", 1),
            track_imaginary=lcfg.get("track_imaginary", True),
            normalise=lcfg.get("normalise", True),
            alpha=lcfg.get("alpha", 0.5),
        )

    elif fusion_type == "bottleneck":
        from mac.fusion.bottleneck import BottleneckFusion
        bcfg = cfg["fusion"].get("bottleneck", {})
        fusion = BottleneckFusion(
            d_common=d_common,
            num_modalities=len(enabled),
            bottleneck_ratio=bcfg.get("bottleneck_ratio", 0.5),
            dropout=bcfg.get("dropout", dropout),
        )

    elif fusion_type == "tmc":
        from mac.fusion.tmc import TMCFusion
        tcfg = cfg["fusion"].get("tmc", {})
        fusion = TMCFusion(
            d_common=d_common,
            num_classes=tcfg.get("num_classes", 9),
            num_modalities=len(enabled),
            modality_ids=enabled,
            dropout=cfg["fusion"].get("dropout", dropout),
        )

    elif fusion_type == "distill_late":
        from mac.fusion.distill_late import DistillLateFusion, EnrichedModalityProjector
        dcfg = cfg["fusion"].get("distill", {})
        enriched_mods = dcfg.get("enriched_modalities", [])
        # Override projector with enriched version
        projector = EnrichedModalityProjector(
            embed_dims=embed_dims,
            d_common=d_common,
            enriched_modalities=enriched_mods,
        )
        fusion = DistillLateFusion(
            d_common=d_common,
            num_modalities=len(enabled),
            num_classes=dcfg.get("num_classes", 9),
            tau=dcfg.get("tau", 3.0),
            enriched_modalities=enriched_mods,
            teacher_modality=dcfg.get("teacher_modality", "video"),
            dropout=cfg["fusion"].get("dropout", dropout),
        )
        fusion.register_modality_classifiers(enabled)

    elif fusion_type == "enriched_late":
        from mac.fusion.distill_late import EnrichedModalityProjector
        from mac.fusion.late import LateFusion
        ecfg = cfg["fusion"].get("enriched", {})
        enriched_mods = ecfg.get("enriched_modalities", [])
        proj_dropout = ecfg.get("projection_dropout", 0.0)
        projector = EnrichedModalityProjector(
            embed_dims=embed_dims,
            d_common=d_common,
            enriched_modalities=enriched_mods,
            dropout=proj_dropout,
        )
        fusion = LateFusion(
            d_common=d_common,
            num_modalities=len(enabled),
            modality_ids=enabled,
            mode=cfg["fusion"].get("mode", "weighted"),
            dropout=dropout,
        )

    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")

    return projector, fusion


class ProjectedFusion(torch.nn.Module):
    """Wraps projector + fusion into a single module for the trainer."""

    def __init__(self, projector: ModalityProjector, fusion, modality_names: list[str]):
        super().__init__()
        self.projector = projector
        self.fusion = fusion
        self.modality_names = modality_names
        # Match BaseFusionModule interface
        self.d_common = fusion.d_common
        self.d_out = fusion.d_out

    def project_and_pool(self, embeddings, modality_ids, masks=None):
        """Project raw embeddings and optionally pool sequences.

        Returns (proj_list, masks) where proj_list contains the projected
        (and pooled, if the fusion module does not support sequences) tensors.
        Useful for inserting gradient hooks between projection and fusion.
        """
        proj_dict = {}
        for emb, mod_id in zip(embeddings, modality_ids):
            proj_dict[mod_id] = emb
        projected = self.projector(proj_dict)
        proj_list = [projected[m] for m in modality_ids]

        # If the fusion module does not support sequence inputs, pool any
        # 3D (B, T, D) projected tensors to 2D (B, D) with mask-aware mean.
        if not getattr(self.fusion, "supports_sequence_input", False):
            pooled = []
            mask_list = masks if masks else [None] * len(proj_list)
            for proj, mask in zip(proj_list, mask_list):
                if proj.dim() == 3:
                    if mask is not None:
                        # mask: (B, T) bool — True = valid token
                        mask_f = mask.unsqueeze(-1).float()  # (B, T, 1)
                        proj = (proj * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
                    else:
                        proj = proj.mean(dim=1)
                pooled.append(proj)
            proj_list = pooled
            masks = None

        return proj_list, masks

    def forward(self, embeddings, modality_ids, masks=None, raw_signals=None):
        proj_list, masks = self.project_and_pool(embeddings, modality_ids, masks)
        return self.fusion(proj_list, modality_ids, masks)

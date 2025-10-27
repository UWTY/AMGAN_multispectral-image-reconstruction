import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

# Ensure project root in path if needed
here = Path(__file__).resolve().parent
if str(here) not in sys.path:
    sys.path.insert(0, str(here))

from MultimodalUnetnewadap import Unet
from MultimodalGAN import MultiScaleDiscriminator


class UnetGAN(nn.Module):
    """
    GAN wrapper using a Multi-Modal Transformer as Generator
    and a Multi-Scale Discriminator for WGAN-GP training.
    """
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        l30_channels: int = 11,
        s1_channels: int = 3,
        planet_channels: int = 7,
        meta_dim: int = 11,
        out_channels: int = 12,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        ndf: int = 64,
        num_scales: int = 3,
        use_spectral_norm: bool = True,
        gp_weight: float = 10.0,
        use_meta=False,
        use_selfattention=False,
        use_spatial_attention=True
    ):
        super().__init__()
        # Generator: Transformer-based
        self.generator = Unet(
            in_channels_l30=l30_channels,
            in_channels_s1=s1_channels,
            in_channels_planet=planet_channels,
            meta_dim=meta_dim,
            out_channels_s30=out_channels,
            base_ch=64,
            use_meta=use_meta,
            use_selfattention=use_selfattention,
            use_spatial_attention=use_spatial_attention
        )
        # Discriminator: Multi-Scale PatchGAN
        self.discriminator = MultiScaleDiscriminator(
            in_channels_l30=l30_channels,
            in_channels_s1=s1_channels,
            in_channels_planet=planet_channels,
            in_channels_s30=out_channels,
            ndf=ndf,
            num_scales=num_scales,
            dropout_rate=dropout,
            use_spectral_norm=use_spectral_norm
        )
        # Gradient penalty coefficient
        self.gp_weight = gp_weight

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Use `.generator(...)` for generation and `.discriminator(...)` for discrimination separately."
        )


# For external scripts to instantiate model
MODEL_CLASS = UnetGAN
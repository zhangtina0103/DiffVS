from __future__ import annotations

import torch
from diffusers import UNet2DConditionModel
from torch import nn


class MarkerTokenEncoder(nn.Module):
    """Learnable target-marker tokens used as diffusion cross-attention context."""

    def __init__(self, marker_names: list[str], cross_attention_dim: int) -> None:
        super().__init__()
        self.marker_names = list(marker_names)
        self.embedding = nn.Embedding(len(self.marker_names), cross_attention_dim)
        nn.init.normal_(self.embedding.weight, std=cross_attention_dim**-0.5)

    def forward(self, marker_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(marker_ids).unsqueeze(1)


def expand_unet_input_channels(
    unet: UNet2DConditionModel,
    in_channels: int,
) -> UNet2DConditionModel:
    """Expand a pretrained UNet input conv for source-latent conditioning.

    The first half of the input receives the noisy target latent. The additional
    channels receive the source H&E latent. Existing pretrained weights are kept
    for the original channels and new channels are zero-initialized, so the model
    starts from the pretrained denoiser behavior.
    """

    if unet.config.in_channels == in_channels:
        return unet
    old_conv = unet.conv_in
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
    )
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.bias.copy_(old_conv.bias)
        keep = min(old_conv.in_channels, in_channels)
        new_conv.weight[:, :keep].copy_(old_conv.weight[:, :keep])
    unet.conv_in = new_conv
    unet.register_to_config(in_channels=in_channels)
    return unet


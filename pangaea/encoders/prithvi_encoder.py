# Adapted from https://github.com/NASA-IMPACT/hls-foundation-os

from logging import Logger
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple
from timm.models.vision_transformer import Block

from pangaea.encoders.base import Encoder
from pangaea.encoders.pos_embed import get_3d_sincos_pos_embed


class Prithvi_Encoder(Encoder):
    """
    Paper: https://arxiv.org/pdf/2310.18660
    Attributes:
        output_layers (int | list[int]): The layers from which to extract the output.
        img_size (int): The size of the input image.
        num_frames (int): The number of frames in the input data.
        patch_size (int): The size of each patch.
        in_chans (int): The number of input channels.
        patch_embed (PatchEmbed): The patch embedding layer.
        cls_token (nn.Parameter): The class token parameter.
        pos_embed (nn.Parameter): The positional embedding parameter.
        blocks (nn.ModuleList): The list of Transformer blocks.
    Methods:
        __init__(self, encoder_weights: str | Path, input_bands: dict[str, list[str]], input_size: int, output_layers: int | list[int], patch_size=16, tubelet_size=1, in_chans=3, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, num_frames=1):
            Initializes the Prithvi_Encoder with the given parameters.
        load_encoder_weights(self, logger: Logger) -> None:
            Loads the encoder weights from a pretrained model and handles any missing or incompatible shapes.
        freeze(self):
            Freezes the parameters of the encoder to prevent them from being updated during training.
        initialize_weights(self):
            Initializes the weights of the encoder, including the positional embeddings and patch embeddings.
        _init_weights(self, m):
            Initializes the weights of the given module using Xavier uniform initialization for Linear layers and constant initialization for LayerNorm layers.
        forward(self, image):
            Performs the forward pass of the encoder, embedding the input patches, adding positional embeddings, and applying the Transformer blocks.
    """

    def __init__(
        self,
        encoder_weights: str | Path,
        input_bands: dict[str, list[str]],
        input_size: int,
        output_dim: int | list[int],
        output_layers: int | list[int],
        download_url: str,
        patch_size=16,
        tubelet_size=1,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        num_frames=1,
    ):
        super().__init__(
            model_name="Prithvi",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=True,
            multi_temporal_output=True,
            pyramid_output=False,
            download_url=download_url,
        )

        self.output_layers = output_layers

        self.img_size = self.input_size
        self.tublet_size = tubelet_size

        if num_frames:
            self.num_frames = num_frames
        else:
            self.num_frames = 1

        self.patch_size = patch_size
        self.in_chans = in_chans
        self.patch_embed = PatchEmbed(
            self.img_size,
            patch_size,
            self.num_frames,
            tubelet_size,
            in_chans,
            embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.initialize_weights()

    def load_encoder_weights(self, logger: Logger) -> None:
        pretrained_model = torch.load(
            self.encoder_weights, map_location="cpu", weights_only=False
        )
        if any(name.startswith("encoder.") for name in pretrained_model.keys()):
            pretrained_model = {
                name.removeprefix("encoder."): value
                for name, value in pretrained_model.items()
                if name.startswith("encoder.")
            }

        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}
        for name, param in self.named_parameters():
            if name not in pretrained_model:
                missing[name] = param.shape
                continue

            pretrained_param = pretrained_model[name]
            if (
                name == "patch_embed.proj.weight"
                and pretrained_param.ndim == param.ndim == 5
                and pretrained_param.shape[0] == param.shape[0]
                and pretrained_param.shape[2:] == param.shape[2:]
                and pretrained_param.shape[1] >= param.shape[1]
            ):
                pretrained_encoder[name] = pretrained_param[:, : param.shape[1], ...]
            elif (
                name == "pos_embed"
                and pretrained_param.ndim == param.ndim == 3
                and pretrained_param.shape[0] == param.shape[0]
                and pretrained_param.shape[2] == param.shape[2]
            ):
                pretrained_encoder[name] = self._interpolate_pos_embed(pretrained_param)
            elif pretrained_param.shape != param.shape:
                incompatible_shape[name] = (param.shape, pretrained_param.shape)
            else:
                pretrained_encoder[name] = pretrained_param

        self.load_state_dict(pretrained_encoder, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)

    def _interpolate_pos_embed(self, pos_embed: torch.Tensor) -> torch.Tensor:
        if pos_embed.shape == self.pos_embed.shape:
            return pos_embed

        cls_pos = pos_embed[:, :1, :]
        patch_pos = pos_embed[:, 1:, :]

        tgt_t, tgt_h, tgt_w = self.patch_embed.grid_size
        spatial_tokens = tgt_h * tgt_w
        src_patch_tokens = patch_pos.shape[1]

        if src_patch_tokens % spatial_tokens != 0:
            return self.pos_embed

        src_t = src_patch_tokens // spatial_tokens
        patch_pos = patch_pos.reshape(1, src_t, tgt_h, tgt_w, self.embed_dim)
        patch_pos = patch_pos.permute(0, 4, 1, 2, 3)
        patch_pos = F.interpolate(
            patch_pos,
            size=(tgt_t, tgt_h, tgt_w),
            mode="trilinear",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 4, 1).reshape(1, -1, self.embed_dim)
        return torch.cat((cls_pos, patch_pos), dim=1)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, image):
        # embed patches
        x = image["optical"]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed

        # apply Transformer blocks

        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.output_layers:
                out = (
                    x[:, 1:, :]
                    .permute(0, 2, 1)
                    .view(
                        x.shape[0],
                        -1,
                        self.num_frames,
                        self.img_size // self.patch_size,
                        self.img_size // self.patch_size,
                    )
                    .squeeze(2)
                    .contiguous()
                )
                output.append(out)

        return output


    def enforce_single_temporal(self):

        self.num_frames = 1

        self.patch_embed = PatchEmbed(
            self.input_size,
            self.patch_size,
            1,
            self.tublet_size,
            self.in_chans,
            self.embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.embed_dim), requires_grad=False
        )

        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

class PatchEmbed(nn.Module):
    """Frames of 2D Images to Patch Embedding
    The 3D version of timm.models.vision_transformer.PatchEmbed
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=3,
        tubelet_size=1,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.grid_size = (
            num_frames // tubelet_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1]),
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # B,C,T,H,W -> B,C,L -> B,L,C
        x = self.norm(x)
        return x

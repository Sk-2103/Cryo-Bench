from logging import Logger
from pathlib import Path

import torch

from .base import Encoder


class THOR_Encoder(Encoder):
    """THOR ViT encoder adapter for PANGAEA decoders."""

    GLB_TO_THOR = {
        "B2": "S2:Blue",
        "B3": "S2:Green",
        "B4": "S2:Red",
        "B8": "S2:NIR",
        "B11": "S2:SWIR1",
        "B12": "S2:SWIR2",
        "ELEVATION": "dem-10m:elevation",
        "SLOPE": "dem-10m:slope",
        "VV": "S1:IW-VV",
        "VH": "S1:IW-VH",
    }

    def __init__(
        self,
        encoder_weights: str | Path | None,
        input_size: int,
        input_bands: dict[str, list[str]],
        embed_dim: int,
        output_layers: list[int],
        output_dim: int,
        download_url: str | None,
        model_type: str = "thor_vit_tiny_encoder_alibi_patch_size_embed_v1",
        patch_size: int = 16,
        gsd: int = 10,
        flexivit_ref_patch_size: int = 4,
        flexivit_ref_grid_size: int = 14,
        flexivit_token_budget: int = 1296,
    ):
        super().__init__(
            model_name="thor_encoder",
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=False,
            encoder_weights=encoder_weights,
            download_url=download_url,
        )

        import thor.models  # noqa: F401
        from thor.core.model_registry import MODELS

        self.patch_size = patch_size
        self.gsd = gsd
        self.grid_size = input_size // patch_size
        self.model_bands = self._model_bands(input_bands)
        self.groups = self._groups(self.model_bands)

        channels = {band: {"GSD": gsd, "patch_size": patch_size} for band in self.model_bands}
        if "S1:IW-VV" in channels:
            channels["S1:IW-VV"]["patch_embed_name"] = "S1:VV"
        if "S1:IW-VH" in channels:
            channels["S1:IW-VH"]["patch_embed_name"] = "S1:VH"

        self.model = MODELS.get_model(model_type)(
            {
                "ground_covers": [input_size * gsd],
                "aggr_type": "subsetmean",
                "cls_token_type": "pooled",
                "use_superposition_encoding": False,
                "use_fuzzy_encoding": False,
                "encoder_pos_type": "alibi",
                "use_flexivit": True,
                "select_patch_strategy": "min",
                "flexivit_ref_patch_size": flexivit_ref_patch_size,
                "flexivit_ref_grid_size": flexivit_ref_grid_size,
                "flexivit_token_budget": flexivit_token_budget,
                "flexivit_patch_size_seqs": [patch_size],
                "channels": channels,
                "groups": self.groups,
            }
        )
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "model"):
            self.model.eval()
        return self

    def _model_bands(self, input_bands: dict[str, list[str]]) -> list[str]:
        bands = []
        for modality in ("optical", "sar"):
            for band in input_bands.get(modality, []):
                if band not in self.GLB_TO_THOR:
                    raise ValueError(
                        f"Band {band} is not supported by the THOR GLB adapter. "
                        "Use B2/B3/B4/B8/B11/B12, ELEVATION/SLOPE, and VV/VH."
                    )
                bands.append(self.GLB_TO_THOR[band])
        return bands

    def _groups(self, model_bands: list[str]) -> list[list[str]]:
        preferred_groups = [
            ["S2:Blue", "S2:Green", "S2:Red", "S2:NIR"],
            ["S2:SWIR1", "S2:SWIR2"],
            ["dem-10m:elevation", "dem-10m:slope"],
            ["S1:IW-VV", "S1:IW-VH"],
        ]
        model_band_set = set(model_bands)
        return [
            [band for band in group if band in model_band_set]
            for group in preferred_groups
            if any(band in model_band_set for band in group)
        ]

    def _images_to_thor(self, images: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        thor_images = {}
        for modality in ("optical", "sar"):
            for index, band in enumerate(self.input_bands.get(modality, [])):
                thor_images[self.GLB_TO_THOR[band]] = images[modality][:, index : index + 1]
        return thor_images

    def _tokens_to_feature_map(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, token_count, dim = tokens.shape
        spatial_tokens = self.grid_size * self.grid_size
        if token_count % spatial_tokens != 0:
            raise ValueError(
                f"THOR returned {token_count} tokens, which is not divisible by "
                f"{spatial_tokens} spatial tokens."
            )
        group_count = token_count // spatial_tokens
        tokens = tokens.view(batch, group_count, spatial_tokens, dim).mean(dim=1)
        return tokens.transpose(1, 2).reshape(
            batch, dim, self.grid_size, self.grid_size
        )

    def forward(self, images: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        thor_images = self._images_to_thor(images)
        outputs = self.model.forward_intermediates(
            thor_images,
            ground_cover=self.input_size * self.gsd,
            indices=self.output_layers,
            norm=True,
            intermediates_only=True,
        )
        return [self._tokens_to_feature_map(output) for output in outputs]

    def load_encoder_weights(self, logger: Logger) -> None:
        if self.encoder_weights is None:
            return

        try:
            checkpoint = torch.load(
                self.encoder_weights, map_location="cpu", weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(self.encoder_weights, map_location="cpu")

        state_dict = {}
        for key, value in checkpoint.items():
            if not key.startswith("encoder."):
                continue
            model_key = key.removeprefix("encoder.")
            if model_key.startswith("band_embed."):
                continue
            state_dict[model_key] = value

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing THOR parameters:\n" + "\n".join(sorted(missing)))
        if unexpected:
            skipped = sorted(unexpected)
            logger.info(
                "Skipped %d unused THOR checkpoint parameters. First entries:\n%s",
                len(skipped),
                "\n".join(skipped[:20]),
            )

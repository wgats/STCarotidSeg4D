import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNETR, DynUNet, ViT

from constants import INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, DEVICE


def create_temporal_encoder_mask(
    time_steps: int, patch_sequence_length: int
) -> torch.Tensor:
    total_sequence_length = time_steps * patch_sequence_length
    mask = torch.full(
        (total_sequence_length, total_sequence_length), float("-inf"), device=DEVICE
    )
    for i in range(patch_sequence_length):
        for t in range(time_steps):
            idx = t * patch_sequence_length + i
            mask[idx, i::patch_sequence_length] = 0.0
    for i in range(total_sequence_length):
        start_idx = (i // patch_sequence_length) * time_steps
        mask[i, start_idx : start_idx + time_steps] = 0.0
    return mask


class SpatioTemporalTransformer(ViT):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] = (128, 128, 16),
        patch_size: Sequence[int] = (16, 16, 1),
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        proj_type: str = "conv",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        mask: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        # constants
        self.patch_size = patch_size
        self.embed_dim = hidden_size
        self.out_channels = out_channels

        temporal_dimension_size = img_size[-1]
        embedded_patch_length = (img_size[0] * img_size[1] * img_size[2]) // (
            patch_size[0] * patch_size[1] * patch_size[2]
        )

        self.attn_mask = (
            create_temporal_encoder_mask(
                time_steps=temporal_dimension_size,
                patch_sequence_length=embedded_patch_length,
            )
            if mask
            else None
        )

        self.norm = nn.LayerNorm(hidden_size)

        self.segmentation_head = SegmentationHead3D(
            in_channels=hidden_size,
            out_channels=self.out_channels,
            patch_size=self.patch_size[0],
        )

    def forward(self, x):
        batch_size, channels, height, width, depth = x.shape
        downsampled_h = height // self.patch_size[0]
        downsampled_w = width // self.patch_size[1]
        downsampled_d = depth // self.patch_size[2]

        x = self.patch_embedding(x)
        for blk in self.blocks:
            x = blk(x, self.attn_mask)
        x = self.norm(x)
        x = x.reshape(
            batch_size, self.embed_dim, downsampled_d, downsampled_h, downsampled_w
        )
        x = self.segmentation_head(x)
        x = x.reshape(batch_size, self.out_channels, height, width, depth)

        return x


class SegmentationHead3D(nn.Module):
    def __init__(self, in_channels=768, out_channels=2, patch_size=16):
        super(SegmentationHead3D, self).__init__()
        num_stages = int(math.log2(patch_size))
        self.upsample_stages = nn.ModuleList()

        self.conv_initial = nn.Conv3d(
            in_channels, 512, kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        self.bn_initial = nn.BatchNorm3d(512)

        current_channels = 512
        for i in range(num_stages):
            out_ch = max(current_channels // 2, out_channels)
            self.upsample_stages.append(
                nn.Sequential(
                    nn.Conv3d(
                        current_channels,
                        out_ch,
                        kernel_size=(1, 3, 3),
                        padding=(0, 1, 1),
                    ),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(),
                    nn.Upsample(
                        scale_factor=(1, 2, 2), mode="trilinear", align_corners=False
                    ),
                )
            )
            current_channels = out_ch

        self.conv_final = nn.Conv3d(
            current_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        self.bn_final = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn_initial(self.conv_initial(x)))
        for stage in self.upsample_stages:
            x = stage(x)
        x = self.bn_final(self.conv_final(x))

        return x


class SegmentationHead2D(nn.Module):
    def __init__(self, in_channels=768, out_channels=2, patch_size=16):
        super(SegmentationHead2D, self).__init__()
        num_stages = int(math.log2(patch_size))
        self.upsample_stages = nn.ModuleList()

        self.conv_initial = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.bn_initial = nn.BatchNorm2d(512)

        current_channels = 512
        for i in range(num_stages):
            out_ch = max(current_channels // 2, out_channels)
            self.upsample_stages.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                )
            )
            current_channels = out_ch

        self.conv_final = nn.Conv2d(
            current_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn_final = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn_initial(self.conv_initial(x)))
        for stage in self.upsample_stages:
            x = stage(x)
        x = self.bn_final(self.conv_final(x))

        return x


class ModelFactory:
    @staticmethod
    def get_model(config: dict):
        match config["name"]:
            case "DynUNet":
                model = DynUNet(
                    spatial_dims=config["spatial_dims"],
                    in_channels=config["in_channels"],
                    out_channels=config["out_channels"],
                    kernel_size=config["kernel_size"],
                    strides=config["strides"],
                    filters=config["filters"],
                    upsample_kernel_size=config["upsample_kernel_size"],
                    norm_name=config["norm_name"],
                    act_name=config["act_name"],
                    res_block=config["res_block"],
                )
                return model
            case "UNETR":
                model = UNETR(
                    in_channels=config["in_channels"],
                    out_channels=config["out_channels"],
                    img_size=config["img_size"],
                )
                return model
            case "SpatioTemporalTransformer":
                model = SpatioTemporalTransformer(
                    in_channels=config["in_channels"],
                    out_channels=config["out_channels"],
                    img_size=config["img_size"],
                    patch_size=config["patch_size"],
                    hidden_size=config["hidden_size"],
                    mlp_dim=config["mlp_dim"],
                    num_layers=config["num_layers"],
                    num_heads=config["num_heads"],
                    dropout_rate=config["dropout_rate"],
                    spatial_dims=config["spatial_dims"],
                    mask=config["mask"],
                )
                return model
            case _:
                return None

    @staticmethod
    def get_config(
        model_name: str, channels: int, depth: int | None = None
    ) -> dict | None:
        match model_name.lower():
            case "unet2d":
                config = {
                    "name": "DynUNet",
                    "spatial_dims": 2,
                    "in_channels": channels,
                    "out_channels": 2,
                    "kernel_size": [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)],
                    "strides": [(1, 1), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
                    "filters": [32, 64, 128, 256, 512, 512],
                    "upsample_kernel_size": [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)],
                    "norm_name": "instance",
                    "act_name": "leakyrelu",
                    "res_block": False,
                }
                return config
            case "unet3d":
                config = {
                    "name": "DynUNet",
                    "spatial_dims": 3,
                    "in_channels": channels,
                    "out_channels": 2,
                    "kernel_size": [
                        (3, 3, 3),
                        (3, 3, 3),
                        (3, 3, 3),
                        (3, 3, 3),
                        (3, 3, 3),
                        (3, 3, 3),
                    ],
                    "strides": [
                        (1, 1, 1),
                        (2, 2, 2),
                        (2, 2, 1),
                        (2, 2, 1),
                        (2, 2, 1),
                        (2, 2, 1),
                    ],
                    "filters": [32, 64, 128, 256, 320, 320],
                    "upsample_kernel_size": [
                        (2, 2, 2),
                        (2, 2, 1),
                        (2, 2, 1),
                        (2, 2, 1),
                        (2, 2, 1),
                    ],
                    "norm_name": "instance",
                    "act_name": "leakyrelu",
                    "res_block": False,
                }
                return config
            case "unetr":
                config = {
                    "name": "UNETR",
                    "in_channels": channels,
                    "out_channels": 2,
                    "img_size": (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, depth),
                }
                return config
            case "spatio_temporal_transformer":
                config = {
                    "name": "SpatioTemporalTransformer",
                    "in_channels": channels,
                    "out_channels": 2,
                    "img_size": (INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, depth),
                    "patch_size": (32, 32, 1),
                    "hidden_size": 768,
                    "mlp_dim": 3072,
                    "num_layers": 12,
                    "num_heads": 12,
                    "dropout_rate": 0.0,
                    "spatial_dims": 3,
                    "mask": False,
                }
                return config
            case _:
                return None

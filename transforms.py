from collections.abc import Hashable, Mapping, Sequence

import monai.transforms as mt
import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import SpatialPadD, InvertibleTransform

from constants import (
    DEVICE,
    IMAGE_KEY,
    INPUT_IMAGE_HEIGHT,
    INPUT_IMAGE_WIDTH,
    KEYS,
    LABEL_KEY,
    PRED_KEY,
)

TRANSFORM_TYPES = ["interpolate", "pad", "pad-crop", "pad-average", "none"]


class AverageSpatialPadD(SpatialPadD, InvertibleTransform):
    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Sequence[int] | int,
        mode: str = "wrap",
        method: str = "end",
    ):
        super().__init__(keys=keys, spatial_size=spatial_size, mode=mode, method=method)

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None
    ) -> dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.keys:
            if key in d:
                original_shape = d[key].shape
                d = super().__call__(d)
                self.push_transform(
                    d, key, extra_info={"original_shape": original_shape}
                )
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            transform_info = self.pop_transform(d, key)
            original_shape = transform_info["extra_info"]["original_shape"]

            original_len = original_shape[-1]
            padded_len = d[key].shape[-1]

            repeated_indices = torch.arange(original_len).repeat(
                (padded_len + original_len - 1) // original_len
            )[:padded_len]

            result = torch.empty(
                *d[key].shape[:-1],
                original_len,
                dtype=d[key].dtype,
                device=d[key].device,
            )

            for i in range(original_len):
                indices = torch.where(repeated_indices == i)[0]
                result[..., i] = d[key][..., indices].mean(dim=-1)
            d[key] = result
        return d


def get_training_transform(transform_type: str, depth: int | None = 16):
    if transform_type not in TRANSFORM_TYPES:
        raise ValueError

    if transform_type == "none":
        train_transform = mt.Compose(
            [
                mt.LoadImageD(
                    KEYS, reader="monai.data.NibabelReader", ensure_channel_first=True
                ),
                mt.SqueezeDimd(KEYS, dim=-2),
                mt.SqueezeDimd(KEYS, dim=-2),
                mt.ToDeviceD(KEYS, DEVICE),
                mt.AsDiscreted(LABEL_KEY, to_onehot=2),
                mt.NormalizeIntensityd(IMAGE_KEY, channel_wise=True),
                mt.RandRotateD(
                    KEYS,
                    range_z=[-np.pi, np.pi],
                    prob=0.5,
                ),
            ]
        )

        return train_transform

    if transform_type == "interpolate":
        pre_processing = mt.ResizeD(
            keys=KEYS,
            spatial_size=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, depth),
            mode="trilinear",
        )
    elif transform_type == "pad":
        pre_processing = mt.SpatialPadD(
            KEYS,
            spatial_size=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, depth),
            mode="wrap",
            method="end",
        )
    elif transform_type == "pad-average":
        pre_processing = AverageSpatialPadD(
            KEYS,
            spatial_size=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, depth),
            mode="wrap",
            method="end",
        )

    train_transform = mt.Compose(
        [
            mt.LoadImageD(
                KEYS, reader="monai.data.NibabelReader", ensure_channel_first=True
            ),
            mt.SqueezeDimd(KEYS, dim=-2),
            mt.SqueezeDimd(KEYS, dim=-2),
            pre_processing,
            mt.ToDeviceD(KEYS, DEVICE),
            mt.AsDiscreted(LABEL_KEY, to_onehot=2),
            mt.NormalizeIntensityd(IMAGE_KEY, channel_wise=True),
            mt.RandRotateD(
                KEYS,
                range_z=[-np.pi, np.pi],
                prob=0.5,
            ),
        ]
    )

    return train_transform


def get_validation_transform(transform_type, depth: int | None = 16):
    if transform_type not in TRANSFORM_TYPES:
        raise ValueError

    if transform_type == "none":
        validation_transform = mt.Compose(
            [
                mt.LoadImageD(
                    KEYS, reader="monai.data.NibabelReader", ensure_channel_first=True
                ),
                mt.SqueezeDimd(KEYS, dim=-2),
                mt.SqueezeDimd(KEYS, dim=-2),
                mt.ToDeviced(KEYS, DEVICE),
                mt.AsDiscreted(LABEL_KEY, to_onehot=2),
                mt.NormalizeIntensityd(IMAGE_KEY, channel_wise=True),
            ]
        )

        return validation_transform

    if transform_type == "interpolate":
        pre_processing = mt.ResizeD(
            keys=KEYS,
            spatial_size=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, depth),
            mode="trilinear",
        )
    elif transform_type in ["pad", "pad-crop"]:
        pre_processing = mt.SpatialPadD(
            KEYS,
            spatial_size=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, depth),
            mode="wrap",
            method="end",
        )
    elif transform_type == "pad-average":
        pre_processing = AverageSpatialPadD(
            KEYS,
            spatial_size=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, depth),
            mode="wrap",
            method="end",
        )

    validation_transform = mt.Compose(
        [
            mt.LoadImageD(
                KEYS, reader="monai.data.NibabelReader", ensure_channel_first=True
            ),
            mt.SqueezeDimd(KEYS, dim=-2),
            mt.SqueezeDimd(KEYS, dim=-2),
            pre_processing,
            mt.ToDeviced(KEYS, DEVICE),
            mt.AsDiscreted(LABEL_KEY, to_onehot=2),
            mt.NormalizeIntensityd(IMAGE_KEY, channel_wise=True),
        ]
    )

    return validation_transform


def get_test_transform(transform_type, depth: int | None = 16):
    if transform_type not in TRANSFORM_TYPES:
        raise ValueError

    if transform_type == "none":
        test_transform = mt.Compose(
            [
                mt.LoadImageD(
                    IMAGE_KEY,
                    reader="monai.data.NibabelReader",
                    ensure_channel_first=True,
                    image_only=False,
                ),
                mt.SqueezeDimd(IMAGE_KEY, dim=-2),
                mt.SqueezeDimd(IMAGE_KEY, dim=-2),
                mt.ToDeviceD(IMAGE_KEY, "cpu"),
                mt.NormalizeIntensityd(IMAGE_KEY, channel_wise=True),
            ]
        )

        return test_transform

    if transform_type == "interpolate":
        pre_processing = mt.ResizeD(
            keys=IMAGE_KEY,
            spatial_size=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, depth),
            mode="trilinear",
        )
    elif transform_type == "pad":
        pre_processing = mt.SpatialPadD(
            IMAGE_KEY,
            spatial_size=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, depth),
            mode="wrap",
            method="end",
        )
    elif transform_type in ["pad", "pad-crop"]:
        pre_processing = mt.SpatialPadD(
            keys=IMAGE_KEY,
            spatial_size=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, depth),
            mode="wrap",
            method="end",
        )
    elif transform_type == "pad-average":
        pre_processing = AverageSpatialPadD(
            keys=IMAGE_KEY,
            spatial_size=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, depth),
            mode="wrap",
            method="end",
        )

    test_transform = mt.Compose(
        [
            mt.LoadImageD(
                IMAGE_KEY,
                reader="monai.data.NibabelReader",
                ensure_channel_first=True,
                image_only=False,
            ),
            mt.SqueezeDimd(IMAGE_KEY, dim=-2),
            mt.SqueezeDimd(IMAGE_KEY, dim=-2),
            pre_processing,
            mt.ToDeviceD(IMAGE_KEY, "cpu"),
            mt.NormalizeIntensityd(IMAGE_KEY, channel_wise=True),
        ]
    )

    return test_transform


def get_raw_transform(transform_type, depth: int | None = 16, remove_dims=True):
    if transform_type not in TRANSFORM_TYPES:
        raise ValueError

    if transform_type == "none":
        return mt.Compose(
            [
                mt.LoadImageD(
                    IMAGE_KEY,
                    reader="monai.data.NibabelReader",
                    ensure_channel_first=True,
                    image_only=False,
                ),
            ]
        )

    if transform_type == "interpolate":
        pre_processing = mt.Compose(
            [
                mt.ResizeD(
                    keys=IMAGE_KEY,
                    spatial_size=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, depth),
                    mode="trilinear",
                ),
            ]
        )

    elif transform_type in ["pad", "pad-crop"]:
        pre_processing = mt.SpatialPadD(
            keys=IMAGE_KEY,
            spatial_size=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, depth),
            mode="wrap",
            method="end",
        )
    elif transform_type == "pad-average":
        pre_processing = AverageSpatialPadD(
            keys=IMAGE_KEY,
            spatial_size=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, depth),
            mode="wrap",
            method="end",
        )

    if remove_dims:
        test_transform = mt.Compose(
            [
                mt.LoadImageD(
                    IMAGE_KEY,
                    reader="monai.data.NibabelReader",
                    ensure_channel_first=True,
                    image_only=False,
                ),
                mt.SqueezeDimd(IMAGE_KEY, dim=-2),
                mt.SqueezeDimd(IMAGE_KEY, dim=-2),
                pre_processing,
            ]
        )
    else:
        test_transform = mt.Compose(
            [
                mt.LoadImageD(
                    IMAGE_KEY,
                    reader="monai.data.NibabelReader",
                    ensure_channel_first=True,
                    image_only=False,
                ),
                pre_processing,
            ]
        )

    return test_transform


def get_post_transform(pre_transform):
    post_transform = mt.Compose(
        [
            mt.InvertD(
                keys=PRED_KEY,
                transform=pre_transform,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
            ),
        ]
    )

    return post_transform


def get_ensemble_post_transform(pre_transform, invert=True):
    if invert:
        post_transform = mt.Compose(
            [
                mt.InvertD(
                    keys=["pred_0", "pred_1", "pred_2", "pred_3", "pred_4"],
                    transform=pre_transform,
                    orig_keys="image",
                    meta_keys=[
                        "pred_0_meta_dict",
                        "pred_1_meta_dict",
                        "pred_2_meta_dict",
                        "pred_3_meta_dict",
                        "pred_4_meta_dict",
                    ],
                    orig_meta_keys="image_meta_dict",
                    meta_key_postfix="meta_dict",
                ),
                mt.VoteEnsembleD(
                    keys=["pred_0", "pred_1", "pred_2", "pred_3", "pred_4"],
                    output_key="validation",
                ),
                mt.AsDiscreted(keys="validation", argmax=True),
            ]
        )
    else:
        post_transform = mt.Compose(
            [
                mt.VoteEnsembleD(
                    keys=["pred_0", "pred_1", "pred_2", "pred_3", "pred_4"],
                    output_key="validation",
                ),
                mt.AsDiscreted(keys="validation", argmax=True),
            ]
        )

    return post_transform


def get_validation_post_transform(pre_transform):
    post_transform = mt.Compose(
        [
            mt.Invertd(
                keys=["validation"],
                transform=pre_transform,
                orig_keys="image",
            ),
            mt.AsDiscreted(keys="validation", argmax=True),
        ]
    )

    return post_transform

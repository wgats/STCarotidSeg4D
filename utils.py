import os
import json
import math
import random
import re
from pathlib import Path

import tqdm
import nibabel as nib
import torch
import numpy as np

from constants import INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH


def save_as_nifti(
    prediction: torch.Tensor,
    image_meta: dict,
    prediction_folder: os.PathLike,
    depth: int = 16,
    post_process: bool = False,
):
    """Convert probability map into a binary mask and save to NFTI.

    Args:
        prediction: Prediction probabilities. Axis 0 should be classes/channels.
        image_meta: Image metadata.
        prediction_folder: Folder to save to.
        depth: Temporal dimension size. Defaults to 16.
        post_process: Do temporal dimension post-processing. Defaults to False.
    """
    # This line converts the activations/probabilities into a binary mask by taking
    #  Class 0 is background, class 1 is lumen.
    prediction_mask = np.int16(np.argmax(prediction.cpu().detach(), axis=0))
    prediction_path = os.path.join(
        prediction_folder,
        os.path.basename(image_meta["filename_or_obj"]).replace("_0000.nii", ".nii"),
    )
    world_matrix = image_meta["affine"]
    if post_process:
        nframes = image_meta["dim"][5]
    else:
        nframes = depth
    img = nib.Nifti1Image(
        prediction_mask.reshape(
            (INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, 1, 1, nframes, 1)
        ),
        world_matrix,
    )
    nib.save(img, prediction_path)


def search_for_data(imgdir: Path, labeldir: Path):
    imgpaths = sorted(list(imgdir.glob("*.nii.gz")))
    labelpaths = sorted(list(labeldir.glob("*.nii.gz")))
    return imgpaths, labelpaths


def train_val_test_split(
    imgpaths: list[Path], labelpaths: list[Path], ratio: float = 0.8, seed: int = 42
):
    print(f"Splitting dataset with train ratio {ratio}, seed {seed}")
    x = set([str(path) for path in imgpaths])
    y = set([str(path) for path in labelpaths])

    pattern = re.compile("(\D{2}\d{3})")
    img_ids = [pattern.search(s).group(1) for s in x]
    label_ids = [pattern.search(s).group(1) for s in y]
    overlap_ids = set(img_ids).intersection(set(label_ids))

    total = len(overlap_ids)
    ntrain = math.floor(ratio * total)
    nval = (total - ntrain) // 2
    ntest = total - (ntrain + nval)
    random.seed(seed)
    samples = random.sample(sorted(list(overlap_ids)), total)

    train_ids = samples[:ntrain]
    print(f"train_ids: {train_ids}")
    val_ids = samples[ntrain : (ntrain + nval)]
    print(f"val_ids: {val_ids}")
    test_ids = samples[-ntest:]
    print(f"test_ids: {test_ids}")

    assert len(set(train_ids).intersection(set(val_ids))) == 0
    assert len(set(train_ids).intersection(set(test_ids))) == 0
    assert len(set(test_ids).intersection(set(val_ids))) == 0

    tr_images = [
        path for path in imgpaths if pattern.search(str(path)).group(1) in train_ids
    ]
    tr_labels = [
        path for path in labelpaths if pattern.search(str(path)).group(1) in train_ids
    ]

    val_images = [
        path for path in imgpaths if pattern.search(str(path)).group(1) in val_ids
    ]
    val_labels = [
        path for path in labelpaths if pattern.search(str(path)).group(1) in val_ids
    ]

    test_images = [
        path for path in imgpaths if pattern.search(str(path)).group(1) in test_ids
    ]
    test_labels = [
        path for path in labelpaths if pattern.search(str(path)).group(1) in test_ids
    ]

    assert len(set(tr_images).intersection(set(val_images))) == 0
    assert len(set(tr_images).intersection(set(test_images))) == 0
    assert len(set(test_images).intersection(set(val_images))) == 0

    return tr_images, tr_labels, val_images, val_labels, test_images, test_labels


def get_test_images(imgpaths: list[Path], ratio: float = 1.0, seed: int = 42):
    x = set([str(path) for path in imgpaths])

    pattern = re.compile("(\D{2}\d{3})")
    img_ids = [pattern.search(s).group(1) for s in x]
    img_ids = set(img_ids)

    total = len(img_ids)
    ntrain = math.floor(ratio * total)
    nval = (total - ntrain) // 2
    ntest = total - (ntrain + nval)
    random.seed(seed)
    samples = random.sample(sorted(list(img_ids)), total)

    test_ids = samples[-ntest:]
    test_images = [
        path for path in imgpaths if pattern.search(str(path)).group(1) in test_ids
    ]

    return test_images


def verify_triple_list(triple_list: list[list]):
    if len(set(list(map(check_path_triple, triple_list)))) != 1:
        return False
    return True


def check_path_triple(triple: list[Path]):
    suffixes = ["0000.nii.gz", "0001.nii.gz", "0002.nii.gz"]
    triple = list(map(str, triple))
    common = [s[:-11] for s in triple]
    endings = [s[-11:] for s in triple]

    if len(set(common)) == 1 and endings == suffixes:
        return True
    else:
        return False


def prep_multichannel_images(imgpaths, num_channels=3):
    imgpaths = sorted(imgpaths)
    train_images = [
        imgpaths[i : i + num_channels] for i in range(0, len(imgpaths), num_channels)
    ]
    if verify_triple_list(train_images) is not True:
        raise ValueError("Image path triplets are incorrect!")
    return train_images


def load_nifti_image_and_swap_axes(original_img_path: Path) -> None:
    # load image
    original_img = nib.load(original_img_path)
    # get array and swap axes
    new_array = original_img.get_fdata().swapaxes(2, 4)
    # copy affine, header and change the data shape
    new_header = original_img.header.copy()
    new_header.set_data_shape(new_array.shape)
    affine = original_img.affine
    # save with new header
    new_image = nib.Nifti1Image(new_array, affine, new_header)
    nib.save(new_image, original_img_path)


def load_splits_from_json(data_path, num_channels=3):
    json_path = Path(data_path) / "splits_final.json"

    with open(json_path, "r") as fp:
        d = json.load(fp)

    splits = []
    num_folds = len(d)

    for i in tqdm.tqdm(range(num_folds), desc="Loading splits_final.json"):
        img_path = Path(data_path) / "images"
        mask_path = Path(data_path) / "labels"
        img_pathlist = sorted(list(Path(img_path).rglob("*.nii.gz")))
        mask_pathlist = sorted(list(Path(mask_path).rglob("*.nii.gz")))

        img_train_paths = [
            path
            for path in img_pathlist
            if "_".join(path.stem.split("_")[:4]) in d[i]["train"]
        ]
        img_train_paths = prep_multichannel_images(img_train_paths, num_channels)

        mask_train_paths = [
            path for path in mask_pathlist if path.stem.split(".")[0] in d[i]["train"]
        ]

        if len(img_train_paths) != len(mask_train_paths):
            raise ValueError

        img_val_paths = [
            path
            for path in img_pathlist
            if "_".join(path.stem.split("_")[:4]) in d[i]["val"]
        ]
        img_val_paths = prep_multichannel_images(img_val_paths, num_channels)

        mask_val_paths = [
            path for path in mask_pathlist if path.stem.split(".")[0] in d[i]["val"]
        ]

        if len(img_val_paths) != len(mask_val_paths):
            raise ValueError

        splits.append(
            (img_train_paths, mask_train_paths, img_val_paths, mask_val_paths)
        )

    return splits

import argparse
import logging
import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from tqdm import tqdm

import transforms
import monai.transforms as mt
from constants import INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH
from utils import get_test_images, load_splits_from_json

IMAGE_KEY = "image"
PRED_KEY = "pred"
DEVICE = "cpu"
FOLDER_SUFFIX = {"pad-average": "_avg", "pad-crop": "_crop"}


def make_sure_folder_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def search_for_test_data_split(image_folder: os.PathLike, split_percentage: float):
    imgpaths = sorted(list(Path(image_folder).glob("*.nii.gz")))
    test_images = get_test_images(imgpaths, ratio=split_percentage)
    return test_images


def search_for_test_data(image_folder):
    imgpaths = sorted(list(Path(image_folder).glob("*.nii.gz")))
    return imgpaths


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
    # This converts the activations/probabilities into a binary mask by taking
    # the argmax along the 0th axis which represents the classes. Class 0
    # is background, and class 1 is lumen.
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


def run_inference(
    model_path,
    eval_images,
    prediction_folder,
    pre_transform,
    post_transform,
    depth,
    post_process,
    slices,
):
    make_sure_folder_exists(prediction_folder)

    if torch.cuda.is_available():
        DEVICE = "cuda"
    model = torch.load(model_path, map_location=torch.device(DEVICE))
    _ = model.eval()

    eval_files = [{IMAGE_KEY: img} for img in eval_images]
    eval_data_set = Dataset(data=eval_files, transform=pre_transform)
    if slices:
        batch_size = 1
    else:
        batch_size = 8
    eval_loader = DataLoader(eval_data_set, batch_size=batch_size, shuffle=False)

    times = []

    with torch.no_grad():
        for eval_data in tqdm(eval_loader):
            start_time = time.time()
            if slices:
                eval_data["validation"] = torch.stack(
                    [
                        model(sample.permute(3, 0, 1, 2)).permute(1, 2, 3, 0)
                        for sample in eval_data[IMAGE_KEY].to(DEVICE)
                    ],
                    dim=0,
                )
            else:
                eval_data["validation"] = model(eval_data[IMAGE_KEY].to(DEVICE))
            times.append(time.time() - start_time)
            outputs = decollate_batch(eval_data)

            saver = mt.SaveImaged(
                keys="validation",
                output_dir=prediction_folder,
                output_postfix="",
                output_ext=".nii.gz",
                separate_folder=False,
                resample=False,
            )

            for item in outputs:
                item = post_transform(item)
                fname = item["image"].meta["filename_or_obj"]
                # Remove '_0000' before .nii.gz
                if fname.endswith("_0000.nii.gz"):
                    item["validation"].meta["filename_or_obj"] = fname.replace(
                        "_0000.nii.gz", ".nii.gz"
                    )
                saver(item)
    return times


def main(
    model_folder,
    model_name,
    dataset_folder,
    output_folder,
    num_channels,
    transform_type,
    depth,
    post_process,
    slices,
):
    logger.info(f"Model folder: {model_folder}")
    logger.info(f"Dataset: {dataset_folder}")
    logger.info(f"Post-processing: {post_process}")
    logger.info(f"Output folder: {output_folder}")

    splits = load_splits_from_json(dataset_folder, num_channels=num_channels)

    for fold in range(len(splits)):
        image_files = splits[fold][2]
        pre_transform = transforms.get_test_transform(transform_type, depth)

        if post_process:
            post_transform = transforms.get_validation_post_transform(pre_transform)
        else:
            post_transform = None

        model_path = Path(model_folder) / f"fold_{fold}" / f"{model_name}.pth"
        times = run_inference(
            model_path,
            image_files,
            output_folder,
            pre_transform,
            post_transform,
            depth,
            post_process,
            slices,
        )
        logger.info(f"Mean inference time: {np.mean(times)}")
        logger.info(f"Median inference time: {np.median(times)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Run inference on validation splits for evaluating the best model.",
    )
    parser.add_argument("-m", "--model_folder")
    parser.add_argument("-n", "--model_name")
    parser.add_argument("-d", "--dataset_folder")
    parser.add_argument("-c", "--channels", required=True, type=int)
    parser.add_argument("-i", "--depth", nargs="?", type=int)
    parser.add_argument(
        "-t",
        "--transform_type",
        choices=["none", "interpolate", "pad", "pad-crop", "pad-average"],
        required=True,
        help="Specify the pre-processing type. Must be one of interpolate, pad",
    )
    parser.add_argument(
        "-p",
        "--post_process",
        action="store_true",
        help="Post process the output segmentation masks.",
    )

    parser.add_argument(
        "-s",
        "--slices",
        action="store_true",
        help="Process in 2D slices, for 2D models only.",
    )
    args = parser.parse_args()

    if args.transform_type in FOLDER_SUFFIX:
        suffix = FOLDER_SUFFIX[args.transform_type]
    else:
        suffix = ""
    output_folder = Path(args.model_folder) / f"validation{suffix}"
    output_folder.mkdir(exist_ok=True, parents=True)
    log_filename = output_folder / "validation_log.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_filename,
        filemode="w",
    )

    logger = logging.getLogger(__name__)

    main(
        args.model_folder,
        args.model_name,
        args.dataset_folder,
        output_folder,
        args.channels,
        args.transform_type,
        args.depth,
        args.post_process,
        args.slices,
    )

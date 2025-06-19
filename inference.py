import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from tqdm import tqdm

import transforms
from utils import get_test_images, prep_multichannel_images, save_as_nifti

IMAGE_KEY = "image"
PRED_KEY = "pred"
DEVICE = "cpu"


def make_sure_folder_exists(path: os.PathLike | str):
    Path(path).mkdir(parents=True, exist_ok=True)


def search_for_test_data_split(image_folder: os.PathLike, split_percentage: float):
    imgpaths = sorted(list(Path(image_folder).glob("*.nii.gz")))
    test_images = get_test_images(imgpaths, ratio=split_percentage)
    return test_images


def search_for_test_data(image_folder):
    imgpaths = sorted(list(Path(image_folder).glob("*.nii.gz")))
    return imgpaths


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
    """_summary_

    Args:
        model_path: Path to the model checkpoint.
        eval_images: _description_
        prediction_folder: _description_
        pre_transform: _description_
        post_transform: _description_
        depth: _description_
        post_process: _description_
        slices: _description_

    Returns:
        _description_
    """
    make_sure_folder_exists(prediction_folder)

    model = torch.load(model_path, map_location=torch.device(DEVICE))
    _ = model.eval()

    eval_files = [{IMAGE_KEY: img} for img in eval_images]
    eval_data_set = Dataset(data=eval_files, transform=pre_transform)
    eval_loader = DataLoader(eval_data_set, batch_size=1, shuffle=False)

    times = []
    for eval_data in tqdm(eval_loader):
        start_time = time.time()
        if slices:
            eval_data["pred"] = torch.stack(
                [
                    model(sample.permute(3, 0, 1, 2)).permute(1, 2, 3, 0)
                    for sample in eval_data[IMAGE_KEY]
                ],
                dim=0,
            )
        else:
            eval_data["pred"] = model(eval_data[IMAGE_KEY])
        times.append(time.time() - start_time)
        if post_transform:
            eval_data = [post_transform(i) for i in decollate_batch(eval_data)]
            save_as_nifti(
                eval_data[0]["pred"],
                eval_data[0]["image_meta_dict"],
                prediction_folder,
                depth,
                post_process,
            )
        else:
            eval_data = decollate_batch(eval_data)
            save_as_nifti(
                eval_data[0]["pred"],
                eval_data[0]["image_meta_dict"],
                prediction_folder,
                depth,
                post_process=False,
            )

    return times


def main(
    image_folder,
    output_folder,
    model_path,
    channels,
    depth,
    transform_type,
    post_process,
    slices,
):
    logger.info(f"Model: {model_path}")
    logger.info(f"Dataset: {image_folder}")
    logger.info(f"Post-processing: {post_process}")
    image_files = search_for_test_data(image_folder)

    pre_transform = transforms.get_test_transform(transform_type, depth)

    if post_process:
        post_transform = transforms.get_post_transform(pre_transform)
    else:
        post_transform = None

    if channels > 1:
        images = prep_multichannel_images(image_files, num_channels=channels)
    else:
        images = image_files

    times = run_inference(
        model_path,
        images,
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
        prog="Run Inference",
        description="This runs the inference on a dataset and saves it as .nii.gz",
    )
    parser.add_argument("-i", "--image_folder")
    parser.add_argument("-o", "--output_folder")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-c", "--channels", required=True, type=int)
    parser.add_argument("-d", "--depth", nargs="?", type=int)
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

    Path(args.output_folder).mkdir(exist_ok=True, parents=True)
    log_filename = Path(args.output_folder) / "inference_log.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_filename,
        filemode="w",
    )

    logger = logging.getLogger(__name__)

    main(
        args.image_folder,
        args.output_folder,
        args.model_path,
        args.channels,
        args.depth,
        args.transform_type,
        args.post_process,
        args.slices,
    )

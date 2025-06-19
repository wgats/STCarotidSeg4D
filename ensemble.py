import argparse
import logging
import os
from pathlib import Path


import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.transforms import SaveImaged

import utils
import transforms


def search_for_test_data(image_folder):
    imgpaths = sorted(list(Path(image_folder).glob("*.nii.gz")))
    return imgpaths


def custom_savepath_fn(meta_dict):
    orig = meta_dict.get("filename_or_obj", "unknown.nii.gz")
    return os.path.basename(str(orig)).replace("_0000", "")


def main(
    model_folder,
    model_name,
    images_folder,
    output_folder,
    num_channels,
    transform_type,
    depth,
    post_process,
    slices,
):
    # load data
    image_files = search_for_test_data(images_folder)
    if num_channels > 1:
        images = utils.prep_multichannel_images(image_files, num_channels=num_channels)
    else:
        images = image_files

    # get transform
    pre_transform = transforms.get_test_transform(transform_type, depth)
    if post_process:
        post_transform = transforms.get_ensemble_post_transform(
            pre_transform, output_folder
        )
    else:
        post_transform = transforms.get_ensemble_post_transform(
            pre_transform, output_folder, invert=False
        )
    if slices or transform_type == "pad-average":
        batch_size = 1
    else:
        batch_size = 8

    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    # load images
    eval_files = [{"image": img} for img in images]
    eval_data_set = CacheDataset(data=eval_files, transform=pre_transform)
    eval_loader = DataLoader(eval_data_set, batch_size=batch_size, shuffle=False)

    # prepare model
    models = [
        torch.load(
            Path(model_folder) / f"fold_{i}" / f"{model_name}.pth",
            map_location=torch.device(DEVICE),
        )
        for i in range(5)
    ]

    # saver
    saver = SaveImaged(
        keys="validation",
        output_dir=output_folder,
        output_postfix="",
        output_ext=".nii.gz",
        separate_folder=False,
        resample=False,
    )
    # start inference
    for eval_data in eval_loader:
        for fold, model in enumerate(models):
            model.eval()
            with torch.no_grad():
                if slices:
                    eval_data[f"pred_{fold}"] = torch.stack(
                        [
                            model(sample.permute(3, 0, 1, 2)).permute(1, 2, 3, 0)
                            for sample in eval_data["image"].to(DEVICE)
                        ],
                        dim=0,
                    )
                else:
                    eval_data[f"pred_{fold}"] = model(eval_data["image"].to(DEVICE))
        outputs = decollate_batch(eval_data)
        for item in outputs:
            item = post_transform(item)
            fname = item["image"].meta["filename_or_obj"]
            if fname.endswith("_0000.nii.gz"):
                item["validation"].meta["filename_or_obj"] = fname.replace(
                    "_0000.nii.gz", ".nii.gz"
                )
            saver(item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Run ensemble inference for 5-fold cross-validation using majority voting.",
    )
    parser.add_argument("-m", "--model_folder")
    parser.add_argument("-n", "--model_name")
    parser.add_argument("-i", "--images_folder")
    parser.add_argument("-o", "--output_folder")
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
    log_filename = Path(args.output_folder) / "validation_log.txt"

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
        args.images_folder,
        args.output_folder,
        args.channels,
        args.transform_type,
        args.depth,
        args.post_process,
        args.slices,
    )

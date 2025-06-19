import argparse
import logging
import math
import shutil
import time
from pathlib import Path
import datetime

import matplotlib.pyplot as plt
import monai.data
import torch
from monai.data import CacheDataset
from sklearn import model_selection
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

import models
import transforms
from constants import (
    BATCH_SIZE,
    DEVICE,
    IMAGE_KEY,
    INIT_LR,
    LABEL_KEY,
    NUM_EPOCHS,
    SEED,
    SPLIT_PCT,
    STOP_CRITERION,
)
from trainers import Trainer, Trainer2D
from utils import prep_multichannel_images, search_for_data, load_splits_from_json


def get_gpu_info(device):
    gpu_info = {
        "name": torch.cuda.get_device_name(device),
        "memory_total": torch.cuda.get_device_properties(device).total_memory
        / (1024**2),
    }

    return gpu_info


def plot_loss(losses, plot_path):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(losses["train_loss"], label="train_loss")
    plt.plot(losses["validation_loss"], label="validation_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(plot_path)
    plt.close("all")


def run_training(
    data_path,
    output_path,
    model_type,
    train_transform,
    validation_transform,
    channels,
    depth,
    fold,
    resume=False,
):
    # Prepare output paths
    if fold is not None:
        output_folder = Path(output_path) / f"fold_{fold}"
    else:
        output_folder = Path(output_path) / "output"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    plot_path = Path(output_folder) / "plot.png"
    model_path = Path(output_folder) / f"{model_type}.pth"
    checkpoint_path = Path(output_folder) / "checkpoint.pth"

    # export constants.txt
    shutil.copy("constants.py", Path(output_folder) / "constants.txt")

    # Set up logging

    # Generate the log filename with current date and time
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_filename = Path(output_folder) / f"train_log_{timestamp}.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_filename,
        filemode="w",
    )

    logger = logging.getLogger(__name__)

    # print training info
    gpu_info = get_gpu_info(DEVICE)
    logger.info("===========GPU INFO=============")
    logger.info(f"GPU Name: {gpu_info['name']}")
    logger.info(f"Memory Total: {gpu_info['memory_total']}")
    logger.info(f"Model Type: {model_type}")
    logger.info("=" * 32)
    logger.info("")

    # Prepare data

    if fold is not None:
        splits = load_splits_from_json(Path(data_path), channels)
        X_train, y_train, X_test, y_test = splits[fold]
    else:
        imgdir = Path(data_path) / "images"
        maskdir = Path(data_path) / "labels"

        imgpaths, labelpaths = search_for_data(imgdir, maskdir)
        if channels == 3:
            train_images = prep_multichannel_images(imgpaths)
        else:
            train_images = imgpaths
        train_masks = labelpaths

        assert len(train_images) == len(train_masks)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            train_images, train_masks, test_size=(1.0 - SPLIT_PCT), random_state=SEED
        )

    train_files = [
        {IMAGE_KEY: img, LABEL_KEY: mask} for img, mask in zip(X_train, y_train)
    ]
    val_files = [{IMAGE_KEY: img, LABEL_KEY: mask} for img, mask in zip(X_test, y_test)]

    train_data_set = CacheDataset(
        data=train_files, transform=train_transform, cache_rate=1.0, num_workers=8
    )
    validation_data_set = CacheDataset(
        data=val_files, transform=validation_transform, cache_rate=1.0, num_workers=8
    )

    batch_size = BATCH_SIZE
    if model_type == "unet2d" or model_type.startswith("flow_transformer"):
        batch_size = 1

    logger.info("===========DATASET=============")
    logger.info(f"Name: {Path(data_path).stem}")
    logger.info(f"Channels: {channels}")
    logger.info(f"Batch size: {batch_size}")

    if fold is not None:
        logger.info(f"Fold: {fold}")
    logger.info("=" * 32)
    logger.info("")

    train_loader = monai.data.DataLoader(
        train_data_set, shuffle=True, batch_size=batch_size
    )
    validation_loader = monai.data.DataLoader(
        validation_data_set, shuffle=False, batch_size=batch_size
    )

    # Prepare loss, optimizer, trainer
    lossFunc = BCEWithLogitsLoss()

    if resume:
        checkpoint = torch.load(checkpoint_path)
        model = checkpoint["model"]
        opt = checkpoint["optimizer"]
        start_epoch = checkpoint["epoch"]
        losses = {
            "train_loss": checkpoint["train_loss"],
            "validation_loss": checkpoint["validation_loss"],
        }
        best_validation_loss = checkpoint["best_validation_loss"]
    else:
        config = models.ModelFactory.get_config(model_type, channels, depth)
        model = models.ModelFactory.get_model(config)
        opt = Adam(model.parameters(), lr=INIT_LR)
        start_epoch = 0
        losses = {"train_loss": [], "validation_loss": []}
        best_validation_loss = math.inf

    if model_type == "unet2d":
        trainer = Trainer2D(model, opt, lossFunc, train_loader, validation_loader)
    else:
        trainer = Trainer(model, opt, lossFunc, train_loader, validation_loader)

    runs_without_improvement = 0

    model.to(DEVICE)
    # Start training
    for e in range(start_epoch, NUM_EPOCHS):
        logger.info(f"EPOCH: {(e + 1)}/{ NUM_EPOCHS}")
        start_time = time.time()

        train_losses = trainer.train_step()
        validation_losses = trainer.evaluation_step()

        end_time = time.time()
        elapsed_time = end_time - start_time

        avg_train_loss = torch.Tensor(train_losses).mean()
        avg_validation_loss = torch.Tensor(validation_losses).mean()
        losses["train_loss"].append(avg_train_loss.cpu().detach().numpy().item())
        losses["validation_loss"].append(
            avg_validation_loss.cpu().detach().numpy().item()
        )

        logger.info(f"Train loss: {avg_train_loss:4f}")
        logger.info(f"Validation loss: {avg_validation_loss:4f}")
        logger.info(f"Epoch time: {elapsed_time:.2f} seconds")

        if best_validation_loss > avg_validation_loss:
            best_validation_loss = avg_validation_loss
            logger.info(f"New best validation loss: {avg_validation_loss:4f}")
            torch.save(model, model_path)
            logger.info("Saved new checkpoint")
            runs_without_improvement = 0
        else:
            runs_without_improvement += 1
        if runs_without_improvement > STOP_CRITERION:
            break

        if (e + 1) % 10 == 0:
            plot_loss(losses, plot_path)
        if (e + 1) % 100 == 0:
            checkpoint = {
                "epoch": e,
                "model": trainer.model,
                "optimizer": trainer.optimizer,
                "train_loss": losses["train_loss"],
                "validation_loss": losses["validation_loss"],
                "best_validation_loss": best_validation_loss,
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info("Saved new milestone checkpoint")

        logger.info("")
    # Plot losses
    plot_loss(losses, plot_path)
    logger.info(f"Final best validation loss: {best_validation_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train vessel segmentation with 2D time-resolved image sequences from 4D Flow MRI.",
        description="This trains the 2D vessel segmenation for time-resolved sequences of images.",
    )
    parser.add_argument("-dp", "--data_path")
    parser.add_argument("-o", "--output_path")
    parser.add_argument(
        "-m",
        "--model_type",
        choices=[
            "unet2d",
            "unet3d",
            "unetr",
            "spatio_temporal_transformer",
        ],
        required=True,
        help="Specify the model type. Must be one of unet2d, unet3d, unetr, spatio_temporal_transformer",
    )
    parser.add_argument(
        "-t",
        "--transform_type",
        choices=["interpolate", "pad"],
        required=True,
        help="Specify the pre-processing type. Must be one of interpolate, pad",
    )
    parser.add_argument(
        "-c",
        "--channels",
        choices=[1, 3],
        type=int,
        required=True,
        help="Specify the number of input channels (2 or 3)",
    )

    parser.add_argument(
        "-d",
        "--depth",
        type=int,
        choices=[16, 32],
        help="Specify the depth of the pre-processing transformation",
    )

    parser.add_argument(
        "-f",
        "--fold",
        type=int,
        help="Use fold from splits_final.json for cross validation.",
    )

    parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="Resume from checkpoint.",
    )

    args = parser.parse_args()

    if args.model_type == "unet2d":
        # select the unet2d transform
        train_transform = transforms.get_training_transform("none")
        validation_transform = transforms.get_validation_transform("none")
    else:
        # else pick based on transform type and depth given
        train_transform = transforms.get_training_transform(
            args.transform_type, int(args.depth)
        )
        validation_transform = transforms.get_validation_transform(
            args.transform_type, int(args.depth)
        )

    run_training(
        args.data_path,
        args.output_path,
        args.model_type,
        train_transform,
        validation_transform,
        args.channels,
        args.depth,
        args.fold,
        args.resume,
    )

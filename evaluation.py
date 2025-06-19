# evaluation.py
import argparse
from glob import glob
from os import makedirs, path
from pathlib import Path
import tqdm

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pingouin as pg

import transforms
from segmentation_results import SegmentationResults
from segmentation_evaluator import SegmentationEvaluator2D
from constants import IMAGE_KEY


def get_pixel_size(data):
    return data.header["pixdim"][1:3]


class VesselSegmentationEvaluationPlotter:
    def __init__(self, output_folder):
        if not path.exists(output_folder):
            makedirs(output_folder)
        self._output_folder = output_folder

    def _plot(self, filename, y_axis, metric_results):
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111)
        ax.set_xticklabels(["lumen"])
        ax.set_ylabel(y_axis)
        plt.subplots_adjust(left=0.215)
        _ = plt.boxplot(np.array(metric_results)[:, 1:], widths=0.5, whis=[1, 99])
        # _ = plt.boxplot(np.array(metric_results), widths=0.5, whis=[1, 99])
        plt.savefig(path.join(self._output_folder, f"{filename}.png"))

    def create_plots(self, data_frame: pd.DataFrame):
        self._plot(
            "Dice Coefficients (time steps)",
            "DC (per time step)",
            get_dice_per_time_step(data_frame),
        )
        self._plot(
            "Hausdorff Distance (time steps)",
            "HD in mm (per time step)",
            get_hd_per_time_step(data_frame),
        )
        self._plot(
            "Average Contour (time steps)",
            "ACD in mm (per time step)",
            get_acd_per_time_step(data_frame),
        )
        self._plot(
            "Dice Coefficients (samples)",
            "DC (per-sample mean)",
            get_mean_dice_per_sample(data_frame),
        )
        self._plot(
            "Average Contour (samples)",
            "ACD in mm (per-sample mean)",
            get_mean_acd_per_sample(data_frame),
        )

    @staticmethod
    def show():
        plt.show()


def evaluate_segmentations(
    prediction_folder,
    gt_folder,
    images_folder,
    velocity_images_folder=None,
    raw_transform=None,
):
    evaluator = SegmentationEvaluator2D(classes=[0, 1])
    segmentation_results = SegmentationResults()

    prediction_paths = glob(path.join(prediction_folder, "*.nii.gz"))

    for prediction_path in tqdm.tqdm(prediction_paths):
        ground_truth_path = path.join(gt_folder, path.basename(prediction_path))

        gt_nib = nib.load(ground_truth_path)
        gt_data = gt_nib.get_fdata().squeeze()
        gt_data = (gt_data > 0.5).astype(np.int16)
        prediction_data = nib.load(prediction_path).get_fdata().squeeze()
        case = Path(prediction_path).parts[-1]

        if velocity_images_folder:
            stem = Path(prediction_path).stem.split(".")[0]
            velocity_image_path = Path(velocity_images_folder) / f"{stem}_0001.nii.gz"
            velocity_image = nib.load(velocity_image_path).get_fdata().squeeze()

            if raw_transform:
                velocity_image = (
                    raw_transform({IMAGE_KEY: velocity_image_path})[IMAGE_KEY]
                    .numpy()
                    .squeeze()
                )
                gt_data = (
                    raw_transform({IMAGE_KEY: ground_truth_path})[IMAGE_KEY]
                    .numpy()
                    .squeeze()
                    .round()  # convert to binary
                )
                prediction_data = (
                    label_transform({IMAGE_KEY: prediction_path})[IMAGE_KEY]
                    .numpy()
                    .squeeze()
                    .round()  # convert to binary
                )

            time_steps = gt_data.shape[-1]

            for time_step in range(time_steps):
                if prediction_data.shape != gt_data.shape:
                    print(f"pred: {prediction_data.shape} != gt: {gt_data.shape}")
                segmentation_result = evaluator.evaluate_with_velocity(
                    velocity_image[:, :, time_step],
                    gt_data[:, :, time_step],
                    prediction_data[:, :, time_step],
                    get_pixel_size(gt_nib),
                    case,
                    time_step,
                    raw_transform,
                )
                segmentation_results.add(segmentation_result)
        else:
            for time_step in range(time_steps):
                segmentation_result = evaluator.evaluate(
                    gt_data[:, :, time_step],
                    prediction_data[:, :, time_step],
                    get_pixel_size(gt_nib),
                    case,
                    time_step,
                )
                segmentation_results.add(segmentation_result)

    return segmentation_results


def get_data_frame(segmentation_results: SegmentationResults) -> pd.DataFrame:
    metrics_data = {
        "sample": [
            result.sample
            for result in segmentation_results._results
            if result.all_valid
        ],
        "time_step": [
            result.time_step
            for result in segmentation_results._results
            if result.all_valid
        ],
        "dice_background": [
            result.dice_coefficients[0]
            for result in segmentation_results._results
            if result.all_valid
        ],
        "dice_lumen": [
            result.dice_coefficients[1]
            for result in segmentation_results._results
            if result.all_valid
        ],
        "hausdorff_distance_background": [
            result.hausdorff_distances[0]
            for result in segmentation_results._results
            if result.all_valid
        ],
        "hausdorff_distance_lumen": [
            result.hausdorff_distances[1]
            for result in segmentation_results._results
            if result.all_valid
        ],
        "average_contour_distance_background": [
            result.average_contour_distances[0]
            for result in segmentation_results._results
            if result.all_valid
        ],
        "average_contour_distance_lumen": [
            result.average_contour_distances[1]
            for result in segmentation_results._results
            if result.all_valid
        ],
        "gt_lumen_diameter": [
            result.lumen_diameters[0]
            for result in segmentation_results._results
            if result.all_valid
        ],
        "pred_lumen_diameter": [
            result.lumen_diameters[1]
            for result in segmentation_results._results
            if result.all_valid
        ],
        "gt_flow_rate": [
            result.flow_rates[0]
            for result in segmentation_results._results
            if result.all_valid
        ],
        "pred_flow_rate": [
            result.flow_rates[1]
            for result in segmentation_results._results
            if result.all_valid
        ],
        "gt_max_velocity": [
            result.max_velocities[0]
            for result in segmentation_results._results
            if result.all_valid
        ],
        "pred_max_velocity": [
            result.max_velocities[1]
            for result in segmentation_results._results
            if result.all_valid
        ],
    }
    df = pd.DataFrame(metrics_data)
    return df


def get_hd_per_time_step(df: pd.DataFrame, class_value=None) -> np.ndarray:
    match class_value:
        case 0:
            return df["hausdorff_distance_background"].values
        case 1:
            return df["hausdorff_distance_lumen"].values
        case _:
            return df[
                ["hausdorff_distance_background", "hausdorff_distance_lumen"]
            ].values


def get_acd_per_time_step(df: pd.DataFrame, class_value=None) -> np.ndarray:
    match class_value:
        case 0:
            return df["average_contour_distance_background"].values
        case 1:
            return df["average_contour_distance_lumen"].values
        case _:
            return df[
                [
                    "average_contour_distance_background",
                    "average_contour_distance_lumen",
                ]
            ].values


def get_dice_per_time_step(df: pd.DataFrame, class_value=None) -> np.ndarray:
    match class_value:
        case 0:
            return df["dice_background"].values
        case 1:
            return df["dice_lumen"].values
        case _:
            return df[["dice_background", "dice_lumen"]].values


def get_mean_hd_per_sample(df: pd.DataFrame, class_value=None) -> np.ndarray:
    match class_value:
        case 0:
            return df.groupby(["sample"])["hausdorff_distance_background"].mean().values
        case 1:
            return df.groupby(["sample"])["hausdorff_distance_lumen"].mean().values
        case _:
            return (
                df.groupby(["sample"])[
                    ["hausdorff_distance_background", "hausdorff_distance_lumen"]
                ]
                .mean()
                .values
            )


def get_max_hd_per_sample(df: pd.DataFrame, class_value=None) -> np.ndarray:
    match class_value:
        case 0:
            return df.groupby(["sample"])["hausdorff_distance_background"].max().values
        case 1:
            return df.groupby(["sample"])["hausdorff_distance_lumen"].max().values
        case _:
            return (
                df.groupby(["sample"])[
                    ["hausdorff_distance_background", "hausdorff_distance_lumen"]
                ]
                .max()
                .values
            )


def get_mean_acd_per_sample(df: pd.DataFrame, class_value=None) -> np.ndarray:
    match class_value:
        case 0:
            return (
                df.groupby(["sample"])["average_contour_distance_background"]
                .mean()
                .values
            )
        case 1:
            return (
                df.groupby(["sample"])["average_contour_distance_lumen"].mean().values
            )
        case _:
            return (
                df.groupby(["sample"])[
                    [
                        "average_contour_distance_background",
                        "average_contour_distance_lumen",
                    ]
                ]
                .mean()
                .values
            )


def get_mean_dice_per_sample(df: pd.DataFrame, class_value=None) -> np.ndarray:
    match class_value:
        case 0:
            return df.groupby(["sample"])["dice_background"].mean().values
        case 1:
            return df.groupby(["sample"])["dice_lumen"].mean().values
        case _:
            return (
                df.groupby(["sample"])[["dice_background", "dice_lumen"]].mean().values
            )


def get_evaluation_results(df: pd.DataFrame) -> pd.DataFrame:
    metrics_data = {
        "dice_background": df["dice_background"].mean(),
        "dice_lumen": df["dice_lumen"].mean(),
        "hausdorff_distance_lumen": df["hausdorff_distance_lumen"].mean(),
        "average_contour_distance_background": df[
            "average_contour_distance_background"
        ].mean(),
        "average_contour_distance_lumen": df["average_contour_distance_lumen"].mean(),
        "median_dice_background": df["dice_background"].median(),
        "median_dice_lumen": df["dice_lumen"].median(),
        "median_hausdorff_distance_background": df[
            "hausdorff_distance_background"
        ].median(),
        "median_hausdorff_distance_lumen": df["hausdorff_distance_lumen"].median(),
        "median_average_countour_distance_background": df[
            "average_contour_distance_background"
        ].median(),
        "median_average_contour_distance_lumen": df[
            "average_contour_distance_lumen"
        ].median(),
        "std_dice_background": df["dice_background"].std(),
        "std_dice_lumen": df["dice_lumen"].std(),
        "std_hausdorff_distance_lumen": df["hausdorff_distance_lumen"].std(),
        "std_average_contour_distance_background": df[
            "average_contour_distance_background"
        ].std(),
        "std_average_contour_distance_lumen": df[
            "average_contour_distance_lumen"
        ].std(),
    }

    return pd.DataFrame(metrics_data, index=[0])


def calculate_icc_lumen(df: pd.DataFrame):
    df_lumen = df.filter(
        items=["gt_lumen_diameter", "pred_lumen_diameter"]
    ).reset_index(names="sample")
    df_long = pd.melt(
        df_lumen, id_vars=["sample"], var_name="rater", value_name="diameter"
    )

    # Calculate ICC
    icc_lumen = pg.intraclass_corr(
        data=df_long, targets="sample", raters="rater", ratings="diameter"
    )

    return icc_lumen


def calculate_icc_flow_rate(df: pd.DataFrame):
    df_flow_rate = df.filter(items=["gt_flow_rate", "pred_flow_rate"]).reset_index(
        names="sample"
    )
    df_long = pd.melt(
        df_flow_rate, id_vars=["sample"], var_name="rater", value_name="flow_rate"
    )

    # Calculate ICC
    icc_flow_rate = pg.intraclass_corr(
        data=df_long,
        targets="sample",
        raters="rater",
        ratings="flow_rate",
        nan_policy="omit",
    )

    return icc_flow_rate


def calculate_icc_max_velocity(df: pd.DataFrame):
    df_max_velocity = df.filter(
        items=["gt_max_velocity", "pred_max_velocity"]
    ).reset_index(names="sample")
    df_long = pd.melt(
        df_max_velocity, id_vars=["sample"], var_name="rater", value_name="max_velocity"
    )

    # Calculate ICC
    icc_max_velocity = pg.intraclass_corr(
        data=df_long,
        targets="sample",
        raters="rater",
        ratings="max_velocity",
        nan_policy="omit",
    )

    return icc_max_velocity


def save_results_to_xlsx(segmentation_results, path_to_save) -> None:
    save_path = Path(path_to_save)
    df = get_data_frame(segmentation_results)
    df.to_excel(save_path / "data.xlsx")

    icc_lumen = calculate_icc_lumen(df)
    icc_lumen.to_excel(save_path / "icc_lumen.xlsx")

    icc_flow_rate = calculate_icc_flow_rate(df)
    icc_flow_rate.to_excel(save_path / "icc_flow_rate.xlsx")

    icc_max_velocity = calculate_icc_max_velocity(df)
    icc_max_velocity.to_excel(save_path / "icc_max_velocity.xlsx")

    results_df = get_evaluation_results(df)
    results_df.to_excel(save_path / "results.xlsx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Evaluate 2D vessel segmentation",
        description="This plots evaluation metrics of the test data and save them to Excel.",
    )
    parser.add_argument("-i", "--images_path")
    parser.add_argument("-gt", "--ground_truth_path")
    parser.add_argument("-p", "--prediction_path")
    parser.add_argument("-s", "--save_path")
    parser.add_argument("-v", "--velocity_images_folder")
    parser.add_argument("-r", "--raw_predictions", action="store_true")
    parser.add_argument(
        "-t",
        "--transform_type",
        choices=["none", "interpolate", "pad", "pad-crop", "pad-average"],
    )
    parser.add_argument("-d", "--depth", type=int)
    parser.add_argument("-e", "--ensemble", action="store_true")
    args = parser.parse_args()

    if args.raw_predictions:
        raw_transform = transforms.get_raw_transform(args.transform_type, args.depth)
    else:
        raw_transform = None

    if args.ensemble:
        label_transform = transforms.get_raw_transform(args.transform_type, args.depth, remove_dims=False)
    else:
        label_transform = None
    segmentation_results = evaluate_segmentations(
        args.prediction_path,
        args.ground_truth_path,
        args.images_path,
        args.velocity_images_folder,
        raw_transform,
    )

    df = get_data_frame(segmentation_results)
    evaluation_plotter = VesselSegmentationEvaluationPlotter(args.save_path)
    evaluation_plotter.create_plots(df)

    save_results_to_xlsx(segmentation_results, args.save_path)

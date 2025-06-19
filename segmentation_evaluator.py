# segmentation_evaluator.py
import itertools
import math

import numpy as np
from evalutils.stats import (
    dice_from_confusion_matrix,
    hausdorff_distance,
    mean_contour_distance,
)
import cv2
from metrics import Metrics


def get_lumen_diameter(img, pixel_dim: float):
    # find largest contour
    contours, hierarchy = cv2.findContours(
        img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    try:
        contour = max(contours, key=len)
    except ValueError:
        return np.nan

    # find minimum enclosing circle
    (x_axis, y_axis), radius = cv2.minEnclosingCircle(contour)

    # pixel_dim is a 2 element array!
    diameter = 2.0 * radius * max(pixel_dim)

    return diameter


def calculate_flow_rate(velocity_image, label, pixel_dim: np.ndarray):
    velocity_in_roi = velocity_image * label
    pixel_area = (pixel_dim * 1e-3).prod()
    # multiply by 6e7 to convert m3/s -> mL/ min
    flow = (pixel_area * velocity_in_roi).sum() * 6e7

    return flow


def calculate_max_velocity(velocity_image: np.ndarray, label: np.ndarray) -> np.float_:
    velocity_in_roi = velocity_image * label
    return np.max(np.abs(velocity_in_roi))


class SegmentationEvaluator2D:
    def __init__(self, classes):
        self._classes = classes

    def validate_inputs(self, truth: np.ndarray, prediction: np.ndarray) -> None:
        if not truth.shape == prediction.shape:
            raise ValueError("Ground truth and prediction do not have the same size")
        if not set(truth.flatten()).issubset(self._classes):
            raise ValueError(f"Truth contains invalid classes: {set(truth.flatten())}")
        if not set(prediction.flatten()).issubset(self._classes):
            raise ValueError("Prediction contains invalid classes")

    def get_confusion_matrix(
        self, truth: np.ndarray, prediction: np.ndarray
    ) -> np.ndarray:
        confusion_matrix = np.zeros((len(self._classes), len(self._classes)))
        for class_predicted, class_truth in itertools.product(
            self._classes, self._classes
        ):
            confusion_matrix[class_truth, class_predicted] = np.sum(
                np.all(
                    np.stack((prediction == class_predicted, truth == class_truth)),
                    axis=0,
                )
            )
        return confusion_matrix

    def evaluate_with_velocity(
        self,
        velocity_image: np.ndarray,
        truth: np.ndarray,
        prediction: np.ndarray,
        pixel_dim: np.ndarray,
        sample,
        time_step,
    ):
        self.validate_inputs(truth, prediction)

        dice_coefficients = self.evaluate_dice_coefficients(truth, prediction)
        hausdorff_distances = self.evaluate_hausdorff_distances(
            truth, prediction, pixel_dim
        )
        mean_contour_distances = self.evaluate_mean_contour_distances(
            truth, prediction, pixel_dim
        )
        lumen_diameters = self.evaluate_lumen_diameters(truth, prediction, pixel_dim)
        flow_rates = self.evaluate_flow_rates(
            velocity_image, truth, prediction, pixel_dim
        )
        max_velocities = self.evaluate_max_velocities(velocity_image, truth, prediction)

        # check if all values are invalid
        all_valid = all(
            [
                math.inf not in dice_coefficients.values(),
                math.inf not in mean_contour_distances.values(),
                math.inf not in hausdorff_distances.values(),
                math.inf not in lumen_diameters.values(),
                math.inf not in flow_rates.values(),
                math.inf not in max_velocities.values(),
            ]
        )

        return Metrics(
            dice_coefficients=dice_coefficients,
            hausdorff_distances=hausdorff_distances,
            average_contour_distances=mean_contour_distances,
            lumen_diameters=lumen_diameters,
            flow_rates=flow_rates,
            max_velocities=max_velocities,
            sample=sample,
            time_step=time_step,
            all_valid=all_valid,
        )

    def evaluate(
        self,
        truth: np.ndarray,
        prediction: np.ndarray,
        pixel_dim: np.ndarray,
        sample,
        time_step,
    ):
        self.validate_inputs(truth, prediction)

        dice_coefficients = self.evaluate_dice_coefficients(truth, prediction)
        hausdorff_distances = self.evaluate_hausdorff_distances(
            truth, prediction, pixel_dim
        )
        mean_contour_distances = self.evaluate_mean_contour_distances(
            truth, prediction, pixel_dim
        )
        lumen_diameters = self.evaluate_lumen_diameters(truth, prediction, pixel_dim)
        flow_rates = self.evaluate_flow_rates(None, truth, prediction, pixel_dim)
        max_velocities = self.evaluate_max_velocities(None, truth, prediction)

        # check if all values are invalid
        all_valid = all(
            [
                math.inf not in dice_coefficients.values(),
                math.inf not in mean_contour_distances.values(),
                math.inf not in hausdorff_distances.values(),
                math.inf not in lumen_diameters.values(),
            ]
        )

        return Metrics(
            dice_coefficients=dice_coefficients,
            hausdorff_distances=hausdorff_distances,
            average_contour_distances=mean_contour_distances,
            lumen_diameters=lumen_diameters,
            flow_rates=flow_rates,
            max_velocities=max_velocities,
            sample=sample,
            time_step=time_step,
            all_valid=all_valid,
        )

    def evaluate_dice_coefficients(self, truth, prediction):
        sorted_dice_coefficients = {}
        confusion_matrix = self.get_confusion_matrix(truth, prediction)
        dice_coefficients = dice_from_confusion_matrix(confusion_matrix)
        for i, class_value in enumerate(self._classes):
            sorted_dice_coefficients[class_value] = dice_coefficients[i]
        return sorted_dice_coefficients

    def evaluate_hausdorff_distances(
        self, truth: np.ndarray, prediction: np.ndarray, pixel_dim
    ):
        hausdorff_distances = {}
        for class_value in self._classes:
            try:
                hausdorff_distances[class_value] = hausdorff_distance(
                    truth == class_value, prediction == class_value, pixel_dim
                )
            except ValueError:
                hausdorff_distances[class_value] = math.inf
        return hausdorff_distances

    def evaluate_mean_contour_distances(
        self, truth: np.ndarray, prediction: np.ndarray, pixel_dim
    ):
        mean_contour_distances = {}
        for class_value in self._classes:
            try:
                mean_contour_distances[class_value] = mean_contour_distance(
                    truth == class_value, prediction == class_value, pixel_dim
                )
            except ValueError:
                mean_contour_distances[class_value] = math.inf
        return mean_contour_distances

    def evaluate_lumen_diameters(
        self, truth: np.ndarray, prediction: np.ndarray, pixel_dim
    ):
        # 0: ground truth diameter
        # 1: prediction diameter
        lumen_diameters = {}

        try:
            lumen_diameters[0] = get_lumen_diameter(truth, pixel_dim)
        except ValueError:
            lumen_diameters[0] = math.inf

        try:
            lumen_diameters[1] = get_lumen_diameter(prediction, pixel_dim)
        except ValueError:
            lumen_diameters[1] = math.inf

        return lumen_diameters

    def evaluate_flow_rates(
        self,
        velocity_image: np.ndarray,
        truth: np.ndarray,
        prediction: np.ndarray,
        pixel_dim: np.ndarray,
    ):
        # 0: ground truth flow rate
        # 1: predicted flow rate
        flow_rates = {}
        if velocity_image is None:
            flow_rates[0] = math.inf
            flow_rates[1] = math.inf
            return flow_rates
        try:
            flow_rates[0] = calculate_flow_rate(velocity_image, truth, pixel_dim)
        except ValueError:
            flow_rates[0] = math.inf
        try:
            flow_rates[1] = calculate_flow_rate(velocity_image, prediction, pixel_dim)
        except ValueError:
            flow_rates[1] = math.inf

        return flow_rates

    def evaluate_max_velocities(
        self,
        velocity_image: np.ndarray,
        truth: np.ndarray,
        prediction: np.ndarray,
    ):
        # 0: ground truth max velocity
        # 1: predicted max velocity
        max_velocities = {}
        if velocity_image is None:
            max_velocities[0] = math.inf
            max_velocities[1] = math.inf
            return max_velocities
        try:
            max_velocities[0] = calculate_max_velocity(velocity_image, truth)
        except ValueError:
            max_velocities[0] = math.inf
        try:
            max_velocities[1] = calculate_max_velocity(velocity_image, prediction)
        except ValueError:
            max_velocities[1] = math.inf

        return max_velocities

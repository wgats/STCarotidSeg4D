# segmentation_results.py
import numpy as np


class SegmentationResults:
    def __init__(self):
        self._results = []

    def add(self, result):
        self._results.append(result)

    @property
    def mean_dice_coefficients(self):
        return [
            np.mean(list(result.dice_coefficients.values()))
            for result in self._results
            if result.all_valid
        ]

    @property
    def mean_hausdorff_distance(self):
        return [
            np.mean(list(result.hausdorff_distances.values()))
            for result in self._results
            if result.all_valid
        ]

    @property
    def mean_average_contour_distances(self):
        return [
            np.mean(list(result.average_contour_distances.values()))
            for result in self._results
            if result.all_valid
        ]

    def dice_coefficients(self, class_value=None):
        if class_value:
            return [
                result.dice_coefficients[class_value]
                for result in self._results
                if result.all_valid
            ]
        return [
            list(result.dice_coefficients.values())
            for result in self._results
            if result.all_valid
        ]

    def hausdorff_distances(self, class_value=None):
        if class_value:
            return [
                result.hausdorff_distances[class_value]
                for result in self._results
                if result.all_valid
            ]
        return [
            list(result.hausdorff_distances.values())
            for result in self._results
            if result.all_valid
        ]

    def average_contour_distances(self, class_value=None):
        if class_value:
            return [
                result.average_contour_distances[class_value]
                for result in self._results
                if result.all_valid
            ]
        return np.nan_to_num(
            np.array(
                [
                    list(result.average_contour_distances.values())
                    for result in self._results
                    if result.all_valid
                ]
            ),
            nan=np.inf,
        )

    def lumen_diameters(self, class_value=None):
        # 0: ground truth diameter
        # 1: predicted diameter
        if class_value:
            return [
                result.lumen_diameters[class_value]
                for result in self._results
                if result.all_valid
            ]
        return np.nan_to_num(
            np.array(
                [
                    list(result.lumen_diameters.values())
                    for result in self._results
                    if result.all_valid
                ]
            ),
            nan=np.inf,
        )

    def flow_rates(self, class_value=None):
        # 0: ground truth diameter
        # 1: predicted diameter
        if class_value:
            return [
                result.flow_rates[class_value]
                for result in self._results
                if result.all_valid
            ]
        return np.nan_to_num(
            np.array(
                [
                    list(result.flow_rates.values())
                    for result in self._results
                    if result.all_valid
                ]
            ),
            nan=np.inf,
        )

    def max_velocities(self, class_value=None):
        # 0: ground truth max velocity
        # 1: predicted max velocity
        if class_value:
            return [
                result.max_velocities[class_value]
                for result in self._results
                if result.all_valid
            ]
        return np.nan_to_num(
            np.array(
                [
                    list(result.max_velocities.values())
                    for result in self._results
                    if result.all_valid
                ]
            ),
            nan=np.inf,
        )

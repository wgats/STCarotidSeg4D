import dataclasses


@dataclasses.dataclass
class Metrics:
    dice_coefficients: dict
    hausdorff_distances: dict
    average_contour_distances: dict
    lumen_diameters: dict
    flow_rates: dict
    max_velocities: dict
    sample: str
    time_step: int
    all_valid: bool

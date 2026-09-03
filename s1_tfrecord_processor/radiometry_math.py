from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


UnknownNoisePolicy = Literal[
    "error",
    "assume_uncorrected",
    "assume_corrected",
]

NonpositivePowerPolicy = Literal[
    "nan",
    "floor",
]


@dataclass(frozen=True)
class RadiometryResult:
    source_power: np.ndarray
    noise_power: np.ndarray
    noise_lut_valid_mask: np.ndarray
    denoised_power: np.ndarray
    calibrated_linear: np.ndarray
    calibrated_db: np.ndarray
    valid_mask: np.ndarray
    noise_was_subtracted: bool
    unknown_noise_policy: UnknownNoisePolicy
    nonpositive_power_policy: NonpositivePowerPolicy


def validate_same_shape(
    *arrays: np.ndarray,
) -> None:
    if not arrays:
        raise ValueError(
            "At least one array is required."
        )

    shapes = {
        array.shape
        for array in arrays
    }

    if len(shapes) != 1:
        raise ValueError(
            "Input arrays must have the same shape: "
            f"{sorted(shapes)}"
        )


def combine_noise_luts(
    *,
    legacy_noise_lut: np.ndarray | None = None,
    range_noise_lut: np.ndarray | None = None,
    azimuth_noise_lut: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build the source-domain thermal-noise power field.

    Legacy products provide one already-combined noise LUT.

    Newer IW/EW GRD products provide separate range and azimuth
    components. Their complete two-dimensional noise field is the
    element-wise product of the interpolated range and azimuth fields.
    """
    has_legacy = legacy_noise_lut is not None
    has_range = range_noise_lut is not None
    has_azimuth = azimuth_noise_lut is not None

    if has_legacy:
        if has_range or has_azimuth:
            raise ValueError(
                "Do not mix the legacy noise LUT with "
                "range/azimuth noise LUTs."
            )

        result = np.asarray(
            legacy_noise_lut,
            dtype=np.float64,
        )

    else:
        if not has_range or not has_azimuth:
            raise ValueError(
                "Modern noise requires both range and "
                "azimuth LUTs."
            )

        range_values = np.asarray(
            range_noise_lut,
            dtype=np.float64,
        )
        azimuth_values = np.asarray(
            azimuth_noise_lut,
            dtype=np.float64,
        )

        validate_same_shape(
            range_values,
            azimuth_values,
        )

        result = (
            range_values
            * azimuth_values
        )

    if not np.isfinite(result).all():
        raise ValueError(
            "Noise LUT contains non-finite values."
        )

    if np.any(result < 0.0):
        raise ValueError(
            "Noise power must not be negative."
        )

    return result.astype(
        np.float32,
        copy=False,
    )


def detected_magnitude_to_power(
    magnitude: np.ndarray,
) -> np.ndarray:
    """
    Convert detected GRD magnitude values to linear source power.
    """
    values = np.asarray(
        magnitude,
        dtype=np.float64,
    )

    result = values * values

    return result.astype(
        np.float32,
        copy=False,
    )


def resolve_noise_subtraction(
    *,
    thermal_noise_correction_performed: bool | None,
    unknown_noise_policy: UnknownNoisePolicy,
) -> bool:
    if thermal_noise_correction_performed is False:
        return True

    if thermal_noise_correction_performed is True:
        return False

    if unknown_noise_policy == "assume_uncorrected":
        return True

    if unknown_noise_policy == "assume_corrected":
        return False

    if unknown_noise_policy != "error":
        raise ValueError(
            "Unsupported unknown-noise policy: "
            f"{unknown_noise_policy!r}"
        )

    raise ValueError(
        "The product does not state whether thermal-noise "
        "correction was already performed. Refusing to "
        "subtract or retain noise without an explicit policy."
    )


def remove_thermal_noise(
    *,
    source_power: np.ndarray,
    noise_power: np.ndarray,
    valid_mask: np.ndarray,
    thermal_noise_correction_performed: bool | None,
    unknown_noise_policy: UnknownNoisePolicy = "error",
    nonpositive_power_policy: NonpositivePowerPolicy = "nan",
    positive_power_floor: float = 1.0e-5,
) -> tuple[np.ndarray, bool, np.ndarray]:
    source = np.asarray(
        source_power,
        dtype=np.float64,
    )
    noise = np.asarray(
        noise_power,
        dtype=np.float64,
    )
    valid = np.asarray(
        valid_mask,
        dtype=bool,
    )

    validate_same_shape(
        source,
        noise,
        valid,
    )

    if nonpositive_power_policy not in {
        "nan",
        "floor",
    }:
        raise ValueError(
            "Unsupported non-positive power policy: "
            f"{nonpositive_power_policy!r}"
        )

    if (
        nonpositive_power_policy == "floor"
        and positive_power_floor <= 0.0
    ):
        raise ValueError(
            "positive_power_floor must be positive."
        )

    subtract_noise = resolve_noise_subtraction(
        thermal_noise_correction_performed=(
            thermal_noise_correction_performed
        ),
        unknown_noise_policy=unknown_noise_policy,
    )

    result = np.full(
        source.shape,
        np.nan,
        dtype=np.float64,
    )

    finite_source = (
        valid
        & np.isfinite(source)
    )

    # In Sentinel-1 denoising annotation, zero-valued LUT
    # samples indicate that a usable noise estimate is not
    # available at that location.
    noise_lut_valid = (
        np.isfinite(noise)
        & (noise > 0.0)
    )

    if subtract_noise:
        processing_mask = (
            finite_source
            & noise_lut_valid
        )

        corrected = (
            source[processing_mask]
            - noise[processing_mask]
        )

        if nonpositive_power_policy == "floor":
            corrected = np.where(
                corrected <= 0.0,
                positive_power_floor,
                corrected,
            )
        else:
            corrected = np.where(
                corrected <= 0.0,
                np.nan,
                corrected,
            )
    else:
        # The measurement has already been denoised, so the
        # annotation LUT is not needed to retain source pixels.
        processing_mask = finite_source
        corrected = source[processing_mask]

    result[processing_mask] = corrected

    return (
        result.astype(
            np.float32,
            copy=False,
        ),
        subtract_noise,
        noise_lut_valid,
    )


def calibrate_linear_power(
    *,
    source_power: np.ndarray,
    calibration_lut: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    power = np.asarray(
        source_power,
        dtype=np.float64,
    )
    calibration = np.asarray(
        calibration_lut,
        dtype=np.float64,
    )
    valid = np.asarray(
        valid_mask,
        dtype=bool,
    )

    validate_same_shape(
        power,
        calibration,
        valid,
    )

    result = np.full(
        power.shape,
        np.nan,
        dtype=np.float64,
    )

    finite_valid = (
        valid
        & np.isfinite(power)
        & np.isfinite(calibration)
        & (calibration > 0.0)
    )

    result[finite_valid] = (
        power[finite_valid]
        / np.square(
            calibration[finite_valid]
        )
    )

    return result.astype(
        np.float32,
        copy=False,
    )


def linear_power_to_db(
    values: np.ndarray,
    *,
    valid_mask: np.ndarray,
) -> np.ndarray:
    linear = np.asarray(
        values,
        dtype=np.float64,
    )
    valid = np.asarray(
        valid_mask,
        dtype=bool,
    )

    validate_same_shape(
        linear,
        valid,
    )

    result = np.full(
        linear.shape,
        np.nan,
        dtype=np.float64,
    )

    positive_valid = (
        valid
        & np.isfinite(linear)
        & (linear > 0.0)
    )

    result[positive_valid] = (
        10.0
        * np.log10(
            linear[positive_valid]
        )
    )

    return result.astype(
        np.float32,
        copy=False,
    )


def process_detected_magnitude(
    *,
    magnitude: np.ndarray,
    calibration_lut: np.ndarray,
    valid_mask: np.ndarray,
    thermal_noise_correction_performed: bool | None,
    legacy_noise_lut: np.ndarray | None = None,
    range_noise_lut: np.ndarray | None = None,
    azimuth_noise_lut: np.ndarray | None = None,
    unknown_noise_policy: UnknownNoisePolicy = "error",
    nonpositive_power_policy: NonpositivePowerPolicy = "nan",
    positive_power_floor: float = 1.0e-5,
) -> RadiometryResult:
    magnitude_values = np.asarray(
        magnitude,
        dtype=np.float64,
    )
    calibration_values = np.asarray(
        calibration_lut,
        dtype=np.float64,
    )
    valid = np.asarray(
        valid_mask,
        dtype=bool,
    )

    noise_power = combine_noise_luts(
        legacy_noise_lut=legacy_noise_lut,
        range_noise_lut=range_noise_lut,
        azimuth_noise_lut=azimuth_noise_lut,
    )

    validate_same_shape(
        magnitude_values,
        calibration_values,
        valid,
        noise_power,
    )

    source_power = (
        detected_magnitude_to_power(
            magnitude_values
        )
    )

    (
        denoised_power,
        subtracted,
        noise_lut_valid_mask,
    ) = remove_thermal_noise(
        source_power=source_power,
        noise_power=noise_power,
        valid_mask=valid,
        thermal_noise_correction_performed=(
            thermal_noise_correction_performed
        ),
        unknown_noise_policy=(
            unknown_noise_policy
        ),
        nonpositive_power_policy=(
            nonpositive_power_policy
        ),
        positive_power_floor=(
            positive_power_floor
        ),
    )

    calibrated_linear = (
        calibrate_linear_power(
            source_power=denoised_power,
            calibration_lut=(
                calibration_values
            ),
            valid_mask=valid,
        )
    )

    calibrated_db = linear_power_to_db(
        calibrated_linear,
        valid_mask=valid,
    )

    final_valid = (
        valid
        & np.isfinite(calibrated_linear)
        & (calibrated_linear > 0.0)
    )

    return RadiometryResult(
        source_power=source_power,
        noise_power=noise_power,
        noise_lut_valid_mask=(
            noise_lut_valid_mask
        ),
        denoised_power=denoised_power,
        calibrated_linear=(
            calibrated_linear
        ),
        calibrated_db=calibrated_db,
        valid_mask=final_valid,
        noise_was_subtracted=subtracted,
        unknown_noise_policy=(
            unknown_noise_policy
        ),
        nonpositive_power_policy=(
            nonpositive_power_policy
        ),
    )

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.warp import reproject

from .inspect_metadata_assets import read_asset_bytes
from .radiometry_lut import (
    RasterWindow,
    interpolate_azimuth_noise_lut,
    interpolate_calibration_lut,
    interpolate_legacy_noise_lut,
    interpolate_range_noise_lut,
)
from .radiometry_math import (
    UnknownNoisePolicy,
    calibrate_linear_power,
    detected_magnitude_to_power,
    linear_power_to_db,
    remove_thermal_noise,
    resolve_noise_subtraction,
)
from .radiometry_metadata import (
    RadiometryMetadata,
    load_radiometry_metadata,
)


@dataclass(frozen=True)
class CalibratedWarpedAsset:
    calibrated_db: np.ndarray
    calibrated_linear: np.ndarray
    valid_mask: np.ndarray
    metadata: dict[str, Any]


def load_remote_radiometry_metadata(
    *,
    product_metadata_href: str,
    calibration_metadata_href: str,
    noise_metadata_href: str,
    polarisation: str,
    s3_client,
) -> RadiometryMetadata:
    product_bytes = read_asset_bytes(
        product_metadata_href,
        s3_client,
    )
    calibration_bytes = read_asset_bytes(
        calibration_metadata_href,
        s3_client,
    )
    noise_bytes = read_asset_bytes(
        noise_metadata_href,
        s3_client,
    )

    return load_radiometry_metadata(
        product_xml=product_bytes,
        calibration_xml=calibration_bytes,
        noise_xml=noise_bytes,
        expected_polarisation=polarisation,
    )


def _require_href(
    value: str | None,
    *,
    label: str,
    polarisation: str,
) -> str:
    if not value:
        raise RuntimeError(
            f"{polarisation} is missing its {label} asset."
        )
    return str(value)


def _validate_source(
    *,
    src: rasterio.DatasetReader,
    metadata: RadiometryMetadata,
    polarisation: str,
    measurement_href: str,
):
    if src.count != 1:
        raise RuntimeError(
            f"{polarisation} measurement was expected to contain one band, "
            f"but contains {src.count}: {measurement_href}"
        )

    if src.nodata is None:
        raise RuntimeError(
            f"{polarisation} measurement has no declared nodata value: "
            f"{measurement_href}"
        )

    gcps, gcp_crs = src.gcps

    if not gcps:
        raise RuntimeError(
            f"{polarisation} measurement contains no GCPs: "
            f"{measurement_href}"
        )

    if gcp_crs is None:
        raise RuntimeError(
            f"{polarisation} measurement has no GCP CRS: "
            f"{measurement_href}"
        )

    if metadata.product.pixel_value not in {None, "Detected"}:
        raise RuntimeError(
            "Expected Sentinel-1 GRD detected magnitude pixels; "
            f"metadata reports {metadata.product.pixel_value!r}."
        )

    if (
        metadata.product.number_of_lines is not None
        and int(metadata.product.number_of_lines) != int(src.height)
    ):
        raise RuntimeError(
            "Product metadata line count does not match measurement TIFF: "
            f"{metadata.product.number_of_lines} != {src.height}"
        )

    if (
        metadata.product.number_of_samples is not None
        and int(metadata.product.number_of_samples) != int(src.width)
    ):
        raise RuntimeError(
            "Product metadata sample count does not match measurement TIFF: "
            f"{metadata.product.number_of_samples} != {src.width}"
        )

    return gcps, gcp_crs


def _source_windows(
    *,
    height: int,
    width: int,
    rows_per_window: int,
):
    for row_start in range(0, height, rows_per_window):
        row_stop = min(height, row_start + rows_per_window)
        yield Window(
            col_off=0,
            row_off=row_start,
            width=width,
            height=row_stop - row_start,
        )


def _noise_power_for_window(
    *,
    metadata: RadiometryMetadata,
    image_shape: tuple[int, int],
    raster_window: RasterWindow,
) -> np.ndarray:
    if metadata.noise.legacy_vectors:
        return interpolate_legacy_noise_lut(
            metadata.noise,
            image_shape=image_shape,
            window=raster_window,
        )

    if not metadata.noise.range_vectors:
        raise RuntimeError(
            "No supported Sentinel-1 thermal-noise LUT was found."
        )

    if not metadata.noise.azimuth_vectors:
        raise RuntimeError(
            "Modern Sentinel-1 noise metadata contains range vectors "
            "without azimuth vectors. Refusing to invent a combination "
            "policy."
        )

    range_noise = interpolate_range_noise_lut(
        metadata.noise,
        image_shape=image_shape,
        window=raster_window,
    )
    azimuth_noise = interpolate_azimuth_noise_lut(
        metadata.noise,
        image_shape=image_shape,
        window=raster_window,
        require_full_coverage=False,
    )

    # Missing azimuth coverage remains NaN. When thermal-noise
    # subtraction is required, remove_thermal_noise() treats it as
    # unavailable denoising information rather than as zero noise.
    return (
        range_noise.astype(np.float64, copy=False)
        * azimuth_noise.astype(np.float64, copy=False)
    ).astype(np.float32)


def _stage_calibrated_linear_raster(
    *,
    measurement_href: str,
    metadata: RadiometryMetadata,
    polarisation: str,
    calibration_lut_name: str,
    unknown_noise_policy: UnknownNoisePolicy,
    rows_per_window: int,
    temp_directory: str | None,
) -> tuple[str, dict[str, Any]]:
    if rows_per_window <= 0:
        raise ValueError(
            "rows_per_window must be greater than zero."
        )

    fd, stage_path = tempfile.mkstemp(
        prefix=f"s1_{polarisation.lower()}_calibrated_",
        suffix=".tif",
        dir=temp_directory,
    )
    os.close(fd)
    os.remove(stage_path)

    stats: dict[str, Any] = {
        "window_count": 0,
        "source_valid_pixel_count": 0,
        "noise_lut_valid_pixel_count": 0,
        "final_valid_pixel_count": 0,
        "nonpositive_after_noise_pixel_count": 0,
    }

    try:
        with rasterio.open(measurement_href) as src:
            gcps, gcp_crs = _validate_source(
                src=src,
                metadata=metadata,
                polarisation=polarisation,
                measurement_href=measurement_href,
            )

            image_shape = (
                int(src.height),
                int(src.width),
            )

            subtract_noise = resolve_noise_subtraction(
                thermal_noise_correction_performed=(
                    metadata.product.thermal_noise_correction_performed
                ),
                unknown_noise_policy=unknown_noise_policy,
            )

            profile = {
                "driver": "GTiff",
                "height": int(src.height),
                "width": int(src.width),
                "count": 1,
                "dtype": "float32",
                "nodata": np.nan,
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 256,
                "compress": "DEFLATE",
                "predictor": 3,
                "BIGTIFF": "YES",
            }

            with rasterio.open(
                stage_path,
                "w",
                **profile,
            ) as dst:
                # Sentinel-1 GRD measurement geolocation is represented
                # by GCPs. Preserve those GCPs on the staged calibrated
                # linear-power raster.
                dst.gcps = (
                    gcps,
                    gcp_crs,
                )

                for source_window in _source_windows(
                    height=int(src.height),
                    width=int(src.width),
                    rows_per_window=rows_per_window,
                ):
                    row_start = int(source_window.row_off)
                    row_stop = row_start + int(source_window.height)
                    column_start = int(source_window.col_off)
                    column_stop = (
                        column_start + int(source_window.width)
                    )

                    raster_window = RasterWindow(
                        row_start=row_start,
                        row_stop=row_stop,
                        column_start=column_start,
                        column_stop=column_stop,
                    )

                    magnitude = src.read(
                        1,
                        window=source_window,
                        out_dtype="float32",
                    )
                    source_mask = (
                        src.read_masks(
                            1,
                            window=source_window,
                        )
                        != 0
                    )
                    source_valid = (
                        source_mask
                        & np.isfinite(magnitude)
                        & (
                            magnitude
                            != np.float32(src.nodata)
                        )
                    )

                    calibration_lut = (
                        interpolate_calibration_lut(
                            metadata.calibration,
                            lut_name=calibration_lut_name,
                            image_shape=image_shape,
                            window=raster_window,
                        )
                    )

                    source_power = (
                        detected_magnitude_to_power(
                            magnitude
                        )
                    )

                    if subtract_noise:
                        noise_power = (
                            _noise_power_for_window(
                                metadata=metadata,
                                image_shape=image_shape,
                                raster_window=raster_window,
                            )
                        )
                    else:
                        # The noise field is not used when product
                        # metadata explicitly states that thermal-noise
                        # correction has already been performed.
                        noise_power = np.ones(
                            magnitude.shape,
                            dtype=np.float32,
                        )

                    (
                        denoised_power,
                        noise_was_subtracted,
                        noise_lut_valid,
                    ) = remove_thermal_noise(
                        source_power=source_power,
                        noise_power=noise_power,
                        valid_mask=source_valid,
                        thermal_noise_correction_performed=(
                            metadata.product
                            .thermal_noise_correction_performed
                        ),
                        unknown_noise_policy=unknown_noise_policy,
                        nonpositive_power_policy="nan",
                    )

                    if noise_was_subtracted != subtract_noise:
                        raise RuntimeError(
                            "Noise-subtraction decision changed while "
                            "processing one measurement."
                        )

                    calibrated_linear = (
                        calibrate_linear_power(
                            source_power=denoised_power,
                            calibration_lut=calibration_lut,
                            valid_mask=source_valid,
                        )
                    )

                    final_valid = (
                        source_valid
                        & np.isfinite(calibrated_linear)
                        & (calibrated_linear > 0.0)
                    )

                    output = np.where(
                        final_valid,
                        calibrated_linear,
                        np.nan,
                    ).astype(np.float32)

                    dst.write(
                        output,
                        1,
                        window=source_window,
                    )

                    stats["window_count"] += 1
                    stats[
                        "source_valid_pixel_count"
                    ] += int(
                        np.count_nonzero(source_valid)
                    )

                    if subtract_noise:
                        stats[
                            "noise_lut_valid_pixel_count"
                        ] += int(
                            np.count_nonzero(
                                source_valid
                                & noise_lut_valid
                            )
                        )
                    else:
                        stats[
                            "noise_lut_valid_pixel_count"
                        ] += int(
                            np.count_nonzero(source_valid)
                        )

                    stats[
                        "final_valid_pixel_count"
                    ] += int(
                        np.count_nonzero(final_valid)
                    )

                    if subtract_noise:
                        stats[
                            "nonpositive_after_noise_pixel_count"
                        ] += int(
                            np.count_nonzero(
                                source_valid
                                & noise_lut_valid
                                & ~np.isfinite(
                                    denoised_power
                                )
                            )
                        )

            stats.update(
                {
                    "source_shape": [
                        int(src.height),
                        int(src.width),
                    ],
                    "source_dtype": str(src.dtypes[0]),
                    "source_nodata": float(src.nodata),
                    "gcp_count": len(gcps),
                    "gcp_crs": gcp_crs.to_string(),
                    "noise_format": (
                        metadata.noise.format_name
                    ),
                    "thermal_noise_correction_performed": (
                        metadata.product
                        .thermal_noise_correction_performed
                    ),
                    "noise_was_subtracted": subtract_noise,
                    "calibration_lut": calibration_lut_name,
                    "rows_per_window": rows_per_window,
                }
            )

        stats["stage_file_size_bytes"] = int(
            os.path.getsize(stage_path)
        )

        return stage_path, stats

    except Exception:
        try:
            os.remove(stage_path)
        except FileNotFoundError:
            pass
        raise


def warp_calibrated_asset_to_grid(
    *,
    measurement_href: str,
    product_metadata_href: str | None,
    calibration_metadata_href: str | None,
    noise_metadata_href: str | None,
    destination_crs: CRS,
    destination_transform: Affine,
    destination_shape: tuple[int, int],
    polarisation: str,
    s3_client,
    calibration_lut_name: str = "sigmaNought",
    unknown_noise_policy: UnknownNoisePolicy = "error",
    rows_per_window: int = 256,
    num_threads: int = 2,
    temp_directory: str | None = None,
) -> CalibratedWarpedAsset:
    """
    Calibrate one Sentinel-1 GRD measurement in source-image space,
    average-reproject calibrated LINEAR power onto the exact Sentinel-2
    target grid using source GCPs, then convert the 128x128 result to dB.

    The complete source raster is never loaded into memory at once.
    """
    polarisation = polarisation.upper()

    product_href = _require_href(
        product_metadata_href,
        label="product metadata",
        polarisation=polarisation,
    )
    calibration_href = _require_href(
        calibration_metadata_href,
        label="calibration metadata",
        polarisation=polarisation,
    )
    noise_href = _require_href(
        noise_metadata_href,
        label="noise metadata",
        polarisation=polarisation,
    )

    metadata = load_remote_radiometry_metadata(
        product_metadata_href=product_href,
        calibration_metadata_href=calibration_href,
        noise_metadata_href=noise_href,
        polarisation=polarisation,
        s3_client=s3_client,
    )

    stage_path: str | None = None

    try:
        stage_path, stage_metadata = (
            _stage_calibrated_linear_raster(
                measurement_href=measurement_href,
                metadata=metadata,
                polarisation=polarisation,
                calibration_lut_name=calibration_lut_name,
                unknown_noise_policy=unknown_noise_policy,
                rows_per_window=rows_per_window,
                temp_directory=temp_directory,
            )
        )

        destination_height, destination_width = (
            destination_shape
        )
        destination_linear = np.full(
            (
                destination_height,
                destination_width,
            ),
            np.nan,
            dtype=np.float32,
        )

        with rasterio.open(stage_path) as staged:
            gcps, gcp_crs = staged.gcps

            if not gcps or gcp_crs is None:
                raise RuntimeError(
                    "The staged calibrated raster did not preserve "
                    "the Sentinel-1 GCP geolocation."
                )

            reproject(
                source=rasterio.band(staged, 1),
                destination=destination_linear,
                gcps=gcps,
                src_crs=gcp_crs,
                src_nodata=np.nan,
                dst_transform=destination_transform,
                dst_crs=destination_crs,
                dst_nodata=np.nan,
                resampling=Resampling.average,
                init_dest_nodata=True,
                num_threads=num_threads,
            )

        destination_valid = (
            np.isfinite(destination_linear)
            & (destination_linear > 0.0)
        )

        destination_db = linear_power_to_db(
            destination_linear,
            valid_mask=destination_valid,
        )

        valid_values = destination_db[
            destination_valid
        ]

        result_metadata: dict[str, Any] = {
            "polarisation": polarisation,
            "measurement_href": measurement_href,
            "product_metadata_href": product_href,
            "calibration_metadata_href": calibration_href,
            "noise_metadata_href": noise_href,
            "value_representation": (
                f"{calibration_lut_name} dB"
            ),
            "processing_order": [
                "detected_magnitude",
                "square_to_source_power",
                "thermal_noise_correction_in_source_domain",
                (
                    f"{calibration_lut_name}_linear_calibration_"
                    "in_source_domain"
                ),
                "gcp_average_reprojection_of_linear_power",
                "10_log10_after_reprojection",
            ],
            "destination_shape": [
                destination_height,
                destination_width,
            ],
            "destination_crs": destination_crs.to_string(),
            "destination_transform": [
                float(value)
                for value in tuple(destination_transform)
            ],
            "resampling": (
                "average on calibrated linear power"
            ),
            "destination_valid_pixel_count": int(
                np.count_nonzero(destination_valid)
            ),
            "destination_invalid_pixel_count": int(
                destination_valid.size
                - np.count_nonzero(destination_valid)
            ),
            "destination_valid_fraction": float(
                destination_valid.mean()
            ),
            "stage": stage_metadata,
        }

        if valid_values.size:
            result_metadata.update(
                {
                    "db_minimum": float(valid_values.min()),
                    "db_maximum": float(valid_values.max()),
                    "db_mean": float(valid_values.mean()),
                    "db_median": float(
                        np.median(valid_values)
                    ),
                }
            )
        else:
            result_metadata.update(
                {
                    "db_minimum": None,
                    "db_maximum": None,
                    "db_mean": None,
                    "db_median": None,
                }
            )

        return CalibratedWarpedAsset(
            calibrated_db=destination_db,
            calibrated_linear=destination_linear,
            valid_mask=destination_valid,
            metadata=result_metadata,
        )

    finally:
        if stage_path is not None:
            try:
                os.remove(stage_path)
            except FileNotFoundError:
                pass

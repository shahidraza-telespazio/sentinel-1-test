from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window

from .inspect_metadata_assets import (
    build_s3_client,
    read_asset_bytes,
)
from .radiometry_lut import (
    CALIBRATION_LUT_NAMES,
    RasterWindow,
    interpolate_azimuth_noise_lut,
    interpolate_calibration_lut,
    interpolate_legacy_noise_lut,
    interpolate_range_noise_lut,
)
from .radiometry_math import (
    RadiometryResult,
    process_detected_magnitude,
)
from .radiometry_metadata import (
    NoiseMetadata,
    RadiometryMetadata,
    load_radiometry_metadata,
)
from .raster_io import aws_rasterio_environment


@dataclass(frozen=True)
class SelectedScene:
    item_id: str
    polarisation: str
    measurement_href: str
    product_metadata_href: str
    calibration_metadata_href: str
    noise_metadata_href: str


@dataclass(frozen=True)
class NoiseAnchor:
    line: int
    pixel: int
    noise_power: float
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Sentinel-1 GRD radiometry on one real "
            "source-image window selected from a metadata-only "
            "validation-scene search. This command does not "
            "perform tile reprojection."
        )
    )
    parser.add_argument(
        "--search-json",
        required=True,
    )
    parser.add_argument(
        "--item-id",
        default=None,
        help=(
            "Specific usable item. When omitted, the first usable "
            "scene/polarisation pair is selected."
        ),
    )
    parser.add_argument(
        "--polarisation",
        default=None,
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--calibration-lut",
        choices=CALIBRATION_LUT_NAMES,
        default="sigmaNought",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "reports/s1_development/"
            "radiometry_validation/source_windows"
        ),
    )
    parser.add_argument(
        "--aws-profile",
        default=None,
    )
    parser.add_argument(
        "--aws-region",
        default="eu-central-1",
    )

    return parser.parse_args()


def load_search(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise ValueError(
            "Validation search JSON must contain an object."
        )

    return value


def matching_scene(
    search: dict[str, Any],
    *,
    item_id: str,
) -> dict[str, Any]:
    matches = [
        scene
        for scene in search.get("scenes", [])
        if scene.get("item_id") == item_id
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one scene entry for {item_id!r}; "
            f"found {len(matches)}."
        )

    return matches[0]


def matching_polarisation(
    scene: dict[str, Any],
    *,
    polarisation: str,
) -> dict[str, Any]:
    requested = polarisation.upper()

    matches = [
        value
        for value in scene.get("polarisations", [])
        if str(value.get("polarisation", "")).upper()
        == requested
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected one polarisation entry for "
            f"{requested!r}; found {len(matches)}."
        )

    return matches[0]


def select_scene(
    search: dict[str, Any],
    *,
    requested_item_id: str | None,
    requested_polarisation: str | None,
) -> SelectedScene:
    usable = search.get(
        "usable_scene_polarisations",
        [],
    )

    if not isinstance(usable, list) or not usable:
        raise RuntimeError(
            "Search JSON contains no usable scene/polarisation pairs."
        )

    if requested_polarisation and not requested_item_id:
        raise ValueError(
            "--polarisation requires --item-id."
        )

    if requested_item_id:
        candidates = [
            value
            for value in usable
            if value.get("item_id")
            == requested_item_id
            and (
                requested_polarisation is None
                or str(
                    value.get("polarisation", "")
                ).upper()
                == requested_polarisation.upper()
            )
        ]
    else:
        candidates = usable[:1]

    if len(candidates) != 1:
        raise ValueError(
            "Scene selection is missing or ambiguous. "
            f"Matching usable pairs: {candidates}"
        )

    selected = candidates[0]
    item_id = str(selected["item_id"])
    polarisation = str(
        selected["polarisation"]
    ).upper()

    scene = matching_scene(
        search,
        item_id=item_id,
    )
    inspection = matching_polarisation(
        scene,
        polarisation=polarisation,
    )

    if (
        inspection.get(
            "usable_for_positive_noise_validation"
        )
        is not True
    ):
        raise ValueError(
            f"{item_id} {polarisation} is not marked usable."
        )

    required = {
        "measurement_href": inspection.get(
            "measurement_href"
        ),
        "product_metadata_href": inspection.get(
            "product_metadata_href"
        ),
        "calibration_metadata_href": inspection.get(
            "calibration_metadata_href"
        ),
        "noise_metadata_href": inspection.get(
            "noise_metadata_href"
        ),
    }

    missing = [
        name
        for name, value in required.items()
        if not value
    ]

    if missing:
        raise ValueError(
            "Selected scene is missing required assets: "
            f"{missing}"
        )

    return SelectedScene(
        item_id=item_id,
        polarisation=polarisation,
        measurement_href=str(
            required["measurement_href"]
        ),
        product_metadata_href=str(
            required["product_metadata_href"]
        ),
        calibration_metadata_href=str(
            required["calibration_metadata_href"]
        ),
        noise_metadata_href=str(
            required["noise_metadata_href"]
        ),
    )


def legacy_noise_anchors(
    metadata: NoiseMetadata,
) -> tuple[NoiseAnchor, ...]:
    anchors: list[NoiseAnchor] = []

    for vector in metadata.legacy_vectors:
        for index in range(
            int(vector.pixels.size)
        ):
            value = float(
                vector.noise_lut[index]
            )

            if value <= 0.0:
                continue

            anchors.append(
                NoiseAnchor(
                    line=int(vector.line),
                    pixel=int(
                        vector.pixels[index]
                    ),
                    noise_power=value,
                    source=(
                        "legacy_noise_vector"
                    ),
                )
            )

    if not anchors:
        raise RuntimeError(
            "Legacy noise metadata has no positive LUT value."
        )

    return tuple(anchors)


def azimuth_value_at(
    metadata: NoiseMetadata,
    *,
    line: int,
    pixel: int,
) -> float | None:
    values: list[float] = []

    for section in metadata.azimuth_vectors:
        if not (
            section.first_azimuth_line
            <= line
            <= section.last_azimuth_line
        ):
            continue

        if not (
            section.first_range_sample
            <= pixel
            <= section.last_range_sample
        ):
            continue

        if not (
            int(section.lines[0])
            <= line
            <= int(section.lines[-1])
        ):
            continue

        value = float(
            np.interp(
                float(line),
                section.lines.astype(
                    np.float64,
                    copy=False,
                ),
                section.noise_azimuth_lut.astype(
                    np.float64,
                    copy=False,
                ),
            )
        )

        if value > 0.0:
            values.append(value)

    if not values:
        return None

    if not np.allclose(
        values,
        values[0],
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError(
            "Overlapping azimuth-noise sections disagree "
            f"at line={line}, pixel={pixel}: {values}"
        )

    return values[0]


def modern_noise_anchors(
    metadata: NoiseMetadata,
) -> tuple[NoiseAnchor, ...]:
    anchors: list[NoiseAnchor] = []

    for vector in metadata.range_vectors:
        for index in range(
            int(vector.pixels.size)
        ):
            range_value = float(
                vector.noise_range_lut[index]
            )

            if range_value <= 0.0:
                continue

            line = int(vector.line)
            pixel = int(vector.pixels[index])

            azimuth_value = azimuth_value_at(
                metadata,
                line=line,
                pixel=pixel,
            )

            if (
                azimuth_value is None
                or azimuth_value <= 0.0
            ):
                continue

            anchors.append(
                NoiseAnchor(
                    line=line,
                    pixel=pixel,
                    noise_power=(
                        range_value
                        * azimuth_value
                    ),
                    source=(
                        "range_times_azimuth"
                    ),
                )
            )

    if not anchors:
        raise RuntimeError(
            "Modern noise metadata has no positive "
            "range × azimuth sample."
        )

    return tuple(anchors)


def noise_anchor_candidates(
    metadata: NoiseMetadata,
) -> tuple[NoiseAnchor, ...]:
    if metadata.legacy_vectors:
        return legacy_noise_anchors(metadata)

    if (
        metadata.range_vectors
        and metadata.azimuth_vectors
    ):
        return modern_noise_anchors(metadata)

    raise ValueError(
        "Unsupported noise metadata layout."
    )


def source_window(
    *,
    anchor: NoiseAnchor,
    image_shape: tuple[int, int],
    window_size: int,
) -> RasterWindow:
    height, width = image_shape

    if window_size <= 0:
        raise ValueError(
            "--window-size must be positive."
        )

    if (
        window_size > height
        or window_size > width
    ):
        raise ValueError(
            f"Window size {window_size} exceeds "
            f"source shape {image_shape}."
        )

    row_start = max(
        0,
        min(
            anchor.line - window_size // 2,
            height - window_size,
        ),
    )
    column_start = max(
        0,
        min(
            anchor.pixel - window_size // 2,
            width - window_size,
        ),
    )

    return RasterWindow(
        row_start=row_start,
        row_stop=row_start + window_size,
        column_start=column_start,
        column_stop=(
            column_start + window_size
        ),
    )


def ranked_noise_anchors(
    *,
    metadata: NoiseMetadata,
    image_shape: tuple[int, int],
    window_size: int,
) -> tuple[NoiseAnchor, ...]:
    height, width = image_shape
    half_window = window_size // 2
    centre_line = (height - 1) / 2.0
    centre_pixel = (width - 1) / 2.0

    unique: dict[
        tuple[int, int],
        NoiseAnchor,
    ] = {}

    for anchor in noise_anchor_candidates(
        metadata
    ):
        if not (
            0 <= anchor.line < height
            and 0 <= anchor.pixel < width
        ):
            continue

        key = (
            anchor.line,
            anchor.pixel,
        )
        previous = unique.get(key)

        if (
            previous is None
            or anchor.noise_power
            > previous.noise_power
        ):
            unique[key] = anchor

    if not unique:
        raise RuntimeError(
            "Noise metadata has no anchors inside the source image."
        )

    def rank(
        anchor: NoiseAnchor,
    ) -> tuple[int, float, float]:
        interior = (
            half_window <= anchor.line
            < height - half_window
            and half_window <= anchor.pixel
            < width - half_window
        )
        distance_squared = (
            (anchor.line - centre_line) ** 2
            + (anchor.pixel - centre_pixel) ** 2
        )

        return (
            0 if interior else 1,
            distance_squared,
            -anchor.noise_power,
        )

    return tuple(
        sorted(
            unique.values(),
            key=rank,
        )
    )


def read_measurement_window(
    *,
    selected: SelectedScene,
    window_size: int,
    metadata: RadiometryMetadata,
    aws_profile: str | None,
    aws_region: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    RasterWindow,
    NoiseAnchor,
    tuple[int, int],
    float | None,
    str,
]:
    minimum_valid_fraction = 0.25
    maximum_attempts = 256

    with aws_rasterio_environment(
        profile=aws_profile,
        region=aws_region,
    ):
        with rasterio.open(
            selected.measurement_href
        ) as source:
            if source.count != 1:
                raise ValueError(
                    "Expected one measurement band; "
                    f"found {source.count}."
                )

            image_shape = (
                int(source.height),
                int(source.width),
            )
            source_nodata = source.nodata
            source_dtype = str(
                source.dtypes[0]
            )
            anchors = ranked_noise_anchors(
                metadata=metadata.noise,
                image_shape=image_shape,
                window_size=window_size,
            )

            best: tuple[
                int,
                np.ndarray,
                np.ndarray,
                RasterWindow,
                NoiseAnchor,
            ] | None = None

            for attempt, anchor in enumerate(
                anchors[:maximum_attempts],
                start=1,
            ):
                window = source_window(
                    anchor=anchor,
                    image_shape=image_shape,
                    window_size=window_size,
                )
                raster_window = Window(
                    col_off=window.column_start,
                    row_off=window.row_start,
                    width=window.width,
                    height=window.height,
                )

                magnitude = source.read(
                    1,
                    window=raster_window,
                    masked=False,
                ).astype(
                    np.float32,
                    copy=False,
                )
                source_mask = (
                    source.read_masks(
                        1,
                        window=raster_window,
                    )
                    > 0
                )
                valid_mask = (
                    source_mask
                    & np.isfinite(magnitude)
                )

                if source_nodata is not None:
                    valid_mask &= (
                        magnitude
                        != float(source_nodata)
                    )

                valid_count = int(
                    valid_mask.sum()
                )

                if (
                    best is None
                    or valid_count > best[0]
                ):
                    best = (
                        valid_count,
                        magnitude,
                        valid_mask,
                        window,
                        anchor,
                    )

                if (
                    valid_count
                    / valid_mask.size
                    >= minimum_valid_fraction
                ):
                    break

    if best is None or best[0] == 0:
        raise RuntimeError(
            "Could not find a positive-noise metadata window "
            "containing any valid source measurement pixels."
        )

    (
        valid_count,
        magnitude,
        valid_mask,
        window,
        anchor,
    ) = best

    valid_fraction = (
        valid_count
        / valid_mask.size
    )

    if valid_fraction < minimum_valid_fraction:
        raise RuntimeError(
            "The best candidate window had only "
            f"{valid_fraction:.2%} valid source pixels; "
            f"at least {minimum_valid_fraction:.2%} is required."
        )

    return (
        magnitude,
        valid_mask,
        window,
        anchor,
        image_shape,
        (
            float(source_nodata)
            if source_nodata is not None
            else None
        ),
        source_dtype,
    )

def interpolate_luts(
    *,
    metadata: RadiometryMetadata,
    image_shape: tuple[int, int],
    window: RasterWindow,
    calibration_lut_name: str,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
]:
    calibration_lut = (
        interpolate_calibration_lut(
            metadata.calibration,
            lut_name=calibration_lut_name,
            image_shape=image_shape,
            window=window,
        )
    )

    if metadata.noise.legacy_vectors:
        return (
            calibration_lut,
            {
                "legacy_noise_lut": (
                    interpolate_legacy_noise_lut(
                        metadata.noise,
                        image_shape=image_shape,
                        window=window,
                    )
                )
            },
        )

    return (
        calibration_lut,
        {
            "range_noise_lut": (
                interpolate_range_noise_lut(
                    metadata.noise,
                    image_shape=image_shape,
                    window=window,
                )
            ),
            "azimuth_noise_lut": (
                interpolate_azimuth_noise_lut(
                    metadata.noise,
                    image_shape=image_shape,
                    window=window,
                )
            ),
        },
    )


def process_window(
    *,
    magnitude: np.ndarray,
    source_valid_mask: np.ndarray,
    calibration_lut: np.ndarray,
    noise_arrays: dict[str, np.ndarray],
    metadata: RadiometryMetadata,
) -> RadiometryResult:
    return process_detected_magnitude(
        magnitude=magnitude,
        calibration_lut=calibration_lut,
        valid_mask=source_valid_mask,
        thermal_noise_correction_performed=(
            metadata.product
            .thermal_noise_correction_performed
        ),
        legacy_noise_lut=noise_arrays.get(
            "legacy_noise_lut"
        ),
        range_noise_lut=noise_arrays.get(
            "range_noise_lut"
        ),
        azimuth_noise_lut=noise_arrays.get(
            "azimuth_noise_lut"
        ),
        unknown_noise_policy="error",
        nonpositive_power_policy="nan",
    )


def finite_statistics(
    values: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int | None]:
    selected = values[
        mask & np.isfinite(values)
    ]

    return {
        "count": int(selected.size),
        "minimum": (
            float(selected.min())
            if selected.size
            else None
        ),
        "maximum": (
            float(selected.max())
            if selected.size
            else None
        ),
        "mean": (
            float(selected.mean())
            if selected.size
            else None
        ),
    }


def validate_equations(
    *,
    magnitude: np.ndarray,
    source_valid_mask: np.ndarray,
    calibration_lut: np.ndarray,
    result: RadiometryResult,
) -> dict[str, bool]:
    if not np.any(source_valid_mask):
        raise AssertionError(
            "Validation window contains no valid source pixels."
        )

    if not result.noise_was_subtracted:
        raise AssertionError(
            "Thermal noise was not subtracted."
        )

    expected_source_power = (
        magnitude.astype(np.float64)
        ** 2
    )

    assert np.allclose(
        result.source_power[
            source_valid_mask
        ],
        expected_source_power[
            source_valid_mask
        ],
        rtol=1e-6,
        atol=1e-5,
    )

    subtraction_domain = (
        source_valid_mask
        & result.noise_lut_valid_mask
    )

    if not np.any(subtraction_domain):
        raise AssertionError(
            "Validation window contains no source pixels with "
            "usable positive noise information."
        )

    expected_denoised = (
        expected_source_power
        - result.noise_power
    )
    positive_after_subtraction = (
        subtraction_domain
        & (expected_denoised > 0.0)
    )
    nonpositive_after_subtraction = (
        subtraction_domain
        & (expected_denoised <= 0.0)
    )

    if not np.any(
        positive_after_subtraction
    ):
        raise AssertionError(
            "Validation window contains no positive signal after "
            "thermal-noise subtraction."
        )

    assert np.allclose(
        result.denoised_power[
            positive_after_subtraction
        ],
        expected_denoised[
            positive_after_subtraction
        ],
        rtol=1e-5,
        atol=1e-4,
    )

    assert np.isnan(
        result.denoised_power[
            nonpositive_after_subtraction
        ]
    ).all()

    expected_linear = (
        expected_denoised
        / np.square(
            calibration_lut.astype(
                np.float64
            )
        )
    )

    assert np.allclose(
        result.calibrated_linear[
            positive_after_subtraction
        ],
        expected_linear[
            positive_after_subtraction
        ],
        rtol=1e-5,
        atol=1e-7,
    )

    expected_db = (
        10.0
        * np.log10(
            expected_linear[
                positive_after_subtraction
            ]
        )
    )

    assert np.allclose(
        result.calibrated_db[
            positive_after_subtraction
        ],
        expected_db,
        rtol=1e-5,
        atol=1e-5,
    )

    assert np.array_equal(
        result.valid_mask,
        positive_after_subtraction,
    )

    return {
        "source_power_equation_passed": True,
        "noise_subtraction_equation_passed": True,
        "calibration_equation_passed": True,
        "db_conversion_equation_passed": True,
        "valid_mask_equation_passed": True,
    }


def main() -> None:
    args = parse_args()

    search_path = Path(
        args.search_json
    )
    search = load_search(
        search_path
    )
    selected = select_scene(
        search,
        requested_item_id=args.item_id,
        requested_polarisation=(
            args.polarisation
        ),
    )

    print(
        "Sentinel-1 real source-window radiometry"
    )
    print("----------------------------------------")
    print(f"Item:         {selected.item_id}")
    print(
        "Polarisation: "
        f"{selected.polarisation}"
    )
    print(
        "Calibration:  "
        f"{args.calibration_lut}"
    )

    s3_client = build_s3_client(
        profile=args.aws_profile,
        region=args.aws_region,
    )

    product_bytes = read_asset_bytes(
        selected.product_metadata_href,
        s3_client,
    )
    calibration_bytes = read_asset_bytes(
        selected.calibration_metadata_href,
        s3_client,
    )
    noise_bytes = read_asset_bytes(
        selected.noise_metadata_href,
        s3_client,
    )

    metadata = load_radiometry_metadata(
        product_xml=product_bytes,
        calibration_xml=calibration_bytes,
        noise_xml=noise_bytes,
        expected_polarisation=(
            selected.polarisation
        ),
    )

    if (
        metadata.product
        .thermal_noise_correction_performed
        is not False
    ):
        raise RuntimeError(
            "This validation requires an explicit "
            "thermalNoiseCorrectionPerformed=false flag."
        )

    if (
        metadata.product.pixel_value
        not in {None, "Detected"}
    ):
        raise RuntimeError(
            "Expected detected GRD pixels; "
            f"received {metadata.product.pixel_value!r}."
        )

    (
        magnitude,
        source_valid_mask,
        window,
        anchor,
        image_shape,
        source_nodata,
        source_dtype,
    ) = read_measurement_window(
        selected=selected,
        window_size=args.window_size,
        metadata=metadata,
        aws_profile=args.aws_profile,
        aws_region=args.aws_region,
    )

    calibration_lut, noise_arrays = (
        interpolate_luts(
            metadata=metadata,
            image_shape=image_shape,
            window=window,
            calibration_lut_name=(
                args.calibration_lut
            ),
        )
    )

    result = process_window(
        magnitude=magnitude,
        source_valid_mask=(
            source_valid_mask
        ),
        calibration_lut=calibration_lut,
        noise_arrays=noise_arrays,
        metadata=metadata,
    )

    validation = validate_equations(
        magnitude=magnitude,
        source_valid_mask=(
            source_valid_mask
        ),
        calibration_lut=calibration_lut,
        result=result,
    )

    source_noise_valid = (
        source_valid_mask
        & result.noise_lut_valid_mask
    )
    missing_noise_count = int(
        np.count_nonzero(
            source_valid_mask
            & ~result.noise_lut_valid_mask
        )
    )
    nonpositive_after_subtraction_count = int(
        np.count_nonzero(
            source_noise_valid
            & ~result.valid_mask
        )
    )

    output_directory = Path(
        args.output_dir
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"{selected.item_id}_"
        f"{selected.polarisation}_"
        f"{args.calibration_lut}_"
        "source_window"
    )
    npz_path = (
        output_directory
        / f"{stem}.npz"
    )
    json_path = (
        output_directory
        / f"{stem}.json"
    )

    np.savez_compressed(
        npz_path,
        magnitude=magnitude,
        source_valid_mask=(
            source_valid_mask
        ),
        calibration_lut=calibration_lut,
        noise_power=result.noise_power,
        noise_lut_valid_mask=(
            result.noise_lut_valid_mask
        ),
        source_power=result.source_power,
        denoised_power=(
            result.denoised_power
        ),
        calibrated_linear=(
            result.calibrated_linear
        ),
        calibrated_db=(
            result.calibrated_db
        ),
        calibrated_valid_mask=(
            result.valid_mask
        ),
        **noise_arrays,
    )

    summary = {
        "selected_scene": asdict(selected),
        "source": {
            "image_shape": list(
                image_shape
            ),
            "window": asdict(window),
            "window_shape": list(
                window.shape
            ),
            "source_dtype": source_dtype,
            "source_nodata": source_nodata,
            "source_valid_count": int(
                source_valid_mask.sum()
            ),
            "magnitude": finite_statistics(
                magnitude,
                source_valid_mask,
            ),
        },
        "metadata": {
            "pixel_value": (
                metadata.product.pixel_value
            ),
            "thermal_noise_correction_performed": (
                metadata.product
                .thermal_noise_correction_performed
            ),
            "noise_format": (
                metadata.noise.format_name
            ),
            "calibration_lut": (
                args.calibration_lut
            ),
        },
        "selected_noise_anchor": (
            asdict(anchor)
        ),
        "processing": {
            "noise_was_subtracted": (
                result.noise_was_subtracted
            ),
            "source_noise_valid_count": int(
                source_noise_valid.sum()
            ),
            "missing_noise_count": (
                missing_noise_count
            ),
            "nonpositive_after_subtraction_count": (
                nonpositive_after_subtraction_count
            ),
            "final_valid_count": int(
                result.valid_mask.sum()
            ),
            "noise_power": finite_statistics(
                result.noise_power,
                result.noise_lut_valid_mask,
            ),
            "calibrated_linear": (
                finite_statistics(
                    result.calibrated_linear,
                    result.valid_mask,
                )
            ),
            "calibrated_db": (
                finite_statistics(
                    result.calibrated_db,
                    result.valid_mask,
                )
            ),
        },
        "validation": validation,
        "scope": {
            "source_window_only": True,
            "tile_reprojection_performed": False,
            "time_series_updated": False,
            "independent_reference_comparison_performed": False,
        },
        "outputs": {
            "npz": str(npz_path),
            "json": str(json_path),
        },
    }

    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(
        "Noise format: "
        f"{metadata.noise.format_name}"
    )
    print(
        "Noise anchor: "
        f"line={anchor.line}, "
        f"pixel={anchor.pixel}, "
        f"power={anchor.noise_power}"
    )
    print(
        "Image shape:  "
        f"{image_shape}"
    )
    print(
        "Window:       "
        f"rows {window.row_start}:"
        f"{window.row_stop}, "
        f"columns {window.column_start}:"
        f"{window.column_stop}"
    )
    print(
        "Source valid: "
        f"{int(source_valid_mask.sum())}"
    )
    print(
        "Noise valid:  "
        f"{int(source_noise_valid.sum())}"
    )
    print(
        "Missing noise:"
        f" {missing_noise_count}"
    )
    print(
        "Final valid:  "
        f"{int(result.valid_mask.sum())}"
    )
    print(
        "dB stats:     "
        f"{summary['processing']['calibrated_db']}"
    )
    print(f"NPZ:          {npz_path}")
    print(f"JSON:         {json_path}")
    print()
    print(
        "Real positive-noise source-window "
        "radiometry validation passed."
    )


if __name__ == "__main__":
    main()

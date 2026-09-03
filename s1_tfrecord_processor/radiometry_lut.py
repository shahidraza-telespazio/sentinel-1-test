from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import numpy as np

from .radiometry_metadata import (
    CalibrationMetadata,
    CalibrationVector,
    LegacyNoiseVector,
    NoiseMetadata,
    NoiseRangeVector,
    RadiometryMetadata,
    load_radiometry_metadata,
)


CALIBRATION_LUT_NAMES = (
    "sigmaNought",
    "betaNought",
    "gamma",
    "dn",
)


@dataclass(frozen=True)
class RasterWindow:
    row_start: int
    row_stop: int
    column_start: int
    column_stop: int

    @property
    def height(self) -> int:
        return self.row_stop - self.row_start

    @property
    def width(self) -> int:
        return self.column_stop - self.column_start

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width


class SparseVector(Protocol):
    line: int
    pixels: np.ndarray


def validate_image_shape(
    image_shape: tuple[int, int],
) -> None:
    height, width = image_shape

    if height <= 0 or width <= 0:
        raise ValueError(
            "Image height and width must be positive."
        )


def normalise_window(
    image_shape: tuple[int, int],
    window: RasterWindow | None,
) -> RasterWindow:
    validate_image_shape(image_shape)
    height, width = image_shape

    if window is None:
        return RasterWindow(
            row_start=0,
            row_stop=height,
            column_start=0,
            column_stop=width,
        )

    if not (
        0 <= window.row_start
        < window.row_stop
        <= height
    ):
        raise ValueError(
            "Window rows are outside the image: "
            f"{window}, image_shape={image_shape}"
        )

    if not (
        0 <= window.column_start
        < window.column_stop
        <= width
    ):
        raise ValueError(
            "Window columns are outside the image: "
            f"{window}, image_shape={image_shape}"
        )

    return window


def require_coordinate_coverage(
    *,
    vector_lines: np.ndarray,
    vectors: Iterable[SparseVector],
    window: RasterWindow,
    label: str,
) -> None:
    requested_first_row = window.row_start
    requested_last_row = window.row_stop - 1

    if (
        requested_first_row < int(vector_lines[0])
        or requested_last_row > int(vector_lines[-1])
    ):
        raise ValueError(
            f"{label} lines do not cover requested rows "
            f"{requested_first_row}..{requested_last_row}; "
            f"available range is "
            f"{int(vector_lines[0])}.."
            f"{int(vector_lines[-1])}."
        )

    requested_first_column = window.column_start
    requested_last_column = window.column_stop - 1

    for vector in vectors:
        first_pixel = int(vector.pixels[0])
        last_pixel = int(vector.pixels[-1])

        if (
            requested_first_column < first_pixel
            or requested_last_column > last_pixel
        ):
            raise ValueError(
                f"{label} vector at line {vector.line} "
                "does not cover requested columns "
                f"{requested_first_column}.."
                f"{requested_last_column}; "
                f"available range is "
                f"{first_pixel}..{last_pixel}."
            )


def interpolate_sparse_vector_window(
    *,
    vectors: tuple[SparseVector, ...],
    values_for_vector: Callable[
        [SparseVector],
        np.ndarray,
    ],
    image_shape: tuple[int, int],
    window: RasterWindow | None = None,
    label: str,
    output_dtype: np.dtype[Any] = np.dtype(
        np.float32
    ),
) -> np.ndarray:
    """
    Interpolate sparse row/column LUT vectors over a requested image
    window.

    Interpolation is linear along the pixel coordinates of each source
    vector and then linear between the source-vector line coordinates.
    No extrapolation is permitted.
    """
    if not vectors:
        raise ValueError(
            f"No {label} vectors were supplied."
        )

    resolved_window = normalise_window(
        image_shape,
        window,
    )

    vector_lines = np.asarray(
        [vector.line for vector in vectors],
        dtype=np.int64,
    )

    if (
        vector_lines.size > 1
        and not np.all(
            np.diff(vector_lines) > 0
        )
    ):
        raise ValueError(
            f"{label} vector lines must be "
            "strictly increasing."
        )

    require_coordinate_coverage(
        vector_lines=vector_lines,
        vectors=vectors,
        window=resolved_window,
        label=label,
    )

    requested_columns = np.arange(
        resolved_window.column_start,
        resolved_window.column_stop,
        dtype=np.float64,
    )

    horizontally_interpolated = np.empty(
        (
            len(vectors),
            resolved_window.width,
        ),
        dtype=np.float64,
    )

    for vector_index, vector in enumerate(
        vectors
    ):
        source_values = np.asarray(
            values_for_vector(vector),
            dtype=np.float64,
        )

        if source_values.shape != vector.pixels.shape:
            raise ValueError(
                f"{label} vector at line {vector.line} "
                "has mismatched pixel and value arrays: "
                f"{vector.pixels.shape} versus "
                f"{source_values.shape}."
            )

        if not np.isfinite(source_values).all():
            raise ValueError(
                f"{label} vector at line {vector.line} "
                "contains non-finite values."
            )

        horizontally_interpolated[
            vector_index
        ] = np.interp(
            requested_columns,
            vector.pixels.astype(
                np.float64,
                copy=False,
            ),
            source_values,
        )

    requested_rows = np.arange(
        resolved_window.row_start,
        resolved_window.row_stop,
        dtype=np.int64,
    )

    upper_indices = np.searchsorted(
        vector_lines,
        requested_rows,
        side="right",
    )
    upper_indices = np.clip(
        upper_indices,
        1,
        len(vector_lines) - 1,
    )
    lower_indices = upper_indices - 1

    lower_lines = vector_lines[
        lower_indices
    ].astype(np.float64)
    upper_lines = vector_lines[
        upper_indices
    ].astype(np.float64)

    denominator = upper_lines - lower_lines

    if np.any(denominator <= 0.0):
        raise ValueError(
            f"{label} interpolation encountered "
            "non-positive line spacing."
        )

    row_weights = (
        requested_rows.astype(np.float64)
        - lower_lines
    ) / denominator

    lower_values = horizontally_interpolated[
        lower_indices
    ]
    upper_values = horizontally_interpolated[
        upper_indices
    ]

    result = (
        lower_values
        + row_weights[:, np.newaxis]
        * (upper_values - lower_values)
    )

    return result.astype(
        output_dtype,
        copy=False,
    )


def calibration_values(
    vector: CalibrationVector,
    lut_name: str,
) -> np.ndarray:
    if lut_name == "sigmaNought":
        return vector.sigma_nought

    if lut_name == "betaNought":
        return vector.beta_nought

    if lut_name == "gamma":
        return vector.gamma

    if lut_name == "dn":
        return vector.dn

    raise ValueError(
        "Unsupported calibration LUT "
        f"{lut_name!r}. Expected one of "
        f"{CALIBRATION_LUT_NAMES}."
    )


def interpolate_calibration_lut(
    metadata: CalibrationMetadata,
    *,
    lut_name: str,
    image_shape: tuple[int, int],
    window: RasterWindow | None = None,
) -> np.ndarray:
    return interpolate_sparse_vector_window(
        vectors=metadata.vectors,
        values_for_vector=lambda vector: (
            calibration_values(
                vector,
                lut_name,
            )
        ),
        image_shape=image_shape,
        window=window,
        label=f"calibration {lut_name}",
    )


def interpolate_legacy_noise_lut(
    metadata: NoiseMetadata,
    *,
    image_shape: tuple[int, int],
    window: RasterWindow | None = None,
) -> np.ndarray:
    if not metadata.legacy_vectors:
        raise ValueError(
            "Noise metadata does not use the "
            "legacy noiseVector format."
        )

    return interpolate_sparse_vector_window(
        vectors=metadata.legacy_vectors,
        values_for_vector=lambda vector: (
            vector.noise_lut
        ),
        image_shape=image_shape,
        window=window,
        label="legacy noise",
    )


def interpolate_range_noise_lut(
    metadata: NoiseMetadata,
    *,
    image_shape: tuple[int, int],
    window: RasterWindow | None = None,
) -> np.ndarray:
    if not metadata.range_vectors:
        raise ValueError(
            "Noise metadata contains no "
            "noiseRangeVector values."
        )

    return interpolate_sparse_vector_window(
        vectors=metadata.range_vectors,
        values_for_vector=lambda vector: (
            vector.noise_range_lut
        ),
        image_shape=image_shape,
        window=window,
        label="range noise",
    )


def interpolate_one_azimuth_section(
    *,
    lines: np.ndarray,
    values: np.ndarray,
    requested_rows: np.ndarray,
) -> np.ndarray:
    if requested_rows.size == 0:
        return np.empty(
            (0,),
            dtype=np.float32,
        )

    first_requested = int(
        requested_rows[0]
    )
    last_requested = int(
        requested_rows[-1]
    )

    if (
        first_requested < int(lines[0])
        or last_requested > int(lines[-1])
    ):
        raise ValueError(
            "Azimuth-noise LUT lines do not "
            "cover the requested row range "
            f"{first_requested}..{last_requested}; "
            f"available range is "
            f"{int(lines[0])}.."
            f"{int(lines[-1])}."
        )

    return np.interp(
        requested_rows.astype(np.float64),
        lines.astype(np.float64),
        values.astype(np.float64),
    ).astype(np.float32)


def interpolate_azimuth_noise_lut(
    metadata: NoiseMetadata,
    *,
    image_shape: tuple[int, int],
    window: RasterWindow | None = None,
    require_full_coverage: bool = True,
) -> np.ndarray:
    """
    Interpolate the newer azimuth-noise LUT sections independently.

    The returned array is the azimuth LUT field only. It is deliberately
    not combined with the range-noise field in this module.
    """
    if not metadata.azimuth_vectors:
        raise ValueError(
            "Noise metadata contains no "
            "noiseAzimuthVector values."
        )

    resolved_window = normalise_window(
        image_shape,
        window,
    )

    requested_rows = np.arange(
        resolved_window.row_start,
        resolved_window.row_stop,
        dtype=np.int64,
    )
    requested_columns = np.arange(
        resolved_window.column_start,
        resolved_window.column_stop,
        dtype=np.int64,
    )

    result = np.full(
        resolved_window.shape,
        np.nan,
        dtype=np.float32,
    )

    coverage_count = np.zeros(
        resolved_window.shape,
        dtype=np.uint8,
    )

    for section in metadata.azimuth_vectors:
        row_mask = (
            requested_rows
            >= section.first_azimuth_line
        ) & (
            requested_rows
            <= section.last_azimuth_line
        )
        column_mask = (
            requested_columns
            >= section.first_range_sample
        ) & (
            requested_columns
            <= section.last_range_sample
        )

        if not row_mask.any() or not column_mask.any():
            continue

        section_rows = requested_rows[
            row_mask
        ]
        row_values = (
            interpolate_one_azimuth_section(
                lines=section.lines,
                values=section.noise_azimuth_lut,
                requested_rows=section_rows,
            )
        )

        row_indices = np.flatnonzero(
            row_mask
        )
        column_indices = np.flatnonzero(
            column_mask
        )

        target = np.ix_(
            row_indices,
            column_indices,
        )

        existing = result[target]
        new_values = np.broadcast_to(
            row_values[:, np.newaxis],
            existing.shape,
        )

        overlap = np.isfinite(existing)

        if (
            overlap.any()
            and not np.allclose(
                existing[overlap],
                new_values[overlap],
                rtol=1e-6,
                atol=1e-7,
            )
        ):
            raise ValueError(
                "Overlapping azimuth-noise "
                "sections contain conflicting "
                "values."
            )

        result[target] = new_values
        coverage_count[target] += 1

    if np.any(coverage_count > 1):
        # Identical overlaps are tolerated but recorded by this check;
        # conflicting overlaps have already raised above.
        pass

    if (
        require_full_coverage
        and not np.isfinite(result).all()
    ):
        missing_count = int(
            np.count_nonzero(
                ~np.isfinite(result)
            )
        )
        raise ValueError(
            "Azimuth-noise sections do not cover "
            "the complete requested window; "
            f"{missing_count} pixels are uncovered."
        )

    return result


def array_summary(
    values: np.ndarray,
) -> dict[str, Any]:
    finite = np.isfinite(values)
    finite_values = values[finite]

    return {
        "shape": [
            int(value)
            for value in values.shape
        ],
        "dtype": str(values.dtype),
        "finite_count": int(
            finite.sum()
        ),
        "non_finite_count": int(
            values.size - finite.sum()
        ),
        "minimum": (
            float(finite_values.min())
            if finite_values.size
            else None
        ),
        "maximum": (
            float(finite_values.max())
            if finite_values.size
            else None
        ),
        "mean": (
            float(finite_values.mean())
            if finite_values.size
            else None
        ),
    }


def interpolate_radiometry_window(
    metadata: RadiometryMetadata,
    *,
    image_shape: tuple[int, int],
    window: RasterWindow,
    calibration_lut_name: str,
) -> dict[str, np.ndarray]:
    result = {
        "calibration_lut": (
            interpolate_calibration_lut(
                metadata.calibration,
                lut_name=calibration_lut_name,
                image_shape=image_shape,
                window=window,
            )
        )
    }

    if metadata.noise.legacy_vectors:
        result["legacy_noise_lut"] = (
            interpolate_legacy_noise_lut(
                metadata.noise,
                image_shape=image_shape,
                window=window,
            )
        )
    else:
        result["range_noise_lut"] = (
            interpolate_range_noise_lut(
                metadata.noise,
                image_shape=image_shape,
                window=window,
            )
        )

        if metadata.noise.azimuth_vectors:
            result["azimuth_noise_lut"] = (
                interpolate_azimuth_noise_lut(
                    metadata.noise,
                    image_shape=image_shape,
                    window=window,
                )
            )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interpolate Sentinel-1 calibration "
            "and noise LUTs over a diagnostic "
            "source-image window. Measurement "
            "pixels are not read or modified."
        )
    )
    parser.add_argument(
        "--product-xml",
        required=True,
    )
    parser.add_argument(
        "--calibration-xml",
        required=True,
    )
    parser.add_argument(
        "--noise-xml",
        required=True,
    )
    parser.add_argument(
        "--polarisation",
        required=True,
    )
    parser.add_argument(
        "--image-height",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--image-width",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--row-start",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--row-stop",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--column-start",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--column-stop",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--calibration-lut",
        choices=CALIBRATION_LUT_NAMES,
        default="sigmaNought",
    )
    parser.add_argument(
        "--output-json",
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_shape = (
        args.image_height,
        args.image_width,
    )
    window = RasterWindow(
        row_start=args.row_start,
        row_stop=args.row_stop,
        column_start=args.column_start,
        column_stop=args.column_stop,
    )

    metadata = load_radiometry_metadata(
        product_xml=args.product_xml,
        calibration_xml=args.calibration_xml,
        noise_xml=args.noise_xml,
        expected_polarisation=(
            args.polarisation
        ),
    )

    arrays = interpolate_radiometry_window(
        metadata,
        image_shape=image_shape,
        window=window,
        calibration_lut_name=(
            args.calibration_lut
        ),
    )

    summary = {
        "polarisation": (
            args.polarisation.upper()
        ),
        "image_shape": [
            args.image_height,
            args.image_width,
        ],
        "window": {
            "row_start": window.row_start,
            "row_stop": window.row_stop,
            "column_start": (
                window.column_start
            ),
            "column_stop": (
                window.column_stop
            ),
            "shape": list(window.shape),
        },
        "calibration_lut": (
            args.calibration_lut
        ),
        "noise_format": (
            metadata.noise.format_name
        ),
        "arrays": {
            name: array_summary(values)
            for name, values in arrays.items()
        },
        "processing_status": {
            "lut_interpolation_complete": True,
            "noise_lut_combination_complete": False,
            "measurement_pixels_read": False,
            "noise_removal_complete": False,
            "radiometric_calibration_complete": False,
            "db_conversion_complete": False,
        },
    }

    print(
        "Sentinel-1 radiometry LUT interpolation"
    )
    print("--------------------------------------")
    print(
        "Polarisation: "
        f"{summary['polarisation']}"
    )
    print(
        "Image shape:  "
        f"{tuple(summary['image_shape'])}"
    )
    print(
        "Window shape: "
        f"{window.shape}"
    )
    print(
        "Noise format: "
        f"{summary['noise_format']}"
    )

    for name, values in arrays.items():
        stats = summary["arrays"][name]
        print()
        print(name)
        print(
            "  shape: "
            f"{tuple(stats['shape'])}"
        )
        print(
            "  min:   "
            f"{stats['minimum']}"
        )
        print(
            "  max:   "
            f"{stats['maximum']}"
        )
        print(
            "  mean:  "
            f"{stats['mean']}"
        )

    print()
    print(
        "No measurement pixels were read "
        "or modified."
    )

    if args.output_json:
        output_path = Path(
            args.output_json
        )
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"JSON: {output_path}")


if __name__ == "__main__":
    main()

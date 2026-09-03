from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .calibrated_raster import (
    warp_calibrated_asset_to_grid,
)
from .extract_dynamic_scene import (
    find_item,
    measurement_assets,
    metadata_assets,
    target_grid,
)
from .inspect_metadata_assets import build_s3_client
from .raster_io import (
    aws_rasterio_environment,
    build_xy_coordinates,
)


DEFAULT_OUTPUT_DIR = (
    "reports/s1_development/calibrated_tiles"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate full-source Sentinel-1 GRD calibration followed "
            "by GCP average reprojection onto one exact Sentinel-2 tile."
        )
    )
    parser.add_argument(
        "--threshold-report",
        required=True,
    )
    parser.add_argument(
        "--item-id",
        required=True,
    )
    parser.add_argument(
        "--polarisation",
        required=True,
    )
    parser.add_argument(
        "--calibration-lut",
        choices=(
            "sigmaNought",
            "betaNought",
            "gamma",
        ),
        default="sigmaNought",
    )
    parser.add_argument(
        "--unknown-noise-policy",
        choices=(
            "error",
            "assume_uncorrected",
            "assume_corrected",
        ),
        default="error",
    )
    parser.add_argument(
        "--rows-per-window",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--temp-directory",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--aws-profile",
        default=None,
    )
    parser.add_argument(
        "--aws-region",
        default="eu-central-1",
    )

    args = parser.parse_args()

    if args.rows_per_window <= 0:
        parser.error(
            "--rows-per-window must be greater than zero"
        )

    if args.num_threads <= 0:
        parser.error(
            "--num-threads must be greater than zero"
        )

    return args


def main() -> None:
    args = parse_args()
    polarisation = args.polarisation.upper()

    report_path = Path(
        args.threshold_report
    )
    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    item = find_item(
        report,
        args.item_id,
    )
    measurements = measurement_assets(
        item
    )

    if polarisation not in measurements:
        raise RuntimeError(
            f"{args.item_id} has no {polarisation} measurement. "
            f"Available: {list(measurements)}"
        )

    metadata = metadata_assets(
        item,
        polarisation,
    )

    destination_crs, destination_transform, shape = (
        target_grid(report)
    )

    x, y = build_xy_coordinates(
        destination_transform,
        int(shape[0]),
        int(shape[1]),
    )

    s3_client = build_s3_client(
        profile=args.aws_profile,
        region=args.aws_region,
    )

    print(
        "Sentinel-1 calibrated tile validation"
    )
    print(
        "--------------------------------------"
    )
    print(f"Item:          {args.item_id}")
    print(f"Polarisation:  {polarisation}")
    print(
        f"Calibration:   {args.calibration_lut}"
    )
    print(f"Target shape:  {shape}")
    print(
        "Processing:    source-domain radiometry "
        "-> linear average warp -> dB"
    )

    with aws_rasterio_environment(
        profile=args.aws_profile,
        region=args.aws_region,
    ):
        result = (
            warp_calibrated_asset_to_grid(
                measurement_href=(
                    measurements[polarisation]
                ),
                product_metadata_href=(
                    metadata["product"]
                ),
                calibration_metadata_href=(
                    metadata["calibration"]
                ),
                noise_metadata_href=(
                    metadata["noise"]
                ),
                destination_crs=destination_crs,
                destination_transform=(
                    destination_transform
                ),
                destination_shape=shape,
                polarisation=polarisation,
                s3_client=s3_client,
                calibration_lut_name=(
                    args.calibration_lut
                ),
                unknown_noise_policy=(
                    args.unknown_noise_policy
                ),
                rows_per_window=(
                    args.rows_per_window
                ),
                num_threads=args.num_threads,
                temp_directory=(
                    args.temp_directory
                ),
            )
        )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"{args.item_id}_"
        f"{polarisation}_"
        f"{args.calibration_lut}_calibrated_tile"
    )

    npz_path = output_dir / f"{stem}.npz"
    json_path = output_dir / f"{stem}.json"

    metadata_output = {
        "item_id": args.item_id,
        "polarisation": polarisation,
        "threshold_report": str(report_path),
        "calibration_lut": args.calibration_lut,
        "dimension_order": ["y", "x"],
        "destination_crs": (
            destination_crs.to_string()
        ),
        "destination_transform": [
            float(value)
            for value in tuple(
                destination_transform
            )
        ],
        "result": result.metadata,
    }

    np.savez_compressed(
        npz_path,
        calibrated_db=result.calibrated_db,
        calibrated_linear=(
            result.calibrated_linear
        ),
        valid_mask=result.valid_mask,
        x=x,
        y=y,
        destination_crs=np.asarray(
            destination_crs.to_string()
        ),
        destination_transform=np.asarray(
            tuple(destination_transform),
            dtype=np.float64,
        ),
        metadata_json=np.asarray(
            json.dumps(
                metadata_output,
                sort_keys=True,
            )
        ),
    )

    json_path.write_text(
        json.dumps(
            metadata_output,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    valid = result.valid_mask
    db_values = result.calibrated_db[
        valid
    ]
    linear_values = (
        result.calibrated_linear[valid]
    )

    if db_values.size == 0:
        raise RuntimeError(
            "Calibrated tile contains no valid pixels."
        )

    print()
    print("Result")
    print(
        "Valid pixels:  "
        f"{int(valid.sum())}/{valid.size}"
    )
    print(
        "Valid fraction: "
        f"{float(valid.mean()) * 100.0:.3f}%"
    )
    print(
        "dB stats:      "
        f"min={float(db_values.min()):.6f}, "
        f"max={float(db_values.max()):.6f}, "
        f"mean={float(db_values.mean()):.6f}"
    )
    print(
        "Linear stats:  "
        f"min={float(linear_values.min()):.9f}, "
        f"max={float(linear_values.max()):.9f}, "
        f"mean={float(linear_values.mean()):.9f}"
    )
    print(
        "Noise format:   "
        f"{result.metadata['stage']['noise_format']}"
    )
    print(
        "Noise removed:  "
        f"{result.metadata['stage']['noise_was_subtracted']}"
    )
    print(
        "Source final:   "
        f"{result.metadata['stage']['final_valid_pixel_count']}"
    )
    print(f"NPZ:            {npz_path}")
    print(f"JSON:           {json_path}")
    print()
    print(
        "Full-source calibrated tile validation passed."
    )


if __name__ == "__main__":
    main()

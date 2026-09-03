from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

from .raster_io import (
    aws_rasterio_environment,
    build_xy_coordinates,
    warp_gcp_asset_to_grid,
)


DEFAULT_REPORT = (
    "reports/s1_prototype/"
    "T26SKG_2025-06-01_to_2025-06-30_intersection_report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one qualifying Sentinel-1 VV/VH scene onto the "
            "exact Sentinel-2 target grid."
        )
    )

    parser.add_argument(
        "--intersection-report",
        default=DEFAULT_REPORT,
    )
    parser.add_argument(
        "--item-id",
        default=None,
        help=(
            "Specific accepted Sentinel-1 item. If omitted, the "
            "highest-coverage accepted item is selected."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="reports/s1_prototype",
    )
    parser.add_argument(
        "--aws-profile",
        default=None,
    )
    parser.add_argument(
        "--aws-region",
        default="eu-central-1",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    if args.num_threads <= 0:
        parser.error("--num-threads must be greater than zero")

    return args


def select_candidate(
    report: dict[str, Any],
    item_id: str | None,
) -> dict[str, Any]:
    accepted_items = report.get("accepted_items", [])

    if not accepted_items:
        raise RuntimeError(
            "The intersection report contains no accepted items."
        )

    if item_id is not None:
        for item in accepted_items:
            if item["item_id"] == item_id:
                return item

        available = [
            item["item_id"]
            for item in accepted_items
        ]

        raise RuntimeError(
            f"Requested item {item_id!r} is not present in the "
            f"accepted items. Available items: {available}"
        )

    return max(
        accepted_items,
        key=lambda item: float(
            item["tile_coverage_fraction"]
        ),
    )


def destination_grid(
    report: dict[str, Any],
) -> tuple[CRS, Affine, tuple[int, int]]:
    target = report["target_grid"]

    destination_crs = CRS.from_user_input(
        target["crs"]
    )

    destination_transform = Affine(
        *[
            float(value)
            for value in target["output_transform"][:6]
        ]
    )

    destination_shape = tuple(
        int(value)
        for value in target["output_shape"]
    )

    if len(destination_shape) != 2:
        raise RuntimeError(
            f"Unexpected destination shape: {destination_shape}"
        )

    return (
        destination_crs,
        destination_transform,
        destination_shape,
    )


def main() -> None:
    args = parse_args()

    report_path = Path(args.intersection_report)

    report = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    candidate = select_candidate(
        report,
        args.item_id,
    )

    (
        destination_crs,
        destination_transform,
        destination_shape,
    ) = destination_grid(report)

    destination_height, destination_width = (
        destination_shape
    )

    x_coordinates, y_coordinates = (
        build_xy_coordinates(
            destination_transform,
            destination_height,
            destination_width,
        )
    )

    print()
    print("Sentinel-1 scene extraction")
    print("---------------------------")
    print(f"Tile:               {report['tile']}")
    print(f"Item:               {candidate['item_id']}")
    print(f"Datetime:           {candidate['datetime_utc']}")
    print(
        "Geometric coverage: "
        f"{candidate['tile_coverage_pct']:.6f}%"
    )
    print(f"Orbit state:        {candidate['orbit_state']}")
    print(f"Relative orbit:     {candidate['relative_orbit']}")
    print(f"Destination CRS:    {destination_crs}")
    print(f"Destination shape:  {destination_shape}")
    print(f"Destination transform:\n{destination_transform}")

    with aws_rasterio_environment(
        profile=args.aws_profile,
        region=args.aws_region,
    ):
        vv = warp_gcp_asset_to_grid(
            href=candidate["vv_href"],
            destination_crs=destination_crs,
            destination_transform=destination_transform,
            destination_shape=destination_shape,
            polarisation="VV",
            num_threads=args.num_threads,
        )

        vh = warp_gcp_asset_to_grid(
            href=candidate["vh_href"],
            destination_crs=destination_crs,
            destination_transform=destination_transform,
            destination_shape=destination_shape,
            polarisation="VH",
            num_threads=args.num_threads,
        )

    joint_valid = (
        vv.valid_mask
        & vh.valid_mask
    )

    either_valid = (
        vv.valid_mask
        | vh.valid_mask
    )

    masks_equal = np.array_equal(
        vv.valid_mask,
        vh.valid_mask,
    )

    joint_valid_fraction = float(
        joint_valid.mean()
    )

    minimum_coverage = float(
        report["requirements"][
            "minimum_tile_coverage_fraction"
        ]
    )

    passes_actual_coverage = (
        joint_valid_fraction
        >= minimum_coverage
    )

    # Shape follows the same spatial/time/band order used by the
    # Sentinel-2 dataset:
    #
    #     time, y, x, band
    #
    # The values are still uncalibrated source measurement values.
    measurement_values = np.stack(
        [vv.values, vh.values],
        axis=-1,
    )[np.newaxis, ...]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_stem = (
        f"{report['tile']}_"
        f"{candidate['item_id']}_"
        "s2_grid_extraction"
    )

    npz_path = output_dir / f"{output_stem}.npz"
    json_path = output_dir / f"{output_stem}.json"

    metadata = {
        "prototype": (
            "Sentinel-1 VV/VH extraction onto Sentinel-2 grid"
        ),
        "tile": report["tile"],
        "item_id": candidate["item_id"],
        "datetime_utc": candidate["datetime_utc"],
        "platform": candidate["platform"],
        "instrument_mode": candidate["instrument_mode"],
        "product_type": candidate["product_type"],
        "orbit_state": candidate["orbit_state"],
        "relative_orbit": candidate["relative_orbit"],
        "absolute_orbit": candidate["absolute_orbit"],
        "polarisations": ["VV", "VH"],
        "value_status": (
            "uncalibrated source measurement values; "
            "linear/dB semantics not yet assumed"
        ),
        "dimensions": {
            "time": 1,
            "y": destination_height,
            "x": destination_width,
            "band": 2,
        },
        "dimension_order": [
            "time",
            "y",
            "x",
            "band",
        ],
        "destination_crs": destination_crs.to_string(),
        "destination_transform": [
            float(value)
            for value in tuple(destination_transform)
        ],
        "geometric_tile_coverage_fraction": float(
            candidate["tile_coverage_fraction"]
        ),
        "geometric_tile_coverage_pct": float(
            candidate["tile_coverage_pct"]
        ),
        "minimum_required_coverage_fraction": (
            minimum_coverage
        ),
        "vv": vv.metadata,
        "vh": vh.metadata,
        "validation": {
            "vv_vh_masks_equal": masks_equal,
            "joint_valid_pixel_count": int(
                joint_valid.sum()
            ),
            "joint_invalid_pixel_count": int(
                joint_valid.size - joint_valid.sum()
            ),
            "joint_valid_fraction": (
                joint_valid_fraction
            ),
            "joint_valid_pct": (
                joint_valid_fraction * 100.0
            ),
            "either_valid_fraction": float(
                either_valid.mean()
            ),
            "passes_actual_coverage": (
                passes_actual_coverage
            ),
            "output_shape_matches_target": (
                measurement_values.shape
                == (
                    1,
                    destination_height,
                    destination_width,
                    2,
                )
            ),
        },
    }

    np.savez_compressed(
        npz_path,
        measurement_values=measurement_values,
        vv=vv.values,
        vh=vh.values,
        vv_valid=vv.valid_mask,
        vh_valid=vh.valid_mask,
        joint_valid=joint_valid,
        x=x_coordinates,
        y=y_coordinates,
        band_name=np.asarray(
            ["VV", "VH"],
            dtype="U2",
        ),
        time=np.asarray(
            [candidate["datetime_utc"]]
        ),
        destination_crs=np.asarray(
            destination_crs.to_string()
        ),
        destination_transform=np.asarray(
            tuple(destination_transform),
            dtype=np.float64,
        ),
        metadata_json=np.asarray(
            json.dumps(
                metadata,
                sort_keys=True,
            )
        ),
    )

    json_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print("===== EXTRACTION VALIDATION =====")
    print(
        "Measurement shape:   "
        f"{measurement_values.shape}"
    )
    print(
        "VV valid coverage:    "
        f"{vv.metadata['valid_pct']:.6f}%"
    )
    print(
        "VH valid coverage:    "
        f"{vh.metadata['valid_pct']:.6f}%"
    )
    print(
        "Joint valid coverage: "
        f"{joint_valid_fraction * 100.0:.6f}%"
    )
    print(
        "VV/VH masks equal:    "
        f"{masks_equal}"
    )
    print(
        "Passes actual 80%:    "
        f"{passes_actual_coverage}"
    )
    print(f"First x coordinate:   {x_coordinates[0]}")
    print(f"Last x coordinate:    {x_coordinates[-1]}")
    print(f"First y coordinate:   {y_coordinates[0]}")
    print(f"Last y coordinate:    {y_coordinates[-1]}")
    print()
    print(f"NPZ output:  {npz_path}")
    print(f"JSON report: {json_path}")

    if not passes_actual_coverage:
        raise RuntimeError(
            "The scene passed geometric filtering but failed "
            "actual joint VV/VH valid-data coverage."
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

from .discovery import KNOWN_POLARISATIONS
from .raster_io import (
    aws_rasterio_environment,
    build_xy_coordinates,
    warp_gcp_asset_to_grid,
)
from .valid_area import warp_gcp_valid_area_fraction_to_grid


DEFAULT_OUTPUT_DIR = "reports/s1_development/scene_extraction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract all available polarisations from one Sentinel-1 "
            "acquisition onto the exact Sentinel-2 target grid."
        )
    )
    parser.add_argument("--threshold-report", required=True)
    parser.add_argument("--item-id", required=True)
    parser.add_argument(
        "--minimum-coverage",
        type=float,
        default=0.80,
        help="Area-weighted valid coverage fraction. Default: 0.80",
    )
    parser.add_argument(
        "--acceptance-rule",
        choices=("report-only", "any", "all"),
        default="report-only",
        help=(
            "How to combine per-polarisation results. The default only "
            "reports them because the final project rule is unresolved."
        ),
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--aws-profile", default=None)
    parser.add_argument("--aws-region", default="eu-central-1")
    parser.add_argument("--num-threads", type=int, default=2)
    args = parser.parse_args()

    if not 0.0 <= args.minimum_coverage <= 1.0:
        parser.error("--minimum-coverage must be between 0 and 1")
    if args.num_threads <= 0:
        parser.error("--num-threads must be greater than zero")

    return args


def find_item(report: dict[str, Any], item_id: str) -> dict[str, Any]:
    matches = [
        item
        for item in report.get("items", [])
        if item.get("item_id") == item_id
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one report item for {item_id!r}; found {len(matches)}"
        )

    item = matches[0]

    if not item.get("eligible_before_coverage", False):
        raise RuntimeError(
            "The requested item is ineligible before coverage filtering: "
            f"{item.get('eligibility_rejection_reasons')}"
        )

    return item


def target_grid(
    report: dict[str, Any],
) -> tuple[CRS, Affine, tuple[int, int]]:
    target = report["target_grid"]
    crs = CRS.from_user_input(target["crs"])
    transform = Affine(
        *[float(value) for value in target["output_transform"][:6]]
    )
    shape = tuple(int(value) for value in target["output_shape"])

    if len(shape) != 2:
        raise RuntimeError(f"Unexpected target shape: {shape}")

    return crs, transform, shape


def report_tile(
    report: dict[str, Any],
    report_path: Path,
) -> str:
    possible_values = [
        report.get("tile"),
    ]

    for section_name in (
        "request",
        "analysis",
        "summary",
    ):
        section = report.get(section_name)

        if isinstance(section, dict):
            possible_values.append(
                section.get("tile")
            )

    for value in possible_values:
        if value is None:
            continue

        tile = str(value).strip().upper()

        if re.fullmatch(
            r"T\d{2}[A-Z]{3}",
            tile,
        ):
            return tile

    # Report directories are named like:
    #
    # T26SKG_2025-06-01_to_2025-06-30
    directory_tile = (
        report_path.parent.name
        .split("_", 1)[0]
        .strip()
        .upper()
    )

    if re.fullmatch(
        r"T\d{2}[A-Z]{3}",
        directory_tile,
    ):
        return directory_tile

    raise RuntimeError(
        "Could not determine the Sentinel-2 tile "
        f"from report: {report_path}"
    )


def polarisation_order(value: str) -> tuple[int, str]:
    try:
        return KNOWN_POLARISATIONS.index(value), value
    except ValueError:
        return len(KNOWN_POLARISATIONS), value


def measurement_assets(item: dict[str, Any]) -> dict[str, str]:
    raw_assets = item.get("measurement_assets")

    if not isinstance(raw_assets, dict):
        raise RuntimeError(
            "The threshold report has no polarisation-keyed "
            "measurement_assets for the requested item."
        )

    assets = {
        str(polarisation).upper(): str(href)
        for polarisation, href in raw_assets.items()
        if href
    }

    if not assets:
        raise RuntimeError("The requested item has no measurement assets.")

    return dict(
        sorted(
            assets.items(),
            key=lambda entry: polarisation_order(entry[0]),
        )
    )


def metadata_assets(
    item: dict[str, Any],
    polarisation: str,
) -> dict[str, str | None]:
    all_metadata = item.get("metadata_assets", {})
    value = (
        all_metadata.get(polarisation, {})
        if isinstance(all_metadata, dict)
        else {}
    )

    return {
        "product": value.get("product") if isinstance(value, dict) else None,
        "calibration": (
            value.get("calibration") if isinstance(value, dict) else None
        ),
        "noise": value.get("noise") if isinstance(value, dict) else None,
    }


def decide_scene_acceptance(
    per_polarisation_pass: dict[str, bool],
    rule: str,
) -> bool | None:
    if rule == "report-only":
        return None
    if rule == "any":
        return any(per_polarisation_pass.values())
    if rule == "all":
        return all(per_polarisation_pass.values())
    raise ValueError(f"Unsupported acceptance rule: {rule}")


def main() -> None:
    args = parse_args()
    report_path = Path(args.threshold_report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tile = report_tile(
        report,
        report_path,
    )
    item = find_item(report, args.item_id)
    assets = measurement_assets(item)
    crs, transform, shape = target_grid(report)
    height, width = shape
    x, y = build_xy_coordinates(transform, height, width)

    print("Sentinel-1 dynamic scene extraction")
    print("-----------------------------------")
    print(f"Tile:               {tile}")
    print(f"Item:               {item['item_id']}")
    print(f"Polarisations:      {list(assets)}")
    print(f"Geometric coverage: {item['tile_coverage_pct']:.6f}%")
    print(f"Valid threshold:    {args.minimum_coverage * 100:.2f}%")
    print(f"Acceptance rule:    {args.acceptance_rule}")

    warped: dict[str, Any] = {}
    valid_area: dict[str, Any] = {}

    with aws_rasterio_environment(
        profile=args.aws_profile,
        region=args.aws_region,
    ):
        for polarisation, href in assets.items():
            print(f"Processing {polarisation}")
            warped[polarisation] = warp_gcp_asset_to_grid(
                href=href,
                destination_crs=crs,
                destination_transform=transform,
                destination_shape=shape,
                polarisation=polarisation,
                num_threads=args.num_threads,
            )
            valid_area[polarisation] = (
                warp_gcp_valid_area_fraction_to_grid(
                    href=href,
                    destination_crs=crs,
                    destination_transform=transform,
                    destination_shape=shape,
                    polarisation=polarisation,
                    num_threads=args.num_threads,
                )
            )

    polarisations = list(assets)
    values = np.stack(
        [warped[p].values for p in polarisations],
        axis=-1,
    )[np.newaxis, ...]
    valid_masks = np.stack(
        [warped[p].valid_mask for p in polarisations],
        axis=-1,
    )[np.newaxis, ...]
    valid_area_fractions = np.stack(
        [valid_area[p].fraction_grid for p in polarisations],
        axis=-1,
    )[np.newaxis, ...]

    per_polarisation: dict[str, Any] = {}
    per_polarisation_pass: dict[str, bool] = {}

    for index, polarisation in enumerate(polarisations):
        occupancy = float(valid_masks[0, :, :, index].mean())
        area_fraction = float(
            valid_area_fractions[0, :, :, index].mean()
        )
        fully_valid = float(
            np.mean(valid_area_fractions[0, :, :, index] >= 1.0 - 1e-6)
        )
        passes = area_fraction >= args.minimum_coverage
        per_polarisation_pass[polarisation] = passes
        per_polarisation[polarisation] = {
            "measurement_href": assets[polarisation],
            "metadata_assets": metadata_assets(item, polarisation),
            "output_pixel_occupancy_fraction": occupancy,
            "output_pixel_occupancy_pct": occupancy * 100.0,
            "area_weighted_valid_fraction": area_fraction,
            "area_weighted_valid_pct": area_fraction * 100.0,
            "fully_valid_output_pixel_fraction": fully_valid,
            "fully_valid_output_pixel_pct": fully_valid * 100.0,
            "passes_coverage_threshold": passes,
            "measurement_warp": warped[polarisation].metadata,
            "valid_area_warp": valid_area[polarisation].metadata,
        }

    occupancy_masks = [warped[p].valid_mask for p in polarisations]
    joint_occupancy = np.logical_and.reduce(occupancy_masks)
    either_occupancy = np.logical_or.reduce(occupancy_masks)

    area_grids = [valid_area[p].fraction_grid for p in polarisations]
    area_grids_equal = all(
        np.allclose(area_grids[0], grid, rtol=0.0, atol=1e-6)
        for grid in area_grids[1:]
    )
    exact_joint_area = (
        float(area_grids[0].mean()) if area_grids_equal else None
    )
    accepted = decide_scene_acceptance(
        per_polarisation_pass,
        args.acceptance_rule,
    )

    metadata = {
        "tile": tile,
        "item_id": item["item_id"],
        "datetime_utc": item.get("datetime_utc"),
        "platform": item.get("platform"),
        "instrument_mode": item.get("instrument_mode"),
        "product_type": item.get("product_type"),
        "orbit_state": item.get("orbit_state"),
        "relative_orbit": item.get("relative_orbit"),
        "absolute_orbit": item.get("absolute_orbit"),
        "polarisations": polarisations,
        "value_status": (
            "uncalibrated source measurement values; not yet noise "
            "corrected, radiometrically calibrated or converted to dB"
        ),
        "dimension_order": ["time", "y", "x", "band"],
        "dimensions": {
            "time": 1,
            "y": height,
            "x": width,
            "band": len(polarisations),
        },
        "destination_crs": crs.to_string(),
        "destination_transform": [float(v) for v in tuple(transform)],
        "geometric_tile_coverage_fraction": float(
            item["tile_coverage_fraction"]
        ),
        "minimum_valid_coverage_fraction": args.minimum_coverage,
        "per_polarisation": per_polarisation,
        "validation": {
            "per_polarisation_pass": per_polarisation_pass,
            "any_available_polarisation_pass": any(
                per_polarisation_pass.values()
            ),
            "all_available_polarisations_pass": all(
                per_polarisation_pass.values()
            ),
            "acceptance_rule": args.acceptance_rule,
            "scene_accepted": accepted,
            "joint_output_pixel_occupancy_fraction": float(
                joint_occupancy.mean()
            ),
            "either_output_pixel_occupancy_fraction": float(
                either_occupancy.mean()
            ),
            "all_area_fraction_grids_equal": area_grids_equal,
            "exact_joint_area_weighted_valid_fraction": exact_joint_area,
            "output_shape_matches_target": values.shape
            == (1, height, width, len(polarisations)),
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{tile}_{item['item_id']}_"
        "dynamic_s2_grid_extraction"
    )
    npz_path = output_dir / f"{stem}.npz"
    json_path = output_dir / f"{stem}.json"

    np.savez_compressed(
        npz_path,
        measurement_values=values,
        valid_mask_values=valid_masks,
        valid_area_fraction_values=valid_area_fractions,
        x=x,
        y=y,
        band_name=np.asarray(polarisations, dtype="U2"),
        time=np.asarray([item.get("datetime_utc")]),
        destination_crs=np.asarray(crs.to_string()),
        destination_transform=np.asarray(tuple(transform), dtype=np.float64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    json_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print()
    print("Coverage results")
    for polarisation, result in per_polarisation.items():
        print(
            f"  {polarisation}: "
            f"{result['area_weighted_valid_pct']:.6f}% "
            f"pass={result['passes_coverage_threshold']}"
        )
    print(f"Output shape: {values.shape}")
    print(f"NPZ:  {npz_path}")
    print(f"JSON: {json_path}")

    if accepted is False:
        raise RuntimeError(
            "The scene failed the selected acceptance rule. Outputs "
            "were retained for inspection."
        )


if __name__ == "__main__":
    main()

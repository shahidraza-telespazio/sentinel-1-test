from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .calibrated_raster import (
    warp_calibrated_asset_to_grid,
)
from .discovery import KNOWN_POLARISATIONS
from .extract_dynamic_scene import (
    decide_scene_acceptance,
    measurement_assets,
    metadata_assets,
    polarisation_order,
    report_tile,
    target_grid,
)
from .inspect_metadata_assets import build_s3_client
from .raster_io import (
    aws_rasterio_environment,
    build_xy_coordinates,
)
from .valid_area import (
    warp_gcp_valid_area_fraction_to_grid,
)


DEFAULT_OUTPUT_DIR = (
    "reports/s1_development/calibrated_time_series"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a calibrated Sentinel-1 GRD tile time series on the "
            "exact Sentinel-2 target grid. Radiometry is applied in the "
            "source image domain, calibrated linear power is "
            "average-reprojected, and dB conversion is applied after "
            "reprojection."
        )
    )
    parser.add_argument(
        "--threshold-report",
        required=True,
    )
    parser.add_argument(
        "--minimum-geometric-coverage",
        type=float,
        default=0.80,
        help=(
            "Fast STAC-footprint prefilter. Default: 0.80"
        ),
    )
    parser.add_argument(
        "--minimum-valid-coverage",
        type=float,
        default=0.80,
        help=(
            "Final per-polarisation area-weighted source-valid "
            "coverage. Default: 0.80"
        ),
    )
    parser.add_argument(
        "--acceptance-rule",
        choices=("any", "all"),
        default="all",
        help=(
            "Rule for combining per-polarisation source-valid coverage. "
            "Default: all. This remains a stakeholder decision."
        ),
    )
    parser.add_argument(
        "--band-layout",
        choices=("union", "canonical"),
        default="union",
        help=(
            "union uses polarisations found in accepted scenes; "
            "canonical always uses VV,VH,HH,HV. Missing bands are NaN."
        ),
    )
    parser.add_argument(
        "--calibration-lut",
        choices=(
            "sigmaNought",
            "betaNought",
            "gamma",
        ),
        default="sigmaNought",
        help=(
            "Radiometric calibration LUT. sigmaNought is the current "
            "development default; the final project quantity remains "
            "a stakeholder decision."
        ),
    )
    parser.add_argument(
        "--unknown-noise-policy",
        choices=(
            "error",
            "assume_uncorrected",
            "assume_corrected",
        ),
        default="error",
        help=(
            "How to handle products without an explicit thermal-noise "
            "correction flag. Default: error."
        ),
    )
    parser.add_argument(
        "--rows-per-window",
        type=int,
        default=256,
        help=(
            "Source rows calibrated at a time. Default: 256."
        ),
    )
    parser.add_argument(
        "--item-id",
        action="append",
        default=None,
        help=(
            "Process only this item ID. May be supplied more than once."
        ),
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help=(
            "Optional diagnostic cap after geometric filtering."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--temp-directory",
        default=None,
        help=(
            "Optional directory for staged calibrated linear GeoTIFFs."
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
    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    for name in (
        "minimum_geometric_coverage",
        "minimum_valid_coverage",
    ):
        value = getattr(args, name)

        if not 0.0 <= value <= 1.0:
            parser.error(
                f"--{name.replace('_', '-')} "
                "must be between 0 and 1"
            )

    if (
        args.max_scenes is not None
        and args.max_scenes <= 0
    ):
        parser.error(
            "--max-scenes must be greater than zero"
        )

    if args.rows_per_window <= 0:
        parser.error(
            "--rows-per-window must be greater than zero"
        )

    if args.num_threads <= 0:
        parser.error(
            "--num-threads must be greater than zero"
        )

    return args


def candidate_items(
    report: dict[str, Any],
    *,
    minimum_geometric_coverage: float,
    requested_item_ids: list[str] | None,
    max_scenes: int | None,
) -> list[dict[str, Any]]:
    requested = set(
        requested_item_ids or []
    )

    if requested:
        available = {
            str(item.get("item_id"))
            for item in report.get(
                "items",
                [],
            )
        }
        missing = sorted(
            requested.difference(available)
        )

        if missing:
            raise RuntimeError(
                "Requested item IDs are absent from the report: "
                + ", ".join(missing)
            )

    items = [
        item
        for item in report.get(
            "items",
            [],
        )
        if item.get(
            "eligible_before_coverage",
            False,
        )
        and float(
            item.get(
                "tile_coverage_fraction",
                0.0,
            )
        )
        >= minimum_geometric_coverage
        and (
            not requested
            or str(
                item.get("item_id")
            )
            in requested
        )
    ]

    items.sort(
        key=lambda item: (
            item.get(
                "datetime_utc"
            )
            is None,
            item.get(
                "datetime_utc"
            )
            or "",
            item.get(
                "item_id"
            )
            or "",
        )
    )

    if max_scenes is not None:
        items = items[:max_scenes]

    return items


def process_scene(
    item: dict[str, Any],
    *,
    destination_crs,
    destination_transform,
    destination_shape: tuple[int, int],
    minimum_valid_coverage: float,
    acceptance_rule: str,
    calibration_lut: str,
    unknown_noise_policy: str,
    rows_per_window: int,
    num_threads: int,
    temp_directory: str | None,
    s3_client,
) -> dict[str, Any]:
    assets = measurement_assets(
        item
    )

    values_by_polarisation: dict[
        str,
        np.ndarray,
    ] = {}
    masks_by_polarisation: dict[
        str,
        np.ndarray,
    ] = {}
    area_by_polarisation: dict[
        str,
        np.ndarray,
    ] = {}
    per_polarisation: dict[
        str,
        Any,
    ] = {}
    pass_by_polarisation: dict[
        str,
        bool,
    ] = {}

    for polarisation, href in (
        assets.items()
    ):
        metadata = metadata_assets(
            item,
            polarisation,
        )

        calibrated = (
            warp_calibrated_asset_to_grid(
                measurement_href=href,
                product_metadata_href=(
                    metadata[
                        "product"
                    ]
                ),
                calibration_metadata_href=(
                    metadata[
                        "calibration"
                    ]
                ),
                noise_metadata_href=(
                    metadata[
                        "noise"
                    ]
                ),
                destination_crs=(
                    destination_crs
                ),
                destination_transform=(
                    destination_transform
                ),
                destination_shape=(
                    destination_shape
                ),
                polarisation=(
                    polarisation
                ),
                s3_client=s3_client,
                calibration_lut_name=(
                    calibration_lut
                ),
                unknown_noise_policy=(
                    unknown_noise_policy
                ),
                rows_per_window=(
                    rows_per_window
                ),
                num_threads=(
                    num_threads
                ),
                temp_directory=(
                    temp_directory
                ),
            )
        )

        valid_area = (
            warp_gcp_valid_area_fraction_to_grid(
                href=href,
                destination_crs=(
                    destination_crs
                ),
                destination_transform=(
                    destination_transform
                ),
                destination_shape=(
                    destination_shape
                ),
                polarisation=(
                    polarisation
                ),
                num_threads=(
                    num_threads
                ),
            )
        )

        area_fraction = float(
            valid_area.fraction_grid.mean()
        )
        passes = (
            area_fraction
            >= minimum_valid_coverage
        )

        values_by_polarisation[
            polarisation
        ] = calibrated.calibrated_db
        masks_by_polarisation[
            polarisation
        ] = calibrated.valid_mask
        area_by_polarisation[
            polarisation
        ] = valid_area.fraction_grid
        pass_by_polarisation[
            polarisation
        ] = passes

        calibrated_valid_fraction = (
            float(
                calibrated.valid_mask.mean()
            )
        )

        per_polarisation[
            polarisation
        ] = {
            "measurement_href": href,
            "metadata_assets": metadata,
            "calibration_lut": (
                calibration_lut
            ),
            "output_value_representation": (
                f"{calibration_lut} dB"
            ),
            "calibrated_output_valid_fraction": (
                calibrated_valid_fraction
            ),
            "calibrated_output_valid_pct": (
                calibrated_valid_fraction
                * 100.0
            ),
            "area_weighted_valid_fraction": (
                area_fraction
            ),
            "area_weighted_valid_pct": (
                area_fraction
                * 100.0
            ),
            "fully_valid_output_pixel_fraction": float(
                np.mean(
                    valid_area.fraction_grid
                    >= 1.0 - 1e-6
                )
            ),
            "passes_coverage_threshold": (
                passes
            ),
            "calibrated_measurement_warp": (
                calibrated.metadata
            ),
            "valid_area_warp": (
                valid_area.metadata
            ),
        }

    accepted = (
        decide_scene_acceptance(
            pass_by_polarisation,
            acceptance_rule,
        )
    )

    if accepted is None:
        raise RuntimeError(
            "A time series requires an explicit "
            "any/all acceptance rule."
        )

    return {
        "item": item,
        "polarisations": list(
            assets
        ),
        "values_by_polarisation": (
            values_by_polarisation
        ),
        "masks_by_polarisation": (
            masks_by_polarisation
        ),
        "area_by_polarisation": (
            area_by_polarisation
        ),
        "per_polarisation": (
            per_polarisation
        ),
        "pass_by_polarisation": (
            pass_by_polarisation
        ),
        "accepted": accepted,
    }


def output_bands(
    accepted_scenes: list[
        dict[str, Any]
    ],
    band_layout: str,
) -> list[str]:
    if band_layout == "canonical":
        return list(
            KNOWN_POLARISATIONS
        )

    observed = {
        polarisation
        for scene in accepted_scenes
        for polarisation
        in scene["polarisations"]
    }

    return sorted(
        observed,
        key=polarisation_order,
    )


def main() -> None:
    args = parse_args()

    report_path = Path(
        args.threshold_report
    )
    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )

    tile = report_tile(
        report,
        report_path,
    )

    (
        destination_crs,
        destination_transform,
        shape,
    ) = target_grid(
        report
    )

    height, width = shape

    x, y = build_xy_coordinates(
        destination_transform,
        height,
        width,
    )

    candidates = candidate_items(
        report,
        minimum_geometric_coverage=(
            args.minimum_geometric_coverage
        ),
        requested_item_ids=(
            args.item_id
        ),
        max_scenes=(
            args.max_scenes
        ),
    )

    if not candidates:
        raise RuntimeError(
            "No eligible scenes passed "
            "the geometric prefilter."
        )

    s3_client = build_s3_client(
        profile=args.aws_profile,
        region=args.aws_region,
    )

    print(
        "Sentinel-1 calibrated tile time series"
    )
    print(
        "--------------------------------------"
    )
    print(
        f"Tile:                       {tile}"
    )
    print(
        "Candidate scenes:           "
        f"{len(candidates)}"
    )
    print(
        "Minimum geometric coverage: "
        f"{args.minimum_geometric_coverage * 100:.2f}%"
    )
    print(
        "Minimum valid coverage:     "
        f"{args.minimum_valid_coverage * 100:.2f}%"
    )
    print(
        "Acceptance rule:            "
        f"{args.acceptance_rule}"
    )
    print(
        "Band layout:                "
        f"{args.band_layout}"
    )
    print(
        "Calibration LUT:            "
        f"{args.calibration_lut}"
    )
    print(
        "Unknown noise policy:       "
        f"{args.unknown_noise_policy}"
    )

    processed: list[
        dict[str, Any]
    ] = []

    with aws_rasterio_environment(
        profile=args.aws_profile,
        region=args.aws_region,
    ):
        for index, item in enumerate(
            candidates,
            start=1,
        ):
            print(
                f"[{index}/{len(candidates)}] "
                f"{item['item_id']}"
            )

            result = process_scene(
                item,
                destination_crs=(
                    destination_crs
                ),
                destination_transform=(
                    destination_transform
                ),
                destination_shape=(
                    shape
                ),
                minimum_valid_coverage=(
                    args.minimum_valid_coverage
                ),
                acceptance_rule=(
                    args.acceptance_rule
                ),
                calibration_lut=(
                    args.calibration_lut
                ),
                unknown_noise_policy=(
                    args.unknown_noise_policy
                ),
                rows_per_window=(
                    args.rows_per_window
                ),
                num_threads=(
                    args.num_threads
                ),
                temp_directory=(
                    args.temp_directory
                ),
                s3_client=s3_client,
            )

            processed.append(
                result
            )

            status = (
                "ACCEPT"
                if result["accepted"]
                else "REJECT"
            )

            coverage_text = ", ".join(
                (
                    f"{polarisation}="
                    f"{details['area_weighted_valid_fraction'] * 100:.3f}%"
                    "/"
                    f"{details['calibrated_output_valid_fraction'] * 100:.3f}%"
                )
                for (
                    polarisation,
                    details,
                )
                in result[
                    "per_polarisation"
                ].items()
            )

            print(
                "  "
                f"{status}: "
                f"source-area/calibrated-output "
                f"{coverage_text}"
            )

    accepted_scenes = [
        scene
        for scene in processed
        if scene["accepted"]
    ]

    if not accepted_scenes:
        raise RuntimeError(
            "Every processed scene failed "
            "the selected valid-coverage rule."
        )

    bands = output_bands(
        accepted_scenes,
        args.band_layout,
    )

    band_index = {
        polarisation: index
        for index, polarisation
        in enumerate(bands)
    }

    scene_count = len(
        accepted_scenes
    )

    values = np.full(
        (
            scene_count,
            height,
            width,
            len(bands),
        ),
        np.nan,
        dtype=np.float32,
    )

    valid_masks = np.zeros(
        (
            scene_count,
            height,
            width,
            len(bands),
        ),
        dtype=bool,
    )

    valid_area_fractions = (
        np.full(
            (
                scene_count,
                height,
                width,
                len(bands),
            ),
            np.nan,
            dtype=np.float32,
        )
    )

    band_present = np.zeros(
        (
            scene_count,
            len(bands),
        ),
        dtype=bool,
    )

    band_pass = np.zeros(
        (
            scene_count,
            len(bands),
        ),
        dtype=bool,
    )

    scene_metadata: list[
        dict[str, Any]
    ] = []

    for (
        scene_index,
        scene,
    ) in enumerate(
        accepted_scenes
    ):
        item = scene["item"]

        for polarisation in (
            scene["polarisations"]
        ):
            output_index = (
                band_index[
                    polarisation
                ]
            )

            values[
                scene_index,
                :,
                :,
                output_index,
            ] = (
                scene[
                    "values_by_polarisation"
                ][polarisation]
            )

            valid_masks[
                scene_index,
                :,
                :,
                output_index,
            ] = (
                scene[
                    "masks_by_polarisation"
                ][polarisation]
            )

            valid_area_fractions[
                scene_index,
                :,
                :,
                output_index,
            ] = (
                scene[
                    "area_by_polarisation"
                ][polarisation]
            )

            band_present[
                scene_index,
                output_index,
            ] = True

            band_pass[
                scene_index,
                output_index,
            ] = (
                scene[
                    "pass_by_polarisation"
                ][polarisation]
            )

        scene_metadata.append(
            {
                "item_id": (
                    item.get(
                        "item_id"
                    )
                ),
                "datetime_utc": (
                    item.get(
                        "datetime_utc"
                    )
                ),
                "platform": (
                    item.get(
                        "platform"
                    )
                ),
                "orbit_state": (
                    item.get(
                        "orbit_state"
                    )
                ),
                "relative_orbit": (
                    item.get(
                        "relative_orbit"
                    )
                ),
                "absolute_orbit": (
                    item.get(
                        "absolute_orbit"
                    )
                ),
                "geometric_tile_coverage_fraction": float(
                    item[
                        "tile_coverage_fraction"
                    ]
                ),
                "available_polarisations": (
                    scene[
                        "polarisations"
                    ]
                ),
                "per_polarisation": (
                    scene[
                        "per_polarisation"
                    ]
                ),
            }
        )

    metadata = {
        "product": (
            "calibrated Sentinel-1 "
            "tile time series"
        ),
        "tile": tile,
        "source_threshold_report": str(
            report_path
        ),
        "value_status": (
            f"{args.calibration_lut} backscatter in dB; "
            "thermal-noise handling follows product metadata; "
            "calibration occurs in source-image space; calibrated "
            "linear power is average-reprojected to the Sentinel-2 "
            "target grid; dB conversion occurs after reprojection"
        ),
        "calibration_lut": (
            args.calibration_lut
        ),
        "unknown_noise_policy": (
            args.unknown_noise_policy
        ),
        "processing_order": [
            "detected_magnitude",
            "square_to_source_power",
            "thermal_noise_correction_in_source_domain",
            (
                f"{args.calibration_lut}_linear_"
                "calibration_in_source_domain"
            ),
            "gcp_average_reprojection_of_linear_power",
            "10_log10_after_reprojection",
        ],
        "dimension_order": [
            "time",
            "y",
            "x",
            "band",
        ],
        "dimensions": {
            "time": scene_count,
            "y": height,
            "x": width,
            "band": len(bands),
        },
        "bands": bands,
        "band_layout": (
            args.band_layout
        ),
        "missing_band_representation": (
            "NaN"
        ),
        "minimum_geometric_coverage_fraction": (
            args.minimum_geometric_coverage
        ),
        "minimum_valid_coverage_fraction": (
            args.minimum_valid_coverage
        ),
        "coverage_basis": (
            "area-weighted source-valid coverage; "
            "calibrated-output valid fraction is also recorded "
            "per polarisation"
        ),
        "acceptance_rule": (
            args.acceptance_rule
        ),
        "candidate_scene_count": len(
            candidates
        ),
        "accepted_scene_count": (
            scene_count
        ),
        "rejected_scene_count": (
            len(processed)
            - scene_count
        ),
        "destination_crs": (
            destination_crs.to_string()
        ),
        "destination_transform": [
            float(value)
            for value in tuple(
                destination_transform
            )
        ],
        "scenes": (
            scene_metadata
        ),
        "rejected_scenes": [
            {
                "item_id": (
                    scene[
                        "item"
                    ].get(
                        "item_id"
                    )
                ),
                "datetime_utc": (
                    scene[
                        "item"
                    ].get(
                        "datetime_utc"
                    )
                ),
                "per_polarisation_pass": (
                    scene[
                        "pass_by_polarisation"
                    ]
                ),
                "per_polarisation": (
                    scene[
                        "per_polarisation"
                    ]
                ),
            }
            for scene in processed
            if not scene["accepted"]
        ],
        "validation_status": {
            "legacy_positive_noise_source_window": True,
            "modern_range_azimuth_source_window": True,
            "modern_full_source_vv_tile": True,
            "modern_full_source_vh_tile": True,
        },
        "open_decisions": {
            "acceptance_rule_is_final": False,
            "band_layout_is_final": False,
            "calibration_quantity_is_final": False,
            "terrain_correction_requirement_is_resolved": False,
            "sentinel_1c_1d_scope_is_resolved": False,
            "duplicate_reprocessing_rule_is_resolved": False,
        },
    }

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"{report_path.parent.name}_"
        f"{args.calibration_lut}_db_"
        "s1_timeseries"
    )

    npz_path = (
        output_dir
        / f"{stem}.npz"
    )
    json_path = (
        output_dir
        / f"{stem}.json"
    )

    np.savez_compressed(
        npz_path,
        backscatter_db_values=(
            values
        ),
        valid_mask_values=(
            valid_masks
        ),
        valid_area_fraction_values=(
            valid_area_fractions
        ),
        band_present=(
            band_present
        ),
        band_pass=(
            band_pass
        ),
        x=x,
        y=y,
        band_name=np.asarray(
            bands,
            dtype="U2",
        ),
        time=np.asarray(
            [
                scene[
                    "datetime_utc"
                ]
                or ""
                for scene
                in scene_metadata
            ],
            dtype="U40",
        ),
        item_id=np.asarray(
            [
                scene[
                    "item_id"
                ]
                or ""
                for scene
                in scene_metadata
            ],
            dtype="U120",
        ),
        platform=np.asarray(
            [
                scene[
                    "platform"
                ]
                or ""
                for scene
                in scene_metadata
            ],
            dtype="U32",
        ),
        orbit_state=np.asarray(
            [
                scene[
                    "orbit_state"
                ]
                or ""
                for scene
                in scene_metadata
            ],
            dtype="U16",
        ),
        relative_orbit=np.asarray(
            [
                -1
                if scene[
                    "relative_orbit"
                ]
                is None
                else int(
                    scene[
                        "relative_orbit"
                    ]
                )
                for scene
                in scene_metadata
            ],
            dtype=np.int64,
        ),
        absolute_orbit=np.asarray(
            [
                -1
                if scene[
                    "absolute_orbit"
                ]
                is None
                else int(
                    scene[
                        "absolute_orbit"
                    ]
                )
                for scene
                in scene_metadata
            ],
            dtype=np.int64,
        ),
        destination_crs=np.asarray(
            destination_crs.to_string()
        ),
        destination_transform=np.asarray(
            tuple(
                destination_transform
            ),
            dtype=np.float64,
        ),
        calibration_lut=np.asarray(
            args.calibration_lut
        ),
        value_units=np.asarray(
            "dB"
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

    finite = np.isfinite(
        values
    )

    print()
    print(
        "Time-series result"
    )
    print(
        f"Accepted scenes: {scene_count}"
    )
    print(
        "Rejected scenes: "
        f"{len(processed) - scene_count}"
    )
    print(
        f"Bands:           {bands}"
    )
    print(
        f"Output shape:    {values.shape}"
    )
    print(
        "Finite values:   "
        f"{int(finite.sum())}/"
        f"{finite.size}"
    )

    if finite.any():
        finite_values = (
            values[finite]
        )
        print(
            "dB stats:        "
            f"min={float(finite_values.min()):.6f}, "
            f"max={float(finite_values.max()):.6f}, "
            f"mean={float(finite_values.mean()):.6f}"
        )

    print(
        f"NPZ:             {npz_path}"
    )
    print(
        f"JSON:            {json_path}"
    )


if __name__ == "__main__":
    main()

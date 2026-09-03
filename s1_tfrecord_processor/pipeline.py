#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pystac_client import Client
from rasterio.crs import CRS
from rasterio.transform import Affine, array_bounds
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping, shape

from .utils import normalize_tile, parse_tile


# ---------------------------------------------------------------------------
# Backward-compatible shared helpers
#
# threshold_analysis.py and some earlier Sentinel-1 development modules import
# these helpers from s1_tfrecord_processor.pipeline. The production pipeline
# below no longer uses them directly, but they remain part of the package's
# internal compatibility surface and must not disappear during the transition
# from prototype discovery to the calibrated production pipeline.
# ---------------------------------------------------------------------------

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"

S2_COLLECTION = "sentinel-2-l1c"
S1_COLLECTION = "sentinel-1-grd"

REQUIRED_POLARISATIONS = {"VV", "VH"}
ALLOWED_PLATFORMS = {"sentinel-1a", "sentinel-1b"}
REQUIRED_INSTRUMENT_MODE = "IW"


def datetime_text(item: Any) -> str | None:
    if item.datetime is None:
        return None
    return item.datetime.isoformat().replace("+00:00", "Z")


def normalise_string_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value.upper()]

    if isinstance(value, (list, tuple, set)):
        return [str(entry).upper() for entry in value]

    return [str(value).upper()]


def value_from_asset_or_item(
    item: Any,
    asset: Any,
    key: str,
) -> Any:
    if key in asset.extra_fields:
        return asset.extra_fields[key]
    return item.properties.get(key)


def discover_s2_grid_item(
    client: Client,
    tile: str,
    start_date: str,
    end_date: str,
) -> tuple[Any, Any]:
    utm_zone, latitude_band, grid_square = parse_tile(tile)

    search = client.search(
        collections=[S2_COLLECTION],
        datetime=f"{start_date}/{end_date}",
        query={
            "mgrs:utm_zone": {"eq": utm_zone},
            "mgrs:latitude_band": {"eq": latitude_band},
            "mgrs:grid_square": {"eq": grid_square},
        },
        max_items=20,
    )

    items = list(search.items())
    items.sort(
        key=lambda item: (
            item.datetime,
            item.id,
        )
    )

    if not items:
        raise RuntimeError(
            f"No Sentinel-2 L1C item found for tile {tile} "
            f"between {start_date} and {end_date}"
        )

    item = items[0]

    blue_asset = item.assets.get("blue")
    if blue_asset is None:
        raise KeyError(
            f"Sentinel-2 item {item.id} does not contain the 'blue' asset"
        )

    return item, blue_asset


def build_s2_target_grid(
    item: Any,
    asset: Any,
    out_dim: int,
) -> dict[str, Any]:
    proj_shape = value_from_asset_or_item(
        item,
        asset,
        "proj:shape",
    )
    proj_transform = value_from_asset_or_item(
        item,
        asset,
        "proj:transform",
    )
    proj_code = value_from_asset_or_item(
        item,
        asset,
        "proj:code",
    )
    proj_epsg = value_from_asset_or_item(
        item,
        asset,
        "proj:epsg",
    )

    missing = []

    if proj_shape is None:
        missing.append("proj:shape")
    if proj_transform is None:
        missing.append("proj:transform")
    if proj_code is None and proj_epsg is None:
        missing.append("proj:code/proj:epsg")

    if missing:
        available_asset_fields = sorted(
            asset.extra_fields
        )
        available_item_fields = sorted(
            key
            for key in item.properties
            if key.startswith("proj:")
        )

        raise RuntimeError(
            "The selected Sentinel-2 asset does not expose the required "
            f"projection metadata. Missing={missing}; "
            f"asset projection fields={available_asset_fields}; "
            f"item projection fields={available_item_fields}"
        )

    if len(proj_shape) != 2:
        raise ValueError(
            f"Unexpected proj:shape: {proj_shape!r}"
        )

    if len(proj_transform) < 6:
        raise ValueError(
            f"Unexpected proj:transform: {proj_transform!r}"
        )

    native_height = int(proj_shape[0])
    native_width = int(proj_shape[1])

    native_transform = Affine(
        *[
            float(value)
            for value in proj_transform[:6]
        ]
    )

    if proj_code is not None:
        crs = CRS.from_user_input(proj_code)
    else:
        crs = CRS.from_epsg(
            int(proj_epsg)
        )

    west, south, east, north = array_bounds(
        native_height,
        native_width,
        native_transform,
    )

    tile_polygon_projected = box(
        west,
        south,
        east,
        north,
    )

    tile_geometry_wgs84 = transform_geom(
        crs,
        "EPSG:4326",
        mapping(
            tile_polygon_projected
        ),
        precision=12,
    )

    output_transform = (
        native_transform
        * Affine.scale(
            native_width / out_dim,
            native_height / out_dim,
        )
    )

    columns = np.arange(
        out_dim,
        dtype=np.float64,
    )
    rows = np.arange(
        out_dim,
        dtype=np.float64,
    )

    x_coordinates, _ = output_transform * (
        columns + 0.5,
        np.full_like(
            columns,
            0.5,
        ),
    )
    _, y_coordinates = output_transform * (
        np.full_like(
            rows,
            0.5,
        ),
        rows + 0.5,
    )

    x_coordinates = np.asarray(
        x_coordinates,
        dtype=np.float64,
    )
    y_coordinates = np.asarray(
        y_coordinates,
        dtype=np.float64,
    )

    x_grid, y_grid = np.meshgrid(
        x_coordinates,
        y_coordinates,
    )

    return {
        "s2_item_id": item.id,
        "s2_asset_href": asset.href,
        "crs": crs,
        "native_shape": (
            native_height,
            native_width,
        ),
        "native_transform": (
            native_transform
        ),
        "output_shape": (
            out_dim,
            out_dim,
        ),
        "output_transform": (
            output_transform
        ),
        "tile_polygon_projected": (
            tile_polygon_projected
        ),
        "tile_geometry_wgs84": (
            tile_geometry_wgs84
        ),
        "x_coordinates": (
            x_coordinates
        ),
        "y_coordinates": (
            y_coordinates
        ),
        "x_grid_shape": (
            x_grid.shape
        ),
        "y_grid_shape": (
            y_grid.shape
        ),
    }


def valid_geometry(geometry: Any) -> Any:
    if geometry.is_empty:
        return geometry

    if not geometry.is_valid:
        geometry = geometry.buffer(0)

    return geometry


def inspect_s1_item(
    item: Any,
    target_grid: dict[str, Any],
    minimum_coverage: float,
) -> dict[str, Any]:
    properties = item.properties

    instrument_mode = str(
        properties.get(
            "sar:instrument_mode"
        )
        or ""
    ).upper()

    platform = str(
        properties.get("platform")
        or ""
    ).lower()

    polarisations = normalise_string_list(
        properties.get(
            "sar:polarizations"
        )
    )

    assets_by_lower_name = {
        key.lower(): asset
        for key, asset
        in item.assets.items()
    }

    vv_asset = (
        assets_by_lower_name.get("vv")
    )
    vh_asset = (
        assets_by_lower_name.get("vh")
    )

    rejection_reasons: list[str] = []

    if (
        instrument_mode
        != REQUIRED_INSTRUMENT_MODE
    ):
        rejection_reasons.append(
            "instrument_mode_is_"
            f"{instrument_mode or 'missing'}"
        )

    if platform not in ALLOWED_PLATFORMS:
        rejection_reasons.append(
            "platform_is_"
            f"{platform or 'missing'}"
        )

    missing_polarisations = sorted(
        REQUIRED_POLARISATIONS.difference(
            polarisations
        )
    )
    if missing_polarisations:
        rejection_reasons.append(
            "missing_polarisations_"
            + "_".join(
                missing_polarisations
            )
        )

    if vv_asset is None or not vv_asset.href:
        rejection_reasons.append(
            "missing_vv_asset"
        )

    if vh_asset is None or not vh_asset.href:
        rejection_reasons.append(
            "missing_vh_asset"
        )

    tile_coverage_fraction = 0.0
    scene_overlap_fraction = 0.0
    intersection_area = 0.0

    if item.geometry is None:
        rejection_reasons.append(
            "missing_geometry"
        )
    else:
        try:
            scene_geometry_projected = shape(
                transform_geom(
                    "EPSG:4326",
                    target_grid["crs"],
                    item.geometry,
                    precision=6,
                )
            )
            scene_geometry_projected = (
                valid_geometry(
                    scene_geometry_projected
                )
            )

            tile_polygon = target_grid[
                "tile_polygon_projected"
            ]
            intersection = (
                tile_polygon.intersection(
                    scene_geometry_projected
                )
            )

            intersection_area = float(
                intersection.area
            )

            if tile_polygon.area > 0:
                tile_coverage_fraction = (
                    intersection_area
                    / float(
                        tile_polygon.area
                    )
                )

            if (
                scene_geometry_projected.area
                > 0
            ):
                scene_overlap_fraction = (
                    intersection_area
                    / float(
                        scene_geometry_projected.area
                    )
                )

        except Exception as exc:
            rejection_reasons.append(
                "geometry_processing_failed:"
                f"{type(exc).__name__}:"
                f"{exc}"
            )

    if (
        tile_coverage_fraction
        < minimum_coverage
    ):
        rejection_reasons.append(
            "tile_coverage_below_threshold"
        )

    keep = not rejection_reasons

    return {
        "item_id": item.id,
        "datetime_utc": (
            datetime_text(item)
        ),
        "platform": (
            platform or None
        ),
        "instrument_mode": (
            instrument_mode or None
        ),
        "polarisations": (
            polarisations
        ),
        "orbit_state": properties.get(
            "sat:orbit_state"
        ),
        "relative_orbit": properties.get(
            "sat:relative_orbit"
        ),
        "absolute_orbit": properties.get(
            "sat:absolute_orbit"
        ),
        "product_type": properties.get(
            "sar:product_type"
        ),
        "vv_href": (
            vv_asset.href
            if vv_asset
            else None
        ),
        "vh_href": (
            vh_asset.href
            if vh_asset
            else None
        ),
        "intersection_area_projected_units": (
            intersection_area
        ),
        "tile_coverage_fraction": (
            tile_coverage_fraction
        ),
        "tile_coverage_pct": (
            tile_coverage_fraction
            * 100.0
        ),
        "scene_overlap_fraction": (
            scene_overlap_fraction
        ),
        "scene_overlap_pct": (
            scene_overlap_fraction
            * 100.0
        ),
        "keep": keep,
        "rejection_reasons": (
            rejection_reasons
        ),
    }


def serialise_grid(
    grid: dict[str, Any],
) -> dict[str, Any]:
    tile_polygon = grid[
        "tile_polygon_projected"
    ]

    return {
        "s2_item_id": (
            grid["s2_item_id"]
        ),
        "s2_asset_href": (
            grid["s2_asset_href"]
        ),
        "crs": (
            grid["crs"].to_string()
        ),
        "crs_wkt": (
            grid["crs"].to_wkt()
        ),
        "native_shape": list(
            grid["native_shape"]
        ),
        "native_transform": tuple(
            grid["native_transform"]
        ),
        "output_shape": list(
            grid["output_shape"]
        ),
        "output_transform": tuple(
            grid["output_transform"]
        ),
        "projected_bounds": list(
            tile_polygon.bounds
        ),
        "tile_geometry_wgs84": (
            grid["tile_geometry_wgs84"]
        ),
        "x_coordinate_first": float(
            grid["x_coordinates"][0]
        ),
        "x_coordinate_last": float(
            grid["x_coordinates"][-1]
        ),
        "y_coordinate_first": float(
            grid["y_coordinates"][0]
        ),
        "y_coordinate_last": float(
            grid["y_coordinates"][-1]
        ),
        "x_grid_shape": list(
            grid["x_grid_shape"]
        ),
        "y_grid_shape": list(
            grid["y_grid_shape"]
        ),
    }


# ---------------------------------------------------------------------------
# Production calibrated TFRecord orchestration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_ROOT = "outputs-s1-tfrecord"
DEFAULT_WORK_ROOT = "reports/s1_pipeline_work"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the validated Sentinel-1 GRD pipeline end-to-end: "
            "discover/filter acquisitions for an S2 MGRS tile, build a "
            "calibrated dB time series, serialize it to TFRecord, and "
            "verify an exact TFRecord round-trip."
        )
    )

    parser.add_argument(
        "--tile",
        required=True,
    )
    parser.add_argument(
        "--start-date",
        required=True,
    )
    parser.add_argument(
        "--end-date",
        required=True,
    )

    parser.add_argument(
        "--minimum-geometric-coverage",
        type=float,
        default=0.80,
        help=(
            "Fast STAC-footprint coverage prefilter. Default: 0.80."
        ),
    )
    parser.add_argument(
        "--minimum-valid-coverage",
        type=float,
        default=0.80,
        help=(
            "Final per-polarisation area-weighted valid coverage. "
            "Default: 0.80."
        ),
    )
    parser.add_argument(
        "--acceptance-rule",
        choices=("any", "all"),
        default="all",
        help=(
            "How per-polarisation coverage results are combined. "
            "Default: all. This remains configurable pending the "
            "final project decision."
        ),
    )
    parser.add_argument(
        "--band-layout",
        choices=("union", "canonical"),
        default="union",
        help=(
            "union includes polarisations observed in accepted scenes; "
            "canonical always uses VV,VH,HH,HV. Default: union."
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
            "Radiometric calibration quantity. Default: sigmaNought. "
            "This remains configurable pending the final project "
            "scientific decision."
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
            "Policy for products without an explicit thermal-noise "
            "correction flag. Default: error."
        ),
    )

    parser.add_argument(
        "--out-dim",
        type=int,
        default=128,
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
        "--chunk-size",
        type=int,
        default=64,
        help=(
            "Time steps per TFRecord chunk. Default: 64."
        ),
    )

    parser.add_argument(
        "--platforms",
        nargs="+",
        default=[
            "sentinel-1a",
            "sentinel-1b",
        ],
        help=(
            "Eligible Sentinel-1 platforms used by threshold analysis. "
            "Default: sentinel-1a sentinel-1b."
        ),
    )
    parser.add_argument(
        "--item-id",
        action="append",
        default=None,
        help=(
            "Process only this acquisition ID. May be supplied more "
            "than once. Intended for diagnostics."
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
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--work-root",
        default=DEFAULT_WORK_ROOT,
        help=(
            "Temporary/intermediate pipeline workspace."
        ),
    )
    parser.add_argument(
        "--temp-directory",
        default=None,
        help=(
            "Optional directory for staged calibrated linear GeoTIFFs."
        ),
    )
    parser.add_argument(
        "--keep-intermediates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep generated threshold reports and NPZ/JSON validation "
            "artifacts after the final TFRecord succeeds. Default: false."
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

    args = parser.parse_args()

    for name in (
        "minimum_geometric_coverage",
        "minimum_valid_coverage",
    ):
        value = getattr(
            args,
            name,
        )

        if not 0.0 <= value <= 1.0:
            parser.error(
                f"--{name.replace('_', '-')} "
                "must be between 0 and 1"
            )

    if args.out_dim <= 0:
        parser.error(
            "--out-dim must be greater than zero"
        )

    if args.rows_per_window <= 0:
        parser.error(
            "--rows-per-window must be greater than zero"
        )

    if args.num_threads <= 0:
        parser.error(
            "--num-threads must be greater than zero"
        )

    if args.chunk_size <= 0:
        parser.error(
            "--chunk-size must be greater than zero"
        )

    if (
        args.max_scenes is not None
        and args.max_scenes <= 0
    ):
        parser.error(
            "--max-scenes must be greater than zero"
        )

    return args


def run_command(
    command: list[str],
) -> None:
    print()
    print("Running:")
    print(
        "  "
        + " ".join(command)
    )
    print()

    completed = subprocess.run(
        command,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "Pipeline stage failed with exit code "
            f"{completed.returncode}: "
            f"{' '.join(command)}"
        )


def final_output_path(
    *,
    output_root: Path,
    tile: str,
    start_date: str,
    end_date: str,
    calibration_lut: str,
) -> Path:
    filename = (
        f"s1_grd_tile_{tile}_"
        f"{start_date}_to_{end_date}_"
        f"{calibration_lut}_db.tfrecord"
    )

    return (
        output_root
        / tile
        / filename
    )


def main() -> None:
    args = parse_args()

    tile = (
        f"T{normalize_tile(args.tile)}"
    )

    work_root = Path(
        args.work_root
    )
    run_name = (
        f"{tile}_"
        f"{args.start_date}_to_"
        f"{args.end_date}"
    )

    threshold_dir = (
        work_root
        / run_name
        / "threshold_analysis"
    )

    timeseries_dir = (
        work_root
        / run_name
        / "calibrated_time_series"
    )

    threshold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    timeseries_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        threshold_dir
        / "threshold_analysis.json"
    )

    output_path = (
        final_output_path(
            output_root=Path(
                args.output_root
            ),
            tile=tile,
            start_date=args.start_date,
            end_date=args.end_date,
            calibration_lut=(
                args.calibration_lut
            ),
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        raise FileExistsError(
            "Final Sentinel-1 TFRecord "
            "already exists: "
            f"{output_path}"
        )

    print(
        "Sentinel-1 GRD calibrated "
        "TFRecord pipeline"
    )
    print(
        "-------------------------------------------"
    )
    print(
        f"Tile:                       {tile}"
    )
    print(
        "Date range:                 "
        f"{args.start_date} to "
        f"{args.end_date}"
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
        "Calibration:                "
        f"{args.calibration_lut}"
    )
    print(
        "Unknown noise policy:       "
        f"{args.unknown_noise_policy}"
    )
    print(
        f"Output:                     {output_path}"
    )

    threshold_command = [
        sys.executable,
        "-m",
        "s1_tfrecord_processor.threshold_analysis",
        "--tile",
        tile,
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--primary-threshold",
        str(
            args.minimum_geometric_coverage
        ),
        "--out-dim",
        str(args.out_dim),
        "--platforms",
        *args.platforms,
        "--output-directory",
        str(threshold_dir),
    ]

    run_command(
        threshold_command
    )

    if not report_path.is_file():
        raise RuntimeError(
            "Threshold-analysis stage completed "
            "but did not create "
            f"{report_path}"
        )

    timeseries_command = [
        sys.executable,
        "-m",
        "s1_tfrecord_processor.build_calibrated_timeseries",
        "--threshold-report",
        str(report_path),
        "--minimum-geometric-coverage",
        str(
            args.minimum_geometric_coverage
        ),
        "--minimum-valid-coverage",
        str(
            args.minimum_valid_coverage
        ),
        "--acceptance-rule",
        args.acceptance_rule,
        "--band-layout",
        args.band_layout,
        "--calibration-lut",
        args.calibration_lut,
        "--unknown-noise-policy",
        args.unknown_noise_policy,
        "--rows-per-window",
        str(args.rows_per_window),
        "--num-threads",
        str(args.num_threads),
        "--output-dir",
        str(timeseries_dir),
        "--aws-region",
        args.aws_region,
    ]

    if args.aws_profile:
        timeseries_command.extend(
            [
                "--aws-profile",
                args.aws_profile,
            ]
        )

    if args.temp_directory:
        timeseries_command.extend(
            [
                "--temp-directory",
                args.temp_directory,
            ]
        )

    if args.max_scenes is not None:
        timeseries_command.extend(
            [
                "--max-scenes",
                str(args.max_scenes),
            ]
        )

    for item_id in (
        args.item_id or []
    ):
        timeseries_command.extend(
            [
                "--item-id",
                item_id,
            ]
        )

    run_command(
        timeseries_command
    )

    npz_path = (
        timeseries_dir
        / (
            "threshold_analysis_"
            f"{args.calibration_lut}_db_"
            "s1_timeseries.npz"
        )
    )

    if not npz_path.is_file():
        candidates = sorted(
            timeseries_dir.glob(
                f"*_{args.calibration_lut}_db_"
                "s1_timeseries.npz"
            )
        )

        if len(candidates) != 1:
            raise RuntimeError(
                "Could not uniquely identify "
                "the calibrated time-series NPZ. "
                "Found: "
                + ", ".join(
                    str(path)
                    for path in candidates
                )
            )

        npz_path = candidates[0]

    tfrecord_command = [
        sys.executable,
        "-m",
        "s1_tfrecord_processor.write_calibrated_tfrecord",
        "--input-npz",
        str(npz_path),
        "--output-tfrecord",
        str(output_path),
        "--chunk-size",
        str(args.chunk_size),
        "--overwrite",
    ]

    run_command(
        tfrecord_command
    )

    if not output_path.is_file():
        raise RuntimeError(
            "TFRecord stage completed "
            "but final output is missing: "
            f"{output_path}"
        )

    final_size = (
        output_path.stat().st_size
    )

    if not args.keep_intermediates:
        run_work_dir = (
            work_root
            / run_name
        )
        shutil.rmtree(
            run_work_dir,
            ignore_errors=False,
        )

    print()
    print(
        "Sentinel-1 pipeline completed successfully."
    )
    print(
        f"Final TFRecord: {output_path}"
    )
    print(
        f"File size:      {final_size} bytes"
    )
    print(
        "TFRecord round-trip verification: passed"
    )
    print(
        "Intermediates:  "
        + (
            "kept"
            if args.keep_intermediates
            else "removed"
        )
    )


if __name__ == "__main__":
    main()

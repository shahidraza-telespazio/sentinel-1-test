from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.session import AWSSession


DEFAULT_REPORT = (
    "reports/s1_prototype/"
    "T30UXC_2025-06-01_to_2025-06-30_intersection_report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the actual VV and VH source rasters for the "
            "highest-coverage Sentinel-1A candidate."
        )
    )

    parser.add_argument(
        "--intersection-report",
        default=DEFAULT_REPORT,
    )
    parser.add_argument(
        "--preview-dim",
        type=int,
        default=512,
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
        "--output",
        default=None,
    )

    args = parser.parse_args()

    if args.preview_dim <= 0:
        parser.error("--preview-dim must be greater than zero")

    return args


def make_boto_session(
    profile: str | None,
    region: str,
) -> boto3.Session:
    if profile:
        return boto3.Session(
            profile_name=profile,
            region_name=region,
        )

    return boto3.Session(region_name=region)


def select_candidate(report: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        item
        for item in report["all_candidates"]
        if item.get("platform") == "sentinel-1a"
        and item.get("instrument_mode") == "IW"
        and item.get("vv_href")
        and item.get("vh_href")
    ]

    if not candidates:
        raise RuntimeError(
            "The intersection report contains no Sentinel-1A "
            "IW candidate with both VV and VH assets."
        )

    return max(
        candidates,
        key=lambda item: float(item["tile_coverage_fraction"]),
    )


def serialise_transform(transform: Any) -> list[float]:
    return [float(value) for value in tuple(transform)]


def inspect_raster(
    href: str,
    polarisation: str,
    boto_session: boto3.Session,
    preview_dim: int,
) -> dict[str, Any]:
    aws_session = AWSSession(
        boto_session,
        requester_pays=True,
    )

    with rasterio.Env(
        aws_session,
        AWS_REGION=boto_session.region_name or "eu-central-1",
        AWS_REQUEST_PAYER="requester",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_PAM_ENABLED="NO",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
    ):
        with rasterio.open(href) as src:
            gcps, gcp_crs = src.gcps

            preview = src.read(
                1,
                out_shape=(preview_dim, preview_dim),
                resampling=Resampling.nearest,
                masked=True,
            )

            valid_mask = src.read_masks(
                1,
                out_shape=(preview_dim, preview_dim),
                resampling=Resampling.nearest,
            )

            valid_values = np.asarray(
                preview.compressed(),
                dtype=np.float64,
            )

            mask_valid = valid_mask != 0
            zero_count = int(
                np.count_nonzero(
                    np.asarray(preview.data) == 0
                )
            )

            mask_flags = [
                flag.name
                for flag in src.mask_flag_enums[0]
            ]

            result: dict[str, Any] = {
                "polarisation": polarisation,
                "href": href,
                "driver": src.driver,
                "count": int(src.count),
                "width": int(src.width),
                "height": int(src.height),
                "shape": [int(src.height), int(src.width)],
                "dtypes": list(src.dtypes),
                "crs": src.crs.to_string() if src.crs else None,
                "crs_wkt": src.crs.to_wkt() if src.crs else None,
                "transform": serialise_transform(src.transform),
                "bounds": [
                    float(src.bounds.left),
                    float(src.bounds.bottom),
                    float(src.bounds.right),
                    float(src.bounds.top),
                ],
                "nodata": (
                    float(src.nodata)
                    if src.nodata is not None
                    else None
                ),
                "nodatavals": [
                    float(value) if value is not None else None
                    for value in src.nodatavals
                ],
                "scales": [float(value) for value in src.scales],
                "offsets": [float(value) for value in src.offsets],
                "units": list(src.units),
                "descriptions": list(src.descriptions),
                "mask_flags": mask_flags,
                "block_shapes": [
                    [int(rows), int(cols)]
                    for rows, cols in src.block_shapes
                ],
                "overviews": [
                    int(value)
                    for value in src.overviews(1)
                ],
                "gcp_count": len(gcps),
                "gcp_crs": (
                    gcp_crs.to_string()
                    if gcp_crs is not None
                    else None
                ),
                "preview": {
                    "shape": [preview_dim, preview_dim],
                    "valid_pixel_count": int(
                        np.count_nonzero(mask_valid)
                    ),
                    "invalid_pixel_count": int(
                        mask_valid.size
                        - np.count_nonzero(mask_valid)
                    ),
                    "valid_fraction": float(mask_valid.mean()),
                    "valid_pct": float(mask_valid.mean() * 100.0),
                    "zero_value_count": zero_count,
                    "zero_value_fraction": float(
                        zero_count / preview.data.size
                    ),
                },
                "dataset_tags": {
                    str(key): str(value)
                    for key, value in src.tags().items()
                },
                "band_tags": {
                    str(key): str(value)
                    for key, value in src.tags(1).items()
                },
            }

            if valid_values.size:
                result["preview"].update(
                    {
                        "valid_min": float(valid_values.min()),
                        "valid_max": float(valid_values.max()),
                        "valid_mean": float(valid_values.mean()),
                        "valid_median": float(
                            np.median(valid_values)
                        ),
                    }
                )
            else:
                result["preview"].update(
                    {
                        "valid_min": None,
                        "valid_max": None,
                        "valid_mean": None,
                        "valid_median": None,
                    }
                )

            return result


def print_raster_summary(result: dict[str, Any]) -> None:
    preview = result["preview"]

    print()
    print(f"===== {result['polarisation']} =====")
    print(f"href:          {result['href']}")
    print(f"driver:        {result['driver']}")
    print(f"shape:         {result['shape']}")
    print(f"dtype:         {result['dtypes']}")
    print(f"crs:           {result['crs']}")
    print(f"transform:     {result['transform']}")
    print(f"bounds:        {result['bounds']}")
    print(f"nodata:        {result['nodata']}")
    print(f"nodatavals:    {result['nodatavals']}")
    print(f"mask flags:    {result['mask_flags']}")
    print(f"GCP count:     {result['gcp_count']}")
    print(f"GCP CRS:       {result['gcp_crs']}")
    print(f"overviews:     {result['overviews']}")
    print(
        "preview valid: "
        f"{preview['valid_pct']:.4f}%"
    )
    print(
        "preview zeros: "
        f"{preview['zero_value_fraction'] * 100.0:.4f}%"
    )
    print(
        "valid values:  "
        f"min={preview['valid_min']} "
        f"max={preview['valid_max']} "
        f"mean={preview['valid_mean']} "
        f"median={preview['valid_median']}"
    )


def main() -> None:
    args = parse_args()

    intersection_report_path = Path(
        args.intersection_report
    )

    report = json.loads(
        intersection_report_path.read_text(
            encoding="utf-8"
        )
    )

    candidate = select_candidate(report)

    boto_session = make_boto_session(
        args.aws_profile,
        args.aws_region,
    )

    print()
    print("Sentinel-1 source-raster inspection")
    print("-----------------------------------")
    print(f"Candidate:      {candidate['item_id']}")
    print(f"Datetime:       {candidate['datetime_utc']}")
    print(
        "STAC coverage:  "
        f"{candidate['tile_coverage_pct']:.6f}%"
    )
    print(f"Orbit state:    {candidate['orbit_state']}")
    print(f"Relative orbit: {candidate['relative_orbit']}")

    vv_result = inspect_raster(
        candidate["vv_href"],
        "VV",
        boto_session,
        args.preview_dim,
    )

    vh_result = inspect_raster(
        candidate["vh_href"],
        "VH",
        boto_session,
        args.preview_dim,
    )

    result = {
        "intersection_report": str(
            intersection_report_path
        ),
        "selected_candidate": candidate,
        "vv": vv_result,
        "vh": vh_result,
    }

    output_path = Path(
        args.output
        or intersection_report_path.with_name(
            intersection_report_path.stem
            + "_source_raster_inspection.json"
        )
    )

    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print_raster_summary(vv_result)
    print_raster_summary(vh_result)

    print()
    print(f"Inspection report: {output_path}")


if __name__ == "__main__":
    main()

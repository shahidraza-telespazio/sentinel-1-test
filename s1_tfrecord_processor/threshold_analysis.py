from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from pystac_client import Client
from rasterio.warp import transform_geom
from shapely.geometry import shape

from .discovery import (
    REQUIRED_PRODUCT_TYPE,
    acquisition_from_item,
    processing_rejection_reasons,
)
from .pipeline import (
    EARTH_SEARCH_URL,
    REQUIRED_INSTRUMENT_MODE,
    S1_COLLECTION,
    build_s2_target_grid,
    datetime_text,
    discover_s2_grid_item,
    normalise_string_list,
    serialise_grid,
    valid_geometry,
)
from .utils import normalize_tile


KNOWN_POLARISATIONS = ("VV", "VH", "HH", "HV")

DEFAULT_PLATFORMS = (
    "sentinel-1a",
    "sentinel-1b",
)

DEFAULT_THRESHOLDS = (
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
)


def parse_thresholds(value: str) -> tuple[float, ...]:
    thresholds: list[float] = []

    for raw_value in value.split(","):
        raw_value = raw_value.strip()

        if not raw_value:
            continue

        try:
            threshold = float(raw_value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid threshold {raw_value!r}"
            ) from exc

        if not 0.0 <= threshold <= 1.0:
            raise argparse.ArgumentTypeError(
                "Every threshold must be between 0 and 1."
            )

        thresholds.append(threshold)

    if not thresholds:
        raise argparse.ArgumentTypeError(
            "At least one threshold must be supplied."
        )

    return tuple(sorted(set(thresholds)))


def parse_date(value: str) -> str:
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a date in YYYY-MM-DD format, received {value!r}"
        ) from exc

    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse how different geometric Sentinel-1 tile-coverage "
            "thresholds affect the number and distribution of acquisitions."
        )
    )

    parser.add_argument("--tile", required=True)
    parser.add_argument(
        "--start-date",
        required=True,
        type=parse_date,
    )
    parser.add_argument(
        "--end-date",
        required=True,
        type=parse_date,
    )

    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=DEFAULT_THRESHOLDS,
        help=(
            "Comma-separated coverage fractions. "
            "Default: 0.50,0.60,0.70,0.75,0.80,0.85,0.90"
        ),
    )
    parser.add_argument(
        "--primary-threshold",
        type=float,
        default=0.80,
        help=(
            "Threshold around which nearby scenes are reported. "
            "Default: 0.80"
        ),
    )
    parser.add_argument(
        "--near-threshold-margin",
        type=float,
        default=0.05,
        help=(
            "Include scenes within this distance of the primary "
            "threshold in the near-threshold report. Default: 0.05"
        ),
    )

    parser.add_argument(
        "--platforms",
        nargs="+",
        default=list(DEFAULT_PLATFORMS),
        help=(
            "Platforms eligible for threshold counts. "
            "Default: sentinel-1a sentinel-1b. "
            "Use '--platforms all' to include every platform."
        ),
    )

    parser.add_argument(
        "--out-dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help=(
            "Optional diagnostic cap on total STAC results. "
            "Omit this option for a complete full-period search."
        ),
    )
    parser.add_argument(
        "--page-limit",
        type=int,
        default=100,
        help=(
            "Requested STAC results per page. "
            "This does not limit the total result count."
        ),
    )
    parser.add_argument(
        "--output-directory",
        default=None,
    )

    args = parser.parse_args()

    if args.start_date > args.end_date:
        parser.error("--start-date must not be later than --end-date")

    if not 0.0 <= args.primary_threshold <= 1.0:
        parser.error("--primary-threshold must be between 0 and 1")

    if not 0.0 <= args.near_threshold_margin <= 1.0:
        parser.error(
            "--near-threshold-margin must be between 0 and 1"
        )

    if args.out_dim <= 0:
        parser.error("--out-dim must be greater than zero")

    if args.max_items is not None and args.max_items <= 0:
        parser.error("--max-items must be greater than zero")

    if args.page_limit <= 0:
        parser.error("--page-limit must be greater than zero")

    normalised_platforms = {
        str(platform).strip().lower()
        for platform in args.platforms
        if str(platform).strip()
    }

    if not normalised_platforms:
        parser.error("--platforms must contain at least one value")

    if "all" in normalised_platforms and len(normalised_platforms) != 1:
        parser.error(
            "'all' cannot be combined with named platforms"
        )

    args.allowed_platforms = (
        None
        if normalised_platforms == {"all"}
        else normalised_platforms
    )

    thresholds = set(args.thresholds)
    thresholds.add(args.primary_threshold)
    args.thresholds = tuple(sorted(thresholds))

    return args


def asset_map(item: Any) -> dict[str, Any]:
    return {
        str(key).lower(): asset
        for key, asset in item.assets.items()
    }


def available_measurement_assets(
    item: Any,
) -> dict[str, str]:
    assets = asset_map(item)
    available: dict[str, str] = {}

    for polarisation in KNOWN_POLARISATIONS:
        asset = assets.get(polarisation.lower())

        if asset is None or not asset.href:
            continue

        available[polarisation] = asset.href

    return available


def expected_metadata_assets(
    item: Any,
    available_polarisations: Sequence[str],
) -> tuple[dict[str, dict[str, str | None]], list[str]]:
    assets = asset_map(item)

    metadata: dict[str, dict[str, str | None]] = {}
    missing: list[str] = []

    for polarisation in available_polarisations:
        suffix = polarisation.lower()

        keys = {
            "product": f"schema-product-{suffix}",
            "calibration": f"schema-calibration-{suffix}",
            "noise": f"schema-noise-{suffix}",
        }

        metadata[polarisation] = {}

        for metadata_type, key in keys.items():
            asset = assets.get(key)

            if asset is None or not asset.href:
                metadata[polarisation][metadata_type] = None
                missing.append(key)
            else:
                metadata[polarisation][metadata_type] = asset.href

    return metadata, sorted(set(missing))


def calculate_geometric_coverage(
    item: Any,
    target_grid: dict[str, Any],
) -> tuple[float, float, float]:
    if item.geometry is None:
        raise ValueError("STAC item has no geometry")

    scene_geometry_projected = shape(
        transform_geom(
            "EPSG:4326",
            target_grid["crs"],
            item.geometry,
            precision=6,
        )
    )
    scene_geometry_projected = valid_geometry(
        scene_geometry_projected
    )

    tile_polygon = target_grid["tile_polygon_projected"]

    if tile_polygon.is_empty or tile_polygon.area <= 0:
        raise ValueError(
            "Sentinel-2 target tile has no positive area"
        )

    intersection = tile_polygon.intersection(
        scene_geometry_projected
    )
    intersection_area = float(intersection.area)

    tile_coverage_fraction = (
        intersection_area / float(tile_polygon.area)
    )

    scene_overlap_fraction = 0.0
    if (
        not scene_geometry_projected.is_empty
        and scene_geometry_projected.area > 0
    ):
        scene_overlap_fraction = (
            intersection_area
            / float(scene_geometry_projected.area)
        )

    if not -1e-9 <= tile_coverage_fraction <= 1.0 + 1e-9:
        raise ValueError(
            "Calculated tile coverage is outside the expected "
            f"range: {tile_coverage_fraction}"
        )

    tile_coverage_fraction = min(
        1.0,
        max(0.0, tile_coverage_fraction),
    )

    return (
        tile_coverage_fraction,
        scene_overlap_fraction,
        intersection_area,
    )


def provisional_acquisition_key(
    *,
    platform: str,
    datetime_utc: str | None,
    end_datetime_utc: str | None,
    instrument_mode: str,
    absolute_orbit: Any,
    available_polarisations: Sequence[str],
) -> str:
    """
    Diagnostic grouping key only.

    It is not used to remove or select any STAC item. It helps reveal
    possible reprocessed or duplicate products sharing the same acquisition
    identity.
    """
    return "|".join(
        (
            platform or "missing",
            datetime_utc or "missing",
            end_datetime_utc or "missing",
            instrument_mode or "missing",
            str(absolute_orbit or "missing"),
            "+".join(available_polarisations) or "missing",
        )
    )


def inspect_item(
    item: Any,
    target_grid: dict[str, Any],
    allowed_platforms: set[str] | None,
) -> dict[str, Any]:
    acquisition = acquisition_from_item(item)

    rejection_reasons = list(
        processing_rejection_reasons(
            acquisition,
            allowed_platforms=allowed_platforms,
        )
    )

    warnings = list(acquisition.warnings)

    tile_coverage_fraction = 0.0
    scene_overlap_fraction = 0.0
    intersection_area = 0.0

    try:
        (
            tile_coverage_fraction,
            scene_overlap_fraction,
            intersection_area,
        ) = calculate_geometric_coverage(
            item,
            target_grid,
        )
    except Exception as exc:
        rejection_reasons.append(
            "geometry_processing_failed:"
            f"{type(exc).__name__}:{exc}"
        )

    measurement_assets = (
        acquisition.measurement_hrefs
    )

    metadata_assets = {
        assets.polarisation: {
            "product": (
                assets.product_metadata_href
            ),
            "calibration": (
                assets.calibration_metadata_href
            ),
            "noise": (
                assets.noise_metadata_href
            ),
        }
        for assets in acquisition.polarisation_assets
    }

    missing_metadata_assets = sorted(
        {
            missing_key
            for assets in acquisition.polarisation_assets
            for missing_key
            in assets.missing_metadata_assets
        }
    )

    available_polarisations = list(
        acquisition.available_polarisations
    )

    polarisation_combination = (
        "+".join(available_polarisations)
        if available_polarisations
        else "NONE"
    )

    acquisition_key = provisional_acquisition_key(
        platform=(
            acquisition.platform or ""
        ),
        datetime_utc=(
            acquisition.datetime_utc
        ),
        end_datetime_utc=(
            acquisition.end_datetime_utc
        ),
        instrument_mode=(
            acquisition.instrument_mode or ""
        ),
        absolute_orbit=(
            acquisition.absolute_orbit
        ),
        available_polarisations=(
            available_polarisations
        ),
    )

    acquisition_year = (
        item.datetime.year
        if item.datetime is not None
        else None
    )

    return {
        "item_id": acquisition.item_id,
        "collection": acquisition.collection,
        "datetime_utc": (
            acquisition.datetime_utc
        ),
        "start_datetime_utc": (
            acquisition.start_datetime_utc
        ),
        "end_datetime_utc": (
            acquisition.end_datetime_utc
        ),
        "year": acquisition_year,
        "platform": acquisition.platform,
        "constellation": (
            acquisition.constellation
        ),
        "instrument_mode": (
            acquisition.instrument_mode
        ),
        "product_type": (
            acquisition.product_type
        ),
        "orbit_state": (
            acquisition.orbit_state
        ),
        "relative_orbit": (
            acquisition.relative_orbit
        ),
        "absolute_orbit": (
            acquisition.absolute_orbit
        ),
        "declared_polarisations": list(
            acquisition.declared_polarisations
        ),
        "available_polarisations": (
            available_polarisations
        ),
        "polarisation_combination": (
            polarisation_combination
        ),
        "measurement_assets": (
            measurement_assets
        ),
        "metadata_assets": metadata_assets,
        "polarisation_assets": [
            assets.to_dict()
            for assets
            in acquisition.polarisation_assets
        ],
        "safe_manifest_href": (
            acquisition.safe_manifest_href
        ),
        "missing_metadata_assets": (
            missing_metadata_assets
        ),
        "intersection_area_projected_units": (
            intersection_area
        ),
        "tile_coverage_fraction": (
            tile_coverage_fraction
        ),
        "tile_coverage_pct": (
            tile_coverage_fraction * 100.0
        ),
        "scene_overlap_fraction": (
            scene_overlap_fraction
        ),
        "scene_overlap_pct": (
            scene_overlap_fraction * 100.0
        ),
        "eligible_before_coverage": (
            not rejection_reasons
        ),
        "eligibility_rejection_reasons": (
            rejection_reasons
        ),
        "warnings": warnings,
        "provisional_acquisition_key": (
            acquisition_key
        ),
    }


def counter_dict(
    values: Iterable[Any],
) -> dict[str, int]:
    counts = Counter(
        "missing"
        if value is None or value == ""
        else str(value)
        for value in values
    )

    return dict(
        sorted(
            counts.items(),
            key=lambda entry: entry[0],
        )
    )


def polarisation_counts(
    records: Sequence[dict[str, Any]],
) -> dict[str, int]:
    counts = Counter()

    for record in records:
        for polarisation in record[
            "available_polarisations"
        ]:
            counts[polarisation] += 1

    return {
        polarisation: int(counts.get(polarisation, 0))
        for polarisation in KNOWN_POLARISATIONS
    }


def parse_datetime(
    value: str | None,
) -> datetime | None:
    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError:
        return None


def unique_acquisition_records(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}

    for record in records:
        key = record["provisional_acquisition_key"]

        existing = by_key.get(key)
        if existing is None:
            by_key[key] = record
            continue

        existing_datetime = parse_datetime(
            existing["datetime_utc"]
        )
        current_datetime = parse_datetime(
            record["datetime_utc"]
        )

        if (
            existing_datetime is None
            or (
                current_datetime is not None
                and current_datetime < existing_datetime
            )
        ):
            by_key[key] = record

    return list(by_key.values())


def time_gap_statistics(
    records: Sequence[dict[str, Any]],
) -> dict[str, float | int | None]:
    unique_records = unique_acquisition_records(records)

    datetimes = sorted(
        value
        for value in (
            parse_datetime(record["datetime_utc"])
            for record in unique_records
        )
        if value is not None
    )

    if len(datetimes) < 2:
        return {
            "acquisition_count_with_datetime": len(datetimes),
            "gap_count": 0,
            "minimum_gap_days": None,
            "median_gap_days": None,
            "mean_gap_days": None,
            "maximum_gap_days": None,
        }

    gaps = [
        (
            datetimes[index]
            - datetimes[index - 1]
        ).total_seconds()
        / 86400.0
        for index in range(1, len(datetimes))
    ]

    return {
        "acquisition_count_with_datetime": len(datetimes),
        "gap_count": len(gaps),
        "minimum_gap_days": min(gaps),
        "median_gap_days": statistics.median(gaps),
        "mean_gap_days": statistics.fmean(gaps),
        "maximum_gap_days": max(gaps),
    }


def percentile(
    values: Sequence[float],
    proportion: float,
) -> float | None:
    if not values:
        return None

    ordered = sorted(float(value) for value in values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * proportion
    lower_index = int(position)
    upper_index = min(
        lower_index + 1,
        len(ordered) - 1,
    )
    fraction = position - lower_index

    return (
        ordered[lower_index]
        + (
            ordered[upper_index]
            - ordered[lower_index]
        )
        * fraction
    )


def coverage_distribution(
    records: Sequence[dict[str, Any]],
) -> dict[str, float | int | None]:
    values = [
        float(record["tile_coverage_fraction"])
        for record in records
    ]

    if not values:
        return {
            "count": 0,
            "minimum": None,
            "p05": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p95": None,
            "maximum": None,
        }

    return {
        "count": len(values),
        "minimum": min(values),
        "p05": percentile(values, 0.05),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.50),
        "mean": statistics.fmean(values),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "maximum": max(values),
    }


def threshold_summary(
    eligible_records: Sequence[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    retained = [
        record
        for record in eligible_records
        if float(record["tile_coverage_fraction"])
        >= threshold
    ]

    unique_records = unique_acquisition_records(retained)

    eligible_count = len(eligible_records)

    return {
        "threshold_fraction": threshold,
        "threshold_pct": threshold * 100.0,
        "retained_stac_item_count": len(retained),
        "retained_provisional_acquisition_count": len(
            unique_records
        ),
        "retained_fraction_of_eligible_items": (
            len(retained) / eligible_count
            if eligible_count
            else 0.0
        ),
        "counts_by_year": counter_dict(
            record["year"]
            for record in retained
        ),
        "counts_by_platform": counter_dict(
            record["platform"]
            for record in retained
        ),
        "counts_by_orbit_state": counter_dict(
            record["orbit_state"]
            for record in retained
        ),
        "counts_by_relative_orbit": counter_dict(
            record["relative_orbit"]
            for record in retained
        ),
        "counts_by_polarisation_combination": counter_dict(
            record["polarisation_combination"]
            for record in retained
        ),
        "counts_by_polarisation": polarisation_counts(
            retained
        ),
        "time_gap_statistics": time_gap_statistics(
            retained
        ),
        "first_datetime_utc": min(
            (
                record["datetime_utc"]
                for record in retained
                if record["datetime_utc"] is not None
            ),
            default=None,
        ),
        "last_datetime_utc": max(
            (
                record["datetime_utc"]
                for record in retained
                if record["datetime_utc"] is not None
            ),
            default=None,
        ),
    }


def duplicate_group_summary(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}

    for record in records:
        grouped.setdefault(
            record["provisional_acquisition_key"],
            [],
        ).append(record)

    duplicates = []

    for key, group in grouped.items():
        if len(group) <= 1:
            continue

        duplicates.append(
            {
                "provisional_acquisition_key": key,
                "item_count": len(group),
                "item_ids": sorted(
                    record["item_id"]
                    for record in group
                ),
            }
        )

    duplicates.sort(
        key=lambda entry: (
            -entry["item_count"],
            entry["provisional_acquisition_key"],
        )
    )

    return duplicates


def validate_analysis(
    eligible_records: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
) -> None:
    for record in eligible_records:
        coverage = float(
            record["tile_coverage_fraction"]
        )

        if not 0.0 <= coverage <= 1.0:
            raise RuntimeError(
                f"Coverage outside [0, 1] for "
                f"{record['item_id']}: {coverage}"
            )

    ordered = sorted(
        summaries,
        key=lambda summary: summary[
            "threshold_fraction"
        ],
    )

    counts = [
        summary["retained_stac_item_count"]
        for summary in ordered
    ]

    if any(
        later > earlier
        for earlier, later in zip(
            counts,
            counts[1:],
        )
    ):
        raise RuntimeError(
            "Threshold counts are not monotonically "
            "non-increasing."
        )


def write_item_csv(
    path: Path,
    records: Sequence[dict[str, Any]],
) -> None:
    fieldnames = (
        "item_id",
        "datetime_utc",
        "end_datetime_utc",
        "year",
        "platform",
        "instrument_mode",
        "product_type",
        "orbit_state",
        "relative_orbit",
        "absolute_orbit",
        "declared_polarisations",
        "available_polarisations",
        "polarisation_combination",
        "tile_coverage_fraction",
        "tile_coverage_pct",
        "scene_overlap_fraction",
        "scene_overlap_pct",
        "eligible_before_coverage",
        "eligibility_rejection_reasons",
        "warnings",
        "missing_metadata_assets",
        "provisional_acquisition_key",
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for record in records:
            row = {
                field: record.get(field)
                for field in fieldnames
            }

            for field in (
                "declared_polarisations",
                "available_polarisations",
                "eligibility_rejection_reasons",
                "warnings",
                "missing_metadata_assets",
            ):
                row[field] = json.dumps(
                    row[field],
                    sort_keys=True,
                )

            writer.writerow(row)


def write_threshold_csv(
    path: Path,
    summaries: Sequence[dict[str, Any]],
) -> None:
    fieldnames = (
        "threshold_fraction",
        "threshold_pct",
        "retained_stac_item_count",
        "retained_provisional_acquisition_count",
        "retained_fraction_of_eligible_items",
        "first_datetime_utc",
        "last_datetime_utc",
        "minimum_gap_days",
        "median_gap_days",
        "mean_gap_days",
        "maximum_gap_days",
        "counts_by_year",
        "counts_by_platform",
        "counts_by_orbit_state",
        "counts_by_relative_orbit",
        "counts_by_polarisation_combination",
        "counts_by_polarisation",
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for summary in summaries:
            gaps = summary["time_gap_statistics"]

            writer.writerow(
                {
                    "threshold_fraction": summary[
                        "threshold_fraction"
                    ],
                    "threshold_pct": summary[
                        "threshold_pct"
                    ],
                    "retained_stac_item_count": summary[
                        "retained_stac_item_count"
                    ],
                    "retained_provisional_acquisition_count": (
                        summary[
                            "retained_provisional_acquisition_count"
                        ]
                    ),
                    "retained_fraction_of_eligible_items": (
                        summary[
                            "retained_fraction_of_eligible_items"
                        ]
                    ),
                    "first_datetime_utc": summary[
                        "first_datetime_utc"
                    ],
                    "last_datetime_utc": summary[
                        "last_datetime_utc"
                    ],
                    "minimum_gap_days": gaps[
                        "minimum_gap_days"
                    ],
                    "median_gap_days": gaps[
                        "median_gap_days"
                    ],
                    "mean_gap_days": gaps[
                        "mean_gap_days"
                    ],
                    "maximum_gap_days": gaps[
                        "maximum_gap_days"
                    ],
                    "counts_by_year": json.dumps(
                        summary["counts_by_year"],
                        sort_keys=True,
                    ),
                    "counts_by_platform": json.dumps(
                        summary["counts_by_platform"],
                        sort_keys=True,
                    ),
                    "counts_by_orbit_state": json.dumps(
                        summary["counts_by_orbit_state"],
                        sort_keys=True,
                    ),
                    "counts_by_relative_orbit": json.dumps(
                        summary[
                            "counts_by_relative_orbit"
                        ],
                        sort_keys=True,
                    ),
                    "counts_by_polarisation_combination": (
                        json.dumps(
                            summary[
                                "counts_by_polarisation_combination"
                            ],
                            sort_keys=True,
                        )
                    ),
                    "counts_by_polarisation": json.dumps(
                        summary[
                            "counts_by_polarisation"
                        ],
                        sort_keys=True,
                    ),
                }
            )


def main() -> None:
    args = parse_args()

    normalised_tile = normalize_tile(args.tile)
    tile_name = f"T{normalised_tile}"

    output_directory = Path(
        args.output_directory
        or (
            "reports/s1_development/"
            "threshold_analysis/"
            f"{tile_name}_{args.start_date}_to_{args.end_date}"
        )
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = Client.open(EARTH_SEARCH_URL)

    s2_item, s2_blue_asset = discover_s2_grid_item(
        client,
        normalised_tile,
        args.start_date,
        args.end_date,
    )

    target_grid = build_s2_target_grid(
        s2_item,
        s2_blue_asset,
        args.out_dim,
    )

    search = client.search(
        collections=[S1_COLLECTION],
        datetime=f"{args.start_date}/{args.end_date}",
        intersects=target_grid["tile_geometry_wgs84"],
        max_items=args.max_items,
        limit=args.page_limit,
    )

    items = list(search.items())
    items.sort(
        key=lambda item: (
            item.datetime is None,
            item.datetime,
            item.id,
        )
    )

    records = [
        inspect_item(
            item,
            target_grid,
            args.allowed_platforms,
        )
        for item in items
    ]

    eligible_records = [
        record
        for record in records
        if record["eligible_before_coverage"]
    ]

    summaries = [
        threshold_summary(
            eligible_records,
            threshold,
        )
        for threshold in args.thresholds
    ]

    validate_analysis(
        eligible_records,
        summaries,
    )

    rejection_counts = Counter(
        reason
        for record in records
        for reason in record[
            "eligibility_rejection_reasons"
        ]
    )

    warning_counts = Counter(
        warning
        for record in records
        for warning in record["warnings"]
    )

    near_threshold_items = sorted(
        (
            record
            for record in eligible_records
            if abs(
                float(
                    record[
                        "tile_coverage_fraction"
                    ]
                )
                - args.primary_threshold
            )
            <= args.near_threshold_margin
        ),
        key=lambda record: (
            record["tile_coverage_fraction"],
            record["datetime_utc"] or "",
            record["item_id"],
        ),
    )

    duplicate_groups = duplicate_group_summary(
        eligible_records
    )

    report = {
        "analysis": {
            "name": (
                "Sentinel-1 geometric tile-coverage "
                "threshold analysis"
            ),
            "tile": tile_name,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "collection": S1_COLLECTION,
            "required_instrument_mode": (
                REQUIRED_INSTRUMENT_MODE
            ),
            "required_product_type": (
                REQUIRED_PRODUCT_TYPE
            ),
            "discovery_model": (
                "Sentinel1Acquisition"
            ),
            "eligible_platforms": (
                "all"
                if args.allowed_platforms is None
                else sorted(args.allowed_platforms)
            ),
            "known_polarisations": list(
                KNOWN_POLARISATIONS
            ),
            "coverage_formula": (
                "intersection_area / "
                "sentinel_2_tile_area"
            ),
            "coverage_type": (
                "STAC footprint geometric coverage; "
                "not raster nodata-aware coverage"
            ),
            "thresholds": list(args.thresholds),
            "primary_threshold": (
                args.primary_threshold
            ),
            "near_threshold_margin": (
                args.near_threshold_margin
            ),
            "max_items": args.max_items,
            "page_limit": args.page_limit,
            "provisional_acquisition_grouping": (
                "Diagnostic only. No STAC item was "
                "removed or selected using this key."
            ),
        },
        "target_grid": serialise_grid(
            target_grid
        ),
        "summary": {
            "intersecting_stac_item_count": len(items),
            "eligible_before_coverage_count": len(
                eligible_records
            ),
            "ineligible_before_coverage_count": (
                len(records) - len(eligible_records)
            ),
            "provisional_unique_acquisition_count": (
                len(
                    unique_acquisition_records(
                        eligible_records
                    )
                )
            ),
            "candidate_duplicate_group_count": len(
                duplicate_groups
            ),
            "counts_by_platform_all_items": counter_dict(
                record["platform"]
                for record in records
            ),
            "counts_by_instrument_mode_all_items": (
                counter_dict(
                    record["instrument_mode"]
                    for record in records
                )
            ),
            "counts_by_polarisation_combination_all_items": (
                counter_dict(
                    record[
                        "polarisation_combination"
                    ]
                    for record in records
                )
            ),
            "eligibility_rejection_reason_counts": dict(
                sorted(rejection_counts.items())
            ),
            "warning_counts": dict(
                sorted(warning_counts.items())
            ),
            "eligible_geometric_coverage_distribution": (
                coverage_distribution(
                    eligible_records
                )
            ),
        },
        "threshold_summaries": summaries,
        "near_primary_threshold_items": (
            near_threshold_items
        ),
        "candidate_duplicate_groups": (
            duplicate_groups
        ),
        "items": records,
    }

    json_path = (
        output_directory
        / "threshold_analysis.json"
    )
    item_csv_path = (
        output_directory
        / "items.csv"
    )
    threshold_csv_path = (
        output_directory
        / "threshold_summary.csv"
    )

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    write_item_csv(
        item_csv_path,
        records,
    )
    write_threshold_csv(
        threshold_csv_path,
        summaries,
    )

    print()
    print("Sentinel-1 coverage-threshold analysis")
    print("--------------------------------------")
    print(f"Tile:                       {tile_name}")
    print(
        "Date range:                 "
        f"{args.start_date} to {args.end_date}"
    )
    print(f"S2 reference item:          {s2_item.id}")
    print(f"Target CRS:                 {target_grid['crs']}")
    print(f"Intersecting STAC items:    {len(items)}")
    print(
        "Eligible before coverage:   "
        f"{len(eligible_records)}"
    )
    print(
        "Ineligible before coverage: "
        f"{len(records) - len(eligible_records)}"
    )

    print()
    print("Threshold results:")

    for summary in summaries:
        gaps = summary["time_gap_statistics"]

        print(
            f"  >= {summary['threshold_pct']:6.2f}% | "
            f"items={summary['retained_stac_item_count']:5d} | "
            "provisional acquisitions="
            f"{summary['retained_provisional_acquisition_count']:5d} | "
            "median gap days="
            f"{gaps['median_gap_days']}"
        )

    print()
    print(
        "Near primary threshold:     "
        f"{len(near_threshold_items)} items"
    )
    print(
        "Candidate duplicate groups: "
        f"{len(duplicate_groups)}"
    )
    print()
    print(f"JSON:          {json_path}")
    print(f"Items CSV:     {item_csv_path}")
    print(f"Threshold CSV: {threshold_csv_path}")


if __name__ == "__main__":
    main()

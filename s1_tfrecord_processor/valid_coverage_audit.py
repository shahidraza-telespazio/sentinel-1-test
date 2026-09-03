from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .extract_one_scene import destination_grid
from .raster_io import (
    aws_rasterio_environment,
    warp_gcp_asset_to_grid,
)
from .valid_area import (
    warp_gcp_valid_area_fraction_to_grid,
)


DEFAULT_AUDIT_THRESHOLDS = (0.60, 0.80)


def parse_fraction_list(value: str) -> tuple[float, ...]:
    fractions: list[float] = []

    for text in value.split(","):
        text = text.strip()

        if not text:
            continue

        try:
            fraction = float(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid fraction {text!r}"
            ) from exc

        if not 0.0 <= fraction <= 1.0:
            raise argparse.ArgumentTypeError(
                "Every fraction must be between 0 and 1."
            )

        fractions.append(fraction)

    if not fractions:
        raise argparse.ArgumentTypeError(
            "At least one fraction is required."
        )

    return tuple(sorted(set(fractions)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Sentinel-1 STAC geometric tile coverage with "
            "nodata-aware valid coverage on the final Sentinel-2 grid."
        )
    )

    parser.add_argument(
        "--threshold-report",
        required=True,
        help=(
            "JSON report created by "
            "s1_tfrecord_processor.threshold_analysis."
        ),
    )
    parser.add_argument(
        "--audit-thresholds",
        type=parse_fraction_list,
        default=DEFAULT_AUDIT_THRESHOLDS,
        help=(
            "Comma-separated thresholds around which scenes are selected. "
            "Default: 0.60,0.80"
        ),
    )
    parser.add_argument(
        "--items-per-side",
        type=int,
        default=2,
        help=(
            "Number of closest scenes below and above each threshold. "
            "Default: 2"
        ),
    )
    parser.add_argument(
        "--gap-threshold",
        type=float,
        default=0.80,
        help=(
            "Include the two acquisitions bordering the longest time gap "
            "at this geometric threshold. Default: 0.80"
        ),
    )
    parser.add_argument(
        "--item-id",
        action="append",
        default=[],
        help=(
            "Explicit STAC item to audit. May be supplied more than once."
        ),
    )
    parser.add_argument(
        "--output-directory",
        default=None,
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
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help=(
            "Write the report and return success even if one or more "
            "measurement assets cannot be audited."
        ),
    )

    args = parser.parse_args()

    if args.items_per_side <= 0:
        parser.error("--items-per-side must be greater than zero")

    if not 0.0 <= args.gap_threshold <= 1.0:
        parser.error("--gap-threshold must be between 0 and 1")

    if args.num_threads <= 0:
        parser.error("--num-threads must be greater than zero")

    args.audit_thresholds = tuple(
        sorted(
            set(args.audit_thresholds)
            | {args.gap_threshold}
        )
    )

    return args


def parse_datetime(
    value: str | None,
) -> datetime | None:
    if not value:
        return None

    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )


def record_sort_key(
    record: dict[str, Any],
) -> tuple[Any, ...]:
    timestamp = parse_datetime(
        record.get("datetime_utc")
    )

    return (
        timestamp is None,
        timestamp,
        str(record.get("item_id") or ""),
    )


def eligible_records(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    records = []

    for record in report.get("items", []):
        if not record.get("eligible_before_coverage"):
            continue

        measurement_assets = record.get(
            "measurement_assets"
        )

        if not isinstance(measurement_assets, dict):
            continue

        if not any(measurement_assets.values()):
            continue

        records.append(record)

    records.sort(key=record_sort_key)
    return records


def add_selection(
    selected: dict[str, dict[str, Any]],
    record: dict[str, Any],
    reason: str,
) -> None:
    item_id = str(record["item_id"])

    if item_id not in selected:
        selected[item_id] = {
            "record": record,
            "selection_reasons": set(),
        }

    selected[item_id][
        "selection_reasons"
    ].add(reason)


def closest_below(
    records: Sequence[dict[str, Any]],
    threshold: float,
    count: int,
) -> list[dict[str, Any]]:
    candidates = [
        record
        for record in records
        if float(
            record["tile_coverage_fraction"]
        ) < threshold
    ]

    candidates.sort(
        key=lambda record: (
            -float(
                record["tile_coverage_fraction"]
            ),
            record_sort_key(record),
        )
    )

    return candidates[:count]


def closest_above(
    records: Sequence[dict[str, Any]],
    threshold: float,
    count: int,
) -> list[dict[str, Any]]:
    candidates = [
        record
        for record in records
        if float(
            record["tile_coverage_fraction"]
        ) >= threshold
    ]

    candidates.sort(
        key=lambda record: (
            float(
                record["tile_coverage_fraction"]
            ),
            record_sort_key(record),
        )
    )

    return candidates[:count]


def longest_gap_pair(
    records: Sequence[dict[str, Any]],
    threshold: float,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    float,
] | None:
    retained = [
        record
        for record in records
        if (
            float(
                record["tile_coverage_fraction"]
            )
            >= threshold
            and parse_datetime(
                record.get("datetime_utc")
            )
            is not None
        )
    ]

    retained.sort(key=record_sort_key)

    longest: tuple[
        dict[str, Any],
        dict[str, Any],
        float,
    ] | None = None

    for previous, current in zip(
        retained,
        retained[1:],
    ):
        previous_time = parse_datetime(
            previous["datetime_utc"]
        )
        current_time = parse_datetime(
            current["datetime_utc"]
        )

        if (
            previous_time is None
            or current_time is None
        ):
            continue

        gap_days = (
            current_time - previous_time
        ).total_seconds() / 86400.0

        if (
            longest is None
            or gap_days > longest[2]
        ):
            longest = (
                previous,
                current,
                gap_days,
            )

    return longest


def select_records(
    report: dict[str, Any],
    thresholds: Sequence[float],
    items_per_side: int,
    gap_threshold: float,
    explicit_item_ids: Sequence[str],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any] | None,
]:
    records = eligible_records(report)

    by_item_id = {
        str(record["item_id"]): record
        for record in report.get("items", [])
    }

    selected: dict[str, dict[str, Any]] = {}

    for threshold in thresholds:
        label = f"{threshold:.4f}"

        for record in closest_below(
            records,
            threshold,
            items_per_side,
        ):
            add_selection(
                selected,
                record,
                f"closest_below_{label}",
            )

        for record in closest_above(
            records,
            threshold,
            items_per_side,
        ):
            add_selection(
                selected,
                record,
                f"closest_at_or_above_{label}",
            )

    gap = longest_gap_pair(
        records,
        gap_threshold,
    )

    gap_summary = None

    if gap is not None:
        previous, current, gap_days = gap

        add_selection(
            selected,
            previous,
            f"longest_gap_start_at_{gap_threshold:.4f}",
        )
        add_selection(
            selected,
            current,
            f"longest_gap_end_at_{gap_threshold:.4f}",
        )

        gap_summary = {
            "threshold_fraction": gap_threshold,
            "threshold_pct": gap_threshold * 100.0,
            "gap_days": gap_days,
            "previous_item_id": previous["item_id"],
            "previous_datetime_utc": previous[
                "datetime_utc"
            ],
            "current_item_id": current["item_id"],
            "current_datetime_utc": current[
                "datetime_utc"
            ],
        }

    missing_explicit_ids = []

    for item_id in explicit_item_ids:
        record = by_item_id.get(item_id)

        if record is None:
            missing_explicit_ids.append(item_id)
            continue

        add_selection(
            selected,
            record,
            "explicit_item_id",
        )

    if missing_explicit_ids:
        raise RuntimeError(
            "Explicit item IDs were not found in the "
            f"threshold report: {missing_explicit_ids}"
        )

    selections = []

    for entry in selected.values():
        selections.append(
            {
                "record": entry["record"],
                "selection_reasons": sorted(
                    entry["selection_reasons"]
                ),
            }
        )

    selections.sort(
        key=lambda entry: record_sort_key(
            entry["record"]
        )
    )

    return selections, gap_summary


def percentage_points(
    value: float,
) -> float:
    return value * 100.0


def combine_masks(
    masks: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if not masks:
        raise ValueError(
            "At least one valid mask is required."
        )

    joint = np.logical_and.reduce(masks)
    either = np.logical_or.reduce(masks)

    return joint, either


def threshold_status(
    *,
    threshold: float,
    geometric_fraction: float,
    polarisation_fractions: dict[str, float],
    joint_fraction: float | None,
    either_fraction: float | None,
    complete: bool,
) -> dict[str, Any]:
    return {
        "threshold_fraction": threshold,
        "threshold_pct": threshold * 100.0,
        "geometric_pass": (
            geometric_fraction >= threshold
        ),
        "per_polarisation_pass": {
            polarisation: fraction >= threshold
            for polarisation, fraction
            in polarisation_fractions.items()
        },
        "all_available_polarisations_pass": (
            all(
                fraction >= threshold
                for fraction
                in polarisation_fractions.values()
            )
            if complete
            and polarisation_fractions
            else None
        ),
        "any_available_polarisation_pass": (
            any(
                fraction >= threshold
                for fraction
                in polarisation_fractions.values()
            )
            if complete
            and polarisation_fractions
            else None
        ),
        "joint_valid_pass": (
            joint_fraction >= threshold
            if complete
            and joint_fraction is not None
            else None
        ),
        "either_valid_pass": (
            either_fraction >= threshold
            if complete
            and either_fraction is not None
            else None
        ),
        "geometric_vs_joint_changed": (
            (
                geometric_fraction >= threshold
            )
            != (
                joint_fraction >= threshold
            )
            if complete
            and joint_fraction is not None
            else None
        ),
        "geometric_vs_any_polarisation_changed": (
            (
                geometric_fraction >= threshold
            )
            != any(
                fraction >= threshold
                for fraction
                in polarisation_fractions.values()
            )
            if complete
            and polarisation_fractions
            else None
        ),
    }


def audit_record(
    *,
    record: dict[str, Any],
    selection_reasons: Sequence[str],
    destination_crs,
    destination_transform,
    destination_shape: tuple[int, int],
    thresholds: Sequence[float],
    num_threads: int,
) -> dict[str, Any]:
    measurement_assets = record.get(
        "measurement_assets",
        {},
    )

    if not isinstance(measurement_assets, dict):
        raise TypeError(
            f"{record['item_id']} has invalid measurement_assets"
        )

    requested_assets = {
        str(polarisation).upper(): str(href)
        for polarisation, href
        in measurement_assets.items()
        if href
    }

    if not requested_assets:
        raise RuntimeError(
            f"{record['item_id']} has no measurement assets"
        )

    requested_assets = dict(
        sorted(requested_assets.items())
    )

    occupancy_metadata: dict[str, Any] = {}
    valid_area_metadata: dict[str, Any] = {}

    occupancy_masks: dict[str, np.ndarray] = {}
    valid_area_fraction_grids: dict[
        str,
        np.ndarray,
    ] = {}

    errors: dict[str, str] = {}

    for polarisation, href in requested_assets.items():
        try:
            occupancy = warp_gcp_asset_to_grid(
                href=href,
                destination_crs=destination_crs,
                destination_transform=(
                    destination_transform
                ),
                destination_shape=destination_shape,
                polarisation=polarisation,
                num_threads=num_threads,
            )

            valid_area = (
                warp_gcp_valid_area_fraction_to_grid(
                    href=href,
                    destination_crs=destination_crs,
                    destination_transform=(
                        destination_transform
                    ),
                    destination_shape=destination_shape,
                    polarisation=polarisation,
                    num_threads=num_threads,
                )
            )

        except Exception as exc:
            errors[polarisation] = (
                f"{type(exc).__name__}:{exc}"
            )
            continue

        occupancy_metadata[polarisation] = (
            occupancy.metadata
        )
        valid_area_metadata[polarisation] = (
            valid_area.metadata
        )

        occupancy_masks[polarisation] = (
            occupancy.valid_mask
        )
        valid_area_fraction_grids[
            polarisation
        ] = valid_area.fraction_grid

    complete = (
        len(valid_area_fraction_grids)
        == len(requested_assets)
        and len(occupancy_masks)
        == len(requested_assets)
        and not errors
    )

    occupancy_fractions = {
        polarisation: float(mask.mean())
        for polarisation, mask
        in occupancy_masks.items()
    }

    area_weighted_fractions = {
        polarisation: float(
            fraction_grid.mean()
        )
        for polarisation, fraction_grid
        in valid_area_fraction_grids.items()
    }

    joint_occupancy_fraction = None
    either_occupancy_fraction = None
    occupancy_masks_equal = None

    area_fraction_grids_equal = None
    joint_area_weighted_fraction = None
    either_area_weighted_fraction = None

    if complete and occupancy_masks:
        ordered_polarisations = list(
            requested_assets
        )

        ordered_occupancy_masks = [
            occupancy_masks[polarisation]
            for polarisation
            in ordered_polarisations
        ]

        (
            joint_occupancy_mask,
            either_occupancy_mask,
        ) = combine_masks(
            ordered_occupancy_masks
        )

        joint_occupancy_fraction = float(
            joint_occupancy_mask.mean()
        )
        either_occupancy_fraction = float(
            either_occupancy_mask.mean()
        )

        occupancy_reference = (
            ordered_occupancy_masks[0]
        )

        occupancy_masks_equal = all(
            np.array_equal(
                occupancy_reference,
                mask,
            )
            for mask
            in ordered_occupancy_masks[1:]
        )

        ordered_area_grids = [
            valid_area_fraction_grids[
                polarisation
            ]
            for polarisation
            in ordered_polarisations
        ]

        area_reference = ordered_area_grids[0]

        area_fraction_grids_equal = all(
            np.allclose(
                area_reference,
                fraction_grid,
                rtol=0.0,
                atol=1e-6,
            )
            for fraction_grid
            in ordered_area_grids[1:]
        )

        # A per-cell mean mask does not by itself contain enough
        # information to calculate the exact intersection or union
        # of two different source masks.
        #
        # When every polarisation has the same fraction grid, their
        # exact joint and union area-weighted fractions are identical.
        if area_fraction_grids_equal:
            joint_area_weighted_fraction = float(
                area_reference.mean()
            )
            either_area_weighted_fraction = (
                joint_area_weighted_fraction
            )

    geometric_fraction = float(
        record["tile_coverage_fraction"]
    )

    per_polarisation_comparison = {}

    for polarisation in requested_assets:
        if (
            polarisation not in area_weighted_fractions
            or polarisation not in occupancy_fractions
        ):
            continue

        area_fraction = (
            area_weighted_fractions[
                polarisation
            ]
        )
        occupancy_fraction = (
            occupancy_fractions[
                polarisation
            ]
        )

        area_metadata = valid_area_metadata[
            polarisation
        ]

        per_polarisation_comparison[
            polarisation
        ] = {
            # Backwards-compatible aliases. These now refer to
            # area-weighted source validity, not binary occupancy.
            "actual_valid_fraction": (
                area_fraction
            ),
            "actual_valid_pct": (
                percentage_points(
                    area_fraction
                )
            ),
            "area_weighted_valid_fraction": (
                area_fraction
            ),
            "area_weighted_valid_pct": (
                percentage_points(
                    area_fraction
                )
            ),
            "output_pixel_occupancy_fraction": (
                occupancy_fraction
            ),
            "output_pixel_occupancy_pct": (
                percentage_points(
                    occupancy_fraction
                )
            ),
            "fully_valid_output_pixel_fraction": (
                area_metadata[
                    "fully_valid_output_pixel_fraction"
                ]
            ),
            "fully_valid_output_pixel_pct": (
                area_metadata[
                    "fully_valid_output_pixel_pct"
                ]
            ),
            "area_weighted_difference_from_geometric_fraction": (
                area_fraction
                - geometric_fraction
            ),
            "area_weighted_difference_from_geometric_pct_points": (
                percentage_points(
                    area_fraction
                    - geometric_fraction
                )
            ),
            # Backwards-compatible aliases.
            "difference_from_geometric_fraction": (
                area_fraction
                - geometric_fraction
            ),
            "difference_from_geometric_pct_points": (
                percentage_points(
                    area_fraction
                    - geometric_fraction
                )
            ),
        }

    status_by_threshold = {
        f"{threshold:.4f}": threshold_status(
            threshold=threshold,
            geometric_fraction=geometric_fraction,
            polarisation_fractions=(
                area_weighted_fractions
            ),
            joint_fraction=(
                joint_area_weighted_fraction
            ),
            either_fraction=(
                either_area_weighted_fraction
            ),
            complete=complete,
        )
        for threshold in thresholds
    }

    return {
        "item_id": record["item_id"],
        "datetime_utc": record.get(
            "datetime_utc"
        ),
        "platform": record.get("platform"),
        "instrument_mode": record.get(
            "instrument_mode"
        ),
        "product_type": record.get(
            "product_type"
        ),
        "orbit_state": record.get(
            "orbit_state"
        ),
        "relative_orbit": record.get(
            "relative_orbit"
        ),
        "absolute_orbit": record.get(
            "absolute_orbit"
        ),
        "available_polarisations": list(
            requested_assets
        ),
        "selection_reasons": list(
            selection_reasons
        ),
        "geometric_coverage_fraction": (
            geometric_fraction
        ),
        "geometric_coverage_pct": (
            percentage_points(
                geometric_fraction
            )
        ),
        "audit_complete": complete,
        "requested_asset_count": len(
            requested_assets
        ),
        "successful_asset_count": len(
            valid_area_fraction_grids
        ),
        "errors": errors,
        "per_polarisation": (
            per_polarisation_comparison
        ),

        "joint_output_pixel_occupancy_fraction": (
            joint_occupancy_fraction
        ),
        "joint_output_pixel_occupancy_pct": (
            percentage_points(
                joint_occupancy_fraction
            )
            if joint_occupancy_fraction
            is not None
            else None
        ),
        "either_output_pixel_occupancy_fraction": (
            either_occupancy_fraction
        ),
        "either_output_pixel_occupancy_pct": (
            percentage_points(
                either_occupancy_fraction
            )
            if either_occupancy_fraction
            is not None
            else None
        ),

        "joint_area_weighted_valid_fraction": (
            joint_area_weighted_fraction
        ),
        "joint_area_weighted_valid_pct": (
            percentage_points(
                joint_area_weighted_fraction
            )
            if joint_area_weighted_fraction
            is not None
            else None
        ),
        "either_area_weighted_valid_fraction": (
            either_area_weighted_fraction
        ),
        "either_area_weighted_valid_pct": (
            percentage_points(
                either_area_weighted_fraction
            )
            if either_area_weighted_fraction
            is not None
            else None
        ),

        # Backwards-compatible aliases now using area-weighted
        # coverage when the polarisation masks are equivalent.
        "joint_valid_fraction": (
            joint_area_weighted_fraction
        ),
        "joint_valid_pct": (
            percentage_points(
                joint_area_weighted_fraction
            )
            if joint_area_weighted_fraction
            is not None
            else None
        ),
        "either_valid_fraction": (
            either_area_weighted_fraction
        ),
        "either_valid_pct": (
            percentage_points(
                either_area_weighted_fraction
            )
            if either_area_weighted_fraction
            is not None
            else None
        ),

        "all_polarisation_masks_equal": (
            occupancy_masks_equal
        ),
        "all_area_fraction_grids_equal": (
            area_fraction_grids_equal
        ),
        "threshold_status": (
            status_by_threshold
        ),
        "warp_metadata": {
            polarisation: {
                "measurement_warp": (
                    occupancy_metadata.get(
                        polarisation
                    )
                ),
                "valid_area_warp": (
                    valid_area_metadata.get(
                        polarisation
                    )
                ),
            }
            for polarisation
            in requested_assets
        },
    }

def write_csv(
    path: Path,
    records: Iterable[dict[str, Any]],
    thresholds: Sequence[float],
) -> None:
    threshold_fields = []

    for threshold in thresholds:
        label = f"{threshold * 100.0:.2f}"

        threshold_fields.extend(
            [
                f"geometric_pass_{label}",
                f"joint_pass_{label}",
                f"any_polarisation_pass_{label}",
                f"geometric_vs_joint_changed_{label}",
            ]
        )

    fieldnames = [
        "item_id",
        "datetime_utc",
        "platform",
        "orbit_state",
        "relative_orbit",
        "available_polarisations",
        "selection_reasons",
        "geometric_coverage_pct",
        "audit_complete",
        "joint_valid_pct",
        "either_valid_pct",
        "all_polarisation_masks_equal",
        "per_polarisation",
        "errors",
        *threshold_fields,
    ]

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
                "item_id": record["item_id"],
                "datetime_utc": record[
                    "datetime_utc"
                ],
                "platform": record["platform"],
                "orbit_state": record[
                    "orbit_state"
                ],
                "relative_orbit": record[
                    "relative_orbit"
                ],
                "available_polarisations": (
                    json.dumps(
                        record[
                            "available_polarisations"
                        ]
                    )
                ),
                "selection_reasons": json.dumps(
                    record["selection_reasons"]
                ),
                "geometric_coverage_pct": (
                    record[
                        "geometric_coverage_pct"
                    ]
                ),
                "audit_complete": record[
                    "audit_complete"
                ],
                "joint_valid_pct": record[
                    "joint_valid_pct"
                ],
                "either_valid_pct": record[
                    "either_valid_pct"
                ],
                "all_polarisation_masks_equal": (
                    record[
                        "all_polarisation_masks_equal"
                    ]
                ),
                "per_polarisation": json.dumps(
                    record["per_polarisation"],
                    sort_keys=True,
                ),
                "errors": json.dumps(
                    record["errors"],
                    sort_keys=True,
                ),
            }

            for threshold in thresholds:
                key = f"{threshold:.4f}"
                label = (
                    f"{threshold * 100.0:.2f}"
                )
                status = record[
                    "threshold_status"
                ][key]

                row[
                    f"geometric_pass_{label}"
                ] = status["geometric_pass"]

                row[
                    f"joint_pass_{label}"
                ] = status["joint_valid_pass"]

                row[
                    f"any_polarisation_pass_{label}"
                ] = status[
                    "any_available_polarisation_pass"
                ]

                row[
                    f"geometric_vs_joint_changed_{label}"
                ] = status[
                    "geometric_vs_joint_changed"
                ]

            writer.writerow(row)



def outcome_counts(
    values: Iterable[bool | None],
) -> dict[str, int]:
    outcomes = list(values)

    changed_count = sum(
        value is True
        for value in outcomes
    )
    unchanged_count = sum(
        value is False
        for value in outcomes
    )
    unknown_count = sum(
        value is None
        for value in outcomes
    )

    if (
        changed_count
        + unchanged_count
        + unknown_count
        != len(outcomes)
    ):
        raise RuntimeError(
            "Unexpected threshold comparison outcome."
        )

    return {
        "known_count": (
            changed_count + unchanged_count
        ),
        "changed_count": changed_count,
        "unchanged_count": unchanged_count,
        "unknown_count": unknown_count,
    }


def summarise_threshold_statuses(
    audited_records: Sequence[dict[str, Any]],
    thresholds: Sequence[float],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}

    for threshold in thresholds:
        key = f"{threshold:.4f}"

        joint_outcomes = outcome_counts(
            record["threshold_status"][key][
                "geometric_vs_joint_changed"
            ]
            for record in audited_records
        )

        any_polarisation_outcomes = (
            outcome_counts(
                record["threshold_status"][key][
                    "geometric_vs_any_polarisation_changed"
                ]
                for record in audited_records
            )
        )

        joint_pass_availability = (
            outcome_counts(
                (
                    None
                    if record[
                        "threshold_status"
                    ][key][
                        "joint_valid_pass"
                    ]
                    is None
                    else False
                )
                for record in audited_records
            )
        )

        any_pass_availability = (
            outcome_counts(
                (
                    None
                    if record[
                        "threshold_status"
                    ][key][
                        "any_available_polarisation_pass"
                    ]
                    is None
                    else False
                )
                for record in audited_records
            )
        )

        summaries[key] = {
            "threshold_fraction": threshold,
            "threshold_pct": threshold * 100.0,
            "audited_item_count": len(
                audited_records
            ),
            "complete_item_count": sum(
                record["audit_complete"]
                for record in audited_records
            ),

            "joint_area_weighted_available_count": (
                joint_pass_availability[
                    "known_count"
                ]
            ),
            "joint_area_weighted_unknown_count": (
                joint_pass_availability[
                    "unknown_count"
                ]
            ),

            "geometric_vs_joint_changed_count": (
                joint_outcomes[
                    "changed_count"
                ]
            ),
            "geometric_vs_joint_unchanged_count": (
                joint_outcomes[
                    "unchanged_count"
                ]
            ),
            "geometric_vs_joint_unknown_count": (
                joint_outcomes[
                    "unknown_count"
                ]
            ),

            "any_polarisation_available_count": (
                any_pass_availability[
                    "known_count"
                ]
            ),
            "any_polarisation_unknown_count": (
                any_pass_availability[
                    "unknown_count"
                ]
            ),

            "geometric_vs_any_polarisation_changed_count": (
                any_polarisation_outcomes[
                    "changed_count"
                ]
            ),
            "geometric_vs_any_polarisation_unchanged_count": (
                any_polarisation_outcomes[
                    "unchanged_count"
                ]
            ),
            "geometric_vs_any_polarisation_unknown_count": (
                any_polarisation_outcomes[
                    "unknown_count"
                ]
            ),
        }

    return summaries

def main() -> None:
    args = parse_args()

    threshold_report_path = Path(
        args.threshold_report
    )

    report = json.loads(
        threshold_report_path.read_text(
            encoding="utf-8"
        )
    )

    selections, gap_summary = select_records(
        report=report,
        thresholds=args.audit_thresholds,
        items_per_side=args.items_per_side,
        gap_threshold=args.gap_threshold,
        explicit_item_ids=args.item_id,
    )

    if not selections:
        raise RuntimeError(
            "No records were selected for valid-coverage auditing."
        )

    (
        destination_crs,
        destination_transform,
        destination_shape,
    ) = destination_grid(report)

    output_directory = Path(
        args.output_directory
        or (
            threshold_report_path.parent
            / "valid_coverage_audit"
        )
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    audited_records: list[dict[str, Any]] = []

    print()
    print("Sentinel-1 nodata-aware coverage audit")
    print("--------------------------------------")
    print(
        f"Tile:              "
        f"{report['analysis']['tile']}"
    )
    print(
        f"Selected items:    {len(selections)}"
    )
    print(
        f"Destination CRS:   {destination_crs}"
    )
    print(
        f"Destination shape: {destination_shape}"
    )

    with aws_rasterio_environment(
        profile=args.aws_profile,
        region=args.aws_region,
    ):
        for index, selection in enumerate(
            selections,
            start=1,
        ):
            record = selection["record"]

            print()
            print(
                f"[{index}/{len(selections)}] "
                f"{record['item_id']}"
            )
            print(
                "  geometric coverage: "
                f"{record['tile_coverage_pct']:.6f}%"
            )
            print(
                "  reasons: "
                f"{selection['selection_reasons']}"
            )

            audit = audit_record(
                record=record,
                selection_reasons=(
                    selection[
                        "selection_reasons"
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
                thresholds=(
                    args.audit_thresholds
                ),
                num_threads=args.num_threads,
            )

            audited_records.append(audit)

            print(
                "  complete: "
                f"{audit['audit_complete']}"
            )
            print(
                "  joint area-weighted valid: "
                f"{audit['joint_valid_pct']}"
            )
            print(
                "  either area-weighted valid: "
                f"{audit['either_valid_pct']}"
            )

            for polarisation, values in (
                audit[
                    "per_polarisation"
                ].items()
            ):
                print(
                    f"  {polarisation}: "
                    f"{values['actual_valid_pct']:.6f}% "
                    "(difference from geometry "
                    f"{values['difference_from_geometric_pct_points']:+.6f} "
                    "percentage points)"
                )

            if audit["errors"]:
                print(
                    f"  errors: {audit['errors']}"
                )

    incomplete = [
        record
        for record in audited_records
        if not record["audit_complete"]
    ]

    threshold_change_counts = (
        summarise_threshold_statuses(
            audited_records,
            args.audit_thresholds,
        )
    )

    output_report = {
        "audit": {
            "name": (
                "Sentinel-1 nodata-aware valid "
                "coverage audit"
            ),
            "threshold_report": str(
                threshold_report_path
            ),
            "tile": report["analysis"]["tile"],
            "date_range": {
                "start": report[
                    "analysis"
                ]["start_date"],
                "end": report[
                    "analysis"
                ]["end_date"],
            },
            "audit_thresholds": list(
                args.audit_thresholds
            ),
            "items_per_side": (
                args.items_per_side
            ),
            "gap_threshold": (
                args.gap_threshold
            ),
            "coverage_definition": (
                "Area-weighted mean of the source validity mask "
                "reprojected onto the final Sentinel-2 target "
                "grid. Source-mask zero values remain data during "
                "average resampling so partially covered output "
                "pixels contribute proportionally."
            ),
            "occupancy_definition": (
                "Fraction of final target-grid pixels receiving "
                "at least one valid source contribution. This is "
                "reported diagnostically but is not used as the "
                "authoritative area-weighted coverage measure."
            ),
            "value_interpretation": (
                "Source values remain uncalibrated. "
                "Only valid masks are assessed."
            ),
        },
        "target_grid": report["target_grid"],
        "longest_gap_selection": gap_summary,
        "summary": {
            "selected_item_count": len(
                selections
            ),
            "complete_item_count": (
                len(audited_records)
                - len(incomplete)
            ),
            "incomplete_item_count": len(
                incomplete
            ),
            "threshold_change_counts": (
                threshold_change_counts
            ),
        },
        "items": audited_records,
    }

    json_path = (
        output_directory
        / "valid_coverage_audit.json"
    )
    csv_path = (
        output_directory
        / "valid_coverage_audit.csv"
    )

    json_path.write_text(
        json.dumps(
            output_report,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    write_csv(
        csv_path,
        audited_records,
        args.audit_thresholds,
    )

    print()
    print("===== AUDIT SUMMARY =====")
    print(
        f"Complete items:   "
        f"{len(audited_records) - len(incomplete)}"
    )
    print(
        f"Incomplete items: {len(incomplete)}"
    )

    for threshold in args.audit_thresholds:
        key = f"{threshold:.4f}"
        summary = threshold_change_counts[key]

        print(
            f"At {threshold * 100.0:.2f}%:"
        )
        print(
            "  geometry/joint: "
            f"changed="
            f"{summary['geometric_vs_joint_changed_count']}, "
            f"unchanged="
            f"{summary['geometric_vs_joint_unchanged_count']}, "
            f"unknown="
            f"{summary['geometric_vs_joint_unknown_count']}"
        )
        print(
            "  geometry/any-polarisation: "
            f"changed="
            f"{summary['geometric_vs_any_polarisation_changed_count']}, "
            f"unchanged="
            f"{summary['geometric_vs_any_polarisation_unchanged_count']}, "
            f"unknown="
            f"{summary['geometric_vs_any_polarisation_unknown_count']}"
        )

    print()
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")

    if incomplete and not args.allow_errors:
        raise RuntimeError(
            f"{len(incomplete)} audited items were incomplete. "
            "The report was written; inspect the asset errors."
        )


if __name__ == "__main__":
    main()

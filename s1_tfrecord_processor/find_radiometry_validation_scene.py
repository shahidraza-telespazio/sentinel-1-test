from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pystac_client import Client

from .discovery import (
    Sentinel1Acquisition,
    acquisition_from_item,
)
from .inspect_metadata_assets import (
    build_s3_client,
    read_asset_bytes,
)
from .radiometry_metadata import (
    NoiseMetadata,
    parse_noise_metadata,
    parse_product_radiometry_metadata,
)


EARTH_SEARCH_URL = (
    "https://earth-search.aws.element84.com/v1"
)
S1_COLLECTION = "sentinel-1-grd"


@dataclass(frozen=True)
class PolarisationInspection:
    polarisation: str
    measurement_href: str | None
    product_metadata_href: str | None
    calibration_metadata_href: str | None
    noise_metadata_href: str | None
    thermal_noise_correction_performed: bool | None
    noise_format: str | None
    noise_minimum: float | None
    noise_maximum: float | None
    positive_noise_value_count: int
    usable_for_positive_noise_validation: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class SceneInspection:
    item_id: str
    datetime: str | None
    platform: str | None
    orbit_state: str | None
    relative_orbit: int | None
    absolute_orbit: int | None
    geometric_coverage_fraction: float | None
    polarisations: tuple[PolarisationInspection, ...]

    @property
    def usable_polarisations(self) -> tuple[str, ...]:
        return tuple(
            inspection.polarisation
            for inspection in self.polarisations
            if inspection.usable_for_positive_noise_validation
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find a Sentinel-1 GRD scene suitable for validating "
            "real positive thermal-noise subtraction. The command "
            "reads metadata XML only and does not read measurement pixels."
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
    )
    parser.add_argument(
        "--item-id",
        action="append",
        default=[],
        help=(
            "Inspect only this item ID. May be supplied more than once."
        ),
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output-json",
        default=(
            "reports/s1_development/"
            "radiometry_validation/"
            "validation_scene_search.json"
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


def report_items(
    report: dict[str, Any],
    *,
    minimum_geometric_coverage: float,
    requested_item_ids: set[str],
    max_scenes: int,
) -> list[dict[str, Any]]:
    raw_items = report.get("items")

    if not isinstance(raw_items, list):
        raw_items = report.get(
            "intersecting_items",
            [],
        )

    if not isinstance(raw_items, list):
        raise ValueError(
            "Threshold report contains no supported item list."
        )

    selected: list[dict[str, Any]] = []

    for item in raw_items:
        item_id = str(
            item.get("item_id", "")
        )

        if not item_id:
            continue

        if (
            requested_item_ids
            and item_id not in requested_item_ids
        ):
            continue

        if item.get(
            "eligible_before_coverage"
        ) is not True:
            continue

        coverage = item.get(
            "tile_coverage_fraction"
        )

        if coverage is None:
            continue

        if (
            float(coverage)
            < minimum_geometric_coverage
        ):
            continue

        selected.append(item)

    selected.sort(
        key=lambda item: (
            str(item.get("datetime", "")),
            str(item.get("item_id", "")),
        )
    )

    if max_scenes <= 0:
        raise ValueError(
            "--max-scenes must be positive."
        )

    return selected[:max_scenes]


def noise_statistics(
    metadata: NoiseMetadata,
) -> tuple[float, float, int]:
    arrays: list[np.ndarray] = []

    arrays.extend(
        vector.noise_lut
        for vector in metadata.legacy_vectors
    )
    arrays.extend(
        vector.noise_range_lut
        for vector in metadata.range_vectors
    )
    arrays.extend(
        vector.noise_azimuth_lut
        for vector in metadata.azimuth_vectors
    )

    if not arrays:
        raise ValueError(
            "Noise metadata contains no LUT arrays."
        )

    minimum = min(
        float(array.min())
        for array in arrays
    )
    maximum = max(
        float(array.max())
        for array in arrays
    )
    positive_count = sum(
        int(np.count_nonzero(array > 0.0))
        for array in arrays
    )

    return minimum, maximum, positive_count


def inspect_polarisation(
    *,
    acquisition: Sentinel1Acquisition,
    polarisation: str,
    s3_client,
) -> PolarisationInspection:
    matching = [
        value
        for value in acquisition.polarisation_assets
        if value.polarisation == polarisation
    ]

    if len(matching) != 1:
        return PolarisationInspection(
            polarisation=polarisation,
            measurement_href=None,
            product_metadata_href=None,
            calibration_metadata_href=None,
            noise_metadata_href=None,
            thermal_noise_correction_performed=None,
            noise_format=None,
            noise_minimum=None,
            noise_maximum=None,
            positive_noise_value_count=0,
            usable_for_positive_noise_validation=False,
            errors=(
                "Polarisation asset mapping is missing or ambiguous.",
            ),
        )

    assets = matching[0]
    errors: list[str] = []

    required_hrefs = {
        "measurement": assets.measurement_href,
        "product metadata": assets.product_metadata_href,
        "calibration metadata": (
            assets.calibration_metadata_href
        ),
        "noise metadata": assets.noise_metadata_href,
    }

    for label, href in required_hrefs.items():
        if not href:
            errors.append(
                f"Missing {label} asset."
            )

    if errors:
        return PolarisationInspection(
            polarisation=polarisation,
            measurement_href=assets.measurement_href,
            product_metadata_href=(
                assets.product_metadata_href
            ),
            calibration_metadata_href=(
                assets.calibration_metadata_href
            ),
            noise_metadata_href=(
                assets.noise_metadata_href
            ),
            thermal_noise_correction_performed=None,
            noise_format=None,
            noise_minimum=None,
            noise_maximum=None,
            positive_noise_value_count=0,
            usable_for_positive_noise_validation=False,
            errors=tuple(errors),
        )

    try:
        product_bytes = read_asset_bytes(
            assets.product_metadata_href,
            s3_client,
        )
        noise_bytes = read_asset_bytes(
            assets.noise_metadata_href,
            s3_client,
        )

        product = (
            parse_product_radiometry_metadata(
                product_bytes
            )
        )
        noise = parse_noise_metadata(
            noise_bytes
        )

        (
            noise_minimum,
            noise_maximum,
            positive_count,
        ) = noise_statistics(noise)

        usable = (
            product
            .thermal_noise_correction_performed
            is False
            and positive_count > 0
            and noise_maximum > 0.0
        )

        return PolarisationInspection(
            polarisation=polarisation,
            measurement_href=assets.measurement_href,
            product_metadata_href=(
                assets.product_metadata_href
            ),
            calibration_metadata_href=(
                assets.calibration_metadata_href
            ),
            noise_metadata_href=(
                assets.noise_metadata_href
            ),
            thermal_noise_correction_performed=(
                product
                .thermal_noise_correction_performed
            ),
            noise_format=noise.format_name,
            noise_minimum=noise_minimum,
            noise_maximum=noise_maximum,
            positive_noise_value_count=(
                positive_count
            ),
            usable_for_positive_noise_validation=usable,
            errors=(),
        )
    except Exception as error:
        return PolarisationInspection(
            polarisation=polarisation,
            measurement_href=assets.measurement_href,
            product_metadata_href=(
                assets.product_metadata_href
            ),
            calibration_metadata_href=(
                assets.calibration_metadata_href
            ),
            noise_metadata_href=(
                assets.noise_metadata_href
            ),
            thermal_noise_correction_performed=None,
            noise_format=None,
            noise_minimum=None,
            noise_maximum=None,
            positive_noise_value_count=0,
            usable_for_positive_noise_validation=False,
            errors=(
                f"{type(error).__name__}: {error}",
            ),
        )


def inspect_scene(
    *,
    item,
    report_item: dict[str, Any],
    s3_client,
) -> SceneInspection:
    acquisition = acquisition_from_item(
        item
    )

    inspections = tuple(
        inspect_polarisation(
            acquisition=acquisition,
            polarisation=polarisation,
            s3_client=s3_client,
        )
        for polarisation in (
            acquisition.available_polarisations
        )
    )

    properties = item.properties
    item_datetime = (
        item.datetime.isoformat()
        if item.datetime is not None
        else (
            properties.get("datetime")
            or properties.get("start_datetime")
        )
    )

    return SceneInspection(
        item_id=acquisition.item_id,
        datetime=(
            str(item_datetime)
            if item_datetime is not None
            else None
        ),
        platform=(
            str(properties.get("platform"))
            if properties.get("platform") is not None
            else None
        ),
        orbit_state=(
            str(properties.get("sat:orbit_state"))
            if properties.get("sat:orbit_state") is not None
            else None
        ),
        relative_orbit=(
            int(properties["sat:relative_orbit"])
            if properties.get("sat:relative_orbit") is not None
            else None
        ),
        absolute_orbit=(
            int(properties["sat:absolute_orbit"])
            if properties.get("sat:absolute_orbit") is not None
            else None
        ),
        geometric_coverage_fraction=(
            float(
                report_item[
                    "tile_coverage_fraction"
                ]
            )
        ),
        polarisations=inspections,
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

    selected_report_items = report_items(
        report,
        minimum_geometric_coverage=(
            args.minimum_geometric_coverage
        ),
        requested_item_ids=set(
            args.item_id
        ),
        max_scenes=args.max_scenes,
    )

    if not selected_report_items:
        raise RuntimeError(
            "No report scenes satisfy the configured filters."
        )

    print(
        "Sentinel-1 radiometry validation-scene search"
    )
    print("---------------------------------------------")
    print(
        "Candidate scenes:           "
        f"{len(selected_report_items)}"
    )
    print(
        "Minimum geometric coverage: "
        f"{args.minimum_geometric_coverage:.2%}"
    )
    print(
        "Measurement pixels read:    no"
    )
    print()

    client = Client.open(
        EARTH_SEARCH_URL
    )
    s3_client = build_s3_client(
        profile=args.aws_profile,
        region=args.aws_region,
    )

    inspections: list[
        SceneInspection
    ] = []

    for index, report_item in enumerate(
        selected_report_items,
        start=1,
    ):
        item_id = str(
            report_item["item_id"]
        )

        print(
            f"[{index}/{len(selected_report_items)}] "
            f"{item_id}"
        )

        items = list(
            client.search(
                collections=[S1_COLLECTION],
                ids=[item_id],
                max_items=1,
            ).items()
        )

        if len(items) != 1:
            print(
                "  SKIP: STAC item lookup "
                f"returned {len(items)} items."
            )
            continue

        inspection = inspect_scene(
            item=items[0],
            report_item=report_item,
            s3_client=s3_client,
        )
        inspections.append(inspection)

        for polarisation in (
            inspection.polarisations
        ):
            if polarisation.errors:
                print(
                    "  "
                    f"{polarisation.polarisation}: "
                    "ERROR "
                    + "; ".join(
                        polarisation.errors
                    )
                )
                continue

            print(
                "  "
                f"{polarisation.polarisation}: "
                "flag="
                f"{polarisation.thermal_noise_correction_performed}, "
                "format="
                f"{polarisation.noise_format}, "
                "noise_max="
                f"{polarisation.noise_maximum}, "
                "positive_values="
                f"{polarisation.positive_noise_value_count}, "
                "usable="
                f"{polarisation.usable_for_positive_noise_validation}"
            )

    usable = [
        {
            "item_id": inspection.item_id,
            "polarisation": polarisation,
        }
        for inspection in inspections
        for polarisation in (
            inspection.usable_polarisations
        )
    ]

    output = {
        "request": {
            "threshold_report": str(
                report_path
            ),
            "minimum_geometric_coverage": (
                args.minimum_geometric_coverage
            ),
            "requested_item_ids": (
                list(args.item_id)
            ),
            "max_scenes": args.max_scenes,
        },
        "summary": {
            "candidate_scene_count": len(
                selected_report_items
            ),
            "inspected_scene_count": len(
                inspections
            ),
            "usable_scene_polarisation_count": len(
                usable
            ),
            "measurement_pixels_read": False,
        },
        "usable_scene_polarisations": usable,
        "scenes": [
            {
                **asdict(inspection),
                "usable_polarisations": list(
                    inspection
                    .usable_polarisations
                ),
            }
            for inspection in inspections
        ],
    }

    output_path = Path(
        args.output_json
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Usable scene/polarisation pairs: "
        f"{len(usable)}"
    )

    for value in usable:
        print(
            "  "
            f"{value['item_id']} "
            f"{value['polarisation']}"
        )

    print(f"JSON: {output_path}")


if __name__ == "__main__":
    main()

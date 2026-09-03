from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen
from xml.etree import ElementTree

import boto3
from pystac_client import Client


EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
S1_COLLECTION = "sentinel-1-grd"

DEFAULT_INTERSECTION_REPORT = (
    "reports/s1_prototype/"
    "T26SKG_2025-06-01_to_2025-06-30_intersection_report.json"
)

KNOWN_POLARISATIONS = ("VV", "VH", "HH", "HV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and inspect the exact Sentinel-1 product, "
            "calibration and noise metadata assets for one STAC item."
        )
    )

    parser.add_argument(
        "--intersection-report",
        default=DEFAULT_INTERSECTION_REPORT,
    )
    parser.add_argument(
        "--item-id",
        default=None,
        help=(
            "Specific accepted item. When omitted, the accepted item "
            "with the highest geometric tile coverage is selected."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="reports/s1_development/metadata_assets",
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


def select_item_id(
    report: dict[str, Any],
    requested_item_id: str | None,
) -> str:
    accepted = report.get("accepted_items", [])

    if not accepted:
        raise RuntimeError(
            "The supplied intersection report contains no accepted items."
        )

    if requested_item_id is not None:
        matching = [
            item
            for item in accepted
            if item.get("item_id") == requested_item_id
        ]

        if len(matching) != 1:
            available = [
                item.get("item_id")
                for item in accepted
            ]

            raise RuntimeError(
                f"Item {requested_item_id!r} is not an accepted item. "
                f"Available accepted items: {available}"
            )

        return requested_item_id

    selected = max(
        accepted,
        key=lambda item: float(
            item["tile_coverage_fraction"]
        ),
    )

    return str(selected["item_id"])


def build_s3_client(
    profile: str | None,
    region: str,
):
    if profile:
        session = boto3.Session(
            profile_name=profile,
            region_name=region,
        )
    else:
        session = boto3.Session(
            region_name=region,
        )

    return session.client("s3")


def read_asset_bytes(
    href: str,
    s3_client,
) -> bytes:
    parsed = urlparse(href)

    if parsed.scheme == "s3":
        response = s3_client.get_object(
            Bucket=parsed.netloc,
            Key=parsed.path.lstrip("/"),
            RequestPayer="requester",
        )
        return response["Body"].read()

    if parsed.scheme in {"http", "https"}:
        with urlopen(href) as response:
            return response.read()

    raise ValueError(
        f"Unsupported metadata asset URL scheme: {href!r}"
    )


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def first_non_empty_text(
    root: ElementTree.Element,
    element_name: str,
) -> str | None:
    for element in root.iter():
        if local_name(element.tag) != element_name:
            continue

        if element.text is None:
            continue

        value = element.text.strip()
        if value:
            return value

    return None


def xml_summary(content: bytes) -> dict[str, Any]:
    root = ElementTree.fromstring(content)

    counts = Counter(
        local_name(element.tag)
        for element in root.iter()
    )

    important_count_names = (
        "calibrationVector",
        "noiseVector",
        "noiseRangeVector",
        "noiseAzimuthVector",
        "geolocationGridPoint",
        "burst",
    )

    selected_value_names = (
        "missionId",
        "productType",
        "mode",
        "swath",
        "polarisation",
        "startTime",
        "stopTime",
        "numberOfSamples",
        "numberOfLines",
        "pixelValue",
        "thermalNoiseCorrectionPerformed",
        "absoluteCalibrationConstant",
    )

    return {
        "root_element": local_name(root.tag),
        "total_elements": int(sum(counts.values())),
        "element_type_count": len(counts),
        "important_element_counts": {
            name: int(counts.get(name, 0))
            for name in important_count_names
        },
        "selected_values": {
            name: first_non_empty_text(root, name)
            for name in selected_value_names
        },
        "all_element_counts": dict(
            sorted(counts.items())
        ),
    }


def available_polarisations(item) -> list[str]:
    property_values = item.properties.get(
        "sar:polarizations",
        [],
    )

    if isinstance(property_values, str):
        property_values = [property_values]

    property_polarisations = {
        str(value).upper()
        for value in property_values
    }

    asset_polarisations = {
        polarisation
        for polarisation in KNOWN_POLARISATIONS
        if polarisation.lower() in item.assets
    }

    return sorted(
        property_polarisations | asset_polarisations,
        key=KNOWN_POLARISATIONS.index,
    )


def metadata_asset_keys(item) -> list[str]:
    keys: list[str] = []

    for key, asset in item.assets.items():
        roles = set(asset.roles or [])
        media_type = asset.media_type or ""

        is_xml = "xml" in media_type.lower()
        is_metadata = "metadata" in roles
        has_expected_name = (
            key == "safe-manifest"
            or key.startswith("schema-")
        )

        if is_xml or is_metadata or has_expected_name:
            keys.append(key)

    return sorted(set(keys))


def validate_expected_assets(
    item,
    polarisations: list[str],
) -> list[str]:
    missing: list[str] = []

    for polarisation in polarisations:
        suffix = polarisation.lower()

        expected = (
            suffix,
            f"schema-product-{suffix}",
            f"schema-calibration-{suffix}",
            f"schema-noise-{suffix}",
        )

        for key in expected:
            if key not in item.assets:
                missing.append(key)

    if "safe-manifest" not in item.assets:
        missing.append("safe-manifest")

    return sorted(set(missing))


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

    item_id = select_item_id(
        report,
        args.item_id,
    )

    client = Client.open(EARTH_SEARCH_URL)

    items = list(
        client.search(
            collections=[S1_COLLECTION],
            ids=[item_id],
            max_items=1,
        ).items()
    )

    if len(items) != 1:
        raise RuntimeError(
            f"Expected exactly one STAC item for {item_id!r}, "
            f"but found {len(items)}."
        )

    item = items[0]

    polarisations = available_polarisations(item)
    missing_expected_assets = validate_expected_assets(
        item,
        polarisations,
    )

    output_directory = (
        Path(args.output_root) / item.id
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    item_json_path = (
        output_directory / "stac_item.json"
    )
    item_json_path.write_text(
        json.dumps(
            item.to_dict(),
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    s3_client = build_s3_client(
        args.aws_profile,
        args.aws_region,
    )

    asset_inventory: dict[str, Any] = {}
    xml_summaries: dict[str, Any] = {}

    print()
    print("Sentinel-1 metadata asset inspection")
    print("------------------------------------")
    print(f"Item:          {item.id}")
    print(f"Collection:    {item.collection_id}")
    print(f"Platform:      {item.properties.get('platform')}")
    print(
        "Polarisations: "
        f"{polarisations}"
    )
    print(
        "Orbit state:   "
        f"{item.properties.get('sat:orbit_state')}"
    )
    print(
        "Relative orbit:"
        f" {item.properties.get('sat:relative_orbit')}"
    )

    print()
    print("===== ALL STAC ASSETS =====")

    for key in sorted(item.assets):
        asset = item.assets[key]

        asset_inventory[key] = {
            "href": asset.href,
            "media_type": asset.media_type,
            "roles": list(asset.roles or []),
            "title": asset.title,
        }

        print(
            f"{key:24s} | "
            f"type={asset.media_type!r} | "
            f"roles={list(asset.roles or [])}"
        )

    print()
    print("===== EXPECTED-ASSET VALIDATION =====")

    if missing_expected_assets:
        print(
            "Missing expected assets: "
            f"{missing_expected_assets}"
        )
    else:
        print(
            "All expected measurement, product, calibration, "
            "noise and manifest assets are present."
        )

    print()
    print("===== XML METADATA =====")

    for key in metadata_asset_keys(item):
        asset = item.assets[key]

        content = read_asset_bytes(
            asset.href,
            s3_client,
        )

        output_path = (
            output_directory / f"{key}.xml"
        )
        output_path.write_bytes(content)

        summary = xml_summary(content)
        xml_summaries[key] = {
            "href": asset.href,
            "saved_path": str(output_path),
            "byte_count": len(content),
            **summary,
        }

        print()
        print(f"[{key}]")
        print(f"root:   {summary['root_element']}")
        print(f"bytes:  {len(content)}")
        print(
            "counts: "
            f"{summary['important_element_counts']}"
        )
        print(
            "values: "
            f"{summary['selected_values']}"
        )

    summary_path = (
        output_directory / "inspection_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            {
                "item_id": item.id,
                "collection": item.collection_id,
                "properties": item.properties,
                "available_polarisations": polarisations,
                "missing_expected_assets": (
                    missing_expected_assets
                ),
                "asset_inventory": asset_inventory,
                "xml_summaries": xml_summaries,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print(f"STAC item: {item_json_path}")
    print(f"Summary:   {summary_path}")

    if missing_expected_assets:
        raise RuntimeError(
            "The tested item is missing expected Sentinel-1 "
            "measurement or metadata assets. See the saved summary."
        )


if __name__ == "__main__":
    main()

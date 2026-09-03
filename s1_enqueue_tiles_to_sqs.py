#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import yaml

from s1_worker_utils import (
    normalize_tile,
    s3_uri,
)


S1_MODE = "S1_GRD_DB"
S1_PRODUCT_LEVEL = "GRD"
S1_OUTPUT_FORMAT = "tfrecord"

CALIBRATION_CHOICES = [
    "sigmaNought",
    "betaNought",
    "gamma",
]
ACCEPTANCE_RULE_CHOICES = [
    "any",
    "all",
]
BAND_LAYOUT_CHOICES = [
    "union",
    "canonical",
]
UNKNOWN_NOISE_POLICY_CHOICES = [
    "error",
    "assume_uncorrected",
    "assume_corrected",
]


def normalise_s1_tile(tile: str) -> str:
    value = str(tile).strip().upper()
    if value.startswith("T"):
        return value
    return f"T{value}"


def processor_output_key(
    output_prefix_base: str,
    tile: str,
    start_date: str,
    end_date: str,
    calibration_lut: str,
) -> str:
    tile_name = normalise_s1_tile(tile)

    prefix = (
        f"{output_prefix_base.rstrip('/')}/"
        f"{tile_name}/"
        f"{start_date}_to_{end_date}/"
        f"{S1_MODE}"
    )

    name = (
        f"s1_grd_tile_{tile_name}_"
        f"{start_date}_to_{end_date}_"
        f"{calibration_lut}_db.tfrecord"
    )

    return f"{prefix}/{name}"


def build_session(
    profile: str | None,
    region: str,
) -> boto3.Session:
    if profile:
        return boto3.Session(
            profile_name=profile,
            region_name=region,
        )

    return boto3.Session(
        region_name=region
    )


def s3_client(
    session: boto3.Session,
    region: str,
):
    return session.client(
        "s3",
        config=Config(
            region_name=region,
            max_pool_connections=32,
            retries={
                "mode": "adaptive",
                "max_attempts": 10,
            },
        ),
    )


def sqs_client(
    session: boto3.Session,
    region: str,
):
    return session.client(
        "sqs",
        config=Config(
            region_name=region,
            max_pool_connections=32,
            retries={
                "mode": "adaptive",
                "max_attempts": 10,
            },
        ),
    )


def load_inventory(
    s3: Any,
    bucket: str,
    key: str,
) -> List[str]:
    obj = s3.get_object(
        Bucket=bucket,
        Key=key,
    )

    data = yaml.safe_load(
        obj["Body"].read()
    )

    # Sentinel-1 is processed against the existing Sentinel-2 MGRS tile
    # inventory, so the historical "sentinel2_tiles" key remains valid.
    tiles = (
        data["sentinel2_tiles"]
        if (
            isinstance(data, dict)
            and "sentinel2_tiles" in data
        )
        else data
    )

    if not isinstance(
        tiles,
        list,
    ):
        raise ValueError(
            "Inventory must be a list of MGRS tiles or a mapping "
            "containing 'sentinel2_tiles'."
        )

    seen = set()
    result: List[str] = []

    for tile in tiles:
        normalised = normalise_s1_tile(
            normalize_tile(
                str(tile)
            )
        )

        if normalised not in seen:
            seen.add(
                normalised
            )
            result.append(
                normalised
            )

    return result


def destination_exists(
    s3: Any,
    bucket: str,
    key: str,
) -> bool:
    try:
        s3.head_object(
            Bucket=bucket,
            Key=key,
        )
        return True

    except ClientError as exc:
        code = (
            exc.response
            .get("Error", {})
            .get("Code", "")
        )

        if code in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return False

        raise


def iter_batches(
    items: Iterable[Dict[str, Any]],
    size: int = 10,
) -> Iterable[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []

    for item in items:
        batch.append(
            item
        )

        if len(batch) == size:
            yield batch
            batch = []

    if batch:
        yield batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enqueue standalone Sentinel-1 GRD TFRecord tile jobs "
            "from the existing MGRS YAML inventory."
        )
    )

    parser.add_argument(
        "--control-bucket",
        required=True,
    )
    parser.add_argument(
        "--inventory-key",
        required=True,
    )
    parser.add_argument(
        "--queue-url",
        required=True,
    )
    parser.add_argument(
        "--output-bucket",
        required=True,
    )
    parser.add_argument(
        "--output-prefix-base",
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
        "--calibration-lut",
        choices=CALIBRATION_CHOICES,
        default="sigmaNought",
    )
    parser.add_argument(
        "--minimum-geometric-coverage",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--minimum-valid-coverage",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--acceptance-rule",
        choices=ACCEPTANCE_RULE_CHOICES,
        default="all",
    )
    parser.add_argument(
        "--band-layout",
        choices=BAND_LAYOUT_CHOICES,
        default="union",
    )
    parser.add_argument(
        "--unknown-noise-policy",
        choices=UNKNOWN_NOISE_POLICY_CHOICES,
        default="error",
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
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    skip_group = (
        parser.add_mutually_exclusive_group()
    )
    skip_group.add_argument(
        "--skip-existing-output",
        dest="skip_existing_output",
        action="store_true",
    )
    skip_group.add_argument(
        "--no-skip-existing-output",
        dest="skip_existing_output",
        action="store_false",
    )

    parser.set_defaults(
        skip_existing_output=True
    )

    args = parser.parse_args()

    if not (
        0.0
        <= args.minimum_geometric_coverage
        <= 1.0
    ):
        parser.error(
            "--minimum-geometric-coverage must be between 0 and 1"
        )

    if not (
        0.0
        <= args.minimum_valid_coverage
        <= 1.0
    ):
        parser.error(
            "--minimum-valid-coverage must be between 0 and 1"
        )

    if (
        args.limit is not None
        and args.limit <= 0
    ):
        parser.error(
            "--limit must be greater than zero"
        )

    return args


def main() -> None:
    args = parse_args()

    session = build_session(
        args.aws_profile,
        args.aws_region,
    )

    s3 = s3_client(
        session,
        args.aws_region,
    )

    sqs = sqs_client(
        session,
        args.aws_region,
    )

    tiles = load_inventory(
        s3,
        args.control_bucket,
        args.inventory_key,
    )

    inventory_uri = s3_uri(
        args.control_bucket,
        args.inventory_key,
    )

    print(
        f"Loaded {len(tiles)} unique MGRS tiles "
        f"from {inventory_uri}",
        flush=True,
    )

    prepared: List[Dict[str, Any]] = []
    skipped_existing = 0

    for idx, tile in enumerate(
        tiles,
        start=1,
    ):
        if (
            args.limit is not None
            and len(prepared) >= args.limit
        ):
            break

        output_key = processor_output_key(
            args.output_prefix_base,
            tile,
            args.start_date,
            args.end_date,
            args.calibration_lut,
        )

        if (
            args.skip_existing_output
            and destination_exists(
                s3,
                args.output_bucket,
                output_key,
            )
        ):
            skipped_existing += 1
            continue

        payload = {
            "tile": tile,
            "processor_family": "S1",
            "mode": S1_MODE,
            "product_level": S1_PRODUCT_LEVEL,
            "output_format": S1_OUTPUT_FORMAT,
            "output_key": output_key,
            "inventory_uri": inventory_uri,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "calibration_lut": (
                args.calibration_lut
            ),
            "minimum_geometric_coverage": (
                args.minimum_geometric_coverage
            ),
            "minimum_valid_coverage": (
                args.minimum_valid_coverage
            ),
            "acceptance_rule": (
                args.acceptance_rule
            ),
            "band_layout": (
                args.band_layout
            ),
            "unknown_noise_policy": (
                args.unknown_noise_policy
            ),
        }

        prepared.append(
            payload
        )

        if idx % 500 == 0:
            print(
                f"Prepared {idx}/{len(tiles)} inventory entries "
                f"-> {len(prepared)} enqueue candidates",
                flush=True,
            )

    if args.dry_run:
        for payload in prepared[
            : min(
                20,
                len(prepared),
            )
        ]:
            print(
                json.dumps(
                    payload,
                    sort_keys=True,
                )
            )

        print(
            "Dry run complete: "
            f"candidates={len(prepared)} "
            f"skipped_existing={skipped_existing} "
            f"total_inventory={len(tiles)}",
            flush=True,
        )
        return

    sent = 0

    for batch_idx, batch in enumerate(
        iter_batches(
            prepared,
            size=10,
        ),
        start=1,
    ):
        entries = [
            {
                "Id": str(index),
                "MessageBody": json.dumps(
                    payload,
                    separators=(",", ":"),
                ),
            }
            for index, payload
            in enumerate(batch)
        ]

        response = sqs.send_message_batch(
            QueueUrl=args.queue_url,
            Entries=entries,
        )

        failed = response.get(
            "Failed",
            [],
        )

        if failed:
            raise RuntimeError(
                "send_message_batch failed: "
                f"{failed}"
            )

        sent += len(
            response.get(
                "Successful",
                [],
            )
        )

        if batch_idx % 100 == 0:
            print(
                f"Enqueued {sent}/{len(prepared)} messages",
                flush=True,
            )

    print(
        "Sentinel-1 enqueue complete: "
        f"sent={sent} "
        f"skipped_existing={skipped_existing} "
        f"total_inventory={len(tiles)} "
        f"queue={args.queue_url}",
        flush=True,
    )


if __name__ == "__main__":
    main()

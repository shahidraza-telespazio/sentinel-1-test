#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import socket
import subprocess
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import boto3


def utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def normalize_tile(tile: str) -> str:
    value = tile.strip().upper()

    if value.startswith("T") and len(value) == 6:
        return value[1:]

    return value


def s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def run_git(
    directory: str,
    args: List[str],
) -> Optional[str]:
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                directory,
                *args,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        return completed.stdout.strip()

    except Exception:
        return None


def file_version_info(
    path: str,
) -> Dict[str, Optional[str]]:
    resolved = Path(path).resolve()
    repo_dir = str(
        resolved.parent
    )

    return {
        "path": str(resolved),
        "sha256": (
            sha256_file(
                str(resolved)
            )
            if resolved.exists()
            else None
        ),
        "git_commit": run_git(
            repo_dir,
            [
                "rev-parse",
                "HEAD",
            ],
        ),
        "git_describe": run_git(
            repo_dir,
            [
                "describe",
                "--tags",
                "--always",
                "--dirty",
            ],
        ),
        "git_branch": run_git(
            repo_dir,
            [
                "rev-parse",
                "--abbrev-ref",
                "HEAD",
            ],
        ),
    }


def get_imds_token(
    timeout: float = 2.0,
) -> Optional[str]:
    request = urllib.request.Request(
        "http://169.254.169.254/latest/api/token",
        method="PUT",
        headers={
            "X-aws-ec2-metadata-token-ttl-seconds": "21600",
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            return (
                response.read()
                .decode("utf-8")
            )

    except Exception:
        return None


def get_imds(
    path: str,
    token: Optional[str],
    timeout: float = 2.0,
) -> Optional[str]:
    request = urllib.request.Request(
        (
            "http://169.254.169.254/latest/"
            f"{path}"
        ),
        headers=(
            {
                "X-aws-ec2-metadata-token": token,
            }
            if token
            else {}
        ),
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            return (
                response.read()
                .decode("utf-8")
            )

    except Exception:
        return None


def discover_instance_identity(
    session: boto3.Session,
    region: str,
) -> Dict[str, Optional[str]]:
    token = get_imds_token()

    instance_id = get_imds(
        "meta-data/instance-id",
        token,
    )

    hostname = socket.gethostname()

    document_raw = get_imds(
        "dynamic/instance-identity/document",
        token,
    )

    instance_name = None
    availability_zone = None

    if document_raw:
        with contextlib.suppress(
            Exception
        ):
            document = json.loads(
                document_raw
            )
            availability_zone = (
                document.get(
                    "availabilityZone"
                )
            )

    if instance_id:
        with contextlib.suppress(
            Exception
        ):
            ec2 = session.client(
                "ec2",
                region_name=region,
            )

            response = (
                ec2.describe_instances(
                    InstanceIds=[
                        instance_id
                    ]
                )
            )

            for reservation in response.get(
                "Reservations",
                [],
            ):
                for instance in reservation.get(
                    "Instances",
                    [],
                ):
                    for tag in (
                        instance.get(
                            "Tags",
                            [],
                        )
                        or []
                    ):
                        if (
                            tag.get("Key")
                            == "Name"
                        ):
                            instance_name = (
                                tag.get(
                                    "Value"
                                )
                            )
                            break

                    if instance_name:
                        break

                if instance_name:
                    break

    return {
        "instance_id": instance_id,
        "instance_name": instance_name,
        "hostname": hostname,
        "availability_zone": (
            availability_zone
        ),
    }

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from s1_worker_utils import (
    discover_instance_identity,
    file_version_info,
    normalize_tile,
    s3_uri,
    utc_now,
)


STOP = threading.Event()
logger = logging.getLogger(__name__)

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

IMDS_BASE_URL = "http://169.254.169.254/latest"
IMDS_TOKEN_TTL_SECONDS = 21600
IMDS_TOKEN_REFRESH_SECONDS = 3600


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


def local_output_path(
    output_root: str,
    tile: str,
    start_date: str,
    end_date: str,
    calibration_lut: str,
) -> str:
    tile_name = normalise_s1_tile(tile)

    name = (
        f"s1_grd_tile_{tile_name}_"
        f"{start_date}_to_{end_date}_"
        f"{calibration_lut}_db.tfrecord"
    )

    return os.path.join(
        output_root,
        tile_name,
        name,
    )


class VisibilityExtender(threading.Thread):
    """Best-effort SQS visibility refresher for long Sentinel-1 tile jobs."""

    def __init__(
        self,
        sqs_client: Any,
        queue_url: str,
        receipt_handle: str,
        visibility_timeout: int,
    ) -> None:
        super().__init__(daemon=True)
        self.sqs_client = sqs_client
        self.queue_url = queue_url
        self.receipt_handle = receipt_handle
        self.visibility_timeout = visibility_timeout
        self._stop_event = threading.Event()
        self._sleep = max(
            30,
            visibility_timeout // 3,
        )

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:  # pragma: no cover - best effort only
        while not self._stop_event.wait(self._sleep):
            try:
                self.sqs_client.change_message_visibility(
                    QueueUrl=self.queue_url,
                    ReceiptHandle=self.receipt_handle,
                    VisibilityTimeout=self.visibility_timeout,
                )
            except Exception as exc:
                logger.warning(
                    "visibility extension failed: %s",
                    exc,
                )


class SpotInterruptionWatcher(threading.Thread):
    """
    Watch EC2 instance metadata for Spot interruption/rebalance signals.

    Safe on On-Demand instances: Spot-specific metadata endpoints return
    HTTP 404 when no Spot signal exists.
    """

    def __init__(
        self,
        poll_seconds: float = 5.0,
    ) -> None:
        super().__init__(daemon=True)

        self.poll_seconds = poll_seconds
        self.triggered = threading.Event()

        self.reason: Optional[str] = None
        self.detail: Optional[str] = None

        self._stop_event = threading.Event()
        self._token: Optional[str] = None
        self._token_refreshed_at = 0.0

    def stop(self) -> None:
        self._stop_event.set()

    def _refresh_token(self) -> None:
        request = urllib.request.Request(
            f"{IMDS_BASE_URL}/api/token",
            method="PUT",
            headers={
                "X-aws-ec2-metadata-token-ttl-seconds": str(
                    IMDS_TOKEN_TTL_SECONDS
                )
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=1.0,
        ) as response:
            self._token = response.read().decode("utf-8")

        self._token_refreshed_at = time.monotonic()

    def _get_metadata(
        self,
        path: str,
    ) -> Optional[str]:
        token_age = (
            time.monotonic()
            - self._token_refreshed_at
        )

        if (
            self._token is None
            or token_age >= IMDS_TOKEN_REFRESH_SECONDS
        ):
            try:
                self._refresh_token()
            except Exception as exc:
                logger.debug(
                    "Unable to refresh IMDSv2 token: %s",
                    exc,
                )
                return None

        request = urllib.request.Request(
            f"{IMDS_BASE_URL}/meta-data/{path}",
            headers={
                "X-aws-ec2-metadata-token": self._token or "",
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=1.0,
            ) as response:
                return (
                    response.read()
                    .decode("utf-8")
                    .strip()
                )

        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None

            if exc.code in {
                401,
                403,
            }:
                self._token = None
                return None

            logger.debug(
                "IMDS metadata request failed for %s: HTTP %s",
                path,
                exc.code,
            )
            return None

        except Exception as exc:
            logger.debug(
                "IMDS metadata request failed for %s: %s",
                path,
                exc,
            )
            return None

    def _trigger(
        self,
        reason: str,
        detail: str,
    ) -> None:
        if self.triggered.is_set():
            return

        self.reason = reason
        self.detail = detail

        logger.warning(
            "EC2 Spot signal detected: reason=%s detail=%s",
            reason,
            detail,
        )

        self.triggered.set()
        STOP.set()

    def run(self) -> None:  # pragma: no cover - depends on EC2 metadata
        while not self._stop_event.is_set():
            interruption = self._get_metadata(
                "spot/instance-action"
            )

            if interruption:
                self._trigger(
                    "spot_interruption_notice",
                    interruption,
                )
                return

            rebalance = self._get_metadata(
                "events/recommendations/rebalance"
            )

            if rebalance:
                self._trigger(
                    "spot_rebalance_recommendation",
                    rebalance,
                )
                return

            if self._stop_event.wait(
                self.poll_seconds
            ):
                return


class QueueWorker:
    def __init__(
        self,
        args: argparse.Namespace,
        extra_processor_args: List[str],
    ) -> None:
        self.args = args
        self.extra_processor_args = [
            value
            for value in extra_processor_args
            if value != "--"
        ]

        self.session = (
            boto3.Session(
                profile_name=args.aws_profile,
                region_name=args.aws_region,
            )
            if args.aws_profile
            else boto3.Session(
                region_name=args.aws_region
            )
        )

        self.identity = discover_instance_identity(
            self.session,
            args.aws_region,
        )

        self.worker_id = (
            f"{self.identity.get('hostname') or 'host'}-"
            f"{os.getpid()}"
        )

        self.run_id = (
            f"{self.worker_id}-"
            f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )

        self.run_manifest_key = (
            f"{args.runs_prefix.rstrip('/')}/"
            f"{self.run_id}/manifest.json"
        )

        self.worker_version = file_version_info(
            __file__
        )
        self.processor_version = file_version_info(
            args.processing_script
        )

        boto_cfg = Config(
            region_name=args.aws_region,
            max_pool_connections=32,
            retries={
                "mode": "adaptive",
                "max_attempts": 10,
            },
        )

        self.s3 = self.session.client(
            "s3",
            config=boto_cfg,
        )
        self.sqs = self.session.client(
            "sqs",
            config=boto_cfg,
        )

        self._processed = 0

        self.spot_watcher = SpotInterruptionWatcher(
            poll_seconds=args.spot_poll_seconds
        )

    def output_key_for_tile(
        self,
        tile: str,
    ) -> str:
        return processor_output_key(
            self.args.output_prefix_base,
            tile,
            self.args.start_date,
            self.args.end_date,
            self.args.calibration_lut,
        )

    def output_uri_for_tile(
        self,
        tile: str,
    ) -> str:
        return s3_uri(
            self.args.output_bucket,
            self.output_key_for_tile(tile),
        )

    def destination_exists(
        self,
        tile: str,
    ) -> bool:
        key = self.output_key_for_tile(tile)

        try:
            self.s3.head_object(
                Bucket=self.args.output_bucket,
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

    def write_run_manifest(self) -> None:
        manifest: Dict[str, Any] = {
            "run_id": self.run_id,
            "created_utc": utc_now(),
            "inventory_uri": self.args.inventory_uri,
            "queue_url": self.args.queue_url,
            "status_prefix": self.args.status_prefix,
            "output_bucket": self.args.output_bucket,
            "output_prefix_base": self.args.output_prefix_base,
            "dispatch_strategy": "sqs",
            "processor_family": "sentinel-1",
            "processor_config": {
                "start_date": self.args.start_date,
                "end_date": self.args.end_date,
                "mode": S1_MODE,
                "product_level": S1_PRODUCT_LEVEL,
                "output_format": S1_OUTPUT_FORMAT,
                "output_root": self.args.output_root,
                "work_root": self.args.work_root,
                "calibration_lut": self.args.calibration_lut,
                "minimum_geometric_coverage": (
                    self.args.minimum_geometric_coverage
                ),
                "minimum_valid_coverage": (
                    self.args.minimum_valid_coverage
                ),
                "acceptance_rule": self.args.acceptance_rule,
                "band_layout": self.args.band_layout,
                "unknown_noise_policy": (
                    self.args.unknown_noise_policy
                ),
                "out_dim": self.args.out_dim,
                "rows_per_window": self.args.rows_per_window,
                "num_threads": self.args.num_threads,
                "chunk_size": self.args.chunk_size,
                "aws_profile": self.args.aws_profile,
                "aws_region": self.args.aws_region,
                "extra_processor_args": (
                    self.extra_processor_args
                ),
            },
            "queue": {
                "visibility_timeout": (
                    self.args.visibility_timeout
                ),
                "wait_time_seconds": (
                    self.args.wait_time_seconds
                ),
                "skip_existing_output": (
                    self.args.skip_existing_output
                ),
                "stop_when_empty": (
                    self.args.stop_when_empty
                ),
                "max_tasks": self.args.max_tasks,
            },
            "worker": {
                "worker_id": self.worker_id,
                "pid": os.getpid(),
                **self.identity,
            },
            "worker_script": self.worker_version,
            "processor_script": self.processor_version,
        }

        self.s3.put_object(
            Bucket=self.args.control_bucket,
            Key=self.run_manifest_key,
            Body=json.dumps(
                manifest,
                indent=2,
            ).encode("utf-8"),
            ContentType="application/json",
        )

        logger.info(
            "Run manifest: %s",
            s3_uri(
                self.args.control_bucket,
                self.run_manifest_key,
            ),
        )

    def write_status(
        self,
        tile: str,
        state: str,
        *,
        started_utc: Optional[str] = None,
        finished_utc: Optional[str] = None,
        elapsed_seconds: Optional[float] = None,
        output_path: Optional[str] = None,
        reason: Optional[str] = None,
        return_code: Optional[int] = None,
        log_tail: Optional[List[str]] = None,
        sqs_message_id: Optional[str] = None,
        sqs_receive_count: Optional[str] = None,
    ) -> None:
        key = (
            f"{self.args.status_prefix.rstrip('/')}/"
            f"{normalise_s1_tile(tile)}.json"
        )

        record: Dict[str, Any] = {
            "tile": normalise_s1_tile(tile),
            "state": state,
            "run_id": self.run_id,
            "run_manifest_uri": s3_uri(
                self.args.control_bucket,
                self.run_manifest_key,
            ),
            "worker": self.worker_id,
            "pid": os.getpid(),
            "updated_utc": utc_now(),
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "elapsed_seconds": elapsed_seconds,
            "output_path": output_path,
            "reason": reason,
            "return_code": return_code,
            "processor_family": "sentinel-1",
            "mode": S1_MODE,
            "product_level": S1_PRODUCT_LEVEL,
            "calibration_lut": (
                self.args.calibration_lut
            ),
            "processor_script": self.processor_version,
            **self.identity,
        }

        if sqs_message_id is not None:
            record["sqs_message_id"] = (
                sqs_message_id
            )

        if sqs_receive_count is not None:
            record["sqs_receive_count"] = (
                sqs_receive_count
            )

        if log_tail:
            record["log_tail"] = log_tail

        self.s3.put_object(
            Bucket=self.args.control_bucket,
            Key=key,
            Body=json.dumps(
                record,
                indent=2,
            ).encode("utf-8"),
            ContentType="application/json",
        )

    def parse_message(
        self,
        body: str,
    ) -> Tuple[str, Dict[str, Any]]:
        try:
            payload = json.loads(body)

            if isinstance(payload, str):
                tile = normalise_s1_tile(
                    normalize_tile(payload)
                )
                return tile, {
                    "tile": tile,
                }

            if not isinstance(
                payload,
                dict,
            ):
                raise ValueError(
                    "message JSON must be an object or string"
                )

        except json.JSONDecodeError:
            tile = normalise_s1_tile(
                normalize_tile(body)
            )
            return tile, {
                "tile": tile,
            }

        raw_tile = (
            payload.get("tile")
            or payload.get("tile_id")
            or payload.get("mgrs_tile")
        )

        if not raw_tile:
            raise ValueError(
                "message missing tile"
            )

        tile = normalise_s1_tile(
            normalize_tile(
                str(raw_tile)
            )
        )

        payload["tile"] = tile
        return tile, payload

    @staticmethod
    def terminate_processor(
        proc: subprocess.Popen,
        grace_seconds: float = 10.0,
    ) -> None:
        if proc.poll() is not None:
            return

        logger.warning(
            "Stopping Sentinel-1 processor process group pid=%s",
            proc.pid,
        )

        try:
            os.killpg(
                proc.pid,
                signal.SIGTERM,
            )
        except ProcessLookupError:
            return

        try:
            proc.wait(
                timeout=grace_seconds
            )
            return

        except subprocess.TimeoutExpired:
            logger.warning(
                "Processor did not exit within %.1fs; sending SIGKILL",
                grace_seconds,
            )

        try:
            os.killpg(
                proc.pid,
                signal.SIGKILL,
            )
        except ProcessLookupError:
            return

        try:
            proc.wait(
                timeout=5.0
            )
        except subprocess.TimeoutExpired:
            logger.error(
                "Processor process group did not exit after SIGKILL"
            )

    def run_tile(
        self,
        tile: str,
    ) -> Tuple[
        str,
        float,
        int,
        List[str],
        Optional[str],
    ]:
        tile_name = normalise_s1_tile(
            tile
        )
        output_uri = self.output_uri_for_tile(
            tile_name
        )

        local_output = local_output_path(
            self.args.output_root,
            tile_name,
            self.args.start_date,
            self.args.end_date,
            self.args.calibration_lut,
        )

        if os.path.isfile(
            local_output
        ):
            logger.warning(
                "Removing stale local Sentinel-1 output before retry: %s",
                local_output,
            )
            os.remove(
                local_output
            )

        run_work_root = os.path.join(
            self.args.work_root,
            self.run_id,
        )

        cmd = [
            sys.executable,
            self.args.processing_script,
            "--tile",
            tile_name,
            "--start-date",
            self.args.start_date,
            "--end-date",
            self.args.end_date,
            "--minimum-geometric-coverage",
            str(
                self.args.minimum_geometric_coverage
            ),
            "--minimum-valid-coverage",
            str(
                self.args.minimum_valid_coverage
            ),
            "--acceptance-rule",
            self.args.acceptance_rule,
            "--band-layout",
            self.args.band_layout,
            "--calibration-lut",
            self.args.calibration_lut,
            "--unknown-noise-policy",
            self.args.unknown_noise_policy,
            "--out-dim",
            str(self.args.out_dim),
            "--rows-per-window",
            str(self.args.rows_per_window),
            "--num-threads",
            str(self.args.num_threads),
            "--chunk-size",
            str(self.args.chunk_size),
            "--output-root",
            self.args.output_root,
            "--work-root",
            run_work_root,
        ]

        if self.args.keep_intermediates:
            cmd.append(
                "--keep-intermediates"
            )

        if self.args.aws_profile:
            cmd.extend(
                [
                    "--aws-profile",
                    self.args.aws_profile,
                ]
            )

        if self.args.aws_region:
            cmd.extend(
                [
                    "--aws-region",
                    self.args.aws_region,
                ]
            )

        cmd.extend(
            self.extra_processor_args
        )

        logger.info(
            "Running: %s",
            " ".join(cmd),
        )

        tail = collections.deque(
            maxlen=200
        )

        start = dt.datetime.now(
            dt.timezone.utc
        )

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        assert proc.stdout is not None

        def _stream_output() -> None:
            try:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    tail.append(
                        line.rstrip("\n")
                    )
            finally:
                proc.stdout.close()

        output_thread = threading.Thread(
            target=_stream_output,
            daemon=True,
        )
        output_thread.start()

        interruption_reason: Optional[str] = None

        while proc.poll() is None:
            if self.spot_watcher.triggered.is_set():
                interruption_reason = (
                    self.spot_watcher.reason
                    or "spot_interruption"
                )

                logger.warning(
                    "Draining worker because of EC2 Spot signal: %s",
                    interruption_reason,
                )

                self.terminate_processor(
                    proc
                )
                break

            if STOP.is_set():
                interruption_reason = (
                    "worker_shutdown"
                )

                logger.warning(
                    "Worker shutdown requested while tile %s is running",
                    tile_name,
                )

                self.terminate_processor(
                    proc
                )
                break

            time.sleep(0.5)

        rc = proc.wait()

        if (
            interruption_reason is None
            and rc != 0
        ):
            if self.spot_watcher.triggered.is_set():
                interruption_reason = (
                    self.spot_watcher.reason
                    or "spot_interruption"
                )
            elif STOP.is_set():
                interruption_reason = (
                    "worker_shutdown"
                )

        output_thread.join(
            timeout=5.0
        )

        if (
            rc == 0
            and interruption_reason is None
        ):
            if not os.path.isfile(
                local_output
            ):
                message = (
                    "Sentinel-1 processor exited successfully but "
                    "the expected TFRecord is missing: "
                    f"{local_output}"
                )
                logger.error(
                    message
                )
                tail.append(
                    message
                )
                rc = 1

            else:
                key = self.output_key_for_tile(
                    tile_name
                )

                logger.info(
                    "Uploading Sentinel-1 TFRecord to s3://%s/%s",
                    self.args.output_bucket,
                    key,
                )

                try:
                    self.s3.upload_file(
                        local_output,
                        self.args.output_bucket,
                        key,
                    )

                    logger.info(
                        "Uploaded Sentinel-1 TFRecord: %s",
                        output_uri,
                    )

                    try:
                        os.remove(
                            local_output
                        )
                    except OSError as exc:
                        logger.warning(
                            "Unable to delete local Sentinel-1 "
                            "TFRecord %s: %s",
                            local_output,
                            exc,
                        )

                except Exception as exc:
                    message = (
                        "Sentinel-1 TFRecord upload failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

                    logger.exception(
                        "Sentinel-1 TFRecord upload failed"
                    )

                    tail.append(
                        message
                    )
                    rc = 1

                if self.spot_watcher.triggered.is_set():
                    interruption_reason = (
                        self.spot_watcher.reason
                        or "spot_interruption"
                    )

                elif STOP.is_set():
                    interruption_reason = (
                        "worker_shutdown"
                    )

        elapsed = round(
            (
                dt.datetime.now(dt.timezone.utc)
                - start
            ).total_seconds(),
            3,
        )

        return (
            output_uri,
            elapsed,
            rc,
            list(tail),
            interruption_reason,
        )

    def classify_failure(
        self,
        rc: int,
        log_tail: List[str],
    ) -> Tuple[str, str]:
        joined = "\n".join(
            log_tail
        )

        no_data_markers = [
            "No Sentinel-1",
            "No scenes remained after filtering.",
            "No acquisitions",
            "Candidate scenes:           0",
        ]

        for marker in no_data_markers:
            if marker in joined:
                return (
                    "no_data",
                    "no_sentinel1_data",
                )

        if rc != 0:
            return (
                "failed",
                f"processor_exit_{rc}",
            )

        return (
            "failed",
            "unknown_failure",
        )

    def delete_message(
        self,
        receipt_handle: str,
    ) -> None:
        self.sqs.delete_message(
            QueueUrl=self.args.queue_url,
            ReceiptHandle=receipt_handle,
        )

    def release_message(
        self,
        receipt_handle: str,
    ) -> None:
        try:
            self.sqs.change_message_visibility(
                QueueUrl=self.args.queue_url,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=0,
            )

            logger.info(
                "Released SQS message for immediate retry"
            )

        except Exception as exc:
            logger.warning(
                "Failed to release SQS message immediately: %s",
                exc,
            )

    def receive_one(
        self,
    ) -> Optional[Dict[str, Any]]:
        response = self.sqs.receive_message(
            QueueUrl=self.args.queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=self.args.wait_time_seconds,
            VisibilityTimeout=self.args.visibility_timeout,
            AttributeNames=[
                "ApproximateReceiveCount",
            ],
        )

        messages = response.get(
            "Messages",
            [],
        )

        return (
            messages[0]
            if messages
            else None
        )

    def validate_message_config(
        self,
        tile: str,
        payload: Dict[str, Any],
    ) -> None:
        expected = {
            "mode": S1_MODE,
            "product_level": S1_PRODUCT_LEVEL,
            "output_format": S1_OUTPUT_FORMAT,
            "start_date": self.args.start_date,
            "end_date": self.args.end_date,
            "calibration_lut": (
                self.args.calibration_lut
            ),
        }

        for key, expected_value in expected.items():
            message_value = payload.get(
                key
            )

            if (
                message_value is not None
                and message_value != expected_value
            ):
                raise ValueError(
                    f"SQS {key} mismatch for {tile}: "
                    f"message={message_value} "
                    f"worker={expected_value}"
                )

    def handle_message(
        self,
        message: Dict[str, Any],
    ) -> None:
        receipt_handle = message[
            "ReceiptHandle"
        ]

        sqs_message_id = message.get(
            "MessageId"
        )

        sqs_receive_count = (
            message.get(
                "Attributes",
                {},
            ).get(
                "ApproximateReceiveCount"
            )
        )

        tile, payload = self.parse_message(
            message["Body"]
        )

        self.validate_message_config(
            tile,
            payload,
        )

        expected_key = self.output_key_for_tile(
            tile
        )

        message_output_key = payload.get(
            "output_key"
        )

        if (
            message_output_key
            and message_output_key != expected_key
        ):
            raise ValueError(
                f"SQS output_key mismatch for {tile}: "
                f"message={message_output_key} "
                f"worker={expected_key}"
            )

        if (
            self.args.skip_existing_output
            and self.destination_exists(tile)
        ):
            logger.info(
                "Skipping tile %s: completed Sentinel-1 output "
                "already exists at %s",
                tile,
                self.output_uri_for_tile(tile),
            )

            self.delete_message(
                receipt_handle
            )
            return

        started_utc = utc_now()

        self.write_status(
            tile,
            "in_progress",
            started_utc=started_utc,
            sqs_message_id=sqs_message_id,
            sqs_receive_count=sqs_receive_count,
        )

        extender = VisibilityExtender(
            sqs_client=self.sqs,
            queue_url=self.args.queue_url,
            receipt_handle=receipt_handle,
            visibility_timeout=self.args.visibility_timeout,
        )

        extender.start()

        try:
            (
                output_path,
                elapsed,
                rc,
                tail,
                interruption_reason,
            ) = self.run_tile(
                tile
            )

            finished_utc = utc_now()

            if interruption_reason is not None:
                extender.stop()
                extender.join(
                    timeout=1.0
                )

                self.write_status(
                    tile,
                    "aborted",
                    started_utc=started_utc,
                    finished_utc=finished_utc,
                    elapsed_seconds=elapsed,
                    output_path=None,
                    reason=interruption_reason,
                    return_code=rc,
                    log_tail=(
                        tail[-50:]
                        if len(tail) > 50
                        else tail
                    ),
                    sqs_message_id=sqs_message_id,
                    sqs_receive_count=sqs_receive_count,
                )

                self.release_message(
                    receipt_handle
                )

                logger.warning(
                    "Tile %s interrupted (%s); "
                    "message released for retry",
                    tile,
                    interruption_reason,
                )

                STOP.set()
                return

            if rc == 0:
                self.write_status(
                    tile,
                    "complete",
                    started_utc=started_utc,
                    finished_utc=finished_utc,
                    elapsed_seconds=elapsed,
                    output_path=output_path,
                    reason=None,
                    return_code=rc,
                    sqs_message_id=sqs_message_id,
                    sqs_receive_count=sqs_receive_count,
                )

                self.delete_message(
                    receipt_handle
                )

                logger.info(
                    "Tile %s complete -> %s (%.1fs)",
                    tile,
                    output_path,
                    elapsed,
                )

            else:
                state, reason = self.classify_failure(
                    rc,
                    tail,
                )

                self.write_status(
                    tile,
                    state,
                    started_utc=started_utc,
                    finished_utc=finished_utc,
                    elapsed_seconds=elapsed,
                    output_path=None,
                    reason=reason,
                    return_code=rc,
                    log_tail=(
                        tail[-50:]
                        if len(tail) > 50
                        else tail
                    ),
                    sqs_message_id=sqs_message_id,
                    sqs_receive_count=sqs_receive_count,
                )

                if state == "no_data":
                    self.delete_message(
                        receipt_handle
                    )

                logger.warning(
                    "Tile %s %s (%s) with exit code %s",
                    tile,
                    state,
                    reason,
                    rc,
                )

        except KeyboardInterrupt:
            finished_utc = utc_now()

            self.write_status(
                tile,
                "aborted",
                started_utc=started_utc,
                finished_utc=finished_utc,
                reason="worker_interrupted",
                sqs_message_id=sqs_message_id,
                sqs_receive_count=sqs_receive_count,
            )

            raise

        except Exception as exc:
            finished_utc = utc_now()

            self.write_status(
                tile,
                "failed",
                started_utc=started_utc,
                finished_utc=finished_utc,
                reason=(
                    "worker_exception:"
                    f"{type(exc).__name__}"
                ),
                log_tail=[
                    repr(exc),
                ],
                sqs_message_id=sqs_message_id,
                sqs_receive_count=sqs_receive_count,
            )

            raise

        finally:
            extender.stop()
            extender.join(
                timeout=1.0
            )

    def _run_loop(self) -> None:
        while not STOP.is_set():
            message = self.receive_one()

            if STOP.is_set():
                if message is not None:
                    self.release_message(
                        message["ReceiptHandle"]
                    )

                logger.warning(
                    "Worker is draining; exiting without accepting new work"
                )
                break

            if message is None:
                if self.args.stop_when_empty:
                    logger.info(
                        "Queue empty; exiting because "
                        "--stop-when-empty was set"
                    )
                    break
                continue

            self.handle_message(
                message
            )

            self._processed += 1

            if (
                self.args.max_tasks is not None
                and self._processed
                >= self.args.max_tasks
            ):
                logger.info(
                    "Processed %s tasks; exiting because "
                    "--max-tasks was reached",
                    self._processed,
                )
                break

    def run(self) -> None:
        self.spot_watcher.start()

        try:
            self.write_run_manifest()
            self._run_loop()

        finally:
            self.spot_watcher.stop()
            self.spot_watcher.join(
                timeout=2.0
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Consume Sentinel-1 MGRS tile jobs from SQS and run the "
            "standalone calibrated Sentinel-1 GRD TFRecord processor."
        )
    )

    parser.add_argument(
        "--queue-url",
        required=True,
    )
    parser.add_argument(
        "--control-bucket",
        required=True,
    )
    parser.add_argument(
        "--processing-script",
        default="sentinel1_tfrecord_processor.py",
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
        "--output-bucket",
        required=True,
    )
    parser.add_argument(
        "--output-prefix-base",
        required=True,
    )
    parser.add_argument(
        "--output-root",
        required=True,
    )
    parser.add_argument(
        "--work-root",
        default="reports/s1_worker_work",
    )
    parser.add_argument(
        "--status-prefix",
        default="status/sentinel-1",
    )
    parser.add_argument(
        "--runs-prefix",
        default="runs/sentinel-1",
    )
    parser.add_argument(
        "--inventory-uri",
        default=None,
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
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
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
        "--visibility-timeout",
        type=int,
        default=3600,
    )
    parser.add_argument(
        "--wait-time-seconds",
        type=int,
        default=20,
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

    parser.add_argument(
        "--stop-when-empty",
        action="store_true",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--spot-poll-seconds",
        type=float,
        default=5.0,
    )

    return parser


def main() -> None:
    parser = build_parser()
    args, extra = parser.parse_known_args()

    if args.spot_poll_seconds <= 0:
        parser.error(
            "--spot-poll-seconds must be greater than zero"
        )

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

    for name in [
        "out_dim",
        "rows_per_window",
        "num_threads",
        "chunk_size",
        "visibility_timeout",
        "wait_time_seconds",
    ]:
        if getattr(
            args,
            name,
        ) <= 0:
            parser.error(
                f"--{name.replace('_', '-')} must be greater than zero"
            )

    if (
        args.max_tasks is not None
        and args.max_tasks <= 0
    ):
        parser.error(
            "--max-tasks must be greater than zero"
        )

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(levelname)s %(message)s"
        ),
        force=True,
    )

    signal.signal(
        signal.SIGTERM,
        _handle_signal,
    )
    signal.signal(
        signal.SIGINT,
        _handle_signal,
    )

    QueueWorker(
        args,
        extra,
    ).run()


def _handle_signal(
    signum: int,
    _frame: Any,
) -> None:
    logger.warning(
        "Received signal %s; draining Sentinel-1 worker "
        "and releasing any in-flight SQS job",
        signum,
    )
    STOP.set()


if __name__ == "__main__":
    main()

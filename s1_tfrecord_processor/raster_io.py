from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import boto3
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.session import AWSSession
from rasterio.transform import Affine
from rasterio.warp import reproject


@dataclass(frozen=True)
class WarpedAsset:
    values: np.ndarray
    valid_mask: np.ndarray
    metadata: dict[str, Any]


@contextmanager
def aws_rasterio_environment(
    profile: str | None = None,
    region: str = "eu-central-1",
) -> Iterator[None]:
    """
    Create a Rasterio/GDAL environment for reading the requester-pays
    Sentinel-1 measurement bucket.
    """
    if profile:
        boto_session = boto3.Session(
            profile_name=profile,
            region_name=region,
        )
    else:
        boto_session = boto3.Session(
            region_name=region,
        )

    aws_session = AWSSession(
        boto_session,
        requester_pays=True,
    )

    with rasterio.Env(
        aws_session,
        AWS_REGION=region,
        AWS_REQUEST_PAYER="requester",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_PAM_ENABLED="NO",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
    ):
        yield


def build_xy_coordinates(
    transform: Affine,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one-dimensional projected x and y pixel-centre coordinates
    from the exact destination transform.
    """
    columns = np.arange(width, dtype=np.float64)
    rows = np.arange(height, dtype=np.float64)

    x_coordinates, _ = transform * (
        columns + 0.5,
        np.full(width, 0.5, dtype=np.float64),
    )

    _, y_coordinates = transform * (
        np.full(height, 0.5, dtype=np.float64),
        rows + 0.5,
    )

    return (
        np.asarray(x_coordinates, dtype=np.float64),
        np.asarray(y_coordinates, dtype=np.float64),
    )


def warp_gcp_asset_to_grid(
    href: str,
    destination_crs: CRS,
    destination_transform: Affine,
    destination_shape: tuple[int, int],
    polarisation: str,
    num_threads: int = 2,
) -> WarpedAsset:
    """
    Warp one Sentinel-1 measurement asset onto an exact target grid.

    The Sentinel-1 GRD TIFFs used here have no normal affine
    geotransform. Their geolocation is represented by GCPs, so those
    GCPs must be supplied to Rasterio/GDAL during reprojection.

    Source nodata is excluded during average resampling. Destination
    pixels that receive no valid source contribution remain NaN.
    """
    destination_height, destination_width = destination_shape

    destination = np.full(
        (destination_height, destination_width),
        np.nan,
        dtype=np.float32,
    )

    with rasterio.open(href) as src:
        if src.count != 1:
            raise RuntimeError(
                f"{polarisation} asset was expected to contain one band, "
                f"but contains {src.count}: {href}"
            )

        gcps, gcp_crs = src.gcps

        if not gcps:
            raise RuntimeError(
                f"{polarisation} asset contains no GCPs: {href}"
            )

        if gcp_crs is None:
            raise RuntimeError(
                f"{polarisation} asset has no GCP CRS: {href}"
            )

        if src.nodata is None:
            raise RuntimeError(
                f"{polarisation} asset has no declared nodata value: {href}"
            )

        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            gcps=gcps,
            src_crs=gcp_crs,
            src_nodata=src.nodata,
            dst_transform=destination_transform,
            dst_crs=destination_crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
            init_dest_nodata=True,
            num_threads=num_threads,
        )

        valid_mask = np.isfinite(destination)
        valid_values = destination[valid_mask]

        metadata: dict[str, Any] = {
            "polarisation": polarisation,
            "href": href,
            "source_shape": [int(src.height), int(src.width)],
            "source_dtype": src.dtypes[0],
            "source_nodata": float(src.nodata),
            "source_crs": (
                src.crs.to_string()
                if src.crs is not None
                else None
            ),
            "source_transform": [
                float(value)
                for value in tuple(src.transform)
            ],
            "gcp_count": len(gcps),
            "gcp_crs": gcp_crs.to_string(),
            "destination_shape": [
                destination_height,
                destination_width,
            ],
            "destination_crs": destination_crs.to_string(),
            "destination_transform": [
                float(value)
                for value in tuple(destination_transform)
            ],
            "resampling": "average",
            "valid_pixel_count": int(valid_mask.sum()),
            "invalid_pixel_count": int(
                valid_mask.size - valid_mask.sum()
            ),
            "valid_fraction": float(valid_mask.mean()),
            "valid_pct": float(valid_mask.mean() * 100.0),
        }

        if valid_values.size:
            metadata.update(
                {
                    "minimum": float(valid_values.min()),
                    "maximum": float(valid_values.max()),
                    "mean": float(valid_values.mean()),
                    "median": float(
                        np.median(valid_values)
                    ),
                }
            )
        else:
            metadata.update(
                {
                    "minimum": None,
                    "maximum": None,
                    "mean": None,
                    "median": None,
                }
            )

    return WarpedAsset(
        values=destination,
        valid_mask=valid_mask,
        metadata=metadata,
    )

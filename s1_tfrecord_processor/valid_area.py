from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from xml.sax.saxutils import escape, quoteattr

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import MaskFlags, Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject


MASK_VALID_VALUE = 255.0


@dataclass(frozen=True)
class ValidAreaResult:
    fraction_grid: np.ndarray
    metadata: dict[str, Any]


def gdal_source_path(href: str) -> str:
    """
    Convert an S3 URI to the GDAL VSI form required inside a VRT.
    """
    parsed = urlparse(href)

    if parsed.scheme == "s3":
        return (
            f"/vsis3/{parsed.netloc}/"
            f"{parsed.path.lstrip('/')}"
        )

    return href


def build_mask_vrt_xml(
    *,
    source_href: str,
    width: int,
    height: int,
    gcps,
    gcp_crs: CRS,
) -> str:
    """
    Build a VRT that exposes band 1's GDAL validity mask as a normal
    Byte raster band.

    Mask values remain data:
        0   = invalid
        255 = valid

    No nodata value is assigned to the VRT band because invalid zero
    values must participate in average resampling.
    """
    source_path = escape(
        gdal_source_path(source_href)
    )

    lines = [
        (
            f'<VRTDataset rasterXSize="{width}" '
            f'rasterYSize="{height}">'
        ),
        (
            "<GCPList Projection="
            f"{quoteattr(gcp_crs.to_wkt())}>"
        ),
    ]

    for index, gcp in enumerate(gcps):
        gcp_id = gcp.id or str(index)

        lines.append(
            "  <GCP"
            f" Id={quoteattr(str(gcp_id))}"
            f' Pixel="{float(gcp.col):.17g}"'
            f' Line="{float(gcp.row):.17g}"'
            f' X="{float(gcp.x):.17g}"'
            f' Y="{float(gcp.y):.17g}"'
            f' Z="{float(gcp.z or 0.0):.17g}"'
            "/>"
        )

    lines.extend(
        [
            "</GCPList>",
            (
                '<VRTRasterBand dataType="Byte" '
                'band="1">'
            ),
            "  <ColorInterp>Gray</ColorInterp>",
            "  <SimpleSource>",
            (
                "    <SourceFilename "
                f'relativeToVRT="0">{source_path}'
                "</SourceFilename>"
            ),
            "    <SourceBand>mask,1</SourceBand>",
            (
                f'    <SrcRect xOff="0" yOff="0" '
                f'xSize="{width}" ySize="{height}"/>'
            ),
            (
                f'    <DstRect xOff="0" yOff="0" '
                f'xSize="{width}" ySize="{height}"/>'
            ),
            "  </SimpleSource>",
            "</VRTRasterBand>",
            "</VRTDataset>",
        ]
    )

    return "\n".join(lines)


def warp_gcp_valid_area_fraction_to_grid(
    *,
    href: str,
    destination_crs: CRS,
    destination_transform: Affine,
    destination_shape: tuple[int, int],
    polarisation: str,
    num_threads: int = 2,
) -> ValidAreaResult:
    """
    Warp the source validity mask onto the target grid using average
    resampling.

    Each output value is a fraction between 0 and 1:

        0.0 = no valid contributing source area
        1.0 = entirely valid contributing source area

    Intermediate values represent partial valid-area coverage.
    """
    destination_height, destination_width = (
        destination_shape
    )

    if destination_height <= 0 or destination_width <= 0:
        raise ValueError(
            f"Invalid destination shape: {destination_shape}"
        )

    temporary_vrt_path: Path | None = None

    try:
        with rasterio.open(href) as src:
            if src.count != 1:
                raise RuntimeError(
                    f"{polarisation} asset must contain exactly "
                    f"one band, found {src.count}: {href}"
                )

            gcps, gcp_crs = src.gcps

            if not gcps:
                raise RuntimeError(
                    f"{polarisation} asset contains no GCPs: "
                    f"{href}"
                )

            if gcp_crs is None:
                raise RuntimeError(
                    f"{polarisation} asset has no GCP CRS: "
                    f"{href}"
                )

            if src.nodata is None:
                raise RuntimeError(
                    f"{polarisation} asset has no declared "
                    f"nodata value: {href}"
                )

            mask_flags = [
                flag.name
                for flag in src.mask_flag_enums[0]
            ]

            if MaskFlags.all_valid in src.mask_flag_enums[0]:
                raise RuntimeError(
                    f"{polarisation} source reports every pixel "
                    "as valid; expected a nodata-derived mask."
                )

            source_width = int(src.width)
            source_height = int(src.height)
            source_nodata = float(src.nodata)

            vrt_xml = build_mask_vrt_xml(
                source_href=href,
                width=source_width,
                height=source_height,
                gcps=gcps,
                gcp_crs=gcp_crs,
            )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".vrt",
            prefix="s1_valid_mask_",
            encoding="utf-8",
            delete=False,
        ) as temporary_vrt:
            temporary_vrt.write(vrt_xml)
            temporary_vrt_path = Path(
                temporary_vrt.name
            )

        destination_mask_average = np.zeros(
            (
                destination_height,
                destination_width,
            ),
            dtype=np.float32,
        )

        with rasterio.open(
            temporary_vrt_path
        ) as mask_source:
            mask_gcps, mask_gcp_crs = (
                mask_source.gcps
            )

            if not mask_gcps:
                raise RuntimeError(
                    "Generated mask VRT contains no GCPs."
                )

            if mask_gcp_crs is None:
                raise RuntimeError(
                    "Generated mask VRT has no GCP CRS."
                )

            reproject(
                source=rasterio.band(
                    mask_source,
                    1,
                ),
                destination=(
                    destination_mask_average
                ),
                gcps=mask_gcps,
                src_crs=mask_gcp_crs,
                # Deliberately no src_nodata:
                # zero mask values are invalid-area data and
                # must participate in the average.
                src_nodata=None,
                dst_transform=(
                    destination_transform
                ),
                dst_crs=destination_crs,
                dst_nodata=0.0,
                resampling=Resampling.average,
                init_dest_nodata=True,
                num_threads=num_threads,
            )

        if not np.all(
            np.isfinite(
                destination_mask_average
            )
        ):
            raise RuntimeError(
                "Warped validity mask contains non-finite "
                "values."
            )

        minimum = float(
            destination_mask_average.min()
        )
        maximum = float(
            destination_mask_average.max()
        )

        tolerance = 1e-4

        if minimum < -tolerance:
            raise RuntimeError(
                "Warped mask contains values below zero: "
                f"{minimum}"
            )

        if maximum > MASK_VALID_VALUE + tolerance:
            raise RuntimeError(
                "Warped mask contains values above 255: "
                f"{maximum}"
            )

        fraction_grid = np.clip(
            destination_mask_average
            / MASK_VALID_VALUE,
            0.0,
            1.0,
        ).astype(
            np.float32,
            copy=False,
        )

        area_weighted_fraction = float(
            fraction_grid.mean()
        )

        occupied_fraction = float(
            np.count_nonzero(
                fraction_grid > 0.0
            )
            / fraction_grid.size
        )

        fully_valid_fraction = float(
            np.count_nonzero(
                np.isclose(
                    fraction_grid,
                    1.0,
                    rtol=0.0,
                    atol=1e-6,
                )
            )
            / fraction_grid.size
        )

        metadata = {
            "polarisation": polarisation,
            "href": href,
            "source_shape": [
                source_height,
                source_width,
            ],
            "source_nodata": source_nodata,
            "source_mask_flags": mask_flags,
            "gcp_count": len(gcps),
            "gcp_crs": gcp_crs.to_string(),
            "destination_shape": [
                destination_height,
                destination_width,
            ],
            "destination_crs": (
                destination_crs.to_string()
            ),
            "destination_transform": [
                float(value)
                for value in tuple(
                    destination_transform
                )
            ],
            "resampling": "average",
            "mask_invalid_value": 0,
            "mask_valid_value": 255,
            "area_weighted_valid_fraction": (
                area_weighted_fraction
            ),
            "area_weighted_valid_pct": (
                area_weighted_fraction * 100.0
            ),
            "occupied_output_pixel_fraction": (
                occupied_fraction
            ),
            "occupied_output_pixel_pct": (
                occupied_fraction * 100.0
            ),
            "fully_valid_output_pixel_fraction": (
                fully_valid_fraction
            ),
            "fully_valid_output_pixel_pct": (
                fully_valid_fraction * 100.0
            ),
            "fraction_grid_minimum": float(
                fraction_grid.min()
            ),
            "fraction_grid_maximum": float(
                fraction_grid.max()
            ),
            "fraction_grid_mean": (
                area_weighted_fraction
            ),
        }

        return ValidAreaResult(
            fraction_grid=fraction_grid,
            metadata=metadata,
        )

    finally:
        if (
            temporary_vrt_path is not None
            and temporary_vrt_path.exists()
        ):
            os.unlink(temporary_vrt_path)

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
from pystac_client import Client
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape

from .utils import normalize_tile, parse_tile


EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
S1_COLLECTION = "sentinel-1-grd"

REQUIRED_INSTRUMENT_MODE = "IW"
REQUIRED_POLARISATIONS = {"VV", "VH"}

# Retaining the original confirmed platform scope for now.
ALLOWED_PLATFORMS = {"sentinel-1a", "sentinel-1b"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan real Sentinel-2 tile polygons for Sentinel-1 IW GRD "
            "scenes satisfying the configured tile-coverage threshold."
        )
    )

    parser.add_argument(
        "--inventory",
        default="inventory/eu_tiles_land_plus_eez.gpkg",
    )
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)

    parser.add_argument(
        "--min-tile-coverage",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=50,
        help=(
            "Maximum number of inventory tiles to inspect. Tiles with "
            "the largest Europe land/EEZ overlap are inspected first."
        ),
    )
    parser.add_argument(
        "--max-items-per-tile",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=10,
        help="Stop after this many qualifying scene/tile matches.",
    )
    parser.add_argument(
        "--output",
        default=(
            "reports/s1_prototype/"
            "sentinel1_qualifying_tile_scan.json"
        ),
    )

    args = parser.parse_args()

    if not 0.0 <= args.min_tile_coverage <= 1.0:
        parser.error("--min-tile-coverage must be between 0 and 1")

    if args.max_tiles <= 0:
        parser.error("--max-tiles must be greater than zero")

    if args.max_items_per_tile <= 0:
        parser.error("--max-items-per-tile must be greater than zero")

    if args.max_matches <= 0:
        parser.error("--max-matches must be greater than zero")

    return args


def tile_crs(tile: str) -> CRS:
    """
    Derive the tile UTM CRS from its MGRS UTM zone and latitude band.

    MGRS latitude bands N-X are in the northern hemisphere.
    Bands C-M are in the southern hemisphere.
    """
    utm_zone, latitude_band, _ = parse_tile(tile)

    northern_hemisphere = latitude_band >= "N"
    epsg = (
        32600 + utm_zone
        if northern_hemisphere
        else 32700 + utm_zone
    )

    return CRS.from_epsg(epsg)


def valid_geometry(geometry: Any) -> Any:
    if geometry.is_empty:
        return geometry

    if not geometry.is_valid:
        geometry = geometry.buffer(0)

    return geometry


def normalise_string_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value.upper()]

    if isinstance(value, (list, tuple, set)):
        return [str(entry).upper() for entry in value]

    return [str(value).upper()]


def inspect_item(
    item: Any,
    tile_polygon_wgs84: dict[str, Any],
    destination_crs: CRS,
    minimum_coverage: float,
) -> dict[str, Any]:
    properties = item.properties

    platform = str(
        properties.get("platform") or ""
    ).lower()

    instrument_mode = str(
        properties.get("sar:instrument_mode") or ""
    ).upper()

    polarisations = normalise_string_list(
        properties.get("sar:polarizations")
    )

    assets = {
        name.lower(): asset
        for name, asset in item.assets.items()
    }

    vv_asset = assets.get("vv")
    vh_asset = assets.get("vh")

    reasons: list[str] = []

    if platform not in ALLOWED_PLATFORMS:
        reasons.append(
            f"platform_is_{platform or 'missing'}"
        )

    if instrument_mode != REQUIRED_INSTRUMENT_MODE:
        reasons.append(
            "instrument_mode_is_"
            f"{instrument_mode or 'missing'}"
        )

    missing_polarisations = sorted(
        REQUIRED_POLARISATIONS.difference(
            polarisations
        )
    )
    if missing_polarisations:
        reasons.append(
            "missing_polarisations_"
            + "_".join(missing_polarisations)
        )

    if vv_asset is None or not vv_asset.href:
        reasons.append("missing_vv_asset")

    if vh_asset is None or not vh_asset.href:
        reasons.append("missing_vh_asset")

    coverage_fraction = 0.0

    if item.geometry is None:
        reasons.append("missing_geometry")
    else:
        try:
            tile_projected = valid_geometry(
                shape(
                    transform_geom(
                        "EPSG:4326",
                        destination_crs,
                        tile_polygon_wgs84,
                        precision=6,
                    )
                )
            )

            scene_projected = valid_geometry(
                shape(
                    transform_geom(
                        "EPSG:4326",
                        destination_crs,
                        item.geometry,
                        precision=6,
                    )
                )
            )

            intersection = tile_projected.intersection(
                scene_projected
            )

            if tile_projected.area > 0:
                coverage_fraction = (
                    float(intersection.area)
                    / float(tile_projected.area)
                )

        except Exception as exc:
            reasons.append(
                "geometry_processing_failed:"
                f"{type(exc).__name__}:{exc}"
            )

    if coverage_fraction < minimum_coverage:
        reasons.append("tile_coverage_below_threshold")

    return {
        "item_id": item.id,
        "datetime_utc": (
            item.datetime.isoformat().replace("+00:00", "Z")
            if item.datetime is not None
            else None
        ),
        "platform": platform or None,
        "instrument_mode": instrument_mode or None,
        "polarisations": polarisations,
        "orbit_state": properties.get("sat:orbit_state"),
        "relative_orbit": properties.get(
            "sat:relative_orbit"
        ),
        "absolute_orbit": properties.get(
            "sat:absolute_orbit"
        ),
        "product_type": properties.get(
            "sar:product_type"
        ),
        "vv_href": (
            vv_asset.href
            if vv_asset is not None
            else None
        ),
        "vh_href": (
            vh_asset.href
            if vh_asset is not None
            else None
        ),
        "tile_coverage_fraction": coverage_fraction,
        "tile_coverage_pct": (
            coverage_fraction * 100.0
        ),
        "keep": not reasons,
        "rejection_reasons": reasons,
    }


def main() -> None:
    args = parse_args()

    inventory_path = Path(args.inventory)
    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tiles = gpd.read_file(inventory_path)

    required_columns = {
        "product_tile",
        "overlap_fraction",
        "geometry",
    }
    missing_columns = required_columns.difference(
        tiles.columns
    )
    if missing_columns:
        raise RuntimeError(
            "Inventory is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if tiles.crs is None:
        raise RuntimeError(
            "Tile inventory has no CRS."
        )

    tiles = tiles.to_crs("EPSG:4326")

    # Prioritise tiles substantially contained in the Europe
    # land-plus-EEZ boundary. This affects scan order only.
    tiles = tiles.sort_values(
        by=["overlap_fraction", "product_tile"],
        ascending=[False, True],
    ).head(args.max_tiles)

    client = Client.open(EARTH_SEARCH_URL)

    matches: list[dict[str, Any]] = []
    tile_summaries: list[dict[str, Any]] = []

    for scan_index, (_, row) in enumerate(
        tiles.iterrows(),
        start=1,
    ):
        product_tile = str(row["product_tile"])
        normalized_tile = normalize_tile(product_tile)
        destination_crs = tile_crs(normalized_tile)

        tile_geometry = mapping(
            valid_geometry(row.geometry)
        )

        try:
            search = client.search(
                collections=[S1_COLLECTION],
                datetime=(
                    f"{args.start_date}/"
                    f"{args.end_date}"
                ),
                intersects=tile_geometry,
                max_items=args.max_items_per_tile,
            )

            items = list(search.items())
            items.sort(
                key=lambda item: (
                    item.datetime is None,
                    item.datetime,
                    item.id,
                )
            )

            inspected = [
                inspect_item(
                    item,
                    tile_geometry,
                    destination_crs,
                    args.min_tile_coverage,
                )
                for item in items
            ]

            qualifying = [
                item
                for item in inspected
                if item["keep"]
            ]

            highest_coverage = max(
                (
                    float(item["tile_coverage_pct"])
                    for item in inspected
                    if item.get("platform")
                    in ALLOWED_PLATFORMS
                ),
                default=0.0,
            )

            tile_summary = {
                "tile": product_tile,
                "tile_crs": destination_crs.to_string(),
                "europe_overlap_fraction": float(
                    row["overlap_fraction"]
                ),
                "intersecting_s1_items": len(items),
                "qualifying_items": len(qualifying),
                "highest_allowed_platform_coverage_pct": (
                    highest_coverage
                ),
                "error": None,
            }

            for item in qualifying:
                matches.append(
                    {
                        "tile": product_tile,
                        "tile_crs": (
                            destination_crs.to_string()
                        ),
                        "tile_geometry_wgs84": (
                            tile_geometry
                        ),
                        **item,
                    }
                )

        except Exception as exc:
            tile_summary = {
                "tile": product_tile,
                "tile_crs": destination_crs.to_string(),
                "europe_overlap_fraction": float(
                    row["overlap_fraction"]
                ),
                "intersecting_s1_items": 0,
                "qualifying_items": 0,
                "highest_allowed_platform_coverage_pct": 0.0,
                "error": (
                    f"{type(exc).__name__}:{exc}"
                ),
            }

        tile_summaries.append(tile_summary)

        print(
            f"[{scan_index}/{len(tiles)}] "
            f"{product_tile} | "
            f"items={tile_summary['intersecting_s1_items']} | "
            f"best="
            f"{tile_summary['highest_allowed_platform_coverage_pct']:.3f}% | "
            f"qualifying={tile_summary['qualifying_items']}"
        )

        if len(matches) >= args.max_matches:
            print(
                "Reached requested qualifying-match limit."
            )
            break

    matches.sort(
        key=lambda item: (
            float(item["tile_coverage_pct"]),
            item["tile"],
            item["item_id"],
        ),
        reverse=True,
    )

    report = {
        "scan": {
            "inventory": str(inventory_path),
            "collection": S1_COLLECTION,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "minimum_tile_coverage_fraction": (
                args.min_tile_coverage
            ),
            "coverage_formula": (
                "intersection_area / sentinel_2_tile_area"
            ),
            "required_instrument_mode": (
                REQUIRED_INSTRUMENT_MODE
            ),
            "required_polarisations": sorted(
                REQUIRED_POLARISATIONS
            ),
            "allowed_platforms": sorted(
                ALLOWED_PLATFORMS
            ),
            "tiles_requested": args.max_tiles,
            "tiles_scanned": len(tile_summaries),
        },
        "summary": {
            "qualifying_scene_tile_matches": len(
                matches
            ),
            "tiles_with_qualifying_matches": len(
                {
                    match["tile"]
                    for match in matches
                }
            ),
        },
        "matches": matches,
        "tile_summaries": tile_summaries,
    }

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print("===== QUALIFYING MATCHES =====")

    if not matches:
        print("No qualifying matches found.")
    else:
        for match in matches:
            print(
                f"{match['tile']} | "
                f"{match['tile_coverage_pct']:.3f}% | "
                f"{match['platform']} | "
                f"{match['orbit_state']} | "
                f"{match['item_id']}"
            )

    print()
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()

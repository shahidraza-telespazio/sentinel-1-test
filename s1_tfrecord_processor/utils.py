from __future__ import annotations

from typing import Tuple


def normalize_tile(tile: str) -> str:
    """
    Normalise an MGRS tile identifier.

    Both T30UXC and 30UXC are accepted. The returned value does not
    contain the leading T, matching the convention currently used by
    the Sentinel-2 processor internally.
    """
    normalized = tile.strip().upper()

    if normalized.startswith("T") and len(normalized) == 6:
        normalized = normalized[1:]

    if len(normalized) != 5:
        raise ValueError(
            f"Expected a five-character MGRS tile such as 30UXC "
            f"or T30UXC, received {tile!r}"
        )

    return normalized


def parse_tile(tile: str) -> Tuple[int, str, str]:
    """
    Split an MGRS tile into UTM zone, latitude band and grid square.

    Example:
        T30UXC -> (30, "U", "XC")
    """
    normalized = normalize_tile(tile)

    try:
        utm_zone = int(normalized[:2])
    except ValueError as exc:
        raise ValueError(
            f"Invalid UTM zone in MGRS tile {tile!r}"
        ) from exc

    latitude_band = normalized[2]
    grid_square = normalized[3:]

    return utm_zone, latitude_band, grid_square

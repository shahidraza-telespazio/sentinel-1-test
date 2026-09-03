from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Iterable


KNOWN_POLARISATIONS = (
    "VV",
    "VH",
    "HH",
    "HV",
)

DEFAULT_ALLOWED_PLATFORMS = frozenset(
    {
        "sentinel-1a",
        "sentinel-1b",
    }
)

REQUIRED_INSTRUMENT_MODE = "IW"
REQUIRED_PRODUCT_TYPE = "GRD"


@dataclass(frozen=True)
class PolarisationAssets:
    polarisation: str
    measurement_href: str
    product_metadata_href: str | None
    calibration_metadata_href: str | None
    noise_metadata_href: str | None
    missing_metadata_assets: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Sentinel1Acquisition:
    item_id: str
    collection: str | None
    datetime_utc: str | None
    start_datetime_utc: str | None
    end_datetime_utc: str | None

    platform: str | None
    constellation: str | None

    instrument_mode: str | None
    product_type: str | None

    orbit_state: str | None
    relative_orbit: int | None
    absolute_orbit: int | None

    declared_polarisations: tuple[str, ...]
    polarisation_assets: tuple[
        PolarisationAssets,
        ...
    ]

    safe_manifest_href: str | None
    warnings: tuple[str, ...]

    @property
    def available_polarisations(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            assets.polarisation
            for assets in self.polarisation_assets
        )

    @property
    def measurement_hrefs(
        self,
    ) -> dict[str, str]:
        return {
            assets.polarisation:
            assets.measurement_href
            for assets in self.polarisation_assets
        }

    @property
    def product_metadata_hrefs(
        self,
    ) -> dict[str, str | None]:
        return {
            assets.polarisation:
            assets.product_metadata_href
            for assets in self.polarisation_assets
        }

    @property
    def calibration_hrefs(
        self,
    ) -> dict[str, str | None]:
        return {
            assets.polarisation:
            assets.calibration_metadata_href
            for assets in self.polarisation_assets
        }

    @property
    def noise_hrefs(
        self,
    ) -> dict[str, str | None]:
        return {
            assets.polarisation:
            assets.noise_metadata_href
            for assets in self.polarisation_assets
        }

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["available_polarisations"] = list(
            self.available_polarisations
        )
        result["measurement_hrefs"] = (
            self.measurement_hrefs
        )
        result["product_metadata_hrefs"] = (
            self.product_metadata_hrefs
        )
        result["calibration_hrefs"] = (
            self.calibration_hrefs
        )
        result["noise_hrefs"] = self.noise_hrefs

        return result


def datetime_text(
    value: Any,
) -> str | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.isoformat().replace(
            "+00:00",
            "Z",
        )

    text = str(value).strip()

    return text or None


def optional_integer(
    value: Any,
) -> int | None:
    if value is None or value == "":
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalise_polarisations(
    value: Any,
) -> tuple[str, ...]:
    if value is None:
        entries: Iterable[Any] = ()
    elif isinstance(value, str):
        entries = (value,)
    elif isinstance(
        value,
        (list, tuple, set),
    ):
        entries = value
    else:
        entries = (value,)

    normalised = {
        str(entry).strip().upper()
        for entry in entries
        if str(entry).strip()
    }

    return tuple(
        sorted(
            normalised,
            key=lambda polarisation: (
                KNOWN_POLARISATIONS.index(
                    polarisation
                )
                if polarisation
                in KNOWN_POLARISATIONS
                else len(KNOWN_POLARISATIONS),
                polarisation,
            ),
        )
    )


def lower_asset_map(
    item: Any,
) -> dict[str, Any]:
    return {
        str(key).lower(): asset
        for key, asset in item.assets.items()
    }


def asset_href(
    assets: dict[str, Any],
    key: str,
) -> str | None:
    asset = assets.get(key.lower())

    if asset is None or not asset.href:
        return None

    return str(asset.href)


def product_type_from_properties(
    properties: dict[str, Any],
) -> tuple[str | None, list[str]]:
    warnings: list[str] = []

    current_value = properties.get(
        "product:type"
    )
    legacy_value = properties.get(
        "sar:product_type"
    )

    current = (
        str(current_value).strip().upper()
        if current_value is not None
        else None
    )
    legacy = (
        str(legacy_value).strip().upper()
        if legacy_value is not None
        else None
    )

    if current and legacy and current != legacy:
        warnings.append(
            "product_type_fields_disagree:"
            f"product:type={current},"
            f"sar:product_type={legacy}"
        )

    if current:
        return current, warnings

    if legacy:
        warnings.append(
            "using_deprecated_sar_product_type"
        )
        return legacy, warnings

    warnings.append("product_type_missing")
    return None, warnings


def polarisation_assets_from_item(
    item: Any,
) -> tuple[PolarisationAssets, ...]:
    assets = lower_asset_map(item)

    results: list[PolarisationAssets] = []

    for polarisation in KNOWN_POLARISATIONS:
        suffix = polarisation.lower()

        measurement_href = asset_href(
            assets,
            suffix,
        )

        if measurement_href is None:
            continue

        metadata_keys = {
            "product": (
                f"schema-product-{suffix}"
            ),
            "calibration": (
                f"schema-calibration-{suffix}"
            ),
            "noise": (
                f"schema-noise-{suffix}"
            ),
        }

        product_href = asset_href(
            assets,
            metadata_keys["product"],
        )
        calibration_href = asset_href(
            assets,
            metadata_keys["calibration"],
        )
        noise_href = asset_href(
            assets,
            metadata_keys["noise"],
        )

        missing = tuple(
            key
            for key, href in (
                (
                    metadata_keys["product"],
                    product_href,
                ),
                (
                    metadata_keys["calibration"],
                    calibration_href,
                ),
                (
                    metadata_keys["noise"],
                    noise_href,
                ),
            )
            if href is None
        )

        results.append(
            PolarisationAssets(
                polarisation=polarisation,
                measurement_href=measurement_href,
                product_metadata_href=(
                    product_href
                ),
                calibration_metadata_href=(
                    calibration_href
                ),
                noise_metadata_href=noise_href,
                missing_metadata_assets=missing,
            )
        )

    return tuple(results)


def acquisition_from_item(
    item: Any,
) -> Sentinel1Acquisition:
    properties = item.properties
    assets = lower_asset_map(item)

    warnings: list[str] = []

    declared_polarisations = (
        normalise_polarisations(
            properties.get(
                "sar:polarizations"
            )
        )
    )

    polarisation_assets = (
        polarisation_assets_from_item(item)
    )

    available_polarisations = {
        assets.polarisation
        for assets in polarisation_assets
    }

    declared_supported = {
        polarisation
        for polarisation
        in declared_polarisations
        if polarisation
        in KNOWN_POLARISATIONS
    }

    declared_without_measurement = sorted(
        declared_supported.difference(
            available_polarisations
        )
    )

    measurements_without_declaration = sorted(
        available_polarisations.difference(
            declared_supported
        )
    )

    if declared_without_measurement:
        warnings.append(
            "declared_polarisations_without_"
            "measurement_assets:"
            + ",".join(
                declared_without_measurement
            )
        )

    if measurements_without_declaration:
        warnings.append(
            "measurement_assets_without_"
            "declaration:"
            + ",".join(
                measurements_without_declaration
            )
        )

    if not polarisation_assets:
        warnings.append(
            "no_supported_measurement_assets"
        )

    for polarisation in polarisation_assets:
        if polarisation.missing_metadata_assets:
            warnings.append(
                "missing_metadata_assets_"
                f"{polarisation.polarisation}:"
                + ",".join(
                    polarisation
                    .missing_metadata_assets
                )
            )

    safe_manifest_href = asset_href(
        assets,
        "safe-manifest",
    )

    if safe_manifest_href is None:
        warnings.append(
            "safe_manifest_missing"
        )

    product_type, product_warnings = (
        product_type_from_properties(
            properties
        )
    )
    warnings.extend(product_warnings)

    instrument_mode_value = properties.get(
        "sar:instrument_mode"
    )
    instrument_mode = (
        str(
            instrument_mode_value
        ).strip().upper()
        if instrument_mode_value is not None
        else None
    )

    platform_value = properties.get(
        "platform"
    )
    platform = (
        str(platform_value).strip().lower()
        if platform_value is not None
        else None
    )

    constellation_value = properties.get(
        "constellation"
    )
    constellation = (
        str(
            constellation_value
        ).strip().lower()
        if constellation_value is not None
        else None
    )

    orbit_state_value = properties.get(
        "sat:orbit_state"
    )
    orbit_state = (
        str(
            orbit_state_value
        ).strip().lower()
        if orbit_state_value is not None
        else None
    )

    return Sentinel1Acquisition(
        item_id=str(item.id),
        collection=(
            str(item.collection_id)
            if item.collection_id
            else None
        ),
        datetime_utc=datetime_text(
            item.datetime
        ),
        start_datetime_utc=datetime_text(
            properties.get(
                "start_datetime"
            )
        ),
        end_datetime_utc=datetime_text(
            properties.get(
                "end_datetime"
            )
        ),
        platform=platform,
        constellation=constellation,
        instrument_mode=instrument_mode,
        product_type=product_type,
        orbit_state=orbit_state,
        relative_orbit=optional_integer(
            properties.get(
                "sat:relative_orbit"
            )
        ),
        absolute_orbit=optional_integer(
            properties.get(
                "sat:absolute_orbit"
            )
        ),
        declared_polarisations=(
            declared_polarisations
        ),
        polarisation_assets=(
            polarisation_assets
        ),
        safe_manifest_href=(
            safe_manifest_href
        ),
        warnings=tuple(
            sorted(set(warnings))
        ),
    )


def processing_rejection_reasons(
    acquisition: Sentinel1Acquisition,
    *,
    allowed_platforms: (
        set[str] | frozenset[str] | None
    ) = DEFAULT_ALLOWED_PLATFORMS,
    required_instrument_mode: str = (
        REQUIRED_INSTRUMENT_MODE
    ),
    required_product_type: str = (
        REQUIRED_PRODUCT_TYPE
    ),
) -> tuple[str, ...]:
    reasons: list[str] = []

    if (
        acquisition.instrument_mode
        != required_instrument_mode.upper()
    ):
        reasons.append(
            "instrument_mode_is_"
            f"{acquisition.instrument_mode or 'missing'}"
        )

    if (
        allowed_platforms is not None
        and acquisition.platform
        not in allowed_platforms
    ):
        reasons.append(
            "platform_is_"
            f"{acquisition.platform or 'missing'}"
        )

    if (
        acquisition.product_type
        != required_product_type.upper()
    ):
        reasons.append(
            "product_type_is_"
            f"{acquisition.product_type or 'missing'}"
        )

    if not acquisition.polarisation_assets:
        reasons.append(
            "no_supported_measurement_assets"
        )

    return tuple(reasons)


def processing_eligible(
    acquisition: Sentinel1Acquisition,
    *,
    allowed_platforms: (
        set[str] | frozenset[str] | None
    ) = DEFAULT_ALLOWED_PLATFORMS,
) -> bool:
    return not processing_rejection_reasons(
        acquisition,
        allowed_platforms=allowed_platforms,
    )

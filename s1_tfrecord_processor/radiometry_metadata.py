from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TypeAlias
from xml.etree import ElementTree as ET

import numpy as np


XmlInput: TypeAlias = str | Path | bytes


@dataclass(frozen=True)
class CalibrationVector:
    line: int
    azimuth_time: str | None
    pixels: np.ndarray
    sigma_nought: np.ndarray
    beta_nought: np.ndarray
    gamma: np.ndarray
    dn: np.ndarray


@dataclass(frozen=True)
class CalibrationMetadata:
    product_type: str | None
    mode: str | None
    polarisation: str | None
    absolute_calibration_constant: float | None
    vectors: tuple[CalibrationVector, ...]


@dataclass(frozen=True)
class LegacyNoiseVector:
    line: int
    azimuth_time: str | None
    pixels: np.ndarray
    noise_lut: np.ndarray


@dataclass(frozen=True)
class NoiseRangeVector:
    line: int
    azimuth_time: str | None
    pixels: np.ndarray
    noise_range_lut: np.ndarray


@dataclass(frozen=True)
class NoiseAzimuthVector:
    first_azimuth_line: int
    last_azimuth_line: int
    first_range_sample: int
    last_range_sample: int
    lines: np.ndarray
    noise_azimuth_lut: np.ndarray


@dataclass(frozen=True)
class NoiseMetadata:
    product_type: str | None
    mode: str | None
    polarisation: str | None
    legacy_vectors: tuple[LegacyNoiseVector, ...]
    range_vectors: tuple[NoiseRangeVector, ...]
    azimuth_vectors: tuple[NoiseAzimuthVector, ...]

    @property
    def format_name(self) -> str:
        if self.legacy_vectors:
            return "legacy_noise_vector"

        if self.range_vectors or self.azimuth_vectors:
            return "range_azimuth_vectors"

        return "none"


@dataclass(frozen=True)
class ProductRadiometryMetadata:
    product_type: str | None
    mode: str | None
    polarisation: str | None
    pixel_value: str | None
    number_of_samples: int | None
    number_of_lines: int | None
    thermal_noise_correction_performed: bool | None


@dataclass(frozen=True)
class RadiometryMetadata:
    product: ProductRadiometryMetadata
    calibration: CalibrationMetadata
    noise: NoiseMetadata


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_xml(value: XmlInput) -> ET.Element:
    if isinstance(value, bytes):
        return ET.fromstring(value)

    path = Path(value)
    return ET.parse(path).getroot()


def iter_named(
    root: ET.Element,
    name: str,
) -> Iterable[ET.Element]:
    for element in root.iter():
        if local_name(element.tag) == name:
            yield element


def first_named(
    root: ET.Element,
    name: str,
) -> ET.Element | None:
    return next(iter_named(root, name), None)


def child_named(
    parent: ET.Element,
    name: str,
) -> ET.Element | None:
    for child in parent:
        if local_name(child.tag) == name:
            return child

    return None


def optional_text(
    root: ET.Element,
    name: str,
) -> str | None:
    element = first_named(root, name)

    if element is None or element.text is None:
        return None

    value = element.text.strip()
    return value or None


def required_child_text(
    parent: ET.Element,
    name: str,
) -> str:
    element = child_named(parent, name)

    if (
        element is None
        or element.text is None
        or not element.text.strip()
    ):
        raise ValueError(
            f"Missing required {name!r} value in "
            f"{local_name(parent.tag)!r}."
        )

    return element.text.strip()


def optional_int(
    root: ET.Element,
    name: str,
) -> int | None:
    value = optional_text(root, name)
    return int(value) if value is not None else None


def optional_float(
    root: ET.Element,
    name: str,
) -> float | None:
    value = optional_text(root, name)
    return float(value) if value is not None else None


def optional_bool(
    root: ET.Element,
    name: str,
) -> bool | None:
    value = optional_text(root, name)

    if value is None:
        return None

    normalised = value.strip().lower()

    if normalised == "true":
        return True

    if normalised == "false":
        return False

    raise ValueError(
        f"Expected true or false for {name!r}, "
        f"received {value!r}."
    )


def int_array(text: str) -> np.ndarray:
    values = np.fromstring(
        text,
        sep=" ",
        dtype=np.int64,
    )

    if values.size == 0:
        raise ValueError(
            "Expected at least one integer value."
        )

    return values


def float_array(text: str) -> np.ndarray:
    values = np.fromstring(
        text,
        sep=" ",
        dtype=np.float64,
    )

    if values.size == 0:
        raise ValueError(
            "Expected at least one floating-point value."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Metadata LUT contains a non-finite value."
        )

    return values


def validate_increasing(
    values: np.ndarray,
    label: str,
) -> None:
    if values.ndim != 1:
        raise ValueError(
            f"{label} must be one-dimensional."
        )

    if values.size > 1 and not np.all(
        np.diff(values) > 0
    ):
        raise ValueError(
            f"{label} must be strictly increasing."
        )


def validate_same_length(
    label: str,
    **arrays: np.ndarray,
) -> None:
    lengths = {
        name: int(value.size)
        for name, value in arrays.items()
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"{label} arrays have different lengths: "
            f"{lengths}"
        )


def normalise_polarisation(
    value: str | None,
) -> str | None:
    return value.strip().upper() if value else None


def parse_calibration_vector(
    element: ET.Element,
) -> CalibrationVector:
    pixels = int_array(
        required_child_text(element, "pixel")
    )
    sigma_nought = float_array(
        required_child_text(
            element,
            "sigmaNought",
        )
    )
    beta_nought = float_array(
        required_child_text(
            element,
            "betaNought",
        )
    )
    gamma = float_array(
        required_child_text(element, "gamma")
    )
    dn = float_array(
        required_child_text(element, "dn")
    )

    validate_increasing(
        pixels,
        "calibration pixels",
    )
    validate_same_length(
        "calibration vector",
        pixels=pixels,
        sigma_nought=sigma_nought,
        beta_nought=beta_nought,
        gamma=gamma,
        dn=dn,
    )

    azimuth = child_named(
        element,
        "azimuthTime",
    )

    return CalibrationVector(
        line=int(
            required_child_text(
                element,
                "line",
            )
        ),
        azimuth_time=(
            azimuth.text.strip()
            if (
                azimuth is not None
                and azimuth.text
                and azimuth.text.strip()
            )
            else None
        ),
        pixels=pixels,
        sigma_nought=sigma_nought,
        beta_nought=beta_nought,
        gamma=gamma,
        dn=dn,
    )


def parse_calibration_metadata(
    value: XmlInput,
) -> CalibrationMetadata:
    root = parse_xml(value)

    vectors = tuple(
        parse_calibration_vector(element)
        for element in iter_named(
            root,
            "calibrationVector",
        )
    )

    if not vectors:
        raise ValueError(
            "Calibration XML contains no "
            "calibrationVector elements."
        )

    lines = np.asarray(
        [vector.line for vector in vectors],
        dtype=np.int64,
    )
    validate_increasing(
        lines,
        "calibration vector lines",
    )

    return CalibrationMetadata(
        product_type=optional_text(
            root,
            "productType",
        ),
        mode=optional_text(root, "mode"),
        polarisation=normalise_polarisation(
            optional_text(
                root,
                "polarisation",
            )
        ),
        absolute_calibration_constant=(
            optional_float(
                root,
                "absoluteCalibrationConstant",
            )
        ),
        vectors=vectors,
    )


def parse_legacy_noise_vector(
    element: ET.Element,
) -> LegacyNoiseVector:
    pixels = int_array(
        required_child_text(element, "pixel")
    )
    noise_lut = float_array(
        required_child_text(
            element,
            "noiseLut",
        )
    )

    validate_increasing(
        pixels,
        "legacy noise pixels",
    )
    validate_same_length(
        "legacy noise vector",
        pixels=pixels,
        noise_lut=noise_lut,
    )

    azimuth = child_named(
        element,
        "azimuthTime",
    )

    return LegacyNoiseVector(
        line=int(
            required_child_text(
                element,
                "line",
            )
        ),
        azimuth_time=(
            azimuth.text.strip()
            if (
                azimuth is not None
                and azimuth.text
                and azimuth.text.strip()
            )
            else None
        ),
        pixels=pixels,
        noise_lut=noise_lut,
    )


def parse_noise_range_vector(
    element: ET.Element,
) -> NoiseRangeVector:
    pixels = int_array(
        required_child_text(element, "pixel")
    )
    noise_lut = float_array(
        required_child_text(
            element,
            "noiseRangeLut",
        )
    )

    validate_increasing(
        pixels,
        "range-noise pixels",
    )
    validate_same_length(
        "range-noise vector",
        pixels=pixels,
        noise_range_lut=noise_lut,
    )

    azimuth = child_named(
        element,
        "azimuthTime",
    )

    return NoiseRangeVector(
        line=int(
            required_child_text(
                element,
                "line",
            )
        ),
        azimuth_time=(
            azimuth.text.strip()
            if (
                azimuth is not None
                and azimuth.text
                and azimuth.text.strip()
            )
            else None
        ),
        pixels=pixels,
        noise_range_lut=noise_lut,
    )


def parse_noise_azimuth_vector(
    element: ET.Element,
) -> NoiseAzimuthVector:
    lines = int_array(
        required_child_text(element, "line")
    )
    noise_lut = float_array(
        required_child_text(
            element,
            "noiseAzimuthLut",
        )
    )

    validate_increasing(
        lines,
        "azimuth-noise lines",
    )
    validate_same_length(
        "azimuth-noise vector",
        lines=lines,
        noise_azimuth_lut=noise_lut,
    )

    return NoiseAzimuthVector(
        first_azimuth_line=int(
            required_child_text(
                element,
                "firstAzimuthLine",
            )
        ),
        last_azimuth_line=int(
            required_child_text(
                element,
                "lastAzimuthLine",
            )
        ),
        first_range_sample=int(
            required_child_text(
                element,
                "firstRangeSample",
            )
        ),
        last_range_sample=int(
            required_child_text(
                element,
                "lastRangeSample",
            )
        ),
        lines=lines,
        noise_azimuth_lut=noise_lut,
    )


def validate_vector_lines(
    vectors: Iterable[
        LegacyNoiseVector | NoiseRangeVector
    ],
    label: str,
) -> None:
    vector_list = list(vectors)

    if not vector_list:
        return

    lines = np.asarray(
        [vector.line for vector in vector_list],
        dtype=np.int64,
    )
    validate_increasing(lines, label)


def parse_noise_metadata(
    value: XmlInput,
) -> NoiseMetadata:
    root = parse_xml(value)

    legacy_vectors = tuple(
        parse_legacy_noise_vector(element)
        for element in iter_named(
            root,
            "noiseVector",
        )
    )
    range_vectors = tuple(
        parse_noise_range_vector(element)
        for element in iter_named(
            root,
            "noiseRangeVector",
        )
    )
    azimuth_vectors = tuple(
        parse_noise_azimuth_vector(element)
        for element in iter_named(
            root,
            "noiseAzimuthVector",
        )
    )

    if legacy_vectors and (
        range_vectors or azimuth_vectors
    ):
        raise ValueError(
            "Noise XML mixes the legacy noiseVector "
            "format with the newer range/azimuth format."
        )

    if not (
        legacy_vectors
        or range_vectors
        or azimuth_vectors
    ):
        raise ValueError(
            "Noise XML contains no supported noise "
            "vectors."
        )

    if azimuth_vectors and not range_vectors:
        raise ValueError(
            "Azimuth-noise vectors are present without "
            "range-noise vectors."
        )

    validate_vector_lines(
        legacy_vectors,
        "legacy noise vector lines",
    )
    validate_vector_lines(
        range_vectors,
        "range-noise vector lines",
    )

    for vector in azimuth_vectors:
        if (
            vector.first_azimuth_line
            > vector.last_azimuth_line
        ):
            raise ValueError(
                "Azimuth-noise first line is after "
                "the last line."
            )

        if (
            vector.first_range_sample
            > vector.last_range_sample
        ):
            raise ValueError(
                "Azimuth-noise first range sample is "
                "after the last range sample."
            )

    return NoiseMetadata(
        product_type=optional_text(
            root,
            "productType",
        ),
        mode=optional_text(root, "mode"),
        polarisation=normalise_polarisation(
            optional_text(
                root,
                "polarisation",
            )
        ),
        legacy_vectors=legacy_vectors,
        range_vectors=range_vectors,
        azimuth_vectors=azimuth_vectors,
    )


def parse_product_radiometry_metadata(
    value: XmlInput,
) -> ProductRadiometryMetadata:
    root = parse_xml(value)

    return ProductRadiometryMetadata(
        product_type=optional_text(
            root,
            "productType",
        ),
        mode=optional_text(root, "mode"),
        polarisation=normalise_polarisation(
            optional_text(
                root,
                "polarisation",
            )
        ),
        pixel_value=optional_text(
            root,
            "pixelValue",
        ),
        number_of_samples=optional_int(
            root,
            "numberOfSamples",
        ),
        number_of_lines=optional_int(
            root,
            "numberOfLines",
        ),
        thermal_noise_correction_performed=(
            optional_bool(
                root,
                "thermalNoiseCorrectionPerformed",
            )
        ),
    )


def validate_consistency(
    metadata: RadiometryMetadata,
    expected_polarisation: str | None,
) -> None:
    expected = normalise_polarisation(
        expected_polarisation
    )

    polarisations = {
        value
        for value in (
            metadata.product.polarisation,
            metadata.calibration.polarisation,
            metadata.noise.polarisation,
        )
        if value is not None
    }

    if len(polarisations) > 1:
        raise ValueError(
            "Product, calibration and noise XMLs "
            "disagree on polarisation: "
            f"{sorted(polarisations)}"
        )

    if (
        expected is not None
        and polarisations
        and expected not in polarisations
    ):
        raise ValueError(
            f"Expected polarisation {expected}, "
            f"metadata reports "
            f"{sorted(polarisations)}."
        )

    product_types = {
        value
        for value in (
            metadata.product.product_type,
            metadata.calibration.product_type,
            metadata.noise.product_type,
        )
        if value is not None
    }

    if len(product_types) > 1:
        raise ValueError(
            "Product, calibration and noise XMLs "
            "disagree on product type: "
            f"{sorted(product_types)}"
        )

    modes = {
        value
        for value in (
            metadata.product.mode,
            metadata.calibration.mode,
            metadata.noise.mode,
        )
        if value is not None
    }

    if len(modes) > 1:
        raise ValueError(
            "Product, calibration and noise XMLs "
            "disagree on acquisition mode: "
            f"{sorted(modes)}"
        )

    if (
        metadata.product.number_of_samples
        is not None
        and metadata.product.number_of_samples <= 0
    ):
        raise ValueError(
            "numberOfSamples must be positive."
        )

    if (
        metadata.product.number_of_lines
        is not None
        and metadata.product.number_of_lines <= 0
    ):
        raise ValueError(
            "numberOfLines must be positive."
        )


def load_radiometry_metadata(
    *,
    product_xml: XmlInput,
    calibration_xml: XmlInput,
    noise_xml: XmlInput,
    expected_polarisation: str | None = None,
) -> RadiometryMetadata:
    metadata = RadiometryMetadata(
        product=parse_product_radiometry_metadata(
            product_xml
        ),
        calibration=parse_calibration_metadata(
            calibration_xml
        ),
        noise=parse_noise_metadata(noise_xml),
    )

    validate_consistency(
        metadata,
        expected_polarisation,
    )

    return metadata


def array_range(
    arrays: Iterable[np.ndarray],
) -> dict[str, int | float | None]:
    array_list = list(arrays)

    if not array_list:
        return {
            "minimum": None,
            "maximum": None,
        }

    minimum = min(
        float(array.min())
        for array in array_list
    )
    maximum = max(
        float(array.max())
        for array in array_list
    )

    return {
        "minimum": minimum,
        "maximum": maximum,
    }


def metadata_summary(
    metadata: RadiometryMetadata,
) -> dict[str, Any]:
    calibration = metadata.calibration
    noise = metadata.noise
    product = metadata.product

    calibration_pixels = [
        vector.pixels
        for vector in calibration.vectors
    ]

    if noise.legacy_vectors:
        noise_lines = [
            vector.line
            for vector in noise.legacy_vectors
        ]
        noise_pixels = [
            vector.pixels
            for vector in noise.legacy_vectors
        ]
    else:
        noise_lines = [
            vector.line
            for vector in noise.range_vectors
        ]
        noise_pixels = [
            vector.pixels
            for vector in noise.range_vectors
        ]

    polarisation = next(
        (
            value
            for value in (
                product.polarisation,
                calibration.polarisation,
                noise.polarisation,
            )
            if value is not None
        ),
        None,
    )

    return {
        "product": {
            "product_type": product.product_type,
            "mode": product.mode,
            "polarisation": polarisation,
            "pixel_value": product.pixel_value,
            "number_of_samples": (
                product.number_of_samples
            ),
            "number_of_lines": (
                product.number_of_lines
            ),
            "thermal_noise_correction_performed": (
                product
                .thermal_noise_correction_performed
            ),
        },
        "calibration": {
            "vector_count": len(
                calibration.vectors
            ),
            "line_range": {
                "minimum": min(
                    vector.line
                    for vector in calibration.vectors
                ),
                "maximum": max(
                    vector.line
                    for vector in calibration.vectors
                ),
            },
            "pixel_range": array_range(
                calibration_pixels
            ),
            "absolute_calibration_constant": (
                calibration
                .absolute_calibration_constant
            ),
            "available_luts": [
                "sigmaNought",
                "betaNought",
                "gamma",
                "dn",
            ],
        },
        "noise": {
            "format": noise.format_name,
            "legacy_vector_count": len(
                noise.legacy_vectors
            ),
            "range_vector_count": len(
                noise.range_vectors
            ),
            "azimuth_vector_count": len(
                noise.azimuth_vectors
            ),
            "line_range": (
                {
                    "minimum": min(noise_lines),
                    "maximum": max(noise_lines),
                }
                if noise_lines
                else {
                    "minimum": None,
                    "maximum": None,
                }
            ),
            "pixel_range": array_range(
                noise_pixels
            ),
            "azimuth_sections": [
                {
                    "first_azimuth_line": (
                        vector.first_azimuth_line
                    ),
                    "last_azimuth_line": (
                        vector.last_azimuth_line
                    ),
                    "first_range_sample": (
                        vector.first_range_sample
                    ),
                    "last_range_sample": (
                        vector.last_range_sample
                    ),
                    "line_count": int(
                        vector.lines.size
                    ),
                }
                for vector in noise.azimuth_vectors
            ],
        },
        "processing_status": {
            "metadata_parsing_complete": True,
            "lut_interpolation_complete": False,
            "noise_removal_complete": False,
            "radiometric_calibration_complete": False,
            "db_conversion_complete": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse and validate Sentinel-1 GRD "
            "product, calibration and noise XMLs. "
            "This command does not alter measurement "
            "pixels."
        )
    )
    parser.add_argument(
        "--product-xml",
        required=True,
    )
    parser.add_argument(
        "--calibration-xml",
        required=True,
    )
    parser.add_argument(
        "--noise-xml",
        required=True,
    )
    parser.add_argument(
        "--polarisation",
        default=None,
    )
    parser.add_argument(
        "--output-json",
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata = load_radiometry_metadata(
        product_xml=args.product_xml,
        calibration_xml=args.calibration_xml,
        noise_xml=args.noise_xml,
        expected_polarisation=(
            args.polarisation
        ),
    )
    summary = metadata_summary(metadata)

    print(
        "Sentinel-1 radiometry metadata"
    )
    print("------------------------------")
    print(
        "Polarisation:              "
        f"{summary['product']['polarisation']}"
    )
    print(
        "Product type:              "
        f"{summary['product']['product_type']}"
    )
    print(
        "Mode:                      "
        f"{summary['product']['mode']}"
    )
    print(
        "Pixel value:               "
        f"{summary['product']['pixel_value']}"
    )
    print(
        "Samples × lines:           "
        f"{summary['product']['number_of_samples']} "
        "× "
        f"{summary['product']['number_of_lines']}"
    )
    print(
        "Noise correction performed:"
        f" "
        f"{summary['product']['thermal_noise_correction_performed']}"
    )
    print(
        "Calibration vectors:       "
        f"{summary['calibration']['vector_count']}"
    )
    print(
        "Noise format:              "
        f"{summary['noise']['format']}"
    )
    print(
        "Legacy noise vectors:      "
        f"{summary['noise']['legacy_vector_count']}"
    )
    print(
        "Range noise vectors:       "
        f"{summary['noise']['range_vector_count']}"
    )
    print(
        "Azimuth noise vectors:     "
        f"{summary['noise']['azimuth_vector_count']}"
    )
    print()
    print(
        "No measurement pixels were modified."
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_text(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"JSON: {output_path}")


if __name__ == "__main__":
    main()

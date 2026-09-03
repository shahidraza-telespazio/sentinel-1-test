from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from tfrecord_xarray_io import (
    read_tfrecord_dataset,
    serialize_dataset_to_tfrecord,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a calibrated Sentinel-1 time-series NPZ into the "
            "project TFRecord xarray format, read it back, and verify an "
            "exact round-trip."
        )
    )
    parser.add_argument(
        "--input-npz",
        required=True,
        help="Calibrated Sentinel-1 time-series NPZ.",
    )
    parser.add_argument(
        "--output-tfrecord",
        default=None,
        help=(
            "Output TFRecord path. Defaults to the input NPZ path with "
            "the .tfrecord suffix."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Number of time steps per TFRecord chunk. Default: 64.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()

    if args.chunk_size <= 0:
        parser.error("--chunk-size must be greater than zero")

    return args


def _scalar_text(array: np.ndarray) -> str:
    value = np.asarray(array)

    if value.ndim != 0:
        raise RuntimeError(
            f"Expected scalar string array, found shape {value.shape}"
        )

    return str(value.item())


def _metadata_from_npz(npz: Any) -> dict[str, Any]:
    if "metadata_json" not in npz.files:
        return {}

    raw = _scalar_text(npz["metadata_json"])

    if not raw:
        return {}

    parsed = json.loads(raw)

    if not isinstance(parsed, dict):
        raise RuntimeError(
            "metadata_json must contain a JSON object."
        )

    return parsed


def _require_keys(npz: Any, required: tuple[str, ...]) -> None:
    missing = [
        name
        for name in required
        if name not in npz.files
    ]

    if missing:
        raise RuntimeError(
            "Input NPZ is missing required arrays: "
            + ", ".join(missing)
        )


def _to_datetime64_ns(values: np.ndarray) -> np.ndarray:
    result: list[np.datetime64] = []

    for value in np.asarray(values).tolist():
        text = str(value).strip()

        if not text:
            result.append(np.datetime64("NaT", "ns"))
            continue

        # NumPy does not need the trailing UTC marker once values are
        # normalised to UTC by the source pipeline.
        if text.endswith("Z"):
            text = text[:-1]

        result.append(
            np.datetime64(text, "ns")
        )

    return np.asarray(
        result,
        dtype="datetime64[ns]",
    )


def build_dataset_from_npz(
    input_path: Path,
) -> xr.Dataset:
    with np.load(
        input_path,
        allow_pickle=False,
    ) as npz:
        _require_keys(
            npz,
            (
                "backscatter_db_values",
                "valid_mask_values",
                "valid_area_fraction_values",
                "band_present",
                "band_pass",
                "x",
                "y",
                "band_name",
                "time",
                "item_id",
                "platform",
                "orbit_state",
                "relative_orbit",
                "absolute_orbit",
                "destination_crs",
                "destination_transform",
                "calibration_lut",
                "value_units",
            ),
        )

        backscatter = np.asarray(
            npz["backscatter_db_values"],
            dtype=np.float32,
        )

        if backscatter.ndim != 4:
            raise RuntimeError(
                "backscatter_db_values must have dimension order "
                "(time, y, x, band); "
                f"found shape {backscatter.shape}"
            )

        n_time, n_y, n_x, n_band = (
            backscatter.shape
        )

        valid_mask = np.asarray(
            npz["valid_mask_values"],
            dtype=bool,
        )
        valid_area = np.asarray(
            npz["valid_area_fraction_values"],
            dtype=np.float32,
        )
        band_present = np.asarray(
            npz["band_present"],
            dtype=bool,
        )
        band_pass = np.asarray(
            npz["band_pass"],
            dtype=bool,
        )

        if valid_mask.shape != backscatter.shape:
            raise RuntimeError(
                "valid_mask_values shape differs from backscatter: "
                f"{valid_mask.shape} != {backscatter.shape}"
            )

        if valid_area.shape != backscatter.shape:
            raise RuntimeError(
                "valid_area_fraction_values shape differs from "
                f"backscatter: {valid_area.shape} != {backscatter.shape}"
            )

        expected_time_band = (
            n_time,
            n_band,
        )

        if band_present.shape != expected_time_band:
            raise RuntimeError(
                "band_present must have shape "
                f"{expected_time_band}; found {band_present.shape}"
            )

        if band_pass.shape != expected_time_band:
            raise RuntimeError(
                "band_pass must have shape "
                f"{expected_time_band}; found {band_pass.shape}"
            )

        x = np.asarray(
            npz["x"],
            dtype=np.float64,
        )
        y = np.asarray(
            npz["y"],
            dtype=np.float64,
        )
        bands = np.asarray(
            npz["band_name"],
        ).astype(str)
        time = _to_datetime64_ns(
            npz["time"]
        )
        item_id = np.asarray(
            npz["item_id"],
        ).astype(str)
        platform = np.asarray(
            npz["platform"],
        ).astype(str)
        orbit_state = np.asarray(
            npz["orbit_state"],
        ).astype(str)
        relative_orbit = np.asarray(
            npz["relative_orbit"],
            dtype=np.int64,
        )
        absolute_orbit = np.asarray(
            npz["absolute_orbit"],
            dtype=np.int64,
        )

        if x.shape != (n_x,):
            raise RuntimeError(
                f"x shape must be {(n_x,)}, found {x.shape}"
            )

        if y.shape != (n_y,):
            raise RuntimeError(
                f"y shape must be {(n_y,)}, found {y.shape}"
            )

        if bands.shape != (n_band,):
            raise RuntimeError(
                "band_name length does not match backscatter band "
                f"dimension: {bands.shape} vs {n_band}"
            )

        for name, values in (
            ("time", time),
            ("item_id", item_id),
            ("platform", platform),
            ("orbit_state", orbit_state),
            ("relative_orbit", relative_orbit),
            ("absolute_orbit", absolute_orbit),
        ):
            if values.shape != (n_time,):
                raise RuntimeError(
                    f"{name} shape must be {(n_time,)}, "
                    f"found {values.shape}"
                )

        destination_crs = _scalar_text(
            npz["destination_crs"]
        )
        destination_transform = tuple(
            float(value)
            for value in np.asarray(
                npz["destination_transform"],
                dtype=np.float64,
            ).tolist()
        )
        calibration_lut = _scalar_text(
            npz["calibration_lut"]
        )
        value_units = _scalar_text(
            npz["value_units"]
        )

        source_metadata = _metadata_from_npz(
            npz
        )

    tile = str(
        source_metadata.get(
            "tile",
            "",
        )
    )

    ds = xr.Dataset(
        data_vars={
            "backscatter_db": (
                (
                    "time",
                    "y",
                    "x",
                    "band",
                ),
                backscatter,
            ),
            "valid_mask": (
                (
                    "time",
                    "y",
                    "x",
                    "band",
                ),
                valid_mask,
            ),
            "valid_area_fraction": (
                (
                    "time",
                    "y",
                    "x",
                    "band",
                ),
                valid_area,
            ),
            "band_present": (
                (
                    "time",
                    "band",
                ),
                band_present,
            ),
            "band_pass": (
                (
                    "time",
                    "band",
                ),
                band_pass,
            ),
        },
        coords={
            "time": (
                "time",
                time,
            ),
            "y": (
                "y",
                y,
            ),
            "x": (
                "x",
                x,
            ),
            "band": (
                "band",
                np.arange(
                    n_band,
                    dtype=np.int16,
                ),
            ),
            "band_name": (
                "band",
                bands,
            ),
            "item_id": (
                "time",
                item_id,
            ),
            "platform": (
                "time",
                platform,
            ),
            "orbit_state": (
                "time",
                orbit_state,
            ),
            "relative_orbit": (
                "time",
                relative_orbit,
            ),
            "absolute_orbit": (
                "time",
                absolute_orbit,
            ),
        },
        attrs={
            "mgrs_tile": tile,
            "source_collection": (
                "sentinel-1-grd"
            ),
            "product_type": "GRD",
            "instrument_mode": "IW",
            "value_variable": (
                "backscatter_db"
            ),
            "value_units": value_units,
            "calibration_lut": (
                calibration_lut
            ),
            "dimension_order": (
                "time,y,x,band"
            ),
            "crs_wkt": (
                destination_crs
            ),
            "destination_crs": (
                destination_crs
            ),
            "dst_transform": (
                destination_transform
            ),
            "processing_order": (
                "detected magnitude -> power -> thermal-noise "
                "correction -> linear radiometric calibration -> "
                "GCP average reprojection -> dB"
            ),
            "source_metadata_json": (
                json.dumps(
                    source_metadata,
                    sort_keys=True,
                )
            ),
        },
    )

    ds["backscatter_db"].attrs.update(
        {
            "long_name": (
                "Sentinel-1 calibrated backscatter"
            ),
            "units": value_units,
            "calibration_lut": (
                calibration_lut
            ),
            "stored_as": (
                "float32_calibrated_backscatter_db"
            ),
            "resampling": (
                "average in calibrated linear power "
                "before dB conversion"
            ),
        }
    )

    ds["valid_mask"].attrs.update(
        {
            "long_name": (
                "valid calibrated output pixel mask"
            ),
        }
    )

    ds[
        "valid_area_fraction"
    ].attrs.update(
        {
            "long_name": (
                "area-weighted source valid coverage fraction "
                "contributing to each output pixel"
            ),
            "units": "fraction",
        }
    )

    ds["band_present"].attrs.update(
        {
            "long_name": (
                "polarisation asset present in acquisition"
            ),
        }
    )

    ds["band_pass"].attrs.update(
        {
            "long_name": (
                "polarisation passed configured coverage threshold"
            ),
        }
    )

    ds["x"].attrs.update(
        {
            "standard_name": (
                "projection_x_coordinate"
            ),
            "units": "m",
        }
    )

    ds["y"].attrs.update(
        {
            "standard_name": (
                "projection_y_coordinate"
            ),
            "units": "m",
        }
    )

    ds["time"].encoding = {
        "units": (
            "nanoseconds since "
            "1970-01-01 00:00:00"
        ),
        "calendar": (
            "proleptic_gregorian"
        ),
        "dtype": "int64",
    }

    return ds


def assert_roundtrip(
    expected: xr.Dataset,
    actual: xr.Dataset,
) -> None:
    xr.testing.assert_identical(
        expected,
        actual,
    )

    expected_names = (
        set(expected.coords)
        | set(expected.data_vars)
    )
    actual_names = (
        set(actual.coords)
        | set(actual.data_vars)
    )

    if expected_names != actual_names:
        raise AssertionError(
            "Variable mismatch: "
            f"expected={sorted(expected_names)} "
            f"actual={sorted(actual_names)}"
        )

    for name in sorted(
        expected_names
    ):
        if (
            dict(
                expected[
                    name
                ].encoding
            )
            != dict(
                actual[
                    name
                ].encoding
            )
        ):
            raise AssertionError(
                f"Encoding mismatch for {name}:\n"
                f"expected="
                f"{expected[name].encoding}\n"
                f"actual="
                f"{actual[name].encoding}"
            )


def main() -> None:
    args = parse_args()

    input_path = Path(
        args.input_npz
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input NPZ does not exist: {input_path}"
        )

    output_path = (
        Path(
            args.output_tfrecord
        )
        if args.output_tfrecord
        else input_path.with_suffix(
            ".tfrecord"
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ds = build_dataset_from_npz(
        input_path
    )

    time_count = int(
        ds.sizes.get(
            "time",
            0,
        )
    )
    chunk_size = max(
        1,
        min(
            args.chunk_size,
            max(
                1,
                time_count,
            ),
        ),
    )

    print(
        "Sentinel-1 TFRecord round-trip validation"
    )
    print(
        "-----------------------------------------"
    )
    print(
        f"Input NPZ:       {input_path}"
    )
    print(
        f"Output TFRecord: {output_path}"
    )
    print(
        f"Dataset sizes:   {dict(ds.sizes)}"
    )
    print(
        "Data variables:  "
        f"{list(ds.data_vars)}"
    )
    print(
        "Bands:           "
        f"{ds['band_name'].values.tolist()}"
    )
    print(
        "Times:           "
        f"{ds['time'].values.astype('datetime64[ns]').tolist()}"
    )
    print(
        f"Chunk size:      {chunk_size}"
    )

    serialize_dataset_to_tfrecord(
        ds,
        str(output_path),
        chunk_dim="time",
        chunk_size=chunk_size,
        overwrite=args.overwrite,
    )

    roundtrip = (
        read_tfrecord_dataset(
            str(output_path)
        )
    )

    assert_roundtrip(
        ds,
        roundtrip,
    )

    expected_values = np.asarray(
        ds[
            "backscatter_db"
        ].values
    )
    actual_values = np.asarray(
        roundtrip[
            "backscatter_db"
        ].values
    )

    if not np.array_equal(
        np.isnan(expected_values),
        np.isnan(actual_values),
    ):
        raise AssertionError(
            "NaN mask changed during TFRecord round-trip."
        )

    finite = np.isfinite(
        expected_values
    )

    if not np.array_equal(
        expected_values[finite],
        actual_values[finite],
    ):
        raise AssertionError(
            "Finite backscatter values changed during "
            "TFRecord round-trip."
        )

    print()
    print("Round-trip result")
    print(
        "Shape:           "
        f"{roundtrip['backscatter_db'].shape}"
    )
    print(
        "Bands:           "
        f"{roundtrip['band_name'].values.tolist()}"
    )
    print(
        "Platforms:       "
        f"{roundtrip['platform'].values.tolist()}"
    )
    print(
        "Orbit states:    "
        f"{roundtrip['orbit_state'].values.tolist()}"
    )
    print(
        "Relative orbits: "
        f"{roundtrip['relative_orbit'].values.tolist()}"
    )
    print(
        "Absolute orbits: "
        f"{roundtrip['absolute_orbit'].values.tolist()}"
    )
    print(
        "Finite values:   "
        f"{int(finite.sum())}/{finite.size}"
    )
    print(
        "File size:       "
        f"{os.path.getsize(output_path)} bytes"
    )
    print()
    print(
        "TFRecord exact round-trip validation passed."
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import io
import json
import os
import shutil
import struct
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from rasterio.transform import Affine


# =========================
# Metadata helpers
# =========================


def _encode_meta(value: Any) -> Any:
    """JSON-safe encoder that preserves tuples and NumPy scalar/dtype metadata."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.dtype):
        return {"__type__": "dtype", "value": str(value)}
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_encode_meta(v) for v in value]}
    if isinstance(value, list):
        return [_encode_meta(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _encode_meta(v) for k, v in value.items()}
    raise TypeError(f"Unsupported metadata type for TFRecord serialization: {type(value)!r}")



def _decode_meta(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_meta(v) for v in value]
    if isinstance(value, dict):
        typ = value.get("__type__")
        if typ == "tuple":
            return tuple(_decode_meta(v) for v in value["items"])
        if typ == "dtype":
            return np.dtype(value["value"])
        return {k: _decode_meta(v) for k, v in value.items()}
    return value



def _json_bytes(data: Mapping[str, Any]) -> np.ndarray:
    payload = json.dumps(_encode_meta(dict(data)), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return np.frombuffer(payload, dtype=np.uint8)



def _from_json_bytes(array: np.ndarray) -> Dict[str, Any]:
    raw = np.asarray(array, dtype=np.uint8).tobytes().decode("utf-8")
    return _decode_meta(json.loads(raw))


# =========================
# CRC32C + TFRecord framing
# =========================


_CRC32C_POLY = 0x82F63B78


def _make_crc32c_table() -> List[int]:
    table: List[int] = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ _CRC32C_POLY
            else:
                crc >>= 1
        table.append(crc & 0xFFFFFFFF)
    return table


_CRC32C_TABLE = _make_crc32c_table()



def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF



def masked_crc32c(data: bytes) -> int:
    crc = crc32c(data)
    return (((crc >> 15) | ((crc << 17) & 0xFFFFFFFF)) + 0xA282EAD8) & 0xFFFFFFFF



def _write_record(handle, payload: bytes) -> None:
    length_bytes = struct.pack("<Q", len(payload))
    handle.write(length_bytes)
    handle.write(struct.pack("<I", masked_crc32c(length_bytes)))
    handle.write(payload)
    handle.write(struct.pack("<I", masked_crc32c(payload)))



def iter_tfrecord_records(path: str) -> Iterator[bytes]:
    with open(path, "rb") as handle:
        while True:
            length_bytes = handle.read(8)
            if not length_bytes:
                break
            if len(length_bytes) != 8:
                raise EOFError("Truncated TFRecord length header.")

            length_crc_bytes = handle.read(4)
            if len(length_crc_bytes) != 4:
                raise EOFError("Truncated TFRecord length CRC.")
            expected_length_crc = masked_crc32c(length_bytes)
            actual_length_crc = struct.unpack("<I", length_crc_bytes)[0]
            if actual_length_crc != expected_length_crc:
                raise ValueError("Invalid TFRecord length CRC32C.")

            length = struct.unpack("<Q", length_bytes)[0]
            payload = handle.read(length)
            if len(payload) != length:
                raise EOFError("Truncated TFRecord payload.")

            payload_crc_bytes = handle.read(4)
            if len(payload_crc_bytes) != 4:
                raise EOFError("Truncated TFRecord payload CRC.")
            expected_payload_crc = masked_crc32c(payload)
            actual_payload_crc = struct.unpack("<I", payload_crc_bytes)[0]
            if actual_payload_crc != expected_payload_crc:
                raise ValueError("Invalid TFRecord payload CRC32C.")

            yield payload


# =========================
# Record payload packing
# =========================


MANIFEST_KEY = "__manifest__"



def _pack_npz_payload(meta: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> bytes:
    to_save: Dict[str, np.ndarray] = {MANIFEST_KEY: _json_bytes(meta)}
    for name, array in arrays.items():
        to_save[name] = np.asarray(array)

    with io.BytesIO() as buffer:
        np.savez_compressed(buffer, **to_save)
        return buffer.getvalue()



def _unpack_npz_payload(payload: bytes) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as npz_file:
        meta = _from_json_bytes(npz_file[MANIFEST_KEY])
        arrays = {name: np.asarray(npz_file[name]) for name in npz_file.files if name != MANIFEST_KEY}
    return meta, arrays


# =========================
# xarray <-> TFRecord serialization
# =========================



def _array_spec(data_array: xr.DataArray, *, chunk_dim: Optional[str]) -> Dict[str, Any]:
    return {
        "dims": list(data_array.dims),
        "attrs": _encode_meta(dict(data_array.attrs)),
        "encoding": _encode_meta(dict(data_array.encoding)),
        "chunked": bool(chunk_dim and chunk_dim in data_array.dims),
    }



def _slice_for_chunk(data_array: xr.DataArray, chunk_dim: str, start: int, stop: int) -> np.ndarray:
    return np.asarray(data_array.isel({chunk_dim: slice(start, stop)}).data)



def serialize_dataset_to_tfrecord(
    ds: xr.Dataset,
    path: str,
    *,
    chunk_dim: str = "time",
    chunk_size: Optional[int] = None,
    overwrite: bool = True,
) -> None:
    """
    Serialize an xarray.Dataset into a TFRecord file.

    Layout:
      - record 0: metadata + all variables/coords that do not depend on `chunk_dim`
      - record N: one chunk per chunk slice containing all variables/coords that depend on `chunk_dim`
    """
    if os.path.exists(path):
        if overwrite:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        else:
            raise FileExistsError(f"TFRecord path already exists: {path}")

    dataset_chunk_dim: Optional[str] = chunk_dim if chunk_dim in ds.dims else None
    if dataset_chunk_dim is None:
        effective_chunk_size = None
        n_chunks = 0
    else:
        full = int(ds.sizes[dataset_chunk_dim])
        effective_chunk_size = max(1, min(int(chunk_size or full), full))
        n_chunks = (full + effective_chunk_size - 1) // effective_chunk_size

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "chunk_dim": dataset_chunk_dim,
        "chunk_size": effective_chunk_size,
        "n_chunks": n_chunks,
        "dims": {name: int(size) for name, size in ds.sizes.items()},
        "attrs": _encode_meta(dict(ds.attrs)),
        "coords": {name: _array_spec(ds.coords[name], chunk_dim=dataset_chunk_dim) for name in ds.coords},
        "data_vars": {name: _array_spec(ds[name], chunk_dim=dataset_chunk_dim) for name in ds.data_vars},
    }

    static_arrays: Dict[str, np.ndarray] = {}
    for name, spec in manifest["coords"].items():
        if not spec["chunked"]:
            static_arrays[f"coord::{name}"] = np.asarray(ds.coords[name].data)
    for name, spec in manifest["data_vars"].items():
        if not spec["chunked"]:
            static_arrays[f"data_var::{name}"] = np.asarray(ds[name].data)

    with open(path, "wb") as handle:
        metadata_record_meta = {"kind": "metadata", "manifest": manifest}
        _write_record(handle, _pack_npz_payload(metadata_record_meta, static_arrays))

        if dataset_chunk_dim is None:
            return

        for chunk_index in range(n_chunks):
            start = chunk_index * effective_chunk_size
            stop = min(start + effective_chunk_size, int(ds.sizes[dataset_chunk_dim]))
            arrays: Dict[str, np.ndarray] = {}

            for name, spec in manifest["coords"].items():
                if spec["chunked"]:
                    arrays[f"coord::{name}"] = _slice_for_chunk(ds.coords[name], dataset_chunk_dim, start, stop)
            for name, spec in manifest["data_vars"].items():
                if spec["chunked"]:
                    arrays[f"data_var::{name}"] = _slice_for_chunk(ds[name], dataset_chunk_dim, start, stop)

            chunk_meta = {
                "kind": "chunk",
                "chunk_index": chunk_index,
                "start": start,
                "stop": stop,
            }
            _write_record(handle, _pack_npz_payload(chunk_meta, arrays))



def read_tfrecord_dataset(path: str) -> xr.Dataset:
    records = list(iter_tfrecord_records(path))
    if not records:
        raise ValueError(f"TFRecord file is empty: {path}")

    metadata_meta, metadata_arrays = _unpack_npz_payload(records[0])
    if metadata_meta.get("kind") != "metadata":
        raise ValueError("First TFRecord record is not a metadata record.")

    manifest = metadata_meta["manifest"]
    chunk_dim = manifest["chunk_dim"]

    static_arrays = metadata_arrays
    chunk_payloads: List[Tuple[int, Dict[str, np.ndarray]]] = []
    for payload in records[1:]:
        meta, arrays = _unpack_npz_payload(payload)
        if meta.get("kind") != "chunk":
            raise ValueError("Encountered a non-chunk record after metadata record.")
        chunk_payloads.append((int(meta["chunk_index"]), arrays))
    chunk_payloads.sort(key=lambda item: item[0])

    def rebuild_array(prefix: str, name: str, spec: Mapping[str, Any]) -> np.ndarray:
        key = f"{prefix}::{name}"
        if not spec["chunked"]:
            return np.asarray(static_arrays[key])
        if not chunk_payloads:
            raise ValueError(f"Missing chunk records for chunked array: {name}")

        axis = list(spec["dims"]).index(chunk_dim)
        parts = [np.asarray(arrays[key]) for _, arrays in chunk_payloads]
        return np.concatenate(parts, axis=axis) if len(parts) > 1 else parts[0]

    coords = {}
    for name, spec in manifest["coords"].items():
        coords[name] = (tuple(spec["dims"]), rebuild_array("coord", name, spec))

    data_vars = {}
    for name, spec in manifest["data_vars"].items():
        data_vars[name] = (tuple(spec["dims"]), rebuild_array("data_var", name, spec))

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=_decode_meta(manifest["attrs"]))

    for name, spec in manifest["coords"].items():
        ds[name].attrs.update(_decode_meta(spec["attrs"]))
        ds[name].encoding = _decode_meta(spec["encoding"])

    for name, spec in manifest["data_vars"].items():
        ds[name].attrs.update(_decode_meta(spec["attrs"]))
        ds[name].encoding = _decode_meta(spec["encoding"])

    return ds


# =========================
# Functions mirroring zarr_processor.py
# =========================



def build_dataset_tfrecord(
    cube: np.ndarray,
    kept: Sequence[Any],
    good_indices: Sequence[int],
    cfg: Any,
    bands: Sequence[str],
    x_easting: np.ndarray,
    y_northing: np.ndarray,
    *,
    iso_to_npdt64ns_fn,
    timestamp_utc_now_fn,
) -> xr.Dataset:
    """
    Drop-in equivalent of build_dataset() from zarr_processor.py.

    The only difference is that the helper functions are injected so this module
    can stay standalone. Inside your existing codebase you can call it with
    iso_to_npdt64ns and timestamp_utc_now.
    """
    if len(good_indices) != cube.shape[0]:
        cube = cube[np.asarray(good_indices, dtype=np.int64)]
        kept = [kept[i] for i in good_indices]

    pid_len = max(len(x.task.system_index) for x in kept) if kept else 1
    pid_arr = np.asarray([x.task.system_index for x in kept], dtype=f"U{pid_len}")
    time_arr = np.asarray([iso_to_npdt64ns_fn(x.task.datetime_utc) for x in kept], dtype="datetime64[ns]")
    cov_arr = np.asarray([np.nan if x.coverage_pct is None else x.coverage_pct for x in kept], dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "reflectance": (("time", "y", "x", "band"), cube),
        },
        coords={
            "time": ("time", time_arr),
            "y": ("y", y_northing.astype(np.float64)),
            "x": ("x", x_easting.astype(np.float64)),
            "band": ("band", np.arange(len(bands), dtype=np.int16)),
            "band_name": ("band", np.asarray(bands, dtype=f"U{max(map(len, bands))}")),
            "system_index": ("time", pid_arr),
            "coverage_pct": ("time", cov_arr),
        },
        attrs={
            "mgrs_tile": cfg.tile,
            "source_collection": cfg.l1c_collection,
            "processing_profile": cfg.profile,
            "filter_strategy": cfg.filter_strategy,
            "read_strategy": cfg.read_strategy,
            "jp2_filter_read_strategy": cfg.jp2_filter_read_strategy,
            "min_valid_fraction": float(cfg.min_valid_fraction),
            "out_dim": int(cfg.out_dim),
            "quantification_value": float(cfg.quantification_value),
            "built_utc": timestamp_utc_now_fn(),
        },
    )

    ds["x"].attrs.update({"standard_name": "projection_x_coordinate", "units": "m"})
    ds["y"].attrs.update({"standard_name": "projection_y_coordinate", "units": "m"})
    ds["coverage_pct"].attrs.update({"units": "percent", "long_name": "scene valid coverage percentage used for filtering"})
    ds["time"].encoding = {
        "units": "nanoseconds since 1970-01-01 00:00:00",
        "calendar": "proleptic_gregorian",
        "dtype": "int64",
    }

    if cube.dtype == np.uint16:
        ds["reflectance"].attrs.update(
            {
                "stored_as": "uint16_scaled_reflectance",
                "scale_factor": 1.0 / float(cfg.quantification_value),
                "add_offset": 0.0,
                "note": "Recover reflectance_float = stored_uint16 * scale_factor + add_offset",
            }
        )
    else:
        ds["reflectance"].attrs.update({"stored_as": "float32_reflectance"})

    # Keep the same chunk intent as the original implementation when dask is available.
    try:
        time_chunk = min(cfg.time_chunk, max(1, cube.shape[0]))
        ds = ds.chunk({"time": time_chunk, "y": cfg.out_dim, "x": cfg.out_dim, "band": len(bands)})
    except Exception:
        pass

    return ds



def write_tfrecord(
    ds: xr.Dataset,
    local_tfrecord: str,
    cfg: Any,
    crs_wkt: str,
    dst_transform: Affine,
    filter_stats: Mapping[str, Any],
) -> None:
    """
    TFRecord equivalent of write_zarr().

    The file contains:
      - record 0: manifest + static coords / attrs / encodings
      - record N: reflectance and time-dependent coords chunked along time
    """
    ds = ds.assign_attrs(
        {
            **dict(ds.attrs),
            "crs_wkt": crs_wkt,
            "dst_transform_gdal": tuple(dst_transform.to_gdal()),
            "scene_count_before_filter": int(filter_stats["n_total"]),
            "scene_count_after_filter": int(filter_stats["n_kept"]),
            "scene_count_skipped_by_filter": int(filter_stats["n_skipped"]),
            "scene_count_after_read": int(filter_stats.get("n_read_ok", 0)),
            "scene_count_failed_read": int(filter_stats.get("n_read_failed", 0)),
            "scene_filter_keep_fraction": (
                float(filter_stats["n_kept"]) / float(filter_stats["n_total"])
                if filter_stats["n_total"] > 0 else 0.0
            ),
            "filter_stats_json": json.dumps(dict(filter_stats), sort_keys=True),
        }
    )

    serialize_dataset_to_tfrecord(
        ds,
        local_tfrecord,
        chunk_dim="time",
        chunk_size=min(cfg.time_chunk, max(1, int(ds.sizes.get("time", 1)))),
        overwrite=bool(cfg.overwrite_local),
    )



def read_tfrecord(local_tfrecord: str) -> xr.Dataset:
    """Inverse of write_tfrecord()."""
    return read_tfrecord_dataset(local_tfrecord)



def assert_dataset_roundtrip_equivalent(expected: xr.Dataset, actual: xr.Dataset) -> None:
    """Verify coords, variables, attrs, and encodings round-trip cleanly."""
    xr.testing.assert_identical(expected, actual)

    expected_names = set(expected.coords) | set(expected.data_vars)
    actual_names = set(actual.coords) | set(actual.data_vars)
    if expected_names != actual_names:
        raise AssertionError(f"Variable mismatch: expected={sorted(expected_names)} actual={sorted(actual_names)}")

    for name in sorted(expected_names):
        if dict(expected[name].encoding) != dict(actual[name].encoding):
            raise AssertionError(
                f"Encoding mismatch for {name}:\n"
                f"expected={expected[name].encoding}\n"
                f"actual={actual[name].encoding}"
            )



def verify_tfrecord_roundtrip(
    ds: xr.Dataset,
    local_tfrecord: str,
    cfg: Any,
    crs_wkt: str,
    dst_transform: Affine,
    filter_stats: Mapping[str, Any],
) -> xr.Dataset:
    """
    Convenience helper: write -> read -> assert equivalence.

    Returns the reconstructed dataset.
    """
    write_tfrecord(ds, local_tfrecord, cfg, crs_wkt, dst_transform, filter_stats)
    roundtrip = read_tfrecord(local_tfrecord)
    expected = ds.assign_attrs(
        {
            **dict(ds.attrs),
            "crs_wkt": crs_wkt,
            "dst_transform_gdal": tuple(dst_transform.to_gdal()),
            "filter_stats_json": json.dumps(dict(filter_stats), sort_keys=True),
        }
    )
    assert_dataset_roundtrip_equivalent(expected, roundtrip)
    return roundtrip

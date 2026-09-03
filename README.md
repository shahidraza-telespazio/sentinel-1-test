# Sentinel-1 GRD Processing

Standalone Sentinel-1 GRD processing implementation for AI4QC2.

This directory is intentionally independent from the existing Sentinel-2 implementation under `NEW_SETUP/`. The Sentinel-1 processor has its own processing pipeline, SQS enqueue script, worker, output naming, and runtime configuration.

## Overview

The processor discovers Sentinel-1 IW GRD acquisitions intersecting a Sentinel-2 MGRS tile, applies radiometric processing, resamples calibrated linear power to the target MGRS grid, converts the result to dB, builds a multi-temporal dataset, and writes the final product as a TFRecord.

The current processing flow is:

```text
Sentinel-1 IW GRD
        |
        v
Discover acquisitions intersecting MGRS tile
        |
        v
Apply geometric coverage threshold
        |
        v
Read detected GRD measurement data
        |
        v
Convert detected magnitude to power
        |
        v
Interpolate calibration LUT
        |
        v
Interpolate thermal-noise LUT
        |
        v
Subtract thermal noise
        |
        v
Apply radiometric calibration in linear domain
        |
        v
Average reproject calibrated linear power
onto the Sentinel-2 MGRS target grid
        |
        v
Convert calibrated power to dB
        |
        v
Apply valid-area acceptance rules
        |
        v
Build multi-date xarray dataset
        |
        v
Write TFRecord
        |
        v
Read TFRecord back and verify exact round trip
```

## Directory structure

```text
Sentinel1_dev/
├── sentinel1_tfrecord_processor.py
├── s1_enqueue_tiles_to_sqs.py
├── s1_worker_sqs_runner.py
├── s1_worker_utils.py
├── tfrecord_xarray_io.py
└── s1_tfrecord_processor/
```

### Main components

`sentinel1_tfrecord_processor.py`

Main command-line entry point for end-to-end Sentinel-1 processing.

`s1_tfrecord_processor/`

Contains Sentinel-1 discovery, coverage analysis, radiometry, reprojection, time-series construction, validation, and TFRecord-writing modules.

`s1_enqueue_tiles_to_sqs.py`

Creates Sentinel-1 processing jobs from the existing MGRS tile inventory and sends them to an SQS queue.

`s1_worker_sqs_runner.py`

Dedicated Sentinel-1 SQS worker. It runs the standalone Sentinel-1 processor, uploads completed TFRecords to S3, manages SQS visibility, and handles EC2 Spot interruption and rebalance events.

`s1_worker_utils.py`

Small Sentinel-1-specific worker utilities. This avoids introducing dependencies on the existing Sentinel-2 worker implementation.

`tfrecord_xarray_io.py`

TFRecord serialization and deserialization utilities used for writing and validating the final xarray dataset.

## Example local processing run

From inside `Sentinel1_dev`:

```bash
python sentinel1_tfrecord_processor.py \
  --tile T30UXC \
  --start-date 2018-03-15 \
  --end-date 2018-03-16 \
  --minimum-geometric-coverage 0.80 \
  --minimum-valid-coverage 0.80 \
  --acceptance-rule all \
  --band-layout union \
  --calibration-lut sigmaNought \
  --unknown-noise-policy error \
  --out-dim 128 \
  --rows-per-window 256 \
  --num-threads 2 \
  --chunk-size 64 \
  --output-root /tmp/s1-output \
  --work-root /tmp/s1-work
```

A successful run produces a TFRecord similar to:

```text
/tmp/s1-output/T30UXC/
s1_grd_tile_T30UXC_2018-03-15_to_2018-03-16_sigmaNought_db.tfrecord
```

## Output dataset

The TFRecord contains an xarray-compatible dataset with calibrated Sentinel-1 backscatter.

Primary variable:

```text
backscatter_db(time, y, x, band)
```

Additional variables include:

```text
valid_mask
valid_area_fraction
band_present
band_pass
```

Metadata includes acquisition information such as:

```text
time
platform
orbit_state
relative_orbit
absolute_orbit
band
calibration method
target CRS
target transform
processing provenance
```

The processor retains all available supported polarisations rather than requiring a fixed VV/VH pair.

## Radiometric processing

The current implementation operates in the linear-power domain until spatial resampling has completed.

Conceptually:

```text
source_power = detected_magnitude²

denoised_power =
    source_power - thermal_noise

calibrated_linear =
    denoised_power / calibration_LUT²

backscatter_db =
    10 * log10(calibrated_linear)
```

Values that are invalid, non-finite, or non-positive after thermal-noise subtraction are represented as no-data.

For modern Sentinel-1 products, combined thermal noise is derived from range and azimuth noise LUTs where available.

For older products, the legacy noise-vector representation is supported.

Importantly, calibrated linear power is spatially averaged before conversion to dB.

The processor does not average dB values directly.

## Target grid

Sentinel-1 acquisitions are processed against the existing Sentinel-2 MGRS grid used by AI4QC2.

Sentinel-2 data is used only to establish the exact spatial target grid for each MGRS tile.

This does not create a runtime dependency on the existing Sentinel-2 processor under `NEW_SETUP/`.

## Coverage filtering

Two different coverage concepts are supported.

### Geometric coverage

Controls whether a Sentinel-1 acquisition covers enough of the target MGRS tile to be considered.

Default:

```text
0.80
```

### Valid calibrated coverage

Controls whether enough valid calibrated data remains after radiometric processing and reprojection.

Default:

```text
0.80
```

Acceptance can be configured using:

```text
--acceptance-rule any
--acceptance-rule all
```

## Band layout

Supported layouts:

```text
union
canonical
```

`union` builds the output band axis from the polarisations available across accepted acquisitions.

This allows acquisitions with different available Sentinel-1 polarisation combinations to participate in the same processing workflow.

## Calibration configuration

Supported calibration LUTs currently include:

```text
sigmaNought
betaNought
gamma
```

`sigmaNought` is currently the development default.

The final calibration quantity should remain configurable until the project confirms the required scientific output definition.

## Unknown noise-correction policy

Supported policies:

```text
error
assume_uncorrected
assume_corrected
```

The default is:

```text
error
```

This deliberately avoids silently making assumptions when Sentinel-1 product metadata does not provide enough information to determine the thermal-noise state.

## SQS processing

Sentinel-1 uses its own enqueue script and worker rather than sharing the Sentinel-2 worker implementation.

Typical job metadata includes:

```json
{
  "tile": "T30UXC",
  "processor_family": "S1",
  "mode": "S1_GRD_DB",
  "product_level": "GRD",
  "output_format": "tfrecord",
  "start_date": "2018-03-15",
  "end_date": "2018-03-16",
  "calibration_lut": "sigmaNought"
}
```

The final S3 key follows this structure:

```text
<output-prefix>/
└── T30UXC/
    └── 2018-03-15_to_2018-03-16/
        └── S1_GRD_DB/
            └── s1_grd_tile_T30UXC_2018-03-15_to_2018-03-16_sigmaNought_db.tfrecord
```

## Spot interruption handling

The Sentinel-1 worker is designed to support EC2 Spot workers.

It monitors EC2 instance metadata for:

```text
Spot instance-action notifications
EC2 rebalance recommendations
```

When an interruption signal is detected, the worker:

```text
stops accepting new work
        |
        v
terminates the active processor process group
        |
        v
stops extending SQS message visibility
        |
        v
records the job as aborted
        |
        v
releases the SQS message
        |
        v
allows another worker to retry the tile
```

The processor is started in its own process session so that the complete processing subprocess tree can be terminated safely.

## Output completion

For Sentinel-1 TFRecord processing, the final S3 TFRecord object is the completion authority.

If the expected output object already exists and skip-existing behaviour is enabled, the worker does not process that tile again.

## Validation completed

The current implementation has been validated with real Sentinel-1 GRD data for tile `T30UXC`.

A test acquisition produced:

```text
Bands:           VV, VH
Output shape:    (1, 128, 128, 2)
Platform:        sentinel-1a
Orbit state:     ascending
Relative orbit:  132
Absolute orbit:  21029
Calibration:     sigmaNought
Units:           dB
```

The generated TFRecord was read back and compared with the source xarray dataset.

Exact round-trip validation passed.

The dedicated Sentinel-1 worker has also been tested through:

```text
worker
→ real Sentinel-1 processor
→ TFRecord generation
→ round-trip verification
→ worker upload path
```

using a local fake-S3 destination.

## Current scientific considerations

The engineering pipeline is operational, but several scientific/project decisions remain deliberately configurable rather than being hard-coded.

These include:

```text
final calibration quantity
terrain-correction / DEM requirements
per-acquisition acceptance semantics
canonical versus union polarisation layout
duplicate/reprocessed product selection
final policy for ambiguous thermal-noise metadata
```

Independent comparison against a trusted reference implementation such as SNAP is recommended before declaring the generated backscatter scientifically production-ready.

## Relationship to Sentinel-2

This implementation intentionally does not modify the existing Sentinel-2 processor.

Repository structure:

```text
AI4QC2-Prod/
├── NEW_SETUP/
│   └── existing Sentinel-2 processing
│
└── Sentinel1_dev/
    └── standalone Sentinel-1 processing
```

This separation allows Sentinel-1 infrastructure, queues, workers, processing parameters, and future deployment changes to evolve independently without increasing risk to the existing Sentinel-2 production workflow.

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Cosmos Multiview: Episode and Sample Candidates

This profile is registered as `cosmos_multiview` using the
`CosmosMultiviewWriter` class in `ovannotators.profile_writers`. Install the
`[cosmos-multiview]` extra for the pinned Lance and PyArrow dependencies.
Creating aligned samples in NumPy array format (NPY) supports CPU processing.

## Purpose and Inputs

Use to register independently timed streams as episodes, then materialize
aligned sample windows for Cosmos 3.5 candidate datasets. Supported task strings
are `t2v`, `i2v`, `transfer`, and `action`. This profile writes candidate resources;
it does not run a Cosmos model or certify canonical ingestion.

Provide explicit dataset/episode identities, authoritative integer nanosecond
timestamps, stream descriptors, named tensors, provenance, ownership, and license
URI. Descriptors declare dtype, shape, units, geometry, and source clock.
Tensor dtype/shape must match. Depth geometry and its measurement convention
must be declared; calibration/pose data comes from producer metadata.

The writer raises `ValueError` for invalid inputs. A release holds one
`dataset_id`, and each episode ID must be unique within it. Each episode must
declare every selected stream and the `reference_stream_id`, and selected
streams must use the episode's source clock. Within an episode, sample
`timestamp_ns` values must be strictly increasing. Each sample must list every
absent selected stream in `missing_data`, and `missing_data` may not name a
stream the sample supplies; with `missing_data_policy="fail"`, any absent
selected stream is rejected.
`end_episode()` also raises `ValueError` when a stream whose descriptor has
`required=True` (the default) has no sample in the episode; listing it in
`missing_data` for every sample does not satisfy this, and an episode needs at
least one sample.

## Output and Data Types

| Resource | Schema summary |
| --- | --- |
| `episodes.lance` | String episode/dataset/source IDs, list of stream types, list of stream structs, dataset-info URI |
| Episode stream struct | String IDs/type/data/calibration/pose/timestamp URIs and derivation ID; float64 frame rate and start/end offsets in seconds |
| `samples.lance` | String sample/episode/task IDs, float64 source window offsets (`source_start_offset_s`, `source_end_offset_s`, in seconds), string `assets_json`, caption struct, sample-config URI, creator |
| Caption struct | Text and language strings plus per-stream string map |
| Source stream resources | NPY arrays, Parquet timestamps, stream JSON, and calibration JSON where provided |
| `objects/` | Content-addressed aligned NPY tensors and Parquet timestamp sidecars |
| `metadata/` | `dataset_info.json` and `sample_config.json` |
| `validation/` | Episode and sample candidate validation JSON |
| `_ovannotators/` | Profile, writer, losses, conformance, checksums, and final `_SUCCESS` |

Materialized tensor shape is `(target_samples, *stream_shape)` with dtype recorded
in the asset JSON. Timing sidecars retain target, native, source, and episode
nanoseconds. `assets_json` records asset URIs, hashes, shape/dtype, and alignment
policy. URI schemes must be explicitly allowed, and output URIs resolve inside
the configured candidate root.

Materialization specifies duration/stride in nanoseconds, reference stream,
target rate, selected streams, encoding, and alignment per stream type.
The current implementation supports `npy` encoding. Alignment policies include
`nearest_timestamp` and `linear_interpolation` (exact matches are used first).
The quality assurance (QA) policy must be owner-supplied and versioned.
Full target-grid stream coverage is required. Missing data either drops a window or fails according to policy.

`register_episodes()` and `materialize_samples()` expose the two stages;
`finalize()` runs both and writes completion evidence. Completed immutable
objects can be reused; unfinished candidates are rejected. Structural validation
checks resources and checksums. Semantic, canonical-loader, and consumer
acceptance remain `not_run` unless validated separately.

For clip-based media conditioning, use [Cosmos Transfer](cosmos_transfer.md).
The tables and candidate semantics here describe this implementation.

## Configuration

| Field | Type | Default |
| --- | --- | --- |
| `output_base_uri` | `str` | `Required` |
| `source_version` | `str` | `Required` |
| `domain` | `str` | `Required` |
| `owner` | `str` | `Required` |
| `license_uri` | `str` | `Required` |
| `default_reference_frame` | `str` | `Required` |
| `supported_uri_schemes` | `tuple[str, ...]` | `Required` |
| `materialization` | `CosmosMaterializationConfig` | `Required` |
| `qa` | `CosmosQAConfig` | `Required` |
| `schema_version` | `str` | `'1.0'` |
| `dataset_info_path` | `str` | `'metadata/dataset_info.json'` |
| `sample_config_path` | `str` | `'metadata/sample_config.json'` |

## Usage

Construct `CosmosMaterializationConfig` and `CosmosQAConfig` using the policy
fields below, pass them as `materialization` and `qa` to
`CosmosMultiviewWriterConfig`, and create `CosmosMultiviewWriter(config)`.
For example, an RGB stream can use `alignment_by_stream_type={"rgb":
"nearest_timestamp"}` and `output_encoding_by_stream_type={"rgb": "npy"}`.
Set `selected_stream_ids` to the episode's stream IDs and use one of them as
`reference_stream_id`. Choose timing and QA tolerances from the source cadence
and owner policy; no universal values are assumed. Then use the
{ref}`episode workflow <writer-episode-workflow>`. Pass `EpisodeEnvelope`
and `SampleEnvelope` objects; these writers do not consume the frame writer's
`render_vars` dictionary.

See the [shared writer workflow](index.md) for input conventions and lifecycle.

### Materialization Policy

| Field | Type | Default |
| --- | --- | --- |
| `version` | `str` | `Required` |
| `duration_ns` | `int` | `Required` |
| `stride_ns` | `int` | `Required` |
| `reference_stream_id` | `str` | `Required` |
| `target_rate_hz` | `float` | `Required` |
| `selected_stream_ids` | `tuple[str, ...]` | `Required` |
| `alignment_by_stream_type` | `Mapping[str, str]` | `Required` |
| `output_encoding_by_stream_type` | `Mapping[str, str]` | `Required` |
| `tasks` | `tuple[str, ...]` | `Required` |
| `created_by` | `str` | `Required` |
| `object_key_policy` | `Literal['content_addressed']` | `'content_addressed'` |

### QA Policy

| Field | Type | Default |
| --- | --- | --- |
| `profile_id` | `str` | `Required` |
| `version` | `str` | `Required` |
| `max_alignment_error_ns` | `int` | `Required` |
| `geometry_reprojection_tolerance_px` | `float` | `Required` |
| `minimum_stream_coverage` | `float` | `Required` |
| `missing_data_policy` | `Literal['drop_sample', 'fail']` | `Required` |
| `warning_policy` | `Literal['forbid', 'owner_approval_required']` | `Required` |

## References

[Cosmos documentation](https://docs.nvidia.com/cosmos/latest/index.html) provides
model and workflow context. The two-table candidate schema above is the contract
of this ovannotators profile; no public Cosmos 3.5 ingestion specification is
linked here. Do not substitute the Transfer 2.5 multiview media layout for it.

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Writer Formats

Select a writer based on your task and dataset requirements. Each page details
the output format, configuration options, usage examples, and limitations for a
specific writer. Links to upstream format specifications are provided for
reference; not every upstream feature is implemented here.

| Writer | Use for | Additional runtime requirements |
| --- | --- | --- |
| [basic](basic.md) | Raw arrays, annotations, previews | None; `[video]` and NVENC for optional video |
| [coco](coco.md) | Detection, instance/panoptic segmentation | None |
| [kitti](kitti.md) | Camera 2D/3D detection | None |
| [yolo](yolo.md) | Detection and instance polygons | None |
| [openlabel](openlabel.md) | ASAM OpenLABEL image boxes and instance polygons | None |
| [bop](bop.md) | Object pose datasets | None |
| [lerobot](lerobot.md) | Simple frame-based episode video | `[lerobot,video]`, NVENC |
| [cosmos_transfer](cosmos_transfer.md) | RGB and visual conditioning clips | `[video]`, NVENC |
| [lerobot_v3](lerobot_v3.md) | Coming soon: timed robot state/action and cameras | Not available in 0.2.0 |
| [cosmos_multiview](cosmos_multiview.md) | Episode registration and aligned candidate samples | `[cosmos-multiview]` |

The `groot_lerobot_v2` and `umi_replay_buffer` episode profiles are described in
the [API reference](../api.rst), not on a page here.

## Frame Writer Workflow

Frame writers require calls to `write_frame(frame_id, render_vars, backend)`
and `finalize(backend)`. Finalization is mandatory because some writers defer
publishing aggregate metadata or buffered videos until this step. All backend
output paths are relative to the configured root directory. Use a distinct
destination for each writer to prevent file overwrites.

This complete CPU example writes an image and a depth array:

```python
import numpy as np
from ovannotators import BasicWriter, BasicWriterConfig, DiskBackend, DiskBackendConfig

backend = DiskBackend(DiskBackendConfig(root_dir="_out/writer_guide", overwrite=True))
writer = BasicWriter(BasicWriterConfig(depth_format="npy", colorize=False))
render_vars = {
    "LdrColor": np.zeros((48, 64, 4), dtype=np.uint8),
    "DistanceToCameraSD": np.full((48, 64), 2.0, dtype=np.float32),
}
writer.write_frame(0, render_vars, backend)
writer.finalize(backend)
print(backend.list())
```

Inspect `0000/ldr_color.png` and `0000/distance_to_camera.npy`. To switch writers,
use the construction shown on its page and supply that writer's required inputs.
`WriterRegistry.list_names()` lists registered frame writers, including installed
plugins; `WriterRegistry.create(name, config)` creates a new registered frame writer.

## Input Conventions

Registered writers support the following input conventions:

- Typical images are uint8 `(H,W,3)` RGB or `(H,W,4)` RGBA.
- Depth is float32 `(H,W)` (or a supported singleton channel).
- Segmentation is uint32 instance/class IDs, with labels mapped using semantic
  ID maps.
- Boxes and camera parameters use decoded structured arrays rather than
  arbitrary float matrices. See {ref}`API types <api-types>` for decoded
  box/camera objects.
- For payload structure details, see the
  [renderer output reference](../render_outputs.rst).
- NumPy and supported Warp arrays can be provided directly without a renderer.

Units of length and coordinate systems are defined by the producer. Writers do
not universally convert stage units, so supply the conventions documented for
your chosen writer. Instance IDs and class IDs serve distinct purposes; retain
their mappings when interpreting labels and masks.

`list_profiles(writer_name)` and `describe_requirements(writer_name, ...)`
report task-specific capture requirements and caller-supplied decisions. See
{ref}`requirements discovery <api-dataset-requirements>`.

(writer-episode-workflow)=
## Episode Writer Workflow

The profile writers use `ProfileWriterRegistry` and explicit envelopes.
`EpisodeEnvelope` declares dataset/sequence/episode identity, task, embodiment,
source clock/origin, streams, and provenance. `StreamDescriptor` supplies each
stream's dtype, shape, units, nominal rate, and geometry. `SampleEnvelope` supplies
source `timestamp_ns`, frame/sample identity, observations, policy signals, and
missing-data declarations. Wrap each payload in `NamedTensor` with its semantics.

After constructing `writer` using the selected profile's config and a `backend`,
and preparing a matching `episode` envelope and its `samples`, the lifecycle is:

```python
writer.begin_episode(episode, backend)
for sample in samples:
    writer.write_sample(sample, backend)
writer.end_episode(backend)
writer.finalize(backend)
result = writer.validate(backend)
if not result.success_marker_allowed:
    raise RuntimeError(result.messages)
```

Repeat begin/write/end for each episode before finalization. Profile artifacts
separate serialization, structure, semantics, canonical loading, and consumer
acceptance. `_SUCCESS` requires serialization and structure to pass; inspect
`conformance.json` for the remaining status fields.

```{toctree}
:maxdepth: 1

basic
coco
kitti
yolo
openlabel
bop
lerobot
cosmos_transfer
lerobot_v3
cosmos_multiview
```

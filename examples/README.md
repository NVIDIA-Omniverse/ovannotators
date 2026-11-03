<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Examples

These examples are written in Python. Run commands from the repository root so
relative scene and output paths resolve correctly.

## CPU quick start (no GPU)

[`quickstart_cpu.py`](quickstart_cpu.py) builds three frames of color, depth,
and normals with NumPy and writes them through `BasicWriter`. It needs no
renderer or GPU.

```bash
uv sync --extra dev-cpu
uv run python examples/quickstart_cpu.py
```

Expected result: the command prints `Wrote 15 files to _out/quickstart_cpu`.
Each frame directory holds `ldr_color.png`, `distance_to_camera.exr`,
`normal.exr`, and colorized PNG previews of depth and normals.

## Minimal renderer capture (NVIDIA GPU)

[`minimal_capture.py`](minimal_capture.py) renders three
[OpenUSD](https://openusd.org/release/index.html) frames with
[`ovrtx`](https://github.com/NVIDIA-Omniverse/ovrtx) through an
application-owned `ovstage` stage, and writes them through `BasicWriter` to
disk.

Prerequisites:

- Python 3.11 or newer
- An NVIDIA RTX-capable GPU, [CUDA 12](https://docs.nvidia.com/cuda/archive/12.9.2/),
  and a compatible driver
- The `dev` extra, which installs `ovrtx` and `ovstage`. The `ovrtx` renderer
  wheels are published on NVIDIA's public package index,
  `https://pypi.nvidia.com/`.

Install and run:

```bash
uv sync --extra dev --extra-index-url https://pypi.nvidia.com/ --index-strategy unsafe-best-match
uv run python examples/minimal_capture.py
```

Expected result: the command prints `Wrote N files to
_out/basic_minimal`. The output directory contains data for three rendered
frames. Inspect the generated images and metadata to complete the runtime
check; a passing unit-test suite does not prove that the local renderer and GPU
stack can produce a frame.

## Advanced: requirements-driven renderer capture

[`requirements_capture.py`](requirements_capture.py) starts from a writer
name instead of a scene that already requests every output. It calls
`list_profiles` and `describe_requirements`, then authors a USD layer over the
bundled scene. That layer contains one RenderProduct with a RenderVar for each
required and recommended output. It deactivates the scene's all-outputs
product, renders the layer with `ovrtx` through an attached `ovstage` stage, and
writes the dataset with the writer built from `requirements.config`.

Prerequisites: an NVIDIA RTX-capable GPU, CUDA 12, and `ovrtx` and `ovstage`
from the NVIDIA package index.

Install and run:

```bash
uv sync --extra dev --extra-index-url https://pypi.nvidia.com/ --index-strategy unsafe-best-match
uv pip install "ovrtx>=0.5,<0.6" "ovstage>=0.2,<0.3"
uv run python examples/requirements_capture.py --writer coco --profile instance_segmentation
```

Expected result: the command prints the COCO profiles, the selected profile's
required and recommended render vars (`LdrColor`,
`SemanticBoundingBox2DTight`, `SemanticInstanceSegmentation`, and
`SemanticIdMap`), and the scene prerequisite for semantic labels. It ends with
`Wrote N files to _out/requirements_capture/coco-instance_segmentation`. Open
`annotations/instances_train.json` and confirm that each annotation has a
`segmentation` mask; open one image under `images/`.

Pass another frame writer and profile, such as `--writer kitti` or
`--writer bop`, to render only that profile's inputs. Profiles that need caller
configuration, non-render inputs such as robot state, or more than one camera
stop with a message naming what is missing.

## WebDataset capture without a renderer

[`webdataset_capture.py`](webdataset_capture.py) synthesizes small
renderer-style frames with NumPy, captures two episodes through `COCOWriter`
into episode-preserving [WebDataset](https://github.com/webdataset/webdataset)
tar shards, then reads the dataset back. It needs no renderer or GPU.

Install and run:

```bash
uv sync --extra dev-cpu
uv run python examples/webdataset_capture.py
```

Expected result: the command prints one line per packaged sample and ends with
`Packaged 5 samples from 2 episodes into 2 shards under
_out/webdataset_capture/example-run`. When the optional `webdataset` reader is
installed (the `dev-cpu` extra includes it), it also prints
`webdataset reader decoded 5 samples`.

Inspect the output:

- `_out/webdataset_capture/example-run/manifest.json` lists both shards with
  their SHA-256 checksums and per-episode sample counts. It is written last.
- `tar tf _out/webdataset_capture/example-run/shard-000000.tar` shows the
  adjacent `sample-<n>.png`, `sample-<n>.json` and `sample-<n>.metadata.json`
  members of `episode001`; `shard-000001.tar` holds `episode002`.
- Extract one shard and open a `.png` image and its `.json` COCO-style record.
- `_out/webdataset_capture/example-run/auxiliary/source-paths.json` maps each
  sample to the native `COCOWriter` file it came from.

Each run deletes and recreates `_out/webdataset_capture`.

## More tasks

- [Tutorials](../docs/public/tutorials.rst) include MS-COCO, multiple
  simultaneous writers, S3, LeRobot, Cosmos Transfer, custom writers, and
  async capture.
- [CPU-only NumPy workflow](../docs/public/tutorials.rst#tutorial-8-without-ovrtx-numpy-arrays-directly)
  shows how to use a writer and disk backend without `ovrtx` or a GPU.
- [Python API reference](../docs/public/api.rst) documents every public class
  used by the examples.
- [WebDataset packaging](../docs/public/tutorials.rst#webdataset-packaging)
  compares managed capture, packaging existing files and
  `export_finalized_dataset` for finalized writer output, lists the built-in
  writer adapters and shows how to read the shards.

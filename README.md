# ovannotators

`ovannotators` is a standalone, receive-only Python library that turns
self-describing renderer output into ML-ready datasets outside the renderer
process. It writes:

- **Perception datasets:** [MS-COCO](https://cocodataset.org/#format-data),
  [KITTI](https://www.cvlibs.net/datasets/kitti/eval_object.php),
  [Ultralytics YOLO](https://docs.ultralytics.com/datasets/detect/),
  [ASAM OpenLABEL](https://www.asam.net/standards/detail/openlabel/),
  [BOP](https://bop.felk.cvut.cz/),
  [DOPE](https://github.com/NVlabs/Deep_Object_Pose),
  [CenterPose](https://github.com/NVlabs/CenterPose), and
  [YCB-Video](https://rse-lab.cs.washington.edu/projects/posecnn/).
- **Robot-learning episodes:**
  [Hugging Face LeRobot](https://huggingface.co/docs/lerobot/index) (v3, its
  LanceDB layout, and the GR00T v2.1 layout), the UMI/Diffusion Policy Zarr
  ReplayBuffer, and
  [NVIDIA Cosmos Transfer](https://docs.nvidia.com/cosmos/latest/transfer2.5/index.html)
  and Cosmos multiview.
- **Raw per-frame output** in a flat layout.

Output goes to disk, [Amazon S3](https://aws.amazon.com/s3/), or
[ovstorage](https://github.com/NVIDIA-Omniverse/ovstorage) through pluggable
backends.

> Pre-release (0.3.0, Python 3.11+). APIs may change before 1.0.

## Start Here

- [Install ovannotators](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/INSTALL.md)
- [Choose and run an example](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/examples/README.md)
- [Browse the documentation](https://nvidia-omniverse.github.io/ovannotators/)
- [Read the Python API reference](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/docs/public/api.rst)
- [Use an agent skill](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/skills/README.md)
- [Get support or report an issue](#support)

## Overview

<!-- DOCS:description:start -->
A standalone library that turns self-describing renderer and simulation output
into ML-ready dataset annotations.

It converts renderer output (RGB, depth, segmentation, bounding boxes, normals,
camera params) into standardized dataset formats and runs outside the renderer
process.
<!-- DOCS:description:end -->

Its Python API reads each frame's render variables as
[NumPy](https://numpy.org/) or [Warp](https://nvidia.github.io/warp/) arrays and
writes datasets through a selected writer and storage backend. `ovannotators`
does not own scene creation or rendering; it consumes frames produced by a
renderer such as [`ovrtx`](https://github.com/NVIDIA-Omniverse/ovrtx).

<!-- DOCS:capabilities:start -->
- **Write standard dataset formats.** Select any format above by its registry
  name, or use `BasicWriter` for flat per-frame output. The
  [API reference](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/docs/public/api.rst)
  lists every registry name and the extra it needs.
- **Package training samples into WebDataset shards.** Capture episodes
  through a writer adapter (built in for Basic and COCO, extensible with your
  own), export finalized output through the same adapter, or package existing
  encoded files into episode-preserving tar shards on any storage backend
  with a checksummed completion manifest.
  See the [WebDataset tutorial](https://nvidia-omniverse.github.io/ovannotators/tutorials.html#webdataset-packaging).
- **Register custom formats and destinations.** Extend the writer and backend
  registries directly or follow the shipped agent skills.
- **Install only the integrations you need.** Storage, robot-learning, GPU
  image-codec, and video-codec dependencies are optional extras.
<!-- DOCS:capabilities:end -->

## Requirements

- Python 3.11 or newer
- Linux (x86_64 or aarch64) or Windows (x86_64)
- [NumPy](https://numpy.org/), [Pillow](https://python-pillow.github.io/),
  [OpenEXR](https://openexr.com/), and
  [Warp](https://nvidia.github.io/warp/) for the core CPU-capable package
- An NVIDIA GPU, compatible driver, and
  [CUDA 12](https://developer.nvidia.com/cuda-toolkit) for the `gpu` and `video`
  extras and for renderer-backed examples

See [Installation](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/INSTALL.md) for the full extras matrix and platform notes.

<!-- DOCS:extras:start -->
| Extra | What it adds | Use when you need |
| --- | --- | --- |
| `s3` | S3 backend support ([boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)) | Writing datasets directly to S3 |
| `ovstorage` | [ovstorage](https://github.com/NVIDIA-Omniverse/ovstorage) Stack adapter | Writing through ovstorage cloud, local, or cached routes |
| `lerobot` | LeRobot Parquet support ([PyArrow](https://arrow.apache.org/docs/python/)) | Writing `lerobot`, `groot_lerobot_v2`, or native `lerobot_v3` datasets (each also needs `video` and an NVIDIA GPU for NVENC video encoding) |
| `cosmos-multiview` | Lance and Parquet tables (`pylance==11.0.0`, `pyarrow==25.0.1`) | Writing `cosmos_multiview` or native `lerobot_lance` datasets (`lerobot_lance` also needs `video` and an NVIDIA GPU for NVENC) |
| `umi` | Zarr v2 and JPEG-XL codecs ([Zarr](https://zarr.readthedocs.io/), `zarr<3`) | Writing the `umi_replay_buffer` profile |
| `gpu` | GPU image codec support ([nvImageCodec](https://docs.nvidia.com/cuda/nvimagecodec/), CUDA 12) | GPU-accelerated image encode/decode |
| `video` | Video codec support ([PyNvVideoCodec](https://docs.nvidia.com/video-technologies/pynvvideocodec/index.html), [NVENC](https://developer.nvidia.com/video-codec-sdk)) | Video-emitting writers on an NVIDIA GPU (H.264, HEVC, AV1) |
| `all-cpu` | `s3`, `ovstorage`, `lerobot`, `cosmos-multiview`, and `umi` | Every CPU-only runtime feature |
| `all-gpu` | `all-cpu` plus `gpu` and `video` | Every CPU and GPU runtime feature |
| `all` | Alias for `all-gpu` plus test tooling | A complete GPU development environment |
<!-- DOCS:extras:end -->

## Discover dataset requirements

Call `ovannotators.list_profiles("coco")` to list supported tasks, or
`ovannotators.describe_requirements("coco")` for the recommended detection
configuration and required RenderVars. Discovery covers frame and episode
writers and reports caller-supplied choices such as robot signals and stage
units. See [Dataset profiles and capture requirements](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/docs/public/api.rst#dataset-profiles-and-capture-requirements).
On a GPU host, [`examples/requirements_capture.py`](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/examples/README.md#advanced-requirements-driven-renderer-capture)
turns a profile's requirements into a RenderProduct and renders it with `ovrtx`.

## Getting Started

Run these commands from the repository root. Start without a GPU; add the
renderer when you have an NVIDIA RTX GPU.

### Without a GPU (CPU only)

Writers accept NumPy arrays from any producer, so no renderer is needed:

```bash
uv sync --extra dev-cpu
uv run python examples/quickstart_cpu.py
```

The command prints `Wrote 15 files to _out/quickstart_cpu`. Each of the three
frame directories holds `ldr_color.png`, `distance_to_camera.exr`,
`normal.exr`, and colorized PNG previews of depth and normals.

### With an NVIDIA GPU (ovrtx and ovstage)

This path renders the bundled USD scene with `ovrtx`. It needs an NVIDIA
RTX-capable GPU, CUDA 12, and a compatible driver.

```bash
uv sync --extra dev --extra-index-url https://pypi.nvidia.com/ --index-strategy unsafe-best-match
uv run python examples/minimal_capture.py
```

The `dev` extra installs `ovrtx` and `ovstage`. The `ovrtx` renderer wheels are
published on NVIDIA's public package index, `https://pypi.nvidia.com/`; the
`ovrtx` entry on PyPI only points there.

The command prints `Wrote N files to _out/basic_minimal`, and that directory
contains three frame groups written by `BasicWriter`. Open one `ldr_color.png`
to confirm that the GPU stack rendered the scene.

## Documentation

| If you want to… | Start here |
| --- | --- |
| Install the package or choose extras | [Installation](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/INSTALL.md) |
| Run an end-to-end or CPU-only workflow | [Examples](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/examples/README.md) and [tutorials](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/docs/public/tutorials.rst) |
| Choose a dataset format or interpret writer output | [Writer formats](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/docs/public/writers/index.md) |
| Select a backend or renderer-output extractor | [Python API reference](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/docs/public/api.rst) |
| Map renderer output to writer inputs | [Renderer output reference](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/docs/public/render_outputs.rst) |
| Diagnose setup or runtime problems | [Troubleshooting](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/docs/public/troubleshooting.rst) |
| Browse the full documentation site | [Documentation map](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/docs/README.md) |

## Examples

The [example index](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/examples/README.md) lists prerequisites, runnable commands,
and expected output. Start with
[`examples/quickstart_cpu.py`](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/examples/quickstart_cpu.py) without a GPU, or
[`examples/minimal_capture.py`](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/examples/minimal_capture.py) for a complete
renderer-to-dataset flow on an NVIDIA GPU.

### Minimal Example (NVIDIA GPU)

```python
from contextlib import ExitStack

import ovrtx
import ovstage
from ovannotators.backends import DiskBackend, DiskBackendConfig
from ovannotators.renderer_outputs import OvrtxOutput
from ovannotators.writers import BasicWriter, BasicWriterConfig

scene_path = "tests/data/minimal_capture.usda"
out_dir = "_out/basic_minimal"
backend = DiskBackend(DiskBackendConfig(root_dir=out_dir, overwrite=True))
writer = BasicWriter(BasicWriterConfig(image_format="png"))
extractor = OvrtxOutput()
render_product = "/Render/RenderProduct_001"

# Register schemas before any USD access; the Renderer must exist before the
# Stage it borrows. ExitStack unwinds in reverse order (detach, stage, renderer)
# and still runs the remaining steps if one of them raises.
ovrtx.register_schema_paths()
renderer = ovrtx.Renderer()
with ExitStack() as cleanup:
    cleanup.callback(renderer.destroy)
    stage = ovstage.Stage("minimal-capture")
    cleanup.callback(stage.destroy)
    renderer.attach_ovstage(stage)
    cleanup.callback(renderer.detach_ovstage)

    outputs = frame = None
    try:
        ordinal = 1
        ovstage.population.open_usd(stage, scene_path, ordinal, 0.0)
        stage.advance_write_floor(ordinal).wait()

        frame_id = 0
        num_frames = 3
        # USD time selects scene samples; a positive delta_time advances the sensor
        # simulation between frames.
        start_tc, end_tc = 0.0, 5.0
        for i in range(num_frames):
            usd_time = start_tc + (end_tc - start_tc) * i / (num_frames - 1)
            ordinal += 1
            ovstage.population.update_from_usd_time(stage, ordinal, usd_time)
            stage.advance_write_floor(ordinal).wait()
            outputs = renderer.step({render_product}, delta_time=0.1, ordinal=ordinal)
            for frame in outputs[render_product].frames:
                writer.write_frame(frame_id, extractor.extract_all(frame), backend)
                frame_id += 1

        outputs = frame = None
        writer.finalize(backend)
    finally:
        # Release frame references before the renderer is torn down.
        outputs = frame = None
print(f"Wrote {len(backend.list())} files to {out_dir}")
```

## Agent Skills

The [complete skill catalog](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/skills/README.md) routes coding agents to tested,
task-specific workflows for capturing a first dataset, planning a full capture,
and adding writers or backends. `skills/` is the canonical source; `.agents/skills` and
`.claude/skills` are read-only aliases.

## Releases

Published changes are recorded in [CHANGELOG.md](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/CHANGELOG.md). The package
uses [Semantic Versioning](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/VERSIONING.md); install a specific release with
`pip install "ovannotators==<version>"`.

Compatibility:

- Python 3.11 or newer
- Linux x86_64, Linux aarch64, or Windows x86_64
- `ovrtx` 0.5.x with `ovstage` 0.2.x, 0.6.x with 0.3.x, or 0.7.x with 0.4.x
  for renderer-backed workflows
- CUDA 12 and a compatible NVIDIA driver for GPU image and video codecs

## Support

Use [GitHub Issues](https://github.com/NVIDIA-Omniverse/ovannotators/issues) for
questions, bug reports, feature suggestions, and usage discussions. Report
security vulnerabilities through the private channels in [SECURITY.md](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/SECURITY.md),
not through a public issue.

## Contributing

The GitHub repository is a read-only public mirror and cannot merge external
pull requests. Read [CONTRIBUTING.md](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/CONTRIBUTING.md) before proposing a change
or filing an implementation request.

## License

`ovannotators` is licensed under the [Apache License 2.0](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/LICENSE). Third-party
attributions are listed in [NOTICE-3rdparty.txt](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/NOTICE-3rdparty.txt). Security
reporting requirements are in [SECURITY.md](https://github.com/NVIDIA-Omniverse/ovannotators/blob/main/SECURITY.md).

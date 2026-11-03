# Installation

## Requirements

- Python 3.11 or newer.
- Target wheel platforms: Linux x86_64, Linux aarch64, and Windows x86_64.
- Core install dependencies: [NumPy](https://numpy.org/),
  [Pillow](https://python-pillow.github.io/),
  [OpenEXR](https://openexr.com/), and [Warp](https://nvidia.github.io/warp/)
  (`warp-lang`).
- The `gpu` and `video` codec extras require an NVIDIA GPU with compatible
  [CUDA 12](https://developer.nvidia.com/cuda-toolkit) and driver support (see
  the extras table below).

## Install

```bash
pip install ovannotators
```

For local development from the repo root, use `uv`:

```bash
uv sync                 # core
uv sync --extra <name>  # add an optional extra (see below)
```

## Install extras

Optional extras (add to your own project with `uv add "ovannotators[<name>]"`
or `pip install "ovannotators[<name>]"`):

| Extra | What it adds | Use when you need |
| --- | --- | --- |
| `s3` | S3 backend support ([boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)) | Writing datasets directly to [Amazon S3](https://aws.amazon.com/s3/) |
| `ovstorage` | [ovstorage](https://github.com/NVIDIA-Omniverse/ovstorage) Stack adapter | Writing through ovstorage cloud, local, or cached routes |
| `lerobot` | [LeRobot](https://huggingface.co/docs/lerobot/index) writer support ([PyArrow](https://arrow.apache.org/docs/python/)) | Writing legacy Hugging Face LeRobot datasets, or the `groot_lerobot_v2` profile (which also needs `video` and an NVIDIA GPU for NVENC) |
| `cosmos-multiview` | Cosmos 3.5 tables (`pylance==11.0.0`, `pyarrow==25.0.1`) | Candidate registration and timestamp-aligned materialization |
| `umi` | Zarr v2 and JPEG-XL codecs ([Zarr](https://zarr.readthedocs.io/), `zarr<3`) | Writing the `umi_replay_buffer` profile |
| `gpu` | GPU image codec support ([nvImageCodec](https://docs.nvidia.com/cuda/nvimagecodec/), CUDA 12) | GPU-accelerated image encode/decode |
| `video` | Video codec support ([PyNvVideoCodec](https://docs.nvidia.com/video-technologies/pynvvideocodec/index.html), [NVENC](https://developer.nvidia.com/video-codec-sdk)) | Video-emitting writers (needs an NVIDIA GPU) |
| `all-cpu` | `s3`, `ovstorage`, `lerobot`, `cosmos-multiview`, and `umi` | Full CPU-only runtime feature set (no GPU) |
| `all-gpu` | `all-cpu` plus `gpu` and `video` | Full GPU runtime feature set |
| `all` | Alias for `all-gpu` plus test tooling (implies GPU/CUDA) | Everything optional; use `all-cpu` for GPU-free |

GPU codec extras require compatible CUDA 12, driver, and GPU support.

Development and test tooling ship in the `test`, `dev-cpu`, and `dev` extras;
see `pyproject.toml` for their exact contents.

### ovstorage connections

Installing `ovannotators[ovstorage]` adds the SDK adapter. The ovstorage Linux
wheels require glibc 2.34 or later (for example, Ubuntu 22.04 or RHEL 9);
`all-cpu` includes this extra. Configure and authenticate an ovstorage Stack
for the destination before using it. The
[OvstorageBackend guide](docs/public/api.rst#ovstoragebackend) links the
upstream Python tutorial, Stack configuration, backend authentication, and
Omniverse Storage Service instructions.

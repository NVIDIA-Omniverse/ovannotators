# Installation

## Requirements

- Python 3.10 or newer.
- Target wheel platforms: Linux x86_64, Linux aarch64, and Windows x86_64.
- Core install dependencies: NumPy, Pillow, OpenEXR, and Warp (`warp-lang`).
- The `gpu` and `video` codec extras require an NVIDIA GPU with compatible
  CUDA 12 and driver support (see the extras table below).

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
| `s3` | S3 backend support (`boto3`) | Writing datasets directly to S3 |
| `storage-api` | HTTP storage backend support (`httpx`) | Writing datasets to HTTP storage APIs |
| `lerobot` | LeRobot writer support (`pyarrow`) | Writing HuggingFace LeRobot datasets |
| `gpu` | GPU image codec support (`nvidia-nvimgcodec-cu12`, CUDA 12) | GPU-accelerated image encode/decode |
| `video` | Video codec support (`PyNvVideoCodec`, NVENC) | Video-emitting writers (needs an NVIDIA GPU) |
| `all-cpu` | `s3`, `storage-api`, and `lerobot` | Full CPU-only runtime feature set (no GPU) |
| `all-gpu` | `all-cpu` plus `gpu` and `video` | Full GPU runtime feature set |
| `all` | Alias for `all-gpu` (implies GPU/CUDA) | Everything optional; use `all-cpu` for GPU-free |

GPU codec extras require compatible CUDA 12, driver, and GPU support.

Development and test tooling ship in the `test`, `dev-cpu`, and `dev` extras;
see `pyproject.toml` for their exact contents.

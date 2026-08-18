# ovannotators — standalone synthetic-data annotation library that turns renderer output into ML-ready datasets

> Convert self-describing renderer output (RGB, depth, segmentation, bounding boxes, normals, camera params) into common ML dataset formats — COCO, KITTI, YOLO, BOP, LeRobot, Cosmos-Transfer — in a few calls, running outside the renderer process. Built on NumPy and Warp, and pairs with USD-native renderers like [`ovrtx`](https://github.com/NVIDIA-Omniverse/ovrtx).
>
> *Pre-release (v0.1.0, Python 3.10+). APIs may change before 1.0.*

---

## 1. What is ovannotators?

<!-- DOCS:description:start -->
A standalone library that turns self-describing renderer and simulation output
into ML-ready dataset annotations.

It converts renderer output (RGB, depth, segmentation, bounding boxes, normals,
camera params) into standardized dataset formats and runs outside the renderer
process.
<!-- DOCS:description:end -->

Its Python API reads each frame's render variables as NumPy/Warp arrays and
writes datasets through pluggable storage backends — disk, S3, or an HTTP
storage API. It consumes the output of renderers like `ovrtx` and slots in as a
peer in the USD-native `ov` library family.

---

## 2. What functionalities are available, and who are the target users?

**What you can do with it:**

<!-- DOCS:capabilities:start -->
- **Write common dataset formats out of the box.** Object detection /
  segmentation (MS-COCO, YOLO), autonomous-driving (KITTI), 6-DoF pose (BOP),
  robot learning (LeRobot), video world-model conditioning (Cosmos-Transfer),
  and a flat-layout `BasicWriter`.
- **Register custom output formats.** Add your own writer via the writer
  registry or entry points — or use the included agent skill to go from dataset
  spec to a custom writer.
- **Enable acceleration and backends as opt-in extras.** GPU image codecs, video
  codec support, and storage backends (S3/HTTP API) install on demand, and
  additional storage backends can be registered for more flexibility.
<!-- DOCS:capabilities:end -->

Optional extras (enable in this repo with `uv sync --extra <name>`; add to your
own project with `uv add "ovannotators[<name>]"` or `pip install "ovannotators[<name>]"`):

<!-- DOCS:extras:start -->
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
<!-- DOCS:extras:end -->

**Who benefits:**

- **Synthetic-data-generation engineers** (robotics, perception, autonomy, digital twins) — turn renderer outputs into COCO/KITTI/YOLO/BOP datasets without wrangling raw image buffers or dataset-format details.
- **Robot-learning teams** — emit HuggingFace LeRobot episodes and Cosmos-Transfer conditioning video from the same render loop.
- **AI coding agents** (and the developers using them) — go from dataset spec to a custom writer or storage backend via the shipped `new-writer` and `new-backend` skills.

---

## 3. Documentation and reference links

- **User Guide & Tutorials:** [`docs/public/tutorials.rst`](docs/public/tutorials.rst)
- **API Reference (Python):** [`docs/public/api.rst`](docs/public/api.rst)
- **Minimal example:** [`examples/minimal_capture.py`](examples/minimal_capture.py)
- **Agent guide:** [`AGENTS.md`](AGENTS.md)
- **Skills for AI coding agents:** [`skills/`](skills/)
- **Compatible renderer (ovrtx):** <https://github.com/NVIDIA-Omniverse/ovrtx>
- **Installation notes:** [`INSTALL.md`](INSTALL.md)
- **Contributing & support:** [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

## 4. System requirements

- Python **3.10+**
- **Linux** (x86_64, aarch64) or **Windows** (x86_64)
- Core dependencies: NumPy, Pillow, OpenEXR, Warp — runs CPU-only by default
- **NVIDIA GPU + CUDA 12** required for the `gpu` (nvImageCodec) and `video` (PyNvVideoCodec/NVENC) extras
- Current release: **ovannotators 0.1.0**

See [`INSTALL.md`](INSTALL.md) for platform and install notes.

---

## 5. Licensing

- **Source code:** Apache-2.0 License — permissive, free for commercial and non-commercial use. See [`LICENSE`](LICENSE).
- Open source, but **not open to external code contributions** — see [`CONTRIBUTING.md`](CONTRIBUTING.md).
- Versioning follows Semantic Versioning — see [`VERSIONING.md`](VERSIONING.md).

---

*ovannotators · ground-truth dataset annotation · Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.*

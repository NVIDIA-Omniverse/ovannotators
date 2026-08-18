# ovannotators

A standalone Python library that handles data parsing and data writing

## Philosophy

- **Agent-First**
  - Easy for agents to discover functionality through namespaced, self-documenting APIs
  - Easy for agents to use without error via consistent patterns and sensible defaults
- **Well Tested** — Comprehensive unit tests covering edge cases and regressions
- **Performant** — Use threaded operations when possible for I/O throughput
- **Extensible** — Projects and agents can register new capabilities to any namespace without modifying this library
- **Minimal Dependencies**
  - [NumPy](https://numpy.org) — array operations and distributions

## Library Goals

- Provide agent-friendly tools to take data from multiple renderers and write it in various formats
    - i.e. this should be an intermediate layer beween the renderers and the written data
- Initial support should include the ovrtx renderer
- Output structure is well defined
- Output can be written to different destinations like disk, S3, Azure, etc.
- Provide agent skills that cover how to effectively use the library for getting data, and writing it
- Library should be fool-proof for agents to use

## Methodology

- Should consist of 3 pieces:
    - **RendererOutput**: A renderer-agnostic protocol for extracting AOV data. Each renderer (ovrtx, srtx, etc.) provides its own implementation that handles renderer-specific extraction and AOV interpretation. Initial implementation: `ovrtx.py`
    - **Writer**: Taking the renderer-agnostic data and parsing it into a useable format
    - **Backend**: Taking the parsed data and saving it to a file in a single or multiple destinations
- RendererOutput implementations handle all renderer-specific logic (DLPack mapping, dtype normalization, struct extraction). Writers receive clean, renderer-agnostic numpy arrays.
- The `ovrtx.py` implementation gets data from the AOVs in [ovrtx](https://github.com/nvidia-omniverse/ovrtx). Future renderers (e.g., srtx) add their own implementation of the same protocol.
- "Writers" structure and parse the data before it is written
- A "Backend" takes the structured data from the "writers" and writes that data to different destinations like the disk or an S3 bucket
- Each layer in the pipeline should be extensible to allow for different or additional versions of each in the future

## Library Functionality

- **RendererOutput** — Renderer-agnostic protocol. Initial implementation (`ovrtx.py`) supports all AOVs that [ovrtx](https://github.com/nvidia-omniverse/ovrtx) currently provides. Additional renderers (srtx, etc.) can be added as new modules implementing the same protocol.
- **Dataset Writers** — Initial support should include the following output formats
    - Basic
    - BOP
    - KITTI
    - LeRobot
    - MS-COCO
    - YOLO
- **Backends** — Initial support should be for writing to disk, storageAPI and S3 buckets.

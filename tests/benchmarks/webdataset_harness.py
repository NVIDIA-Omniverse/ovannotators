# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare loose files with WebDataset shards built from the same encoded samples.

The pytest-benchmark module ``test_bench_webdataset.py`` uses this harness with
small defaults. For a fuller single-machine report, run::

    PYTHONPATH=src python -m tests.benchmarks.webdataset_harness \\
        --dataset-mib 1024 --shard-sizes "16 MiB,128 MiB,512 MiB"

Each layout phase runs in a fresh spawned process so its peak RSS is isolated.
Results describe one machine, cache state and reader configuration; they are
not a general claim about either layout.
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing
import os
import platform
import queue as queue_module
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import tracemalloc
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ovannotators import SampleFiles, WebDatasetPackager, WebDatasetPackagerConfig
from tests.webdataset_matrix import RecordingStorage

MIB = 1024**2
DEFAULT_SHARD_SIZES = "16 MiB,128 MiB"
DEFAULT_DATASET_MIB = 48
READER_CONFIG = (
    "single process; warm page cache after one unmeasured pass; raw bytes, no "
    "decoding; tarfile: tarfile.open(path) streaming every member; webdataset: "
    "wds.WebDataset(urls, shardshuffle=False) with default handlers; loose: "
    "recursive scandir listing then one open/read per file"
)


@dataclass(frozen=True)
class UnitSpec:
    """Encoded sample shape for one sample unit.

    Attributes:
        name: ``frame`` (one small frame per sample) or ``episode`` (one
            larger multi-frame sample).
        frames_per_episode: Samples per packaged episode.
        side: Square image side in pixels.
        stacked_frames: Frames stacked into one ``frames.npy`` component
            (episode unit only).
    """

    name: str
    frames_per_episode: int
    side: int
    stacked_frames: int = 0


UNITS = {
    "frame": UnitSpec("frame", frames_per_episode=32, side=256),
    "episode": UnitSpec("episode", frames_per_episode=1, side=256, stacked_frames=32),
}


@dataclass
class Corpus:
    """Loose encoded samples shared by every layout.

    Attributes:
        root: Loose dataset root.
        episodes: Episode ID mapped to its ordered samples.
        files: Number of loose component files.
        bytes: Total encoded payload bytes.
        write_requests: Storage ``write`` calls used to publish loose files.
    """

    root: Path
    episodes: dict[str, list[SampleFiles]]
    files: int
    bytes: int
    write_requests: int

    @property
    def samples(self) -> int:
        return sum(len(samples) for samples in self.episodes.values())


def hardware() -> dict[str, Any]:
    """Describe the measuring machine."""
    info = {
        "machine": platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cpu": platform.processor() or platform.machine(),
        "memory_gib": None,
    }
    if sys.platform == "darwin":
        for key, name in (("cpu", "machdep.cpu.brand_string"), ("mem", "hw.memsize")):
            try:
                value = subprocess.run(
                    ["sysctl", "-n", name],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10,
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                continue
            if key == "cpu":
                info["cpu"] = value
            else:
                info["memory_gib"] = round(int(value) / 1024**3, 1)
        info["system"] = f"macOS {platform.mac_ver()[0]}"
    elif hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            info["memory_gib"] = round(pages / 1024**3, 1)
        except (ValueError, OSError):
            pass
    try:
        import webdataset

        info["webdataset"] = webdataset.__version__
    except (ImportError, AttributeError):
        info["webdataset"] = "unknown"
    return info


def _encoded_pool(spec: UnitSpec, count: int = 8) -> list[dict[str, bytes]]:
    from PIL import Image

    rng = np.random.default_rng(0)
    pool = []
    for _ in range(count):
        components = {}
        if spec.stacked_frames:
            frames = rng.integers(
                0, 256, (spec.stacked_frames, spec.side, spec.side, 3), dtype=np.uint8
            )
            buffer = io.BytesIO()
            np.save(buffer, frames)
            components["frames.npy"] = buffer.getvalue()
        else:
            image = rng.integers(0, 256, (spec.side, spec.side, 3), dtype=np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="PNG")
            components["rgb.png"] = buffer.getvalue()
            depth = rng.uniform(0.1, 50.0, (spec.side // 2, spec.side // 2))
            buffer = io.BytesIO()
            np.save(buffer, depth.astype(np.float32))
            components["depth.npy"] = buffer.getvalue()
        pool.append(components)
    return pool


def build_corpus(root: Path, unit: str, dataset_mib: float) -> Corpus:
    """Write a loose dataset of about ``dataset_mib`` encoded MiB."""
    spec = UNITS[unit]
    pool = _encoded_pool(spec)
    per_sample = sum(len(b) for b in pool[0].values())
    samples = max(1, round(dataset_mib * MIB / per_sample))
    storage = RecordingStorage(root)
    episodes: dict[str, list[SampleFiles]] = {}
    files = total = 0
    for index in range(samples):
        episode = f"{index // spec.frames_per_episode:05d}"
        key = index % spec.frames_per_episode
        components = dict(pool[index % len(pool)])
        components["json"] = json.dumps(
            {"episode": episode, "sample": key, "unit": unit}
        ).encode()
        paths = {}
        for suffix, payload in components.items():
            relative = f"episode-{episode}/sample-{key}.{suffix}"
            storage.write(relative, payload)
            paths[suffix] = root / relative
            files += 1
            total += len(payload)
        episodes.setdefault(episode, []).append(SampleFiles(key, paths, {"unit": unit}))
    return Corpus(root, episodes, files, total, storage.calls["write"])


def _tree_bytes(root: Path) -> int:
    total = 0
    for directory, _, names in os.walk(root):
        for name in names:
            try:
                total += os.stat(os.path.join(directory, name)).st_size
            except FileNotFoundError:
                pass
    return total


class _TempDiskProbe:
    """Track peak bytes under a temporary root by polling and at each publish."""

    def __init__(self, root: Path, interval: float = 0.02) -> None:
        self.root = root
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def sample(self) -> None:
        self.peak = max(self.peak, _tree_bytes(self.root))

    def _poll(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def __enter__(self) -> _TempDiskProbe:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


class _ProbedStorage(RecordingStorage):
    def __init__(self, root: Path, probe: _TempDiskProbe | None) -> None:
        super().__init__(root)
        self.probe = probe

    def write(self, path: str, data: bytes) -> None:
        if self.probe is not None:
            self.probe.sample()
        super().write(path, data)


def package(
    corpus: Corpus, destination: Path, shard_size: str, temp_root: Path | None = None
) -> dict[str, Any]:
    """Package the corpus; return shard paths, request counts and temp peak."""
    if destination.exists():
        shutil.rmtree(destination)
    probe = None
    previous = tempfile.tempdir
    if temp_root is not None:
        temp_root.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(temp_root)
        probe = _TempDiskProbe(temp_root)
    storage = _ProbedStorage(destination, probe)
    try:
        with probe if probe is not None else nullcontext():
            with WebDatasetPackager(
                storage,
                WebDatasetPackagerConfig(
                    max_shard_bytes=shard_size, dataset_id="bench"
                ),
            ) as packager:
                for episode, samples in corpus.episodes.items():
                    packager.pack_episode(episode, samples)
    finally:
        tempfile.tempdir = previous
    manifest = json.loads((destination / "bench/manifest.json").read_text())
    return {
        "shards": [str(destination / s["path"]) for s in manifest["shards"]],
        "shard_bytes": sum(s["bytes"] for s in manifest["shards"]),
        "requests": dict(storage.calls),
        "temp_peak_bytes": probe.peak if probe is not None else None,
    }


def read_loose(root: Path) -> dict[str, int]:
    """List and read every loose file; return sample, byte and request counts."""
    paths = []
    stack = [root]
    while stack:
        with os.scandir(stack.pop()) as entries:
            for entry in entries:
                if entry.is_dir():
                    stack.append(Path(entry.path))
                else:
                    paths.append(entry.path)
    paths.sort()
    samples = set()
    total = 0
    for path in paths:
        with open(path, "rb") as stream:
            total += len(stream.read())
        name = Path(path)
        samples.add((name.parent.name, name.name.split(".", 1)[0]))
    return {"samples": len(samples), "bytes": total, "objects_opened": len(paths)}


def read_tarfile(shards: list[str]) -> dict[str, int]:
    """Stream every member of every shard with ``tarfile``."""
    samples = set()
    total = 0
    for shard in shards:
        with tarfile.open(shard, mode="r|") as archive:
            for member in archive:
                total += len(archive.extractfile(member).read())
                samples.add(member.name.split(".", 1)[0])
    return {"samples": len(samples), "bytes": total, "objects_opened": len(shards)}


def read_webdataset(shards: list[str]) -> dict[str, int]:
    """Iterate raw samples with the official WebDataset reader."""
    import webdataset as wds

    samples = total = 0
    for sample in wds.WebDataset(shards, shardshuffle=False):
        samples += 1
        total += sum(len(v) for k, v in sample.items() if not k.startswith("__"))
    return {"samples": samples, "bytes": total, "objects_opened": len(shards)}


def measure(function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run once for time/CPU and once under tracemalloc for peak allocations."""
    wall = time.perf_counter()
    cpu = time.process_time()
    result = function()
    result["wall_s"] = time.perf_counter() - wall
    result["cpu_s"] = time.process_time() - cpu
    tracemalloc.start()
    try:
        function()
        result["py_peak_bytes"] = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return result


def _maxrss_bytes() -> int:
    try:
        import resource  # POSIX only; the harness must still import on Windows
    except ImportError:
        return 0
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def _phase(queue: Any, name: str, kwargs: dict[str, Any]) -> None:
    baseline = _maxrss_bytes()
    if name == "read_loose":
        read_loose(Path(kwargs["root"]))  # warm the page cache
        result = measure(lambda: read_loose(Path(kwargs["root"])))
    elif name == "read_tarfile":
        read_tarfile(kwargs["shards"])
        result = measure(lambda: read_tarfile(kwargs["shards"]))
    elif name == "read_webdataset":
        read_webdataset(kwargs["shards"])
        result = measure(lambda: read_webdataset(kwargs["shards"]))
    else:
        corpus = kwargs["corpus"]
        destination = Path(kwargs["destination"])
        shard_size = kwargs["shard_size"]
        # Time without the temporary-disk probe, then probe in a separate run.
        result = measure(lambda: package(corpus, destination, shard_size))
        probed = package(corpus, destination, shard_size, Path(kwargs["temp_root"]))
        result["temp_peak_bytes"] = probed["temp_peak_bytes"]
    result["rss_baseline_bytes"] = baseline
    result["rss_peak_bytes"] = _maxrss_bytes()
    queue.put(result)


PHASE_TIMEOUT_S = 3600.0
_POLL_S = 0.5


def run_isolated(
    name: str, *, timeout_s: float = PHASE_TIMEOUT_S, **kwargs: Any
) -> dict[str, Any]:
    """Run one phase in a spawned process so its peak RSS is isolated.

    Raises ``RuntimeError`` if the phase process dies before reporting a result
    or does not report within ``timeout_s`` seconds.
    """
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_phase, args=(queue, name, kwargs))
    process.start()
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            try:
                result = queue.get(timeout=_POLL_S)
                break
            except queue_module.Empty:
                pass
            if not process.is_alive():
                # The result may have been flushed just before the process exited.
                try:
                    result = queue.get(timeout=_POLL_S)
                    break
                except queue_module.Empty:
                    process.join()
                    raise RuntimeError(
                        f"benchmark phase {name} exited with code "
                        f"{process.exitcode} before reporting a result"
                    ) from None
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"benchmark phase {name} did not report within {timeout_s:g} s"
                )
        process.join(timeout=30)  # let a reported phase exit on its own
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        queue.close()
    if process.exitcode:
        raise RuntimeError(f"benchmark phase {name} failed")
    return result


def _mib(value: float | None) -> str:
    return "-" if value is None else f"{value / MIB:,.1f}"


def report(work: Path, dataset_mib: float, shard_sizes: list[str]) -> str:
    """Run every unit, layout and shard size; return a Markdown report."""
    lines = [
        "Single-machine CPU measurement; not a general claim about either layout.",
        "",
        "Hardware: " + json.dumps(hardware()),
        f"Reader configuration: {READER_CONFIG}.",
        "Requests: storage calls counted by a recording backend while publishing; "
        "reads count objects opened (one GET each on an object store).",
        "Peak RSS is the spawned phase process maximum (includes interpreter and "
        "imports, shown as baseline); Python peak is tracemalloc.",
        "",
        "| unit | layout | shard | samples | objects | samples/s | MiB/s | CPU s "
        "| RSS peak / base MiB | Py peak MiB | temp peak MiB | publish requests |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for unit in UNITS:
        corpus = build_corpus(work / unit / "loose", unit, dataset_mib)

        def row(layout, shard, result, objects, publish, temp=None):
            rate = result["samples"] / result["wall_s"]
            throughput = result["bytes"] / MIB / result["wall_s"]
            lines.append(
                f"| {unit} | {layout} | {shard} | {result['samples']} | {objects} | "
                f"{rate:,.0f} | {throughput:,.0f} | {result['cpu_s']:.2f} | "
                f"{_mib(result['rss_peak_bytes'])} / "
                f"{_mib(result['rss_baseline_bytes'])} | "
                f"{_mib(result['py_peak_bytes'])} | {_mib(temp)} | {publish} |"
            )

        loose = run_isolated("read_loose", root=str(corpus.root))
        assert loose["samples"] == corpus.samples and loose["bytes"] == corpus.bytes
        row("loose read", "-", loose, corpus.files, f"write={corpus.write_requests}")
        for shard_size in shard_sizes:
            destination = work / unit / f"shards-{shard_size.replace(' ', '')}"
            packed = run_isolated(
                "package",
                corpus=corpus,
                destination=str(destination),
                temp_root=str(work / "tmp"),
                shard_size=shard_size,
            )
            requests = ", ".join(
                f"{k}={v}" for k, v in sorted(packed["requests"].items())
            )
            packed["samples"] = corpus.samples
            packed["bytes"] = corpus.bytes
            count = len(packed["shards"])
            row(
                "package",
                shard_size,
                packed,
                count + 1,
                requests,
                packed["temp_peak_bytes"],
            )
            for layout, phase in (
                ("tarfile read", "read_tarfile"),
                ("wds read", "read_webdataset"),
            ):
                result = run_isolated(phase, shards=packed["shards"])
                assert result["samples"] == corpus.samples
                row(layout, shard_size, result, count, "-")
            shutil.rmtree(destination)
    return "\n".join(lines)


def settings_from_env() -> tuple[float, list[str]]:
    """Read benchmark sizes from ``OVANNOTATORS_WDS_BENCH_*`` variables."""
    dataset = float(
        os.environ.get("OVANNOTATORS_WDS_BENCH_DATASET_MIB", DEFAULT_DATASET_MIB)
    )
    sizes = os.environ.get("OVANNOTATORS_WDS_BENCH_SHARD_SIZES", DEFAULT_SHARD_SIZES)
    return dataset, [size.strip() for size in sizes.split(",") if size.strip()]


def main(argv: list[str] | None = None) -> None:
    """Print a Markdown loose-versus-shard report."""
    dataset, sizes = settings_from_env()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-mib", type=float, default=dataset)
    parser.add_argument("--shard-sizes", default=",".join(sizes))
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args(argv)
    shard_sizes = [s.strip() for s in args.shard_sizes.split(",") if s.strip()]
    with tempfile.TemporaryDirectory(prefix="wds-bench-", dir=args.work_dir) as work:
        print(report(Path(work), args.dataset_mib, shard_sizes))


if __name__ == "__main__":
    main()

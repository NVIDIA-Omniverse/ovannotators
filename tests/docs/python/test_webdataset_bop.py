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

"""Executable BOP WebDataset capture and reader example."""

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.cpu_only]


def test_bop_webdataset_capture_and_read(tmp_path, sample_render_vars):
    output_dir = tmp_path / "capture"
    episodes = [("scene-a", [sample_render_vars, sample_render_vars])]
    # [snippet:doc-webdataset-bop-capture]
    from ovannotators import DatasetCapture
    from ovannotators.backends import (
        DiskBackend,
        DiskBackendConfig,
        WebDatasetPackager,
        WebDatasetPackagerConfig,
    )
    from ovannotators.writers import BOPWriter, BOPWriterConfig

    storage = DiskBackend(DiskBackendConfig(root_dir=output_dir))
    packager = WebDatasetPackager(
        storage, WebDatasetPackagerConfig(dataset_id="bop-run")
    )
    writer = BOPWriter(BOPWriterConfig(scene_id=1, depth_scale=1.0))
    with DatasetCapture(writer=writer, packager=packager) as dataset:
        for episode_id, frames in episodes:
            with dataset.episode(episode_id) as episode:
                for frame_index, render_vars in enumerate(frames):
                    episode.write_frame(frame_index, render_vars)
    # [/snippet:doc-webdataset-bop-capture]

    shard_urls = [
        f"file:{(output_dir / 'bop-run/shard-000000.tar').resolve().as_posix()}"
    ]
    # [snippet:doc-webdataset-bop-read]
    import io

    import numpy as np
    import webdataset as wds
    from PIL import Image

    def png(payload):  # keeps 16-bit depth as uint16
        return np.asarray(Image.open(io.BytesIO(payload)))

    for sample in wds.WebDataset(shard_urls, shardshuffle=False).decode():
        record = sample["json"]
        camera = record["scene_camera"]  # cam_K, cam_R_w2c, cam_t_w2c (mm)
        depth_mm = png(sample["depth.png"]) * camera["depth_scale"]
        print(sample["__key__"], depth_mm.shape, depth_mm.min(), depth_mm.max())
        for pose, mask in zip(record["scene_gt"], record["masks"]):
            visible = png(sample[mask["mask_visib"]]) if mask["mask_visib"] else None
            print(pose["obj_id"], pose["cam_t_m2c"], visible is not None)
    # [/snippet:doc-webdataset-bop-read]
    assert png(sample["depth.png"]).dtype == np.uint16
    assert depth_mm.shape == (480, 640)
    assert [m["obj_id"] for m in record["masks"]] == [1, 2]
    assert visible.shape == (480, 640)
    assert record["scene_id"] == 1

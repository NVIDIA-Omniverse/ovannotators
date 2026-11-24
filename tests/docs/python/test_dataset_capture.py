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

"""CPU-only capture examples for the ``ovannotators-dataset-quickstart`` and ``ovannotators-dataset-capture`` skills.

Each snippet is self-contained so an agent can copy it into a script that runs
against ``pip install ovannotators`` without a renderer or GPU.
"""

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.cpu_only]


def test_discover_writer_inputs(capsys):
    # [snippet:doc-capture-discover]
    from ovannotators import (
        ProfileWriterRegistry,
        WriterRegistry,
        describe_requirements,
        list_profiles,
    )

    print(WriterRegistry.list_names())  # frame writers: ['basic', 'bop', 'coco', ...]
    print(ProfileWriterRegistry.list_names())  # episode writers: ['lerobot_v3', ...]
    print(list_profiles("coco"))  # ['detection', 'instance_segmentation', 'panoptic']
    requirements = describe_requirements("coco", profile="detection")
    print(sorted(requirements.render_vars))  # inputs every frame must carry
    print(sorted(requirements.recommended_render_vars))  # inputs that add labels
    # [/snippet:doc-capture-discover]

    out = capsys.readouterr().out
    assert "'coco'" in out
    assert "'lerobot_v3'" in out
    assert "['LdrColor', 'SemanticBoundingBox2DTight']" in out
    assert "['SemanticIdMap']" in out


def test_quickstart_coco_cpu(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    # [snippet:doc-quickstart-coco-cpu]
    import numpy as np
    from ovannotators import DiskBackend, DiskBackendConfig, WriterRegistry

    # SemanticBoundingBox2DTight records as ovrtx produces them: 128-bit
    # instance and semantic IDs as four little-endian uint32 words, then the
    # tight pixel extents.
    bbox2d_dtype = np.dtype(
        [
            ("instance_id", "<u4", (4,)),
            ("semantic_id", "<u4", (4,)),
            ("x_min", "<i4"),
            ("y_min", "<i4"),
            ("x_max", "<i4"),
            ("y_max", "<i4"),
        ]
    )

    def synthetic_frame(frame_id: int, rng: np.random.Generator) -> dict:
        """Return one renderer-style frame with two labeled boxes."""
        boxes = np.zeros(2, dtype=bbox2d_dtype)
        for index, (semantic_id, x, y) in enumerate([(1, 10, 20), (2, 120, 60)]):
            boxes[index]["instance_id"] = (index + 1, 0, 0, 0)
            boxes[index]["semantic_id"] = (semantic_id, 0, 0, 0)
            x += 8 * frame_id
            boxes[index]["x_min"], boxes[index]["y_min"] = x, y
            boxes[index]["x_max"], boxes[index]["y_max"] = x + 60, y + 80
        return {
            "LdrColor": rng.integers(0, 256, (240, 320, 4), dtype=np.uint8),
            "SemanticBoundingBox2DTight": boxes,
        }

    out_dir = "_out/dataset_quickstart"
    backend = DiskBackend(DiskBackendConfig(root_dir=out_dir, overwrite=True))
    writer = WriterRegistry.create("coco")
    rng = np.random.default_rng(0)
    for frame_id in range(3):
        writer.write_frame(frame_id, synthetic_frame(frame_id, rng), backend)
    writer.finalize(backend)
    print(f"Wrote {len(backend.list())} files to {out_dir}")
    # [/snippet:doc-quickstart-coco-cpu]

    # [snippet:doc-quickstart-verify-coco]
    import json
    from pathlib import Path

    root = Path("_out/dataset_quickstart")
    coco = json.loads((root / "annotations/instances_train.json").read_text())
    images = sorted(p.name for p in (root / "images").iterdir())
    print(f"{len(coco['images'])} images, {len(coco['annotations'])} annotations")
    print("categories:", [c["name"] for c in coco["categories"]])
    assert [image["file_name"] for image in coco["images"]] == images
    for annotation in coco["annotations"]:
        x, y, width, height = annotation["bbox"]
        assert width > 0 and height > 0
    # [/snippet:doc-quickstart-verify-coco]

    out = capsys.readouterr().out
    assert "Wrote 4 files to _out/dataset_quickstart" in out
    assert "3 images, 6 annotations" in out
    assert len(coco["categories"]) == 2

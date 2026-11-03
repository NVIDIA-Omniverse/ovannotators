# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tested discovery-to-dataset examples."""

import json

import pytest


@pytest.mark.integration
@pytest.mark.cpu_only
def test_coco_requirements_to_dataset(tmp_path, sample_render_vars):
    output_dir = tmp_path
    render_vars = sample_render_vars
    # [snippet:doc-dataset-requirements]
    from ovannotators import describe_requirements, list_profiles
    from ovannotators.backends import DiskBackend, DiskBackendConfig
    from ovannotators.writers import WriterRegistry

    profiles = list_profiles("coco")
    print(profiles)  # ['detection', 'instance_segmentation', 'panoptic']
    requirements = describe_requirements(
        "coco", profile="instance_segmentation", image_format="png"
    )
    writer = WriterRegistry.create("coco", requirements.config)
    backend = DiskBackend(DiskBackendConfig(root_dir=output_dir))
    requested = requirements.render_vars | requirements.recommended_render_vars
    frame = {name: render_vars[name] for name in requested}
    writer.write_frame(0, frame, backend)
    writer.finalize(backend)
    # [/snippet:doc-dataset-requirements]

    assert profiles == ["detection", "instance_segmentation", "panoptic"]
    assert backend.exists("images/000000.png")
    annotations = json.loads(backend.read("annotations/instances_train.json"))
    assert annotations["annotations"]
    assert all(item["segmentation"]["counts"] for item in annotations["annotations"])
    assert {item["name"] for item in annotations["categories"]} == {"car", "person"}


def test_depth_requirements():
    # [snippet:doc-depth-requirements]
    from ovannotators import describe_requirements

    requirements = describe_requirements("cosmos_transfer")
    depth = requirements.representations["depth"]
    print(depth["capture"], depth["storage"], depth["units"])
    # [/snippet:doc-depth-requirements]

    assert depth["capture"] == "DistanceToCameraSD"
    assert "8-bit" in depth["storage"]
    assert depth["units"] == "clip-normalized, not physical depth"

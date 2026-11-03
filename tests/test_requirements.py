# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration-aware requirements discovery and writer contract checks."""

from dataclasses import dataclass

import pytest

import ovannotators as o


@pytest.mark.parametrize(
    "name",
    [
        "basic",
        "coco",
        "yolo",
        "openlabel",
        "kitti",
        "bop",
        "lerobot",
        "cosmos_transfer",
        "cosmos_multiview",
    ],
)
def test_every_public_writer_has_options_and_a_default(name):
    result = o.describe_requirements(name)
    assert result.profile == o.list_profiles(name)[0]
    assert result.config_type is not None
    assert result.options
    assert (result.config is None) == bool(result.required_options)


def test_coco_profiles_and_option_discovery():
    assert o.list_profiles("coco") == ["detection", "instance_segmentation", "panoptic"]
    r = o.describe_requirements("coco", "instance_segmentation", image_format="png")
    assert r.render_vars == {
        "LdrColor",
        "SemanticBoundingBox2DTight",
        "SemanticInstanceSegmentation",
    }
    assert r.recommended_render_vars == {"SemanticIdMap"}
    assert r.minimum_camera_views == 1
    assert r.config.include_segmentation and r.config.image_format == "png"
    assert next(item for item in r.options if item.name == "image_format").choices == (
        "jpg",
        "png",
    )
    assert (
        o.describe_requirements(
            "coco", config=o.COCOWriterConfig(include_segmentation=True)
        ).profile
        == "instance_segmentation"
    )


@pytest.mark.parametrize("profile", ["detection", "instance_segmentation", "panoptic"])
def test_coco_discovery_agrees_with_writer(profile, disk_backend):
    r = o.describe_requirements("coco", profile)
    writer = o.WriterRegistry.create("coco", r.config)
    assert writer.required_render_vars == r.render_vars
    if profile != "detection":
        with pytest.raises(
            o.MissingRenderVarError, match="SemanticInstanceSegmentation"
        ):
            writer.write_frame(
                0, {"LdrColor": None, "SemanticBoundingBox2DTight": None}, disk_backend
            )
        assert disk_backend.list() == []


@pytest.mark.parametrize(
    "name,overrides",
    [
        ("yolo", {"task": "segment"}),
        ("openlabel", {"include_segmentation": True}),
        ("kitti", {"write_depth": True}),
        ("cosmos_transfer", {"conditioning_types": ["canny"], "canny_source": "rgb"}),
        (
            "cosmos_transfer",
            {"conditioning_types": ["canny"], "canny_source": "normals"},
        ),
        ("cosmos_transfer", {"conditioning_types": ["shaded_segmentation"]}),
    ],
)
def test_configured_frame_requirements_agree_with_writer(name, overrides):
    r = o.describe_requirements(name, **overrides)
    writer = o.WriterRegistry.create(name, r.config)
    assert r.render_vars == writer.required_render_vars


def test_canny_only_needs_color(disk_backend, sample_render_vars):
    r = o.describe_requirements("cosmos_transfer", conditioning_types=["canny"])
    assert r.render_vars == {"LdrColor"}
    assert "depth" not in r.representations
    writer = o.WriterRegistry.create("cosmos_transfer", r.config)
    writer.write_frame(0, {"LdrColor": sample_render_vars["LdrColor"]}, disk_backend)
    assert len(writer._rgb_frames) == 1


def test_shaded_missing_normals_fails_before_accumulation(
    disk_backend, sample_render_vars
):
    r = o.describe_requirements(
        "cosmos_transfer", conditioning_types=["shaded_segmentation"]
    )
    writer = o.WriterRegistry.create("cosmos_transfer", r.config)
    with pytest.raises(o.MissingRenderVarError, match="NormalSD"):
        writer.write_frame(
            0,
            {
                key: value
                for key, value in sample_render_vars.items()
                if key != "NormalSD"
            },
            disk_backend,
        )
    assert not writer._rgb_frames


def test_depth_representations_are_explicit():
    assert (
        "8-bit"
        in o.describe_requirements("cosmos_transfer").representations["depth"][
            "storage"
        ]
    )
    assert (
        o.describe_requirements(
            "kitti", write_depth=True, depth_scale=512
        ).representations["depth"]["stored_units_per_metre"]
        == 512
    )


def test_basic_default_and_explicit_selection():
    assert o.describe_requirements("basic").render_vars == {"LdrColor"}
    assert o.describe_requirements("basic", render_vars=["PointCloud"]).render_vars == {
        "PointCloud"
    }
    assert o.describe_requirements("basic", render_vars=None).render_vars == set()


def test_legacy_camera_keys_do_not_claim_independent_views():
    r = o.describe_requirements("lerobot", camera_keys=["front", "wrist"])
    assert r.minimum_camera_views == 1
    assert r.camera_stream_ids == ()
    assert "same incoming image" in r.notes[0]


def test_episode_profiles_do_not_invent_stream_schema():
    assert o.describe_requirements("cosmos_multiview").minimum_camera_views is None


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("missing", {}),
        ("coco", {"profile": "keypoints"}),
        ("coco", {"profile": ""}),
        ("coco", {"profile": "detection", "include_segmentation": True}),
        ("coco", {"config": o.YOLOWriterConfig()}),
        ("coco", {"image_format": "exr"}),
        ("coco", {"made_up": True}),
        ("coco", {"include_segmentation": "false"}),
        ("cosmos_transfer", {"conditioning_types": ["bogus"]}),
        ("cosmos_transfer", {"conditioning_types": "depth"}),
        ("lerobot", {"camera_keys": []}),
        ("lerobot", {"fps": 0}),
    ],
)
def test_invalid_options_fail_clearly(name, kwargs):
    with pytest.raises(o.ConfigurationError):
        o.describe_requirements(name, **kwargs)


def test_discovery_does_not_construct_writers(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("writer construction is forbidden during discovery")

    for name in o.WriterRegistry.list_names():
        monkeypatch.setattr(o.WriterRegistry.get(name), "__init__", fail)
        assert o.describe_requirements(name).writer_name == name
    for name in o.ProfileWriterRegistry.list_names():
        monkeypatch.setattr(o.ProfileWriterRegistry.get(name), "__init__", fail)
        assert o.describe_requirements(name).writer_name == name


def test_defaults_and_input_configs_are_isolated():
    config = {"render_vars": ["LdrColor"]}
    first = o.describe_requirements("basic", config=config)
    first.config.render_vars.append("NormalSD")
    assert config["render_vars"] == ["LdrColor"]
    assert o.describe_requirements("basic").render_vars == {"LdrColor"}


def test_extension_discovery_and_legacy_fallback(monkeypatch):
    @dataclass
    class Config:
        image_format: str = "png"

    class Custom:
        name = "discovery_test"
        required_render_vars = frozenset({"CustomColor"})

    monkeypatch.setitem(o.WriterRegistry._registry, Custom.name, Custom)
    assert o.describe_requirements(Custom.name).minimum_camera_views is None
    assert o.describe_requirements(Custom.name).notes
    o.register_dataset_profiles(
        Custom,
        Config,
        (o.DatasetProfile("default", "Custom format"),),
        lambda profile, values: o.DatasetRequirements(
            Custom.name,
            profile,
            render_vars=Custom.required_render_vars,
            representations={"image": values["image_format"]},
        ),
    )
    assert o.describe_requirements(Custom.name).config.image_format == "png"
    assert (
        o.describe_requirements(Custom.name, image_format="jpg").representations[
            "image"
        ]
        == "jpg"
    )
    with pytest.raises(o.ConfigurationError, match="already registered"):
        o.register_dataset_profiles(
            Custom, Config, (o.DatasetProfile("default", "Custom"),), lambda p, v: None
        )


def test_panoptic_only_config_remains_supported():
    r = o.describe_requirements(
        "coco", config=o.COCOWriterConfig(include_panoptic=True)
    )
    assert r.profile == "panoptic"
    assert not r.config.include_segmentation
    assert "SemanticInstanceSegmentation" in r.render_vars
    options = {item.name: item for item in r.options}
    assert options["include_panoptic"].fixed_by_profile
    assert not options["include_segmentation"].fixed_by_profile
    assert o.describe_requirements("coco", "panoptic").config.include_segmentation


def test_sensor_only_basic_capture_does_not_need_a_camera():
    r = o.describe_requirements("basic", render_vars=["PointCloud"])
    assert r.minimum_camera_views == 0
    assert r.sensor_requirements

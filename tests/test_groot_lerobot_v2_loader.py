# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned Isaac-GR00T dataset, processor, collator, and batch gate."""

from __future__ import annotations

from dataclasses import replace
from importlib import metadata
import json
from unittest.mock import patch

import numpy as np
import pytest

from ovannotators import DiskBackend, DiskBackendConfig, GrootLeRobotV2Writer
from ovannotators.profile_writers.groot_lerobot_v2 import GROOT_RELEASE_COMMIT
from tests.test_groot_lerobot_v2 import _config


def _assert_pinned_groot_install() -> None:
    """Require immutable PEP 610 provenance for the selected GR00T checkout."""
    try:
        distribution = metadata.distribution("gr00t")
    except metadata.PackageNotFoundError:
        pytest.fail("importable gr00t has no installed distribution metadata")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        pytest.fail("gr00t install has no direct_url.json commit provenance")
    direct_url = json.loads(direct_url_text)
    commit_id = direct_url.get("vcs_info", {}).get("commit_id")
    assert commit_id == GROOT_RELEASE_COMMIT, (
        "canonical acceptance requires Isaac-GR00T commit "
        f"{GROOT_RELEASE_COMMIT}, got {commit_id!r}"
    )


class _Tokenizer:
    padding_side = "right"


class _LocalVlmProcessor:
    """Avoid model downloads while retaining the real GR00T preparation path."""

    def __init__(self) -> None:
        self.tokenizer = _Tokenizer()

    def apply_chat_template(self, conversation, **kwargs):
        del conversation, kwargs
        return "<image> pick cube"

    def __call__(self, *, text, images, return_tensors, padding):
        import torch

        assert return_tensors == "pt"
        assert padding is True
        batch_size = len(text)
        return {
            "input_ids": torch.ones((batch_size, 3), dtype=torch.int64),
            "attention_mask": torch.ones((batch_size, 3), dtype=torch.int64),
            "pixel_values": torch.zeros((len(images), 3, 2, 2)),
        }


@pytest.mark.unit
def test_pinned_install_check_requires_exact_pep610_commit(monkeypatch):
    class Distribution:
        def __init__(self, commit_id: str) -> None:
            self.commit_id = commit_id

        def read_text(self, filename: str) -> str:
            assert filename == "direct_url.json"
            return json.dumps({"vcs_info": {"commit_id": self.commit_id}})

    monkeypatch.setattr(
        metadata, "distribution", lambda name: Distribution(GROOT_RELEASE_COMMIT)
    )
    _assert_pinned_groot_install()

    monkeypatch.setattr(
        metadata, "distribution", lambda name: Distribution("unreviewed-commit")
    )
    with pytest.raises(AssertionError, match="canonical acceptance requires"):
        _assert_pinned_groot_install()


@pytest.mark.e2e
@pytest.mark.gpu
def test_pinned_groot_pipeline_produces_real_collated_batch(
    tmp_path, priority_profile_fixture
):
    """A missing selected GR00T/GPU environment skips, never passes conformance."""
    pytest.importorskip(
        "gr00t", reason="pinned Isaac-GR00T n1.7 environment unavailable"
    )
    _assert_pinned_groot_install()

    import torch
    from gr00t.data.dataset.sharded_mixture_dataset import ShardedMixtureDataset
    from gr00t.data.dataset.sharded_single_step_dataset import (
        ShardedSingleStepDataset,
    )
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    from torch.utils.data import DataLoader

    embodiment = EmbodimentTag.NEW_EMBODIMENT
    config = replace(_config(), embodiment=embodiment.value)
    backend = DiskBackend(DiskBackendConfig(tmp_path, overwrite=True))
    writer = GrootLeRobotV2Writer(config)
    source_episode = priority_profile_fixture.episodes[0]
    episode = replace(
        source_episode,
        embodiment=embodiment.value,
        streams=tuple(
            replace(stream, shape=(192, 192, 3))
            if stream.stream_id in ("camera.front", "camera.wrist")
            else stream
            for stream in source_episode.streams
        ),
    )
    writer.begin_episode(episode, backend)
    for sample in priority_profile_fixture.samples[0]:
        observations = dict(sample.observations)
        for stream_id in ("camera.front", "camera.wrist"):
            tensor = observations[stream_id]
            observations[stream_id] = replace(
                tensor,
                data=np.full((192, 192, 3), int(tensor.data[0, 0, 0]), dtype=np.uint8),
            )
        writer.write_sample(replace(sample, observations=observations), backend)
    writer.end_episode(backend)
    writer.finalize(backend)

    modality_config = writer.canonical_modality_config()
    dataset = ShardedSingleStepDataset(
        dataset_path=tmp_path,
        embodiment_tag=embodiment,
        modality_configs=modality_config,
        shard_size=8,
        episode_sampling_rate=1.0,
        seed=0,
    )
    statistics = dataset.get_dataset_statistics()
    local_vlm = _LocalVlmProcessor()
    with patch(
        "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor",
        return_value=local_vlm,
    ):
        processor = Gr00tN1d7Processor(
            modality_configs={embodiment.value: modality_config},
            statistics={embodiment.value: statistics},
            use_percentiles=config.normalization.use_percentiles,
            clip_outliers=config.normalization.clip_outliers,
            image_crop_size=[192, 192],
            image_target_size=[192, 192],
            formalize_language=False,
            model_name="local-conformance-stub",
            max_state_dim=4,
            max_action_dim=4,
            max_action_horizon=4,
            use_relative_action=True,
            embodiment_id_mapping={embodiment.value: 0},
        )

    mixture = ShardedMixtureDataset(
        datasets=[dataset],
        weights=[1.0],
        processor=processor,
        seed=0,
        training=False,
        num_shards_per_epoch=1,
    )
    loader = DataLoader(
        mixture,
        batch_size=1,
        num_workers=0,
        collate_fn=processor.collator,
    )
    batch = next(iter(loader))["inputs"]

    assert {"state", "action", "action_mask", "embodiment_id"} <= set(batch)
    assert {"input_ids", "attention_mask", "pixel_values"} <= set(batch)
    assert batch["state"].shape == (1, 1, 4)
    assert batch["action"].shape == (1, 4, 4)
    assert batch["action_mask"].shape == batch["action"].shape
    expected_mask = torch.zeros((1, 4, 4))
    expected_mask[:, : config.action_horizon, :2] = 1
    torch.testing.assert_close(batch["action_mask"], expected_mask)
    assert batch["embodiment_id"].tolist() == [0]
    assert np.asarray(statistics["relative_action"]["arm_action"]["mean"]).shape == (
        2,
        1,
    )

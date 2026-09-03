# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from physicalai.inference.constants import IMAGES, STATE, TASK, TOKENIZED_PROMPT, TOKENIZED_PROMPT_MASK
from physicalai.inference.manifest import ComponentSpec
from physicalai.inference.component_factory import instantiate_component
from physicalai.inference.preprocessors import MolmoAct2ModelInputs, MolmoAct2Preprocessor
from physicalai.inference.preprocessors.molmoact2_inputs import (
    MolmoAct2InputConfig,
    build_batched_images,
    expand_image_placeholders,
)
from physicalai.inference.preprocessors.molmoact2_image import MolmoAct2ImageProcessor
from physicalai.inference.postprocessors import MolmoAct2Postprocessor


def _prepare(**kwargs) -> MolmoAct2Preprocessor:
    return MolmoAct2Preprocessor(
        image_keys=["top", "wrist"],
        image_size=(28, 28),
        num_state_tokens=4,
        setup_type="tabletop",
        control_mode="joint",
        **kwargs,
    )


def _assemble(**kwargs) -> MolmoAct2ModelInputs:
    params = {
        "max_action_dim": 4,
        "action_dim": 2,
        "bos_token_id": 1,
        "pad_token_id": 0,
        "image_placeholder_token_id": 99,
        "image_start_token_id": 10,
        "image_end_token_id": 12,
        "image_patch_id": 11,
        "image_col_id": 13,
        "low_res_image_start_token_id": 10,
        "image_token_ids": [10, 11, 12, 13],
        "image_size": (28, 28),
        "patch_size": 14,
        "pooling_size": (2, 2),
    }
    params.update(kwargs)
    return MolmoAct2ModelInputs(**params)


def _observation(*, image_count: int = 1) -> dict:
    values = {
        STATE: np.array([[-1.0, 1.0]], dtype=np.float32),
        TASK: ["Task: Pick up."],
    }
    for index, key in enumerate(["top", "wrist"][:image_count]):
        values[f"{IMAGES}.{key}"] = np.full((1, 3, 28, 28), index * 255, dtype=np.uint8)
    return values


class TestMolmoAct2Preprocessor:
    def test_builds_prompt_and_packs_ordered_cameras(self) -> None:
        processor = _prepare()
        result = processor(_observation(image_count=2))

        assert result[IMAGES].shape == (2, 1, 3, 28, 28)
        assert float(result[IMAGES][0].max()) == 0.0
        assert float(result[IMAGES][1].min()) == 1.0
        assert result[TASK][0].startswith("Image 1<|image|>Image 2<|image|>")
        assert "The task is to pick up." in result[TASK][0]
        assert "<state_0><state_3>" in result[TASK][0]

    def test_nested_image_fallback_is_sorted_but_explicit_order_is_preserved(self) -> None:
        images = {
            "wrist": np.full((1, 3, 28, 28), 255, dtype=np.uint8),
            "top": np.zeros((1, 3, 28, 28), dtype=np.uint8),
        }
        inputs = {STATE: np.zeros((1, 2), dtype=np.float32), TASK: "move", IMAGES: images}

        fallback = MolmoAct2Preprocessor(image_keys=[], image_size=(28, 28))(inputs)[IMAGES]
        explicit = MolmoAct2Preprocessor(image_keys=["wrist", "top"], image_size=(28, 28))(inputs)[IMAGES]

        assert float(fallback[0].max()) == 0.0
        assert float(fallback[1].min()) == 1.0
        assert float(explicit[0].min()) == 1.0
        assert float(explicit[1].max()) == 0.0

    def test_accepts_batched_channels_last_camera_frames(self) -> None:
        observation = _observation(image_count=2)
        observation[f"{IMAGES}.top"] = np.zeros((1, 28, 28, 3), dtype=np.uint8)
        observation[f"{IMAGES}.wrist"] = np.full((1, 28, 28, 3), 255, dtype=np.uint8)

        result = _prepare()(observation)

        assert result[IMAGES].shape == (2, 1, 3, 28, 28)
        assert float(result[IMAGES][0].max()) == 0.0
        assert float(result[IMAGES][1].min()) == 1.0

    def test_applies_masked_normalization_and_joint_transform(self) -> None:
        processor = MolmoAct2Preprocessor(
            image_keys=[],
            image_size=(28, 28),
            state_stats={"q01": [0.0, 0.0], "q99": [2.0, 2.0], "mask": [True, False]},
            adapt_to_so101=True,
            joint_signs=[1.0, -1.0],
            joint_offsets=[0.0, 2.0],
        )
        result = processor({
            STATE: np.array([[1.0, 1.0]], dtype=np.float32),
            TASK: "move",
            IMAGES: np.zeros((1, 3, 28, 28), dtype=np.uint8),
        })
        assert "<state_128><state_255>" in result[TASK][0]

    def test_supports_mean_std_normalization(self) -> None:
        processor = MolmoAct2Preprocessor(
            image_keys=[],
            image_size=(28, 28),
            num_state_tokens=4,
            state_stats={"mean": [0.0], "std": [2.0]},
            normalization_mode="MEAN_STD",
        )
        result = processor({
            STATE: np.array([[2.0]], dtype=np.float32),
            TASK: "move",
            IMAGES: np.zeros((1, 3, 28, 28), dtype=np.uint8),
        })

        assert "<state_3>" in result[TASK][0]

    def test_preserves_masked_tokenizer_padding(self) -> None:
        result = _assemble()({
            TOKENIZED_PROMPT: np.array([[99, 5, 0, 0]], dtype=np.int64),
            TOKENIZED_PROMPT_MASK: np.array([[1, 1, 0, 0]], dtype=np.bool_),
            IMAGES: np.zeros((1, 1, 3, 28, 28), dtype=np.float32),
        })

        assert result["input_ids"].shape == (1, 7)
        assert result["attention_mask"].shape == (1, 7)
        assert int(result["attention_mask"].sum()) == 5

    def test_placeholder_expansion_uses_configured_padding(self) -> None:
        config = MolmoAct2InputConfig(
            pad_token_id=7,
            image_placeholder_token_id=99,
            image_patch_id=11,
            image_start_token_id=10,
            image_end_token_id=12,
        )

        input_ids, attention_mask, _ = expand_image_placeholders(
            config=config,
            input_ids=np.array([[99, 5], [99, 99]], dtype=np.int64),
            attention_mask=np.ones((2, 2), dtype=np.int64),
            image_grids=np.array([[1, 1, 0, 0]] * 3, dtype=np.int64),
        )

        assert input_ids[0].tolist() == [10, 11, 12, 5, 7, 7]
        assert attention_mask[0].tolist() == [1, 1, 1, 1, 0, 0]

    def test_build_batched_images_supports_multi_crop_grids(self) -> None:
        config = MolmoAct2InputConfig(
            pad_token_id=0,
            image_placeholder_token_id=99,
            image_patch_id=11,
            image_start_token_id=10,
            image_end_token_id=12,
        )

        images, pooling = build_batched_images(
            config,
            input_ids=np.array([[10, 11, 12, 10, 11, 12], [10, 11, 12, 10, 11, 12]]),
            pixel_values=np.arange(8, dtype=np.float32).reshape(2, 4, 1),
            image_token_pooling=np.arange(10, dtype=np.int64).reshape(10, 1) % 4,
            image_grids=np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.int64),
            image_num_crops=np.ones(2, dtype=np.int64),
        )

        assert images.shape == (2, 1, 4, 1)
        assert pooling.shape == (2, 5, 1)

    def test_rejects_placeholder_image_mismatch(self) -> None:
        with pytest.raises(ValueError, match="placeholders"):
            _assemble()({
                TOKENIZED_PROMPT: np.array([[99, 99, 5]], dtype=np.int64),
                TOKENIZED_PROMPT_MASK: np.ones((1, 3), dtype=np.bool_),
                IMAGES: np.zeros((1, 1, 3, 28, 28), dtype=np.float32),
            })

    def test_nullable_single_crop_columns_follow_image_setting(self) -> None:
        result = _assemble(use_single_crop_col_tokens=None)({
            TOKENIZED_PROMPT: np.array([[99]], dtype=np.int64),
            TOKENIZED_PROMPT_MASK: np.array([[1]], dtype=np.bool_),
            IMAGES: np.zeros((1, 1, 3, 28, 28), dtype=np.float32),
        })

        assert result["input_ids"].tolist() == [[1, 10, 11, 13, 12]]

    def test_rejects_missing_state(self) -> None:
        with pytest.raises(ValueError, match="state"):
            _prepare()({TASK: ["move"], IMAGES: np.zeros((1, 3, 28, 28), dtype=np.uint8)})

    def test_model_inputs_is_distinct_class(self) -> None:
        assert MolmoAct2ModelInputs is not MolmoAct2Preprocessor


class TestMolmoAct2ImageProcessor:
    def test_resize_mode_matches_patch_layout(self) -> None:
        processor = MolmoAct2ImageProcessor(
            crop_mode="resize",
            size={"height": 28, "width": 28},
            patch_size=14,
            pooling_size=(2, 2),
        )

        result = processor(np.zeros((2, 3, 28, 28), dtype=np.float32))

        assert result["pixel_values"].shape == (2, 4, 588)
        assert result["image_token_pooling"].tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
        assert result["image_grids"].tolist() == [[1, 1, 0, 0], [1, 1, 0, 0]]


class TestMolmoAct2ManifestPipeline:
    def test_processes_observation_and_action(self, monkeypatch) -> None:
        from physicalai.inference.preprocessors.hf_tokenizer import HFTokenizer

        class TransformersTokenizer:
            name_or_path = "allenai/MolmoAct2"
            config = type("Config", (), {"revision": "1dbc166cf8765166998eff31ade2eb64c8a40076"})()

            def __call__(self, tasks, **kwargs):
                del tasks, kwargs
                return {
                    "input_ids": np.array([[154629, 7, 0, 0]], dtype=np.int64),
                    "attention_mask": np.array([[1, 1, 0, 0]], dtype=np.int64),
                }

        monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: TransformersTokenizer())
        specs = [
            ComponentSpec(
                type="molmoact2",
                image_keys=["top"],
                state_stats={"q01": [-1.0, -1.0], "q99": [1.0, 1.0]},
                image_size=(28, 28),
            ),
            ComponentSpec(
                type="hf_tokenizer",
                tokenizer_name="allenai/MolmoAct2",
                revision="1dbc166cf8765166998eff31ade2eb64c8a40076",
                max_token_len=4,
            ),
            ComponentSpec(
                type="molmoact2_inputs",
                max_action_dim=4,
                action_dim=2,
                bos_token_id=1,
                pad_token_id=0,
                image_placeholder_token_id=154629,
                image_start_token_id=154624,
                image_end_token_id=154625,
                image_patch_id=154626,
                image_col_id=154627,
                low_res_image_start_token_id=154628,
                frame_start_token_id=154631,
                frame_end_token_id=154632,
                image_low_res_id=154630,
                image_size=(28, 28),
                patch_size=14,
                pooling_size=(2, 2),
                image_token_ids=[154624, 154625, 154626, 154627, 154628],
            ),
        ]
        values = {
            "state": np.array([[0.0, 0.5]], dtype=np.float32),
            "task": ["pick up the block"],
            "images.top": np.zeros((1, 3, 28, 28), dtype=np.uint8),
        }
        preprocessor = instantiate_component(specs[0])
        assert isinstance(preprocessor, MolmoAct2Preprocessor)
        values = preprocessor(values)

        tokenizer = instantiate_component(specs[1])
        assert isinstance(tokenizer, HFTokenizer)
        values = tokenizer(values)

        model_inputs = instantiate_component(specs[2])
        assert isinstance(model_inputs, MolmoAct2ModelInputs)
        assert model_inputs._layout.frame_start_token_id == 154631
        assert model_inputs._layout.frame_end_token_id == 154632
        assert model_inputs._layout.image_low_res_id == 154630
        values = model_inputs(values)

        assert set(values) == {"input_ids", "attention_mask", "images", "token_pooling", "action_dim_is_pad", "token_type_ids"}
        assert values["images"].shape == (1, 1, 4, 588)
        assert values["action_dim_is_pad"].tolist() == [[False, False, True, True]]

        postprocessor = instantiate_component(
            ComponentSpec(
                type="molmoact2_postprocess",
                action_stats={"q01": [0.0, 0.0], "q99": [2.0, 2.0]},
            ),
        )
        assert isinstance(postprocessor, MolmoAct2Postprocessor)
        result = postprocessor({"action": np.array([[[0.0, 1.0]]], dtype=np.float32)})
        np.testing.assert_allclose(result["action"], np.array([[[1.0, 2.0]]], dtype=np.float32))
